"""Phase 1 orchestrator: Methodology Shadow Mode.

What it does, per character:

1. **After every activity** — builds the problem signature from the activity
   record, asks the policy which methodology *would* have been selected, and
   records that selection. Then it attributes the activity's **objective
   outcome** to that methodology and updates its value/confidence.
2. **After REVIEW** — one budgeted LLM call turns the week's evidence into
   candidate methodologies (`METHOD_PROPOSED`, value 0, confidence 0).

What it never does: change a prompt, a plan, an action, a state delta or an
existing skill number. The switch (`world.cognition.methodology_shadow`,
default off) turns the whole thing into a no-op.

Two conventions worth knowing when reading the events:

- **Attribution is counterfactual.** No method has been adopted yet (that is
  phase 4), so an observed outcome is attributed to the method the policy would
  have picked. Every such event carries
  `attribution="shadow_counterfactual"`, so real adoption evidence can later be
  separated instead of silently mixed in.
- **A method proposed at REVIEW gets no value from that same week.** Value only
  comes from outcomes of *later* activities, which keeps "reflection cannot
  reinforce a method" true in practice: extraction happens after the week it
  analyses, and the policy can only select methods that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.agents.cognition import materializer
from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.methodology_policy import (
    context_signature_from_activity,
    deterministic_rng,
    exploration_rate_for,
    select_methods,
)
from src.agents.cognition.models import Methodology, method_id_for
from src.agents.cognition.prompts import (
    build_method_extraction_prompt,
    format_week_evidence,
    parse_method_extraction,
)
from src.agents.cognition.reward_model import (
    DEFAULT_REWARD_CALIBER,
    SPECIALIZED_CONTEXT_MARGIN,
    SPECIALIZED_MIN_CONTEXT_SAMPLES,
    alternative_caliber,
    baseline_tracker_from_events,
    goal_baseline_tracker_from_events,
    live_reward_for,
    shadow_baseline_for,
    signals_from_activity_record,
    update_context_value,
    update_value,
    weights_for_caliber,
)
from src.utils import get_logger

SHADOW_LOGGER = get_logger("cognition", quiet=True)


@dataclass
class ShadowConfig:
    """`world.cognition` configuration with inert defaults."""

    enabled: bool = False
    max_extract_calls_per_week: int = 1
    # Phase 5: which centred-reward variant the shadow score should use.
    # Phase 5 follow-up (KI-19): which formula moves a value in this run.
    # Directly constructed configs stay inert (`off`); `from_world_config`
    # supplies the run's default.
    reward_caliber: str = "off"
    specialized_margin: float = SPECIALIZED_CONTEXT_MARGIN
    specialized_min_samples: int = SPECIALIZED_MIN_CONTEXT_SAMPLES
    goal_progress_weight: float = 0.0

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "ShadowConfig":
        section = (world_cfg or {}).get("cognition") or {}
        try:
            cap = int(section.get("max_extract_calls_per_week", 1))
        except (TypeError, ValueError):
            cap = 1
        try:
            margin = float(
                section.get("method_specialized_margin", SPECIALIZED_CONTEXT_MARGIN)
            )
        except (TypeError, ValueError):
            margin = SPECIALIZED_CONTEXT_MARGIN
        try:
            min_samples = int(
                section.get(
                    "method_specialized_min_samples", SPECIALIZED_MIN_CONTEXT_SAMPLES
                )
            )
        except (TypeError, ValueError):
            min_samples = SPECIALIZED_MIN_CONTEXT_SAMPLES
        try:
            goal_weight = float(section.get("goal_progress_weight", 0.0))
        except (TypeError, ValueError):
            goal_weight = 0.0
        return ShadowConfig(
            enabled=bool(section.get("methodology_shadow", False)),
            max_extract_calls_per_week=max(0, cap),
            reward_caliber=str(
                section.get("reward_caliber", DEFAULT_REWARD_CALIBER)
                or DEFAULT_REWARD_CALIBER
            ).lower(),
            specialized_margin=margin,
            specialized_min_samples=max(1, min_samples),
            goal_progress_weight=max(0.0, goal_weight),
        )


class MethodologyShadow:
    """Shadow-mode capability tracking for one character."""

    def __init__(
        self,
        *,
        dm,
        clock,
        agent_name: str,
        model: str,
        traits: Optional[Dict[str, float]] = None,
        config: Optional[ShadowConfig] = None,
        language: str = "en",
    ) -> None:
        self.dm = dm
        self.clock = clock
        self.agent_name = agent_name
        self.model = model
        self.traits = traits or {}
        self.config = config or ShadowConfig()
        self.language = language
        self.store = CapabilityEventStore(dm)
        self._methods: Optional[List[Methodology]] = None
        self._view: Optional[Dict[str, Any]] = None
        self._extract_calls: Dict[str, int] = {}

    # -- gating ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # -- candidate methods -------------------------------------------------
    def _capability_view(self) -> Dict[str, Any]:
        if self._view is None:
            self._view = materializer.build_capability_view(self.dm)
        return self._view

    def methods(self) -> List[Methodology]:
        """Current candidate set, folded from the event stream (cached)."""
        if self._methods is None:
            self._methods = materializer.load_methodologies(self._capability_view())
        return self._methods

    def selectable_methods(self) -> List[Methodology]:
        """Candidates that may receive evidence from the *current* week.

        A method proposed by this week's REVIEW is excluded: the design document
        forbids reflection from reinforcing a method, and the cheapest way to
        keep that true is to never let a same-week activity feed it.
        """
        week = self._week_key()
        view_methods = (self._capability_view().get("methodologies") or {})
        out: List[Methodology] = []
        for method in self.methods():
            proposed_week = str(
                (view_methods.get(method.method_id) or {}).get("proposed_week") or ""
            )
            if proposed_week and proposed_week == week:
                continue
            out.append(method)
        return out

    def _invalidate(self) -> None:
        self._methods = None
        self._view = None
        self.store.invalidate()

    def _method_by_id(self, method_id: str) -> Optional[Methodology]:
        for method in self.methods():
            if method.method_id == method_id:
                return method
        return None

    # -- (1) per activity --------------------------------------------------
    def observe_activity(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Shadow-select a method for a finished activity and score its outcome.

        Returns the value-update record when one was written, else None.
        """
        if not self.enabled:
            return None

        activity_id = str(record.get("activity_id") or "")
        candidates = self.selectable_methods()
        if not candidates:
            return None

        signals = signals_from_activity_record(record)
        primary_skill = self._dominant_skill(record)
        signature = context_signature_from_activity(record, skill_id=primary_skill)

        selection = select_methods(
            candidates,
            signature,
            exploration_rate=exploration_rate_for(
                creativity=self.traits.get("creativity"),
                curiosity=self.traits.get("curiosity"),
            ),
            rng=deterministic_rng(
                self.agent_name,
                self.clock.get_time(),
                activity_id,
                signature.context_key(),
            ),
        )
        if selection.empty or selection.primary is None:
            return None

        for role, scored in (("primary", selection.primary), ("supporting", selection.supporting)):
            if scored is None:
                continue
            self.store.append(
                "METHOD_SELECTED",
                method_id=scored.method.method_id,
                activity_id=activity_id,
                payload={
                    "role": role,
                    "context_key": selection.context_key,
                    "explored": selection.explored,
                    "score": round(scored.score, 4),
                    "shadow": True,
                },
            )

        primary = selection.primary.method
        reward = live_reward_for(
            signals,
            skill_id=primary.skill_id,
            context_key=selection.context_key,
            tracker=baseline_tracker_from_events(self.store.events()),
            goal_tracker=goal_baseline_tracker_from_events(self.store.events()),
            weights=weights_for_caliber(
                self.config.reward_caliber,
                goal_progress_weight=self.config.goal_progress_weight,
            ),
        )
        shadow = shadow_baseline_for(
            signals,
            skill_id=primary.skill_id,
            context_key=selection.context_key,
            tracker=baseline_tracker_from_events(self.store.events()),
            goal_tracker=goal_baseline_tracker_from_events(self.store.events()),
            weights=weights_for_caliber(
                alternative_caliber(self.config.reward_caliber),
                goal_progress_weight=self.config.goal_progress_weight,
            ),
            caliber=alternative_caliber(self.config.reward_caliber),
        )
        self.store.append(
            "METHOD_OUTCOME_OBSERVED",
            method_id=primary.method_id,
            activity_id=activity_id,
            payload={
                "context_key": selection.context_key,
                "reward": round(reward.total, 4),
                "components": reward.to_dict(),
                "success": reward.total > 0,
                "attribution": "shadow_counterfactual",
                "activity_type": signals.activity_type,
                "skill_gains": signals.skill_gains,
                "delta_vitality": signals.delta_vitality,
                "delta_money": signals.delta_money,
                "delta_fulfillment": dict(signals.delta_fulfillment),
                "turns": signals.turns,
                "rejections": signals.verification_rejections,
                "week": self._week_key(),
                # KI-8 shadow, same as the real-adoption path.
                **shadow.to_payload(),
            },
        )

        update = update_value(
            value=primary.global_value,
            practice_count=primary.practice_count,
            success_count=primary.success_count,
            reward=reward.total,
            current_status=primary.status,
            context_values=update_context_value(
                primary.context_values,
                context_key=selection.context_key,
                reward=reward.total,
            ),
        )
        context_values = update_context_value(
            primary.context_values,
            context_key=selection.context_key,
            reward=reward.total,
        )
        payload = update.to_dict()
        payload.update(
            {
                "context_values": context_values,
                "last_used": str(self.clock.get_time()),
                "evidence_kind": "shadow_counterfactual",
                "week": self._week_key(),
            }
        )
        record_out, created = self.store.append(
            "METHOD_VALUE_UPDATED",
            method_id=primary.method_id,
            activity_id=activity_id,
            payload=payload,
        )
        self._invalidate()
        if created:
            SHADOW_LOGGER.info(
                f"[{self.agent_name}] shadow value {primary.method_id}: "
                f"{primary.global_value:.3f} -> {update.value:.3f} "
                f"(reward {reward.total:+.3f}, n={update.practice_count})"
            )
        return record_out

    # -- (2) weekly extraction --------------------------------------------
    def after_review(
        self, *, weekly_summary: str = "", week_records: Optional[List[Dict]] = None
    ) -> int:
        """Extract candidate methodologies from the week; returns events written."""
        if not self.enabled:
            return 0
        if not self._budget_available():
            SHADOW_LOGGER.info(
                f"[{self.agent_name}] extraction budget exhausted for {self._week_key()}"
            )
            return 0

        records = week_records if week_records is not None else self._week_activity_records()
        if not records:
            return 0

        skills = materializer.skills_from_state(self.dm)
        known_skills = sorted(skills.keys())
        existing_titles = [m.title for m in self.methods()]

        from src.agents.cognition.models import CONTEXT_TAGS

        prompt = build_method_extraction_prompt(
            char=self.agent_name,
            skill_list=known_skills,
            week_evidence=format_week_evidence(records),
            existing_methods=existing_titles,
            allowed_tags=list(CONTEXT_TAGS),
            language=self.language,
        )

        self._consume_budget()
        methods = self._call_extraction(prompt, known_skills=known_skills)
        if not methods:
            return 0

        written = 0
        refined = 0
        refiner = getattr(self.dm, "method_refiner", None)
        evidence_memory_ids = self._week_memory_ids()
        for item in methods:
            skill_id = str(item.get("skill_id") or "").strip()
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            method_id = method_id_for(skill_id or "unmapped", title)
            if self._method_by_id(method_id) is not None:
                continue
            # Phase 5 version evolution: a later week re-deriving a method that
            # is already known under a different wording is a *new version of the
            # same method*, not a new method. Without this the same recipe
            # accumulates as near-duplicates ("同义方法重复率").
            if refiner is not None:
                try:
                    if refiner({"title": title, "skill_id": skill_id, **item}) is not None:
                        refined += 1
                        continue
                except Exception as e:  # a refinement must never break extraction
                    SHADOW_LOGGER.warning(f"[{self.agent_name}] refinement failed: {e!r}")
            payload = dict(item)
            payload.update(
                {
                    "title": title,
                    "skill_id": skill_id,
                    "source_type": "practice_reflection",
                    "status": "proposed",
                    "global_value": 0.0,
                    "confidence": 0.0,
                    "week": self._week_key(),
                    # KI-10 (phase 5): the memories consolidated from the same
                    # week's evidence are this method's evidence base, so an
                    # adoption of it can mark them as actually used.
                    "source_memory_ids": list(evidence_memory_ids),
                }
            )
            _, created = self.store.append(
                "METHOD_PROPOSED",
                method_id=method_id,
                payload=payload,
            )
            if created:
                written += 1
        if written or refined:
            self._invalidate()
            SHADOW_LOGGER.info(
                f"[{self.agent_name}] proposed {written} method(s) in {self._week_key()}"
                + (f", refined {refined} existing one(s)" if refined else "")
            )
        return written

    # -- LLM call ----------------------------------------------------------
    def _call_extraction(self, prompt: str, *, known_skills: List[str]):
        """One extraction call. Never raises: a bad answer means "no methods"."""
        from src.utils import get_response_with_retry

        def _validator(response: str, **kwargs):
            return parse_method_extraction(response, known_skills=known_skills)

        try:
            messages = [{"role": "system", "content": prompt}]
            data = get_response_with_retry(
                post_processing_funcs=[_validator],
                model=self.model,
                messages=messages,
                max_retry=2,
            )
            # token 账：认知层抽取调用不进 generation 日志，没有这一行就无法回答
            # "新功能多花了多少 token"（见 src/agents/cognition/usage.py）。
            from src.agents.cognition.usage import record_llm_usage

            record_llm_usage(
                self.dm,
                "method_extraction",
                messages=messages,
                response=data,
                model=self.model,
            )
        except Exception as e:  # pragma: no cover - defensive: shadow never breaks a run
            SHADOW_LOGGER.warning(
                f"[{self.agent_name}] method extraction failed: {e!r}"
            )
            return []
        if not isinstance(data, dict):
            return []
        methods = data.get("methods")
        return methods if isinstance(methods, list) else []

    # -- helpers -----------------------------------------------------------
    def _week_key(self) -> str:
        t = self.clock.get_time()
        return f"Y{t.year}-W{t.week:02d}"

    def _budget_available(self) -> bool:
        used = self._extract_calls.get(self._week_key(), 0)
        return used < self.config.max_extract_calls_per_week

    def _consume_budget(self) -> None:
        key = self._week_key()
        self._extract_calls[key] = self._extract_calls.get(key, 0) + 1

    def _week_memory_ids(self, *, limit: int = 3) -> List[str]:
        """Memories consolidated from this same week's evidence (KI-10).

        Phase 4 left `used_in_plan` at zero for every review-derived method,
        because only idea-conversion methods carried `source_memory_ids`. The
        memories written by this week's consolidation are exactly the evidence
        the extraction read, so linking them is what lets an adopted method say
        which memories actually informed the plan.
        """
        week = self._week_key()
        try:
            from src.agents.cognition.memory_views import load_memories

            memories = [
                m
                for m in load_memories(self.dm)
                if str(getattr(m, "created_at", "") or "").startswith(week)
            ]
        except Exception:  # memories are optional for extraction
            return []
        memories.sort(key=lambda m: (-float(getattr(m, "strength", 0.0) or 0.0), m.memory_id))
        return [m.memory_id for m in memories[: max(0, limit)]]

    def _dominant_skill(self, record: Dict[str, Any]) -> str:
        signals = signals_from_activity_record(record)
        best, best_value = "", 0.0
        for name, value in signals.skill_gains.items():
            if value > best_value:
                best, best_value = name, value
        return best

    def _week_activity_records(self) -> List[Dict[str, Any]]:
        """This week's activity records, straight from the ledger."""
        path = self.dm.root / "activity.jsonl"
        if not path.exists():
            return []
        import json

        week = self._week_key()
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            time_str = str(record.get("time") or "")
            if time_str.startswith(week):
                out.append(record)
        return out

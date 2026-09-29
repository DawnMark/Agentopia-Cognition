"""Phase 4: passive method hints, and what the character does with them.

The first phase that changes behaviour, and it does so the narrow way the design
document prescribes:

    offer a small menu of methods the character could use this week
    -> the character is free to ignore all of them
    -> if it says it will use one, that adoption is recorded
    -> if the adopted method is then practised, the practice outcome is
       attributed to it as **real** evidence

Three rules keep this honest:

1. **Being offered is not reinforcement** (invariant #11, the recall-side
   counterpart of #6): `METHOD_HINTED` moves no number; only a practice outcome
   does.
2. **Real and counterfactual evidence stay separable**: the shadow tracker from
   phase 1 attributes outcomes to the method that *would* have been picked; an
   activity claimed by a real adoption is never also counted counterfactually,
   and both kinds carry an explicit `attribution`.
3. **A hint is a menu, not an instruction**: the prompt says so, and the renderer
   includes what the method is for, its first steps, a check and a failure mode,
   so the character can judge it.

Only structure-complete methods are offered (KI-6), and offers are scored against
the character's own skills, so the menu is about things it could actually do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, TYPE_CHECKING

from src.agents.cognition import materializer
from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.memory_store import MemoryEventStore
from src.agents.cognition.methodology_policy import (
    ContextSignature,
    context_signature_from_activity,
    deterministic_rng,
    exploration_rate_for,
    score_methods,
)
from src.agents.cognition.models import Methodology
from src.agents.cognition.offer_bandit import (
    DEFAULT_LOW_MONEY_THRESHOLD,
    MethodStat,
    consecutive_weeks_used,
    method_stats,
    origin_of,
    practice_priority,
    rank_offers,
    week_signature,
)
from src.agents.cognition.reward_model import (
    DEFAULT_REWARD_CALIBER,
    SPECIALIZED_CONTEXT_MARGIN,
    SPECIALIZED_MIN_CONTEXT_SAMPLES,
    BaselineTracker,
    alternative_caliber,
    baseline_tracker_from_events,
    goal_baseline_tracker_from_events,
    live_reward_for,
    shadow_baseline_for,
    weights_for_caliber,
    signals_from_activity_record,
    update_context_value,
    update_value,
)
from src.utils import get_logger

HINT_LOGGER = get_logger("cognition", quiet=True)

DEFAULT_TOP_K = 3
DEFAULT_MAX_CHARS = 1400
DEFAULT_EXPLORATION_SLOTS = 1

# Statuses that may be offered. Archived/deprecated methods are not suggestions.
OFFERABLE_STATUSES = ("proposed", "learned", "tested", "validated", "specialized")
EXPLORABLE_STATUSES = ("proposed", "learned")

# Why the character did not take an offered method up (phase 5). All three are
# the character's own outcome, never the offerer's assumption about its motives.
DECLINE_REASONS = ("no_declaration", "chose_other", "no_plan")

# A method with no first step cannot be acted on.
MIN_STEPS = 1
MAX_STEPS_SHOWN = 3

_ADOPT_PATTERN = re.compile(r"<method>(.*?)</method>", re.IGNORECASE | re.DOTALL)

HINT_BLOCK_ZH = """## 你这次可以考虑的方法（可选，不是命令）

下面这些方法来自你自己过去的经历。**用不用完全由你决定**；如果都不合适，忽略它们就好。

{offers}

如果你这周打算试着用其中某一个，请在你的计划里写一行 `<method>方法的标题</method>`；
不打算用就不要写。只写一个。"""

HINT_BLOCK_EN = """## Methods you may want to try this week (optional, not an order)

These come from your own past experience. **It is entirely your choice** whether to use
any of them; ignore them if none fits.

{offers}

If you plan to try one of them this week, write one line `<method>the method's title</method>`
in your plan. If not, write nothing. At most one."""


@dataclass
class HintConfig:
    """`world.cognition` hint settings, inert by default."""

    enabled: bool = False
    top_k: int = DEFAULT_TOP_K
    max_chars: int = DEFAULT_MAX_CHARS
    exploration_slots: int = DEFAULT_EXPLORATION_SLOTS
    # Phase 5: the contextual bandit that decides *what to offer*. Off means the
    # phase-4 menu shape exactly, so a run without it stays comparable.
    bandit: bool = False
    low_money_threshold: float = DEFAULT_LOW_MONEY_THRESHOLD
    # Phase 5 observation: record the situation the menu was offered in, and the
    # character's answer to every offered method. On whenever hints are on —
    # it writes events, it does not change the menu.
    record_declines: bool = True
    # Phase 5 probe: which centred-reward variant the *shadow* score on each
    # practice should use. Off keeps the design document's floored quality
    # mapping (see `reward_model.BASELINE_FULL_SCALE_SIGNED`).
    # Phase 5 follow-up (KI-19, user decision 2026-09-27): which formula actually
    # moves a method's value. `signed` = centred performance as the score itself,
    # `quality` = the design document's floored mapping, `off` = phases 1-3.
    # A directly constructed config stays on `off` (inert, as every config here
    # is); `from_world_config` supplies the run's default.
    reward_caliber: str = "off"
    # Design §3's goal-progress weight. Recorded on every outcome either way; it
    # stays 0 by default because the probe on run 09271745 showed the extra
    # positive term dilutes the separation the signed caliber buys.
    goal_progress_weight: float = 0.0
    # `specialized` thresholds (phase 5, reward design §3.3).
    specialized_margin: float = SPECIALIZED_CONTEXT_MARGIN
    specialized_min_samples: int = SPECIALIZED_MIN_CONTEXT_SAMPLES

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "HintConfig":
        section = (world_cfg or {}).get("cognition") or {}

        def _int(key: str, default: int) -> int:
            try:
                return int(section.get(key, default))
            except (TypeError, ValueError):
                return default

        def _float(key: str, default: float) -> float:
            try:
                return float(section.get(key, default))
            except (TypeError, ValueError):
                return default

        return HintConfig(
            enabled=bool(section.get("method_hints", False)),
            top_k=max(1, _int("method_hints_top_k", DEFAULT_TOP_K)),
            max_chars=max(200, _int("method_hints_max_chars", DEFAULT_MAX_CHARS)),
            exploration_slots=max(0, _int("method_hints_exploration_slots", DEFAULT_EXPLORATION_SLOTS)),
            bandit=bool(section.get("method_hints_bandit", False)),
            low_money_threshold=_float(
                "method_hints_low_money", DEFAULT_LOW_MONEY_THRESHOLD
            ),
            record_declines=bool(section.get("method_hints_record_declines", True)),
            reward_caliber=str(
                section.get("reward_caliber", DEFAULT_REWARD_CALIBER) or DEFAULT_REWARD_CALIBER
            ).lower(),
            goal_progress_weight=_float("goal_progress_weight", 0.0),
            specialized_margin=_float("method_specialized_margin", SPECIALIZED_CONTEXT_MARGIN),
            specialized_min_samples=max(
                1, _int("method_specialized_min_samples", SPECIALIZED_MIN_CONTEXT_SAMPLES)
            ),
        )


@dataclass
class HintOffer:
    """One method offered to the character this week."""

    method_id: str
    title: str
    skill_id: str
    status: str
    role: str  # "exploit" | "explore" | "filler"
    score: float
    steps: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)
    source_idea_id: str = ""
    # "own_idea" when the method came from the character's own idea (KI-5 →
    # KI-6 → hint). Shown in the menu so the loop this phase exists for is
    # visible to the character and to acceptance.
    origin: str = "practice"
    # Phase 5: the situation this offer was conditioned on, and the parts of its
    # score. Recorded so a later week can explain why the menu changed.
    context_key: str = ""
    parts: Dict[str, float] = field(default_factory=dict)
    ignored_streak: int = 0
    # Phase 5 follow-up: the character does not remember its own menu history, so
    # telling it "you have been using this one for N weeks" is information it
    # cannot otherwise have. Still an annotation on a menu, never an order.
    weeks_in_a_row: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "title": self.title,
            "skill_id": self.skill_id,
            "status": self.status,
            "role": self.role,
            "origin": self.origin,
            "score": round(self.score, 4),
            "context_key": self.context_key,
            "ignored_streak": int(self.ignored_streak),
        }


@dataclass
class Adoption:
    """A method the character said it would use this week."""

    method_id: str
    title: str
    skill_id: str
    matched_by: str  # "id" | "title"
    week: str
    context_key: str = ""
    source_memory_ids: List[str] = field(default_factory=list)
    source_idea_id: str = ""
    claimed_activity_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "title": self.title,
            "matched_by": self.matched_by,
            "week": self.week,
            "context_key": self.context_key,
            "claimed_activity_id": self.claimed_activity_id,
        }


def normalize_title(text: str) -> str:
    """Loose title comparison: whitespace, punctuation and case insensitive."""
    return re.sub(r"[\s，。、：:；;（）()【】\[\]！!？?\"'“”‘’]", "", str(text)).lower()


class MethodHintProvider:
    """Offers methods and tracks what the character does with them."""

    def __init__(
        self,
        *,
        dm,
        clock,
        agent_name: str,
        traits: Optional[Dict[str, float]] = None,
        config: Optional[HintConfig] = None,
        language: str = "en",
    ) -> None:
        self.dm = dm
        self.clock = clock
        self.agent_name = agent_name
        self.traits = traits or {}
        self.config = config or HintConfig()
        self.language = language
        self.capability = CapabilityEventStore(dm)
        self.memories = MemoryEventStore(dm)
        self._offers: List[HintOffer] = []
        self._adoption: Optional[Adoption] = None
        self._claimed_activities: Set[str] = set()
        self._cached_methods: Optional[List[Methodology]] = None
        self._offered_ids: Optional[Set[str]] = None
        self._cached_skills: List[str] = []
        # Phase 5: the situation signature this week's menu was built from, and
        # the offer statistics it was scored against.
        self._signature: Optional[ContextSignature] = None
        self._stats: Optional[Dict[str, MethodStat]] = None
        self._baseline: Optional[BaselineTracker] = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # -- offering ----------------------------------------------------------
    def offerable_methods(self) -> List[Methodology]:
        """Methods that may be offered: complete, actionable, not retired."""
        if self._cached_methods is not None:
            return self._cached_methods
        view = materializer.build_capability_view(self.dm)
        self._cached_skills = sorted((view.get("skills") or {}).keys())
        out: List[Methodology] = []
        for entry in (view.get("methodologies") or {}).values():
            if entry.get("structure_incomplete"):
                continue  # KI-6: an incomplete recipe must not be suggested
            if entry.get("status") not in OFFERABLE_STATUSES:
                continue
            if len(entry.get("steps") or []) < MIN_STEPS:
                continue
            payload = {
                k: v
                for k, v in entry.items()
                if k
                not in (
                    "selections",
                    "applications",
                    "evidence",
                    "last_selection",
                    "skill_mapped",
                    "structure_incomplete",
                    "structure_missing",
                )
            }
            try:
                out.append(Methodology.from_dict(payload))
            except (KeyError, ValueError):
                continue
        self._cached_methods = out
        return out

    def select_offers(self, methods: Optional[Sequence[Methodology]] = None) -> List[HintOffer]:
        """Top-K menu: the best method plus a small exploration slot.

        Phase 4 shape (bandit off): the best-scoring method, one exploration
        slot, then fillers. Phase 5 shape (bandit on): a contextual bandit ranks
        the whole candidate set by `exploitation(situation) + UCB bonus`, so the
        menu rotates when the character's situation changes or when showing the
        same method stops working.
        """
        candidates = list(methods if methods is not None else self.offerable_methods())
        if not candidates:
            return []

        signature = self._week_signature()
        if self.config.bandit:
            return self._select_offers_bandit(candidates, signature)
        return self._select_offers_phase4(candidates, signature)

    def _select_offers_phase4(
        self, candidates: Sequence[Methodology], signature: ContextSignature
    ) -> List[HintOffer]:
        """The phase-4 menu, unchanged (invariant #10: off means old behaviour)."""
        scored = score_methods(candidates, signature)
        rng = deterministic_rng(self.agent_name, self._week_key(), "hints")
        exploration_rate = exploration_rate_for(
            creativity=self.traits.get("creativity"),
            curiosity=self.traits.get("curiosity"),
        )

        offers: List[HintOffer] = []
        taken: Set[str] = set()

        def add(scored_item, role: str) -> None:
            method = scored_item.method
            if method.method_id in taken or len(offers) >= self.config.top_k:
                return
            taken.add(method.method_id)
            offers.append(self._offer_from(method, role=role, score=scored_item.score))

        if scored:
            # Phase 4.5: a method grown from the character's own lesson takes the
            # leading slot whenever one exists and has not been practised yet —
            # it is the one the character is most likely to act on, and practice
            # is the only thing that can validate it.
            lesson_first = sorted(
                scored,
                key=lambda s: (
                    -practice_priority(s.method, practised=int(s.method.practice_count or 0)),
                    -s.score,
                    s.method.method_id,
                ),
            )
            add(lesson_first[0], "exploit")

        explorable = [s for s in scored if s.method.status in EXPLORABLE_STATUSES]
        if explorable and exploration_rate > 0:
            # The exploration slot gives the character's *own* material priority:
            # an idea or lesson that never reaches the menu can never guide an
            # action, which would break the memory → idea → method → action loop
            # this phase exists for (measured on run 09261617: 25 of 26 offers
            # were review-derived methods, only 1-2 idea-derived ones).
            already_offered = self._offered_method_ids()
            own_lessons = [
                s
                for s in explorable
                if origin_of(s.method) == "own_lesson"
                and s.method.method_id not in already_offered
            ]
            own_ideas = [
                s
                for s in explorable
                if str(getattr(s.method, "source_type", "")) == "idea_conversion"
                and s.method.method_id not in already_offered
            ]
            pool = own_lessons or own_ideas or explorable
            slots = self.config.exploration_slots
            for _ in range(slots):
                pick = pool[rng.randrange(len(pool))] if pool else None
                if pick is None:
                    break
                before = len(offers)
                add(pick, "explore")
                if len(offers) == before:
                    # The draw collided with an offer already made; fall back to
                    # the plain pool so the slot is not wasted.
                    remaining = [s for s in explorable if s.method.method_id not in taken]
                    if not remaining:
                        break
                    add(remaining[rng.randrange(len(remaining))], "explore")
                    if len(offers) == before:
                        break
                pool = [s for s in pool if s.method.method_id not in taken] or pool

        for item in scored:
            if len(offers) >= self.config.top_k:
                break
            add(item, "filler")

        return offers

    def _select_offers_bandit(
        self, candidates: Sequence[Methodology], signature: ContextSignature
    ) -> List[HintOffer]:
        """Phase 5: the offer is a contextual-bandit decision (design doc 阶段 5)."""
        stats = self._offer_stats()
        ranked = rank_offers(
            candidates,
            signature,
            stats=stats,
            traits=self.traits,
            top_k=self.config.top_k,
        )
        offers: List[HintOffer] = []
        for item in ranked:
            offers.append(
                self._offer_from(
                    item.method,
                    role=item.role,
                    score=item.score,
                    parts=dict(item.parts),
                    ignored_streak=item.ignored_streak,
                )
            )
        return offers

    def _offer_from(
        self,
        method: Methodology,
        *,
        role: str,
        score: float,
        parts: Optional[Dict[str, float]] = None,
        ignored_streak: int = 0,
        weeks_in_a_row: Optional[int] = None,
    ) -> HintOffer:
        if weeks_in_a_row is None:
            weeks_in_a_row = consecutive_weeks_used(
                self._offer_stats().get(method.method_id)
            )
        return HintOffer(
            method_id=method.method_id,
            title=method.title or method.description[:40],
            skill_id=method.skill_id,
            status=method.status,
            role=role,
            score=score,
            steps=list(method.steps)[:MAX_STEPS_SHOWN],
            checks=list(method.checks)[:1],
            failure_modes=list(method.failure_modes)[:1],
            applicable_contexts=list(method.applicable_contexts)[:3],
            source_memory_ids=list(method.source_memory_ids),
            source_idea_id=method.source_idea_id or "",
            origin=origin_of(method),
            context_key=(self._signature.context_key() if self._signature else ""),
            parts=parts or {},
            ignored_streak=int(ignored_streak),
            weeks_in_a_row=int(weeks_in_a_row),
        )

    # -- the situation the menu is offered in (phase 5) ---------------------
    def _week_signature(self) -> ContextSignature:
        """Situation the menu is conditioned on; built once per week."""
        if self._signature is not None:
            return self._signature
        try:
            state = self.dm.read_state(exclude_cur_t=False)
        except Exception:  # pragma: no cover - a missing state is not fatal
            state = {}
        self._signature = week_signature(
            skills=list(self._cached_skills),
            state=state if isinstance(state, dict) else {},
            goal_memories=self._goal_memories(),
            recent_records=self._recent_activity_records(),
            low_money_threshold=self.config.low_money_threshold,
        )
        return self._signature

    def _goal_memories(self) -> List[Dict[str, Any]]:
        """The character's own stated goals, strongest first (phase 2 memories)."""
        try:
            from src.agents.cognition.memory_views import build_memory_views

            view = build_memory_views(self.dm).get("memories") or {}
        except Exception:  # pragma: no cover - memories are optional
            return []
        goals: List[Dict[str, Any]] = []
        for entry in (view.get("memories") or {}).values():
            if str(entry.get("kind") or "") != "goal":
                continue
            if str(entry.get("status") or "active") != "active":
                continue
            if not str(entry.get("content") or "").strip():
                continue
            goals.append(entry)
        goals.sort(
            key=lambda e: (-float(e.get("strength") or 0.0), str(e.get("memory_id") or ""))
        )
        return goals

    def _recent_activity_records(self) -> List[Dict[str, Any]]:
        """Last week's finished activities: what the coming week will look like."""
        import json

        path = self.dm.root / "activity.jsonl"
        if not path.exists():
            return []
        week = self._week_key()
        try:
            year, week_no = week.split("-")
            number = int(week_no.lstrip("W"))
        except (ValueError, IndexError):
            return []
        previous = f"{year}-W{max(0, number - 1):02d}"
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("time") or "").startswith(previous):
                out.append(record)
        return out

    def _offer_stats(self) -> Dict[str, MethodStat]:
        """Offer/adoption/decline history, folded from the capability events."""
        if self._stats is None:
            self._stats = method_stats(self.capability.events())
        return self._stats

    def _offered_method_ids(self) -> Set[str]:
        """Methods this character has already been shown (any week)."""
        if self._offered_ids is None:
            self._offered_ids = {
                str(event.get("method_id"))
                for event in self.capability.events()
                if event.get("type") == "METHOD_HINTED" and event.get("method_id")
            }
        return self._offered_ids

    def render(self, offers: Sequence[HintOffer]) -> str:
        """Render the bounded prompt block (menu, never an instruction)."""
        if not offers:
            return ""
        lines: List[str] = []
        for index, offer in enumerate(offers, start=1):
            head = f"{index}. {offer.title}"
            if offer.origin == "own_lesson":
                # The character's own words, so it can recognise the lesson it
                # wrote in its notes (the menu stays a menu: no obligation).
                head += "（你自己在笔记里总结的教训）"
            elif offer.origin == "own_idea":
                head += "（你自己最近想到的做法）"
            if offer.weeks_in_a_row >= 2:
                # Measured on run 09271745: the character adopted the same method
                # every week for five weeks and had no way of knowing it — its
                # own menu history is not part of its memory. Saying so is
                # information, not an instruction; it may keep using it.
                head += f"（你已经连续 {offer.weeks_in_a_row} 周用这个了）"
            if offer.applicable_contexts:
                head += f"（适用：{'、'.join(offer.applicable_contexts)}）"
            lines.append(head)
            if offer.steps:
                lines.append("   步骤：" + " → ".join(offer.steps[:MAX_STEPS_SHOWN]))
            if offer.checks:
                lines.append(f"   自检：{offer.checks[0]}")
            if offer.failure_modes:
                lines.append(f"   常见失败：{offer.failure_modes[0]}")
        body = "\n".join(lines)
        template = HINT_BLOCK_ZH if str(self.language).lower() in ("zh", "cn") else HINT_BLOCK_EN
        block = template.format(offers=body)
        if len(block) > self.config.max_chars:
            block = block[: self.config.max_chars].rstrip() + "\n…"
        return block

    def prepare_week(self) -> str:
        """Select, record and render this week's offers (once per week)."""
        if not self.enabled:
            return ""
        self._signature = None  # the situation is re-read for every week
        self._stats = None
        offers = self.select_offers()
        self._offers = offers
        self._adoption = None
        self._cached_methods = None  # offer from a fresh view
        self._offered_ids = None
        if not offers:
            return ""
        block = self.render(offers)
        week = self._week_key()
        signature = self._signature or ContextSignature()
        for offer in offers:
            self.capability.append(
                "METHOD_HINTED",
                method_id=offer.method_id,
                payload={
                    "week": week,
                    "role": offer.role,
                    "origin": offer.origin,
                    "score": round(offer.score, 4),
                    "status": offer.status,
                    "block_chars": len(block),
                    # Phase 5 observation: the situation the offer was made in,
                    # and the score parts it was made from. Without these a
                    # decline cannot be told apart from "the menu never fitted".
                    "context_key": signature.context_key(),
                    "context_tags": list(signature.context_tags),
                    "goal": signature.goal,
                    "constraints": dict(signature.constraints),
                    "score_parts": {
                        k: round(float(v), 4) for k, v in (offer.parts or {}).items()
                    },
                    "ignored_streak": int(offer.ignored_streak),
                    "bandit": bool(self.config.bandit),
                },
                # One offer per method per week; re-entering the stage must not
                # duplicate the offer.
                idempotency_key=f"METHOD_HINTED:{offer.method_id}:{week}",
            )
        HINT_LOGGER.info(
            f"[{self.agent_name}] offered {len(offers)} method(s) in {week} "
            f"({len(block)} chars, situation "
            f"{'+'.join(signature.context_tags) or 'default'})"
        )
        return block

    # -- adoption ----------------------------------------------------------
    def parse_adoption(self, text: str) -> Optional[Adoption]:
        """Find the character's own declaration, matched against what was offered."""
        match = _ADOPT_PATTERN.search(str(text or ""))
        if not match:
            return None
        raw = match.group(1).strip()
        if not raw:
            return None
        normalized = normalize_title(raw)

        for offer in self._offers:
            if raw == offer.method_id or offer.method_id in raw:
                return self._adoption_from(offer, "id")
        for offer in self._offers:
            title = normalize_title(offer.title)
            if title and (title in normalized or normalized in title):
                return self._adoption_from(offer, "title")
        HINT_LOGGER.info(
            f"[{self.agent_name}] declared a method that was not offered: {raw[:60]!r}"
        )
        return None

    def _adoption_from(self, offer: HintOffer, matched_by: str) -> Adoption:
        return Adoption(
            method_id=offer.method_id,
            title=offer.title,
            skill_id=offer.skill_id,
            matched_by=matched_by,
            week=self._week_key(),
            # Phase 5: the adoption is conditioned on the *situation* the menu
            # was offered in, not on the method's own tags (phase 4 used the
            # latter, which made the context key useless for the bandit).
            context_key=offer.context_key or "+".join(offer.applicable_contexts) or "default",
            source_memory_ids=list(offer.source_memory_ids),
            source_idea_id=offer.source_idea_id,
        )

    def record_adoption(self, text: str) -> Optional[Adoption]:
        """Record the character's answer to this week's menu.

        Phase 5 also records the *other* half: every offered method the
        character did not take up gets a `METHOD_DECLINED`. That is the only
        negative signal the offering policy has about its own choices, and
        phase 4 could not see it at all (26 of 27 person-weeks adopted
        something).
        """
        if not self.enabled:
            return None
        adoption = self.parse_adoption(text)
        if adoption is not None:
            self._adoption = adoption
            _, created = self.capability.append(
                "METHOD_SELECTED",
                method_id=adoption.method_id,
                payload={
                    "week": adoption.week,
                    "role": "primary",
                    "shadow": False,
                    "source": "hint",
                    "adoption": True,
                    "matched_by": adoption.matched_by,
                    "context_key": adoption.context_key,
                },
                idempotency_key=f"METHOD_SELECTED:{adoption.method_id}:{adoption.week}:hint",
            )
            if created:
                HINT_LOGGER.info(
                    f"[{self.agent_name}] adopted {adoption.method_id} "
                    f"({adoption.title[:40]}) for {adoption.week}"
                )
                # KI-10: the memories behind an adopted method are actually used,
                # so they earn the "used in plan" factor (placement alone does
                # not).
                for memory_id in adoption.source_memory_ids:
                    self.memories.used_in_plan(
                        memory_id=memory_id,
                        week=adoption.week,
                        method_id=adoption.method_id,
                        idea_id=adoption.source_idea_id,
                    )
        self._record_declines(text, adoption)
        return adoption

    def _record_declines(self, text: str, adoption: Optional[Adoption]) -> int:
        """Record what the character did *not* take up (phase 5 observation)."""
        if not self.config.record_declines or not self._offers:
            return 0
        week = self._week_key()
        empty_plan = not str(text or "").strip()
        count = 0
        for offer in self._offers:
            if adoption is not None and offer.method_id == adoption.method_id:
                continue
            if empty_plan:
                reason = "no_plan"
            elif adoption is not None:
                reason = "chose_other"
            else:
                reason = "no_declaration"
            assert reason in DECLINE_REASONS  # the vocabulary stays closed
            _, created = self.capability.append(
                "METHOD_DECLINED",
                method_id=offer.method_id,
                payload={
                    "week": week,
                    "reason": reason,
                    "role": offer.role,
                    "origin": offer.origin,
                    "context_key": offer.context_key,
                    "ignored_streak": int(offer.ignored_streak),
                    "adopted_method_id": (adoption.method_id if adoption else ""),
                },
                idempotency_key=f"METHOD_DECLINED:{offer.method_id}:{week}",
            )
            if created:
                count += 1
        if count:
            HINT_LOGGER.info(
                f"[{self.agent_name}] declined {count} offered method(s) in {week} "
                f"({('no plan' if empty_plan else ('chose another' if adoption else 'no declaration'))})"
            )
        return count

    @property
    def adoption(self) -> Optional[Adoption]:
        return self._adoption

    # -- practice ----------------------------------------------------------
    def observe_activity(self, record: Dict[str, Any]) -> bool:
        """Attribute an activity's outcome to the adopted method.

        Returns True when this activity was claimed as the practice of the
        adopted method, so the shadow tracker skips it (one practice must not
        produce both a real and a counterfactual update).
        """
        if not self.enabled or self._adoption is None:
            return False
        activity_id = str(record.get("activity_id") or "")
        if not activity_id or activity_id in self._claimed_activities:
            return False
        if self._adoption.claimed_activity_id:
            return False  # one practice per adoption

        signals = signals_from_activity_record(record)
        if not self._is_relevant(record, signals):
            return False

        week = self._week_key()
        method = self._method(self._adoption.method_id)
        context_key = self._adoption.context_key or "default"

        self.capability.append(
            "METHOD_APPLIED",
            method_id=self._adoption.method_id,
            activity_id=activity_id,
            payload={
                "week": week,
                "source": "hint",
                "binding": "skill_or_context_match",
                "activity_type": signals.activity_type,
            },
            idempotency_key=f"METHOD_APPLIED:{self._adoption.method_id}:{activity_id}",
        )

        reward = live_reward_for(
            signals,
            skill_id=self._adoption.skill_id,
            context_key=context_key,
            tracker=self._baseline_tracker(),
            goal_tracker=self._goal_baseline_tracker(),
            weights=weights_for_caliber(
                self.config.reward_caliber,
                goal_progress_weight=self.config.goal_progress_weight,
            ),
        )
        shadow = shadow_baseline_for(
            signals,
            skill_id=self._adoption.skill_id,
            context_key=context_key,
            tracker=self._baseline_tracker(),
            goal_tracker=self._goal_baseline_tracker(),
            weights=weights_for_caliber(
                alternative_caliber(self.config.reward_caliber),
                goal_progress_weight=self.config.goal_progress_weight,
            ),
            caliber=alternative_caliber(self.config.reward_caliber),
        )
        self.capability.append(
            "METHOD_OUTCOME_OBSERVED",
            method_id=self._adoption.method_id,
            activity_id=activity_id,
            payload={
                "week": week,
                "context_key": context_key,
                "reward": round(reward.total, 4),
                "components": reward.to_dict(),
                "success": reward.total > 0,
                "attribution": "real_adoption",
                "activity_type": signals.activity_type,
                "skill_gains": signals.skill_gains,
                "delta_vitality": signals.delta_vitality,
                "delta_money": signals.delta_money,
                "delta_fulfillment": dict(signals.delta_fulfillment),
                "turns": signals.turns,
                "rejections": signals.verification_rejections,
                # KI-8 shadow: what the centred formula would have said. Written
                # for the review run; never used to move a value yet.
                **shadow.to_payload(),
            },
            idempotency_key=f"METHOD_OUTCOME_OBSERVED:{self._adoption.method_id}:{activity_id}:real",
        )

        if method is not None:
            context_values = update_context_value(
                method.context_values,
                context_key=context_key,
                reward=reward.total,
            )
            update = update_value(
                value=method.global_value,
                practice_count=method.practice_count,
                success_count=method.success_count,
                reward=reward.total,
                current_status=method.status,
                context_values=context_values,
                specialized_margin=self.config.specialized_margin,
                specialized_min_samples=self.config.specialized_min_samples,
            )
            payload = update.to_dict()
            payload.update(
                {
                    "context_values": context_values,
                    "last_used": str(self.clock.get_time()),
                    "evidence_kind": "real_adoption",
                    "week": week,
                }
            )
            self.capability.append(
                "METHOD_VALUE_UPDATED",
                method_id=self._adoption.method_id,
                activity_id=activity_id,
                payload=payload,
                idempotency_key=(
                    f"METHOD_VALUE_UPDATED:{self._adoption.method_id}:{activity_id}:real"
                ),
            )
            HINT_LOGGER.info(
                f"[{self.agent_name}] real practice of {self._adoption.method_id}: "
                f"reward {reward.total:+.3f}, value {method.global_value:.3f} -> "
                f"{update.value:.3f} (n={update.practice_count})"
            )

        self._adoption.claimed_activity_id = activity_id
        self._claimed_activities.add(activity_id)
        self._cached_methods = None  # the value just changed
        return True

    def _is_relevant(self, record: Dict[str, Any], signals) -> bool:
        """Does this activity plausibly practise the adopted method?"""
        if self._adoption is None:
            return False
        skill = self._adoption.skill_id
        if skill and signals.gain_for(skill) > 0:
            return True
        if skill and skill in (signals.skill_gains or {}):
            return True
        context_tags = set((self._adoption.context_key or "").split("+"))
        if context_tags and context_tags != {"default"}:
            signature = context_signature_from_activity(record, skill_id=skill)
            if context_tags & set(signature.context_tags):
                return True
        return False

    def _method(self, method_id: str) -> Optional[Methodology]:
        for method in self.offerable_methods():
            if method.method_id == method_id:
                return method
        return None

    def _baseline_tracker(self) -> BaselineTracker:
        """The KI-8 baseline, rebuilt from this character's own outcomes.

        Rebuilt (not cached) because the stream grows with every practice: the
        baseline a method is scored against must already contain this week's
        earlier outcomes, and rebuilding from events is the only version of that
        which survives a resume.
        """
        return baseline_tracker_from_events(self.capability.events())

    def _goal_baseline_tracker(self) -> BaselineTracker:
        """The character's own usual goal progress (separate series)."""
        return goal_baseline_tracker_from_events(self.capability.events())

    def _week_key(self) -> str:
        t = self.clock.get_time()
        return f"Y{t.year}-W{t.week:02d}"

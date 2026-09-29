"""Weekly idea pipeline (design doc 8.6-8.9, phase 3).

Runs after memory consolidation, once per character per week:

1. read the memories and the relation graph built by phase 2;
2. find motif candidates with rules (deterministic, no LLM);
3. phrase the top few with **one** budgeted LLM call;
4. gate each one through the five checks and score IdeaPotential;
5. store at most `idea_max_per_week` ideas (0-2 in practice, fewer when the
   character is exhausted or timid);
6. let at most one idea become a *candidate methodology* (value 0, status
   proposed) — the only route by which an idea can ever reach capability.

Everything is shadow: prompts, plans, state and skill numbers are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.idea_engine import (
    MAX_CANDIDATES_FOR_PHRASING,
    Idea,
    IdeaCandidate,
    IdeaConfig,
    IdeaWorldState,
    build_idea,
    evaluate_candidate,
    find_motif_candidates,
    weekly_idea_budget,
    world_state_from,
)
from src.agents.cognition.idea_models import method_id_for_idea
from src.agents.cognition.idea_store import IdeaEventStore, write_idea_view
from src.agents.cognition.memory_models import MemoryItem
from src.agents.cognition.memory_views import build_memory_views, load_memories
from src.agents.cognition.prompts import (
    build_idea_prompt,
    build_methodization_prompt,
    parse_idea_response,
    parse_methodization_response,
)
from src.utils import get_logger

IDEA_LOGGER = get_logger("cognition", quiet=True)

# Only ideas at least this promising may create a candidate methodology.
CONVERSION_MIN_POTENTIAL = 0.05
CONVERSIONS_PER_WEEK = 1

# A method must carry a usable structure (KI-6). When the methodisation call
# cannot fill it, the method is still stored but marked and logged as a program
# error so acceptance runs can attribute it.
METHOD_MIN_STEPS = 2


def _structure_gaps(method: Dict[str, Any]) -> List[str]:
    """Fields a usable method must have (KI-6); empty list means complete."""
    gaps: List[str] = []
    if not str(method.get("title") or "").strip():
        gaps.append("title")
    steps = [s for s in (method.get("steps") or []) if str(s).strip()]
    if len(steps) < METHOD_MIN_STEPS:
        gaps.append("steps")
    if not [c for c in (method.get("checks") or []) if str(c).strip()]:
        gaps.append("checks")
    if not [f for f in (method.get("failure_modes") or []) if str(f).strip()]:
        gaps.append("failure_modes")
    if not [c for c in (method.get("applicable_contexts") or []) if str(c).strip()]:
        gaps.append("applicable_contexts")
    return gaps


class IdeaEngine:
    """Weekly idea generation for one character."""

    def __init__(
        self,
        *,
        dm,
        clock,
        agent_name: str,
        model: str,
        traits: Optional[Dict[str, float]] = None,
        config: Optional[IdeaConfig] = None,
        language: str = "en",
    ) -> None:
        self.dm = dm
        self.clock = clock
        self.agent_name = agent_name
        self.model = model
        self.traits = traits or {}
        self.config = config or IdeaConfig()
        self.language = language
        self.store = IdeaEventStore(dm)
        self.capability = CapabilityEventStore(dm)
        self._calls: Dict[str, int] = {}
        self._conversion_calls: Dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # -- weekly entry point -------------------------------------------------
    def weekly(self, *, vitality: Optional[float] = None) -> Dict[str, int]:
        """Generate this week's ideas. Returns counts for logging/tests."""
        result = {
            "candidates": 0,
            "stored": 0,
            "rejected": 0,
            "refined": 0,
            "converted": 0,
            "structure_incomplete": 0,
            "skipped_reason": 0,
        }
        if not self.enabled:
            return result
        if not self._budget_available():
            IDEA_LOGGER.info(
                f"[{self.agent_name}] idea budget exhausted for {self._week_key()}"
            )
            self._record_skip("extract_budget_exhausted")
            return result

        memories = load_memories(self.dm)
        if len(memories) < 2:
            self._record_skip("not_enough_memories", {"memories": len(memories)})
            return result
        graph = build_memory_views(self.dm)["graph"]
        edges = graph.get("edges") or []

        candidates = find_motif_candidates(
            memories,
            edges,
            traits=self.traits,
            max_candidates=MAX_CANDIDATES_FOR_PHRASING,
        )
        result["candidates"] = len(candidates)
        if not candidates:
            self._record_skip("no_candidates", {"memories": len(memories), "edges": len(edges)})
            return result

        world = world_state_from(self.dm)
        if vitality is not None:
            world.vitality = float(vitality)
        budget = weekly_idea_budget(self.config, vitality=world.vitality, traits=self.traits)
        if budget <= 0:
            IDEA_LOGGER.info(
                f"[{self.agent_name}] no idea budget this week (vitality={world.vitality})"
            )
            self._record_skip(
                "no_budget",
                {"vitality": round(float(world.vitality), 2), "candidates": len(candidates)},
            )
            return result

        self._consume_budget()
        phrased_map = self._phrase(candidates, memories, world)
        if not phrased_map:
            self._record_skip("phrasing_failed", {"candidates": len(candidates)})
            return result

        existing_ideas = self._existing_ideas()
        existing_methods = self._existing_methods()

        scored: List[tuple] = []
        for index, candidate in enumerate(candidates):
            payload = phrased_map.get(index)
            if payload is None:
                continue
            decision = evaluate_candidate(
                candidate,
                content=str(payload.get("content") or ""),
                test_plan=str(payload.get("test_plan") or ""),
                requires=payload.get("requires") or {},
                memories=memories,
                existing_ideas=existing_ideas,
                existing_methods=existing_methods,
                world_state=world,
                traits=self.traits,
                min_potential=self.config.min_potential,
            )
            preview = build_idea(
                candidate=candidate,
                phrased=payload,
                persona=self.agent_name,
                created_at=str(self.clock.get_time()),
                expires_at=self._expiry(),
                decision=decision,
            )
            if not decision.accepted:
                if "duplicates_existing_idea_semantics" in decision.reasons and decision.duplicate_of:
                    # KI-5: the same insight, re-derived from a new partner
                    # memory. Record the refinement; do not store a second idea.
                    result["refined"] += 1
                    self.store.refined(
                        idea_id=decision.duplicate_of,
                        week=self._week_key(),
                        duplicate_of=decision.duplicate_of,
                        candidate=candidate.to_dict(),
                    )
                    continue
                result["rejected"] += 1
                self.store.rejected(
                    idea_id=preview.idea_id,
                    week=self._week_key(),
                    reasons=decision.reasons,
                    candidate=candidate.to_dict(),
                    # Diagnostics: without these a `below_min_potential`
                    # rejection is a dead end (run 09261721 evidence).
                    potential=round(float(decision.potential.potential), 6),
                    score_components=decision.potential.to_dict(),
                    feasibility=round(float(decision.feasibility), 4),
                )
                continue
            scored.append((decision.potential.potential, candidate, payload, decision, preview))

        scored.sort(key=lambda item: (-item[0], item[4].idea_id))
        for _potential, candidate, payload, decision, _preview in scored[:budget]:
            idea = build_idea(
                candidate=candidate,
                phrased=payload,
                persona=self.agent_name,
                created_at=str(self.clock.get_time()),
                expires_at=self._expiry(),
                decision=decision,
            )
            _, created = self.store.created(idea, week=self._week_key())
            if not created:
                continue
            self.store.scored(
                idea_id=idea.idea_id,
                week=self._week_key(),
                potential=idea.potential,
                components=idea.score_components,
            )
            result["stored"] += 1
            existing_ideas.append(idea)

        result["converted"] = self._convert(existing_ideas, result=result)

        if result["stored"] or result["rejected"] or result["converted"]:
            write_idea_view(self.dm)
            IDEA_LOGGER.info(
                f"[{self.agent_name}] ideas {self._week_key()}: "
                f"candidates={result['candidates']} stored={result['stored']} "
                f"rejected={result['rejected']} converted={result['converted']}"
            )
        return result

    # -- steps --------------------------------------------------------------
    def _phrase(
        self,
        candidates: Sequence[IdeaCandidate],
        memories: Sequence[MemoryItem],
        world: IdeaWorldState,
    ) -> Dict[int, Dict[str, Any]]:
        """One LLM call to phrase the candidates; failures mean "no ideas"."""
        from src.utils import get_response_with_retry

        by_id = {m.memory_id: m for m in memories}
        payload_candidates: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            entry = candidate.to_dict()
            entry["memory_contents"] = [
                (by_id[mid].content[:120] if mid in by_id else mid)
                for mid in candidate.memory_ids
            ]
            entry["hint"] = candidate.hint
            payload_candidates.append(entry)

        prompt = build_idea_prompt(
            char=self.agent_name,
            candidates=payload_candidates,
            skills=sorted(world.skills),
            entities=sorted(world.entities),
            money=world.deposit,
            language=self.language,
        )

        def _validator(response: str, **kwargs):
            return parse_idea_response(response)

        try:
            messages = [{"role": "system", "content": prompt}]
            data = get_response_with_retry(
                post_processing_funcs=[_validator],
                model=self.model,
                messages=messages,
                max_retry=2,
            )
            from src.agents.cognition.usage import record_llm_usage

            record_llm_usage(
                self.dm,
                "idea_phrasing",
                messages=messages,
                response=data,
                model=self.model,
            )
        except Exception as e:  # pragma: no cover - shadow never breaks a run
            IDEA_LOGGER.warning(f"[{self.agent_name}] idea phrasing failed: {e!r}")
            return {}
        if not isinstance(data, dict):
            return {}

        out: Dict[int, Dict[str, Any]] = {}
        for item in data.get("ideas") or []:
            try:
                index = int(item.get("candidate_index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates) and index not in out:
                out[index] = item
        return out

    def _convert(self, ideas: Sequence[Idea], *, result: Optional[Dict[str, int]] = None) -> int:
        """Let the best fresh idea become a candidate methodology (value 0)."""
        eligible = [
            idea
            for idea in ideas
            if idea.status == "candidate"
            and not idea.candidate_method_id
            and idea.potential >= CONVERSION_MIN_POTENTIAL
            and idea.test_plan
        ]
        if not eligible:
            return 0
        # Ideas grown from the character's own lessons convert first: the whole
        # point of the lesson bridge is that "what I concluded" reaches the
        # menu, and there is only one conversion slot per week by default.
        eligible.sort(
            key=lambda i: (
                0 if i.motif == "lesson_application" else 1,
                -i.potential,
                i.idea_id,
            )
        )

        written = 0
        for idea in eligible[:CONVERSIONS_PER_WEEK]:
            skill_id = (idea.related_skill_ids or ["unmapped"])[0]
            method_id = method_id_for_idea(idea.idea_id, skill_id)
            # KI-6: an idea is a hypothesis, a method is a recipe. The recipe is
            # written by one budgeted call; whatever it cannot fill is reported.
            methodised = self._methodize(idea, skill_id=skill_id)
            gaps = _structure_gaps(methodised)
            if gaps:
                IDEA_LOGGER.error(
                    "[%s] methodisation produced an incomplete method: "
                    "method_id=%s idea_id=%s week=%s motif=%s missing=%s model=%s",
                    self.agent_name,
                    method_id,
                    idea.idea_id,
                    self._week_key(),
                    idea.motif,
                    ",".join(gaps),
                    self.model,
                )
            payload = {
                "skill_id": skill_id,
                "title": methodised.get("title") or idea.content[:24],
                "description": methodised.get("description") or idea.content,
                "source_type": "idea_conversion",
                "status": "proposed",
                "steps": methodised.get("steps")
                or ([idea.test_plan] if idea.test_plan else []),
                "checks": methodised.get("checks") or [],
                "failure_modes": methodised.get("failure_modes") or [],
                "applicable_contexts": methodised.get("applicable_contexts") or [],
                "contraindications": methodised.get("contraindications") or [],
                "source_idea_id": idea.idea_id,
                "source_memory_ids": list(idea.source_memory_ids),
                # Which motif produced the idea. The hint provider uses it to
                # give the character's own lessons practice priority (phase 4.5).
                "source_motif": idea.motif,
                "week": self._week_key(),
                # A converted idea is a candidate, never evidence: the value and
                # confidence of a methodology may only move through practice.
                "global_value": 0.0,
                "confidence": 0.0,
            }
            if gaps and result is not None:
                result["structure_incomplete"] = (
                    result.get("structure_incomplete", 0) + 1
                )
                payload["structure_incomplete"] = True
                payload["structure_missing"] = gaps
            _, created = self.capability.append(
                "METHOD_PROPOSED",
                method_id=method_id,
                payload=payload,
                # Week-independent on purpose: one idea yields one candidate
                # methodology, however often the pipeline re-runs it.
                idempotency_key=f"METHOD_PROPOSED:{method_id}:-:idea-conversion",
            )
            if not created:
                continue
            self.store.converted(
                idea_id=idea.idea_id, week=self._week_key(), method_id=method_id
            )
            idea.candidate_method_id = method_id
            written += 1
        return written

    # -- helpers ------------------------------------------------------------
    def _methodize(self, idea: Idea, *, skill_id: str) -> Dict[str, Any]:
        """One budgeted LLM call that turns the idea into a usable method (KI-6).

        Failures are not fatal: the caller stores the method with whatever the
        idea already provides and marks the structure as incomplete.
        """
        from src.utils import get_response_with_retry

        if not self._conversion_budget_available():
            IDEA_LOGGER.info(
                f"[{self.agent_name}] methodisation budget exhausted for {self._week_key()}"
            )
            return {}
        self._consume_conversion_budget()

        memories: List[MemoryItem] = []
        try:
            by_id = {m.memory_id: m for m in load_memories(self.dm)}
            memories = [by_id[mid] for mid in idea.source_memory_ids if mid in by_id]
        except Exception:  # pragma: no cover - defensive
            memories = []
        prompt = build_methodization_prompt(
            idea_content=idea.content,
            test_plan=idea.test_plan,
            motif=idea.motif,
            skill=skill_id,
            shared_focus=str(getattr(idea, "shared_focus", "") or ""),
            goal=str(getattr(idea, "goal", "") or ""),
            obstacles=list(getattr(idea, "obstacles", []) or []),
            resources=list(getattr(idea, "resources", []) or []),
            memories=[m.content[:160] for m in memories],
            language=self.language,
        )

        def _validator(response: str, **kwargs):
            return parse_methodization_response(response)

        try:
            messages = [{"role": "system", "content": prompt}]
            data = get_response_with_retry(
                post_processing_funcs=[_validator],
                model=self.model,
                messages=messages,
                max_retry=2,
            )
            from src.agents.cognition.usage import record_llm_usage

            record_llm_usage(
                self.dm,
                "idea_methodisation",
                messages=messages,
                response=data,
                model=self.model,
            )
        except Exception as e:  # pragma: no cover - shadow never breaks a run
            IDEA_LOGGER.warning(f"[{self.agent_name}] methodisation failed: {e!r}")
            return {}
        return data if isinstance(data, dict) else {}

    def _conversion_budget_available(self) -> bool:
        return (
            self._conversion_calls.get(self._week_key(), 0)
            < self.config.max_conversion_calls_per_week
        )

    def _consume_conversion_budget(self) -> None:
        key = self._week_key()
        self._conversion_calls[key] = self._conversion_calls.get(key, 0) + 1

    def _record_skip(self, reason: str, detail: Optional[Dict[str, Any]] = None) -> None:
        """Persist the fact that this week produced nothing, and why."""
        try:
            self.store.skipped(week=self._week_key(), reason=reason, detail=detail or {})
        except Exception as e:  # pragma: no cover - observability must not break a run
            IDEA_LOGGER.warning(f"[{self.agent_name}] recording idea skip failed: {e!r}")

    def _existing_ideas(self) -> List[Idea]:
        from src.agents.cognition.idea_store import load_ideas

        return load_ideas(self.dm)

    def _existing_methods(self) -> List[Any]:
        from src.agents.cognition import materializer

        return materializer.load_methodologies(materializer.build_capability_view(self.dm))

    def _week_key(self) -> str:
        t = self.clock.get_time()
        return f"Y{t.year}-W{t.week:02d}"

    def _expiry(self) -> str:
        t = self.clock.get_time()
        week = t.week + 4
        year = t.year
        n_week = 10
        try:
            from src.config import get_config

            n_week = int(get_config()["world"]["time"]["n_week"])
        except Exception:
            pass
        while week > n_week:
            week -= n_week
            year += 1
        return f"Y{year}-W{week:02d}"

    def _budget_available(self) -> bool:
        return (
            self._calls.get(self._week_key(), 0)
            < self.config.max_extract_calls_per_week
        )

    def _consume_budget(self) -> None:
        key = self._week_key()
        self._calls[key] = self._calls.get(key, 0) + 1

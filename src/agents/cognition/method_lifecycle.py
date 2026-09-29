"""Phase 5: the rest of a methodology's life — specialised, archived, versioned.

Design doc 阶段 5 lists three things beyond the bandit itself: **method
archiving**, **method version evolution**, and (from the reward design, §3.3)
the `specialized` state. Phase 1 already produced `proposed → tested →
validated` and a `deprecated` path in `reward_model.status_for`; the states below
were declared in `models.METHOD_STATUSES` but nothing ever wrote them, which is
why "同义方法重复率" and "长期低 value 方法" were unanswerable.

Three rules, all deterministic and all rebuildable from the event stream:

1. **`specialized`** — a method whose value in *one* situation is clearly above
   its pooled value is not "good", it is "good here". Marked from evidence, with
   a sample floor so one lucky week cannot specialise anything.
2. **`archived`** — a method that keeps failing and has not been used for weeks
   leaves the menu, but is never deleted (invariant #4: forgetting lowers
   accessibility, it does not erase history). Archived methods stay in the view
   with the week and reason, and a later refinement can bring one back.
3. **versioned** — when a later week re-derives a method that is already known
   under a slightly different wording, that is a **refinement of the same
   method**, not a new one: `METHOD_REFINED` with `version + 1` and
   `parent_method_id` pointing at itself. This is what keeps "同义方法重复率"
   from growing with every extraction.

The module decides; it does not schedule itself. `MethodLifecycle.settle_week()`
is the writer, called from the weekly settlement next to the memory settle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.agents.cognition.models import Methodology
from src.agents.cognition.reward_model import (
    DEPRECATED_MAX_VALUE,
    DEPRECATED_MIN_PRACTICES,
)

# A method is "good *here*" when one context beats its pooled value by this much.
SPECIALIZED_CONTEXT_MARGIN = 0.2
SPECIALIZED_MIN_CONTEXT_SAMPLES = 3

# A failing method leaves the menu after this many weeks without a practice.
ARCHIVE_AFTER_UNUSED_WEEKS = 6

# Two titles this close are the same method, differently worded. The blend of
# Jaccard and containment (with a floor on shared material) is the same rule the
# lesson bridge uses for "is this a rewrite?" — CJK titles are short, and pure
# Jaccard under-reports an obvious refinement like
# 「先定冲突再写场景」 -> 「先定冲突再排场景顺序」.
REFINEMENT_TITLE_THRESHOLD = 0.6
MIN_SHARED_BIGRAMS = 3

_WEEK_RE = re.compile(r"^Y(\d{4})-W(\d{1,2})$")
_TITLE_NOISE = re.compile(r"[\s，。、：:；;（）()【】\[\]！!？?\"'“”‘’\-—_/·]+")


# --- configuration -----------------------------------------------------------


@dataclass
class LifecycleConfig:
    """`world.cognition` lifecycle settings, inert by default."""

    enabled: bool = False
    specialized_margin: float = SPECIALIZED_CONTEXT_MARGIN
    specialized_min_samples: int = SPECIALIZED_MIN_CONTEXT_SAMPLES
    archive_after_unused_weeks: int = ARCHIVE_AFTER_UNUSED_WEEKS
    refinement_threshold: float = REFINEMENT_TITLE_THRESHOLD
    weeks_per_year: int = 10

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "LifecycleConfig":
        section = (world_cfg or {}).get("cognition") or {}
        time_section = (world_cfg or {}).get("time") or {}

        def _float(key: str, default: float) -> float:
            try:
                return float(section.get(key, default))
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int) -> int:
            try:
                return int(section.get(key, default))
            except (TypeError, ValueError):
                return default

        try:
            weeks_per_year = int(time_section.get("n_week", 10))
        except (TypeError, ValueError):
            weeks_per_year = 10

        return LifecycleConfig(
            enabled=bool(section.get("method_lifecycle", False)),
            specialized_margin=_float("method_specialized_margin", SPECIALIZED_CONTEXT_MARGIN),
            specialized_min_samples=max(
                1, _int("method_specialized_min_samples", SPECIALIZED_MIN_CONTEXT_SAMPLES)
            ),
            archive_after_unused_weeks=max(
                1, _int("method_archive_after_weeks", ARCHIVE_AFTER_UNUSED_WEEKS)
            ),
            refinement_threshold=_float(
                "method_refinement_threshold", REFINEMENT_TITLE_THRESHOLD
            ),
            weeks_per_year=max(1, weeks_per_year),
        )


# --- week arithmetic ---------------------------------------------------------


def week_number(week: str, *, weeks_per_year: int = 10) -> Optional[int]:
    """`Y2020-W03` -> an absolute week counter.

    `weeks_per_year` comes from `world.time.n_week` (10 in this world): a
    calendar year is not 100 weeks, so the counter has to be told how long one is.
    """
    match = _WEEK_RE.match(str(week or "").strip())
    if not match:
        return None
    span = max(1, int(weeks_per_year))
    return int(match.group(1)) * span + int(match.group(2))


def weeks_between(earlier: str, later: str, *, weeks_per_year: int = 10) -> Optional[int]:
    """How many simulated weeks separate two week keys; `None` if unparsable."""
    a = week_number(earlier, weeks_per_year=weeks_per_year)
    b = week_number(later, weeks_per_year=weeks_per_year)
    if a is None or b is None:
        return None
    return max(0, b - a)


def week_of_layout(time_str: str) -> str:
    """`Y2020-W03-activity-D2` -> `Y2020-W03`."""
    parts = str(time_str or "").split("-")
    if len(parts) >= 2 and parts[0].startswith("Y") and parts[1].startswith("W"):
        return f"{parts[0]}-{parts[1]}"
    return ""


# --- specialised -------------------------------------------------------------


@dataclass
class Specialization:
    """The situation where a method does clearly better than it does overall."""

    context_key: str
    value: float
    count: int
    margin: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_key": self.context_key,
            "value": round(self.value, 4),
            "count": int(self.count),
            "margin": round(self.margin, 4),
        }


def strongest_context(
    context_values: Dict[str, Any],
    *,
    pooled: float,
    margin: float = SPECIALIZED_CONTEXT_MARGIN,
    min_samples: int = SPECIALIZED_MIN_CONTEXT_SAMPLES,
) -> Optional[Specialization]:
    """The strongest context clearly above the pooled value, or `None`.

    Kept separate from `specialization_of` so the value update (which has the
    numbers but not a `Methodology` object) can ask the same question.
    """
    best: Optional[Specialization] = None
    for key, entry in sorted((context_values or {}).items()):
        count = int((entry or {}).get("count", 0))
        if count < max(1, min_samples):
            continue
        value = float((entry or {}).get("value", 0.0))
        delta = value - float(pooled)
        if delta < margin:
            continue
        if best is None or value > best.value:
            best = Specialization(
                context_key=str(key), value=value, count=count, margin=delta
            )
    return best


def specialization_of(
    method: Methodology,
    *,
    margin: float = SPECIALIZED_CONTEXT_MARGIN,
    min_samples: int = SPECIALIZED_MIN_CONTEXT_SAMPLES,
) -> Optional[Specialization]:
    """Find the one context where the method is clearly above its own average.

    A method that is uniformly good is `validated`, not `specialized`: the margin
    has to be relative to the method's own pooled value, never absolute.
    """
    return strongest_context(
        method.context_values,
        pooled=float(method.global_value),
        margin=margin,
        min_samples=min_samples,
    )


# --- archiving ---------------------------------------------------------------


def archive_reason(
    method: Methodology,
    *,
    current_week: str,
    after_weeks: int = ARCHIVE_AFTER_UNUSED_WEEKS,
    weeks_per_year: int = 10,
) -> Optional[str]:
    """Why this method should leave the menu now, or `None`.

    Only *failing* methods are archived (`deprecated`, or the same numbers that
    would deprecate them), and only after they have gone unused for long enough
    that the character plausibly stopped reaching for them. A method with no
    practice evidence is never archived: it has not had its chance yet.
    """
    if str(method.status) in ("archived",):
        return None
    if int(method.practice_count or 0) <= 0:
        return None

    failing = (
        str(method.status) == "deprecated"
        or (
            int(method.practice_count) >= DEPRECATED_MIN_PRACTICES
            and float(method.global_value) <= DEPRECATED_MAX_VALUE
        )
    )
    if not failing:
        return None

    last_used = str(method.last_used or "")
    last_week = week_of_layout(last_used)
    if not last_week:
        return "never_used_but_failing"
    elapsed = weeks_between(last_week, current_week, weeks_per_year=weeks_per_year)
    if elapsed is None:
        return None
    if elapsed < max(1, after_weeks):
        return None
    return f"failing_and_unused_for_{elapsed}_weeks"


# --- version evolution -------------------------------------------------------


def normalize_title(text: str) -> str:
    return _TITLE_NOISE.sub("", str(text or "")).lower()


def _bigrams(text: str) -> set[str]:
    normalized = normalize_title(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def title_similarity(a: str, b: str) -> float:
    """How much two method titles say the same thing.

    `max(Jaccard, containment)`, with a floor on shared material: containment is
    what recognises a refinement that only *adds* detail, and the floor stops a
    short generic title from matching every longer one. Robust for short CJK
    titles, where bigrams are the natural unit.
    """
    left, right = _bigrams(a), _bigrams(b)
    if not left or not right:
        return 0.0
    shared = left & right
    if len(shared) < MIN_SHARED_BIGRAMS:
        return 0.0
    jaccard = len(shared) / len(left | right)
    containment = len(shared) / min(len(left), len(right))
    return round(max(jaccard, containment), 4)


def refinement_target(
    title: str,
    methods: Iterable[Methodology],
    *,
    threshold: float = REFINEMENT_TITLE_THRESHOLD,
    exclude: Sequence[str] = (),
) -> Optional[Tuple[Methodology, float]]:
    """The existing method a newly extracted title is really a new version of.

    Deterministic: the highest similarity wins, ties break on `method_id`, and a
    title that is identical (same id) is not a refinement — that case is already
    deduplicated by identity.
    """
    best: Optional[Tuple[Methodology, float]] = None
    skip = set(exclude)
    for method in methods:
        if method.method_id in skip:
            continue
        score = title_similarity(title, method.title)
        if score < threshold:
            continue
        if best is None or score > best[1] or (
            score == best[1] and method.method_id < best[0].method_id
        ):
            best = (method, score)
    return best


@dataclass
class Refinement:
    """A same-method, better-worded version derived from later evidence."""

    method_id: str
    version: int
    parent_method_id: str
    title: str
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "parent_method_id": self.parent_method_id,
            "title": self.title,
            "similarity": round(self.similarity, 4),
        }


def plan_refinement(
    candidate: Dict[str, Any],
    methods: Iterable[Methodology],
    *,
    threshold: float = REFINEMENT_TITLE_THRESHOLD,
) -> Optional[Refinement]:
    """Turn an extracted candidate into a refinement of an existing method."""
    title = str(candidate.get("title") or "").strip()
    if not title:
        return None
    match = refinement_target(title, methods, threshold=threshold)
    if match is None:
        return None
    method, score = match
    return Refinement(
        method_id=method.method_id,
        version=int(method.version or 1) + 1,
        parent_method_id=method.parent_method_id or method.method_id,
        title=title,
        similarity=score,
    )


# --- the weekly writer -------------------------------------------------------


class MethodLifecycle:
    """Weekly settlement of method states (archive + refine).

    Writes only `METHOD_ARCHIVED` and `METHOD_REFINED`; both are idempotent per
    (method, week), so a resumed week repeats nothing.
    """

    def __init__(self, *, dm, clock, config: Optional[LifecycleConfig] = None) -> None:
        self.dm = dm
        self.clock = clock
        self.config = config or LifecycleConfig()
        from src.agents.cognition.event_store import CapabilityEventStore

        self.store = CapabilityEventStore(dm)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _week_key(self) -> str:
        t = self.clock.get_time()
        return f"Y{t.year}-W{t.week:02d}"

    def methods(self) -> List[Methodology]:
        from src.agents.cognition import materializer

        return materializer.load_methodologies(materializer.build_capability_view(self.dm))

    def archive_failing_methods(self) -> int:
        """Retire failing, long-unused methods (they stay in the view)."""
        if not self.enabled:
            return 0
        week = self._week_key()
        written = 0
        for method in self.methods():
            reason = archive_reason(
                method,
                current_week=week,
                after_weeks=self.config.archive_after_unused_weeks,
            )
            if reason is None:
                continue
            _, created = self.store.append(
                "METHOD_ARCHIVED",
                method_id=method.method_id,
                payload={
                    "week": week,
                    "reason": reason,
                    "global_value": method.global_value,
                    "practice_count": method.practice_count,
                    "last_used": method.last_used,
                },
                idempotency_key=f"METHOD_ARCHIVED:{method.method_id}:{week}",
            )
            written += int(created)
        return written

    def record_refinement(self, candidate: Dict[str, Any]) -> Optional[Refinement]:
        """Fold a re-derived method into the version of the one already known."""
        if not self.enabled:
            return None
        plan = plan_refinement(
            candidate,
            self.methods(),
            threshold=self.config.refinement_threshold,
        )
        if plan is None:
            return None
        payload = {
            "week": self._week_key(),
            "title": plan.title,
            "parent_method_id": plan.parent_method_id,
            "similarity": round(plan.similarity, 4),
            "source_type": "refinement",
        }
        for key in ("description", "applicable_contexts", "contraindications",
                    "steps", "checks", "failure_modes", "skill_id"):
            if candidate.get(key):
                payload[key] = candidate[key]
        _, created = self.store.append(
            "METHOD_REFINED",
            method_id=plan.method_id,
            payload=payload,
            idempotency_key=(
                f"METHOD_REFINED:{plan.method_id}:v{plan.version}:{self._week_key()}"
            ),
        )
        return plan if created else None

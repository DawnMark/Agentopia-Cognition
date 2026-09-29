"""Skill-name families: a side mapping, not a migration (phase-5 follow-up).

The design document's concern (doc §3.2 / open question) was that "未知 skill
名称可以直接创建，容易产生同义词碎片和数值通胀". Phase 5 finally measured it,
and the measurement splits the concern in two:

1. **Inside a character there is almost no fragmentation.** Across both
   acceptance runs, every method's `skill_id` except one was already a skill the
   character had — the extraction prompt's "use existing skill names" is being
   obeyed. `unmapped_skill_methods` is therefore a non-issue in practice.
2. **Across characters there is real fragmentation, and it is created by the
   environment model**, not by the cognition layer: God coins `长跑` for one
   character, `长跑耐力` for another and `长跑与体能` for a third, and
   `电动车维修` / `电动车保养`, `太极` / `太极拳`, `厨艺` / `烹饪` the same way.
   Those names live in `state.jsonl` and professions, salaries and reward maths
   key off them, so renaming them is a phase-6 migration, not a side effect.

So this module does the *side mapping* the design allows for a first version:
it groups names into capability families for measurement and reporting, and
never rewrites `state.jsonl`, an event or a method. The world supplies the
families in `data/<world>/skill_aliases.json`; anything a world author has not
mapped is reported with a concrete suggestion instead of being guessed at.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ALIAS_FILENAME = "skill_aliases.json"

_PUNCT = re.compile(r"[\s，。、；：！？·\"'“”‘’（）()【】\[\]《》<>…—\-_/\\,.!?:;]+")
# Names with too little material in common are not "the same skill, spelled
# differently" and must not be merged by a heuristic.
MIN_SHARED_CHARS = 2


def normalize_skill(value: Any) -> str:
    """Case/punctuation-insensitive form used for family lookup."""
    return _PUNCT.sub("", str(value or "").strip().lower())


@dataclass
class SkillFamilies:
    """Canonical capability families, with the fragments observed per family."""

    family_of: Dict[str, str] = field(default_factory=dict)
    members: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def families(self) -> List[str]:
        return sorted(self.members)

    def canonical(self, name: str) -> str:
        """The family a skill name belongs to, or the name itself when unmapped."""
        text = str(name or "").strip()
        if not text:
            return ""
        return self.family_of.get(normalize_skill(text), text)

    def merged(self, name: str) -> bool:
        return normalize_skill(name) in self.family_of and self.family_of[
            normalize_skill(name)
        ] != str(name).strip()

    def fragments(self, names: Iterable[str]) -> Dict[str, List[str]]:
        """Group observed names by family; only families with >1 fragment matter."""
        grouped: Dict[str, set] = {}
        for name in names:
            if not str(name or "").strip():
                continue
            grouped.setdefault(self.canonical(name), set()).add(str(name).strip())
        return {
            family: sorted(members)
            for family, members in sorted(grouped.items())
            if len(members) > 1
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "families": len(self.members),
            "mapped_names": len(self.family_of),
            "members": {k: sorted(v) for k, v in sorted(self.members.items())},
        }

    @staticmethod
    def from_world_dir(world_dir: Path) -> "SkillFamilies":
        """Load `data/<world>/skill_aliases.json`; an absent file means no mapping."""
        path = Path(world_dir) / ALIAS_FILENAME
        if not path.exists():
            return SkillFamilies()
        try:
            payload = json.loads(io.open(path, encoding="utf-8").read())
        except (OSError, ValueError):
            return SkillFamilies()
        out = SkillFamilies()
        for entry in payload.get("skills") or []:
            canonical = str(entry.get("canonical") or "").strip()
            if not canonical:
                continue
            names = [canonical, *(entry.get("aliases") or [])]
            out.members[canonical] = sorted({str(n).strip() for n in names if str(n).strip()})
            for name in names:
                text = normalize_skill(name)
                if text:
                    out.family_of[text] = canonical
        return out

    @staticmethod
    def from_data_manager(dm) -> "SkillFamilies":
        """The families for the world this DataManager belongs to."""
        try:
            world_dir = Path("data") / str(dm.world)
        except Exception:  # pragma: no cover - defensive
            return SkillFamilies()
        return SkillFamilies.from_world_dir(world_dir)


def similar_names(name: str, candidates: Sequence[str], *, limit: int = 3) -> List[str]:
    """The candidate skills a name most looks like, for an actionable report.

    Deliberately crude and deterministic: shared characters, longest first. It is
    a *suggestion* for a world author to turn into an alias, never a merge that
    happens on its own — guessing that 「太极」and 「太极拳」are one capability is
    reasonable, guessing it for 「体能训练」and 「长跑与体能」is a judgement call.
    """
    target = set(normalize_skill(name))
    if len(target) < MIN_SHARED_CHARS:
        return []
    scored: List[tuple[int, str]] = []
    for candidate in candidates:
        other = set(normalize_skill(candidate))
        shared = len(target & other)
        if shared >= MIN_SHARED_CHARS:
            scored.append((shared, str(candidate)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:limit]]

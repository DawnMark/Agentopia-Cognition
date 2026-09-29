"""Per-persona "impressions": the alias book used when organising memories (KI-2).

The problem this solves
-----------------------
Phase 2 validation used to compare every extracted entity against an exact-match
allow-list built from the world's *internal* names (persona directory names,
contact file stems, possessions, location keys). In a Chinese world those keys
are English ids (`Breakfast_Stall_Street`) while the character naturally writes
"早点摊"; the character also writes short names ("日和" for "吉日和"). The result
was measured at a **98–99% entity drop rate**, which stripped the relation graph
of its strongest anchor and starved the idea motifs (KI-2, KI-3, KI-4, KI-7).

The design (user decision, 2026-09-26)
--------------------------------------
Every persona keeps an **impressions document** under `memory/`:

    persona/<name>/memory/impressions.json           materialised view (rebuildable)
    persona/<name>/memory/impression_events.jsonl    append-only source of truth

Content: the canonical entities this persona knows (people / places /
possessions), a dictionary of *aliases* → canonical name, and the names it has
mentioned but that cannot be resolved yet ("soft entities", kept instead of
dropped).

Aliases come from two places, both offline and deterministic:

1. **world seed** — persona names, contact partners, possessions, location keys
   and display names, plus the world's optional `entity_aliases.json`
   (`data/<world>/entity_aliases.json`) which a world author uses to give
   human-readable names to machine-readable ids;
2. **runtime learning** — when a week's extraction produces a name that is not
   canonical but *does* resolve to one (by alias, containment or a learned
   alias), the mapping is written back as an `ALIAS_LEARNED` event, so the next
   week resolves it for free.

Everything here is pure/deterministic and works without an LLM call; nothing in
this module touches prompts, plans, state or skill numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from src.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

IMPRESSION_LOGGER = get_logger("cognition", quiet=True)

VIEW_FILENAME = "impressions.json"
EVENT_FILENAME = "impression_events.jsonl"
WORLD_ALIAS_FILENAME = "entity_aliases.json"

IMPRESSION_EVENT_TYPES = (
    "IMPRESSION_SEEDED",
    "ALIAS_LEARNED",
    "SOFT_ENTITY_SEEN",
)

# Uncertainty discounts. A canonical entity is trusted; a soft entity is kept so
# the information is not lost, but it makes the memory slightly less certain.
SOFT_ENTITY_CONFIDENCE_FACTOR = 0.9

_PUNCT = re.compile(r"[\s，。、；：！？·\"'“”‘’（）()【】\[\]《》<>…—\-_/\\,.!?:;]+")
# Names shorter than this are too ambiguous to match by containment.
MIN_CONTAINMENT_LEN = 2
MAX_ENTITY_LEN = 40


def normalize_name(value: Any) -> str:
    """Case/punctuation-insensitive form used for alias lookup."""
    text = str(value or "").strip().lower()
    return _PUNCT.sub("", text)


def name_alias_candidates(name: str) -> List[str]:
    """Deterministic short forms of a name ("吉日和" → 日和, 上官霄月 → 霄月).

    Both the world's characters and its locations are written by the model the
    way people actually speak: the surname or the machine-readable prefix is
    dropped. Only unambiguous, purely additive candidates are generated here;
    collisions are resolved by the caller (an alias that maps to two canonicals
    is discarded).
    """
    name = str(name or "").strip()
    if not name:
        return []
    out = [name]
    core = name.split("/")[-1].strip()
    if core and core != name:
        out.append(core)
    cjk = bool(core) and all("一" <= ch <= "鿿" for ch in core)
    short_forms = [core] if core else []
    if cjk and len(core) >= 3:
        short_forms.append(core[-2:])
        if len(core) >= 4:
            short_forms.append(core[-3:])
    out.extend(short_forms)
    # A home is spoken about both as "吉日和的房间" and as "日和屋"; machine
    # readable ids never take a Chinese suffix.
    for form in short_forms:
        if form and all("一" <= ch <= "鿿" for ch in form):
            out.extend([f"{form}的房间", f"{form}屋", f"{form}家"])
    return [v for v in dict.fromkeys(v for v in out if v)]

def _world_alias_entries(source: Path) -> List[Dict[str, Any]]:
    if not source.exists():
        return []
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = data.get("entities") if isinstance(data, dict) else data
    return [e for e in (entries or []) if isinstance(e, dict) and e.get("canonical")]


@dataclass
class Impressions:
    """Folded view: canonical entities, alias dictionary, soft entities."""

    persona: str = ""
    canonical: Dict[str, List[str]] = field(default_factory=dict)  # kind -> names
    aliases: Dict[str, str] = field(default_factory=dict)  # normalized -> canonical
    soft_entities: Dict[str, int] = field(default_factory=dict)  # name -> times seen
    seeded: bool = False
    updated_at: str = ""

    # -- queries -----------------------------------------------------------
    def canonical_names(self) -> Set[str]:
        out: Set[str] = set()
        for names in self.canonical.values():
            out.update(names)
        return out

    def known(self) -> Set[str]:
        """Everything that counts as a known entity for validation."""
        return set(self.canonical_names()) | set(self.aliases.values())

    def resolve(self, name: Any) -> Tuple[Optional[str], str]:
        """Return `(canonical, how)`; `how` is exact | alias | containment | ""."""
        raw = str(name or "").strip()
        if not raw:
            return None, ""
        normalized = normalize_name(raw)
        if not normalized:
            return None, ""
        canonicals = self.canonical_names()
        if raw in canonicals:
            return raw, "exact"
        hit = self.aliases.get(normalized)
        if hit:
            return hit, "alias"
        # Containment: "九亭地铁站" ↔ "JiuTing_Metro_Station"'s alias "地铁站".
        if len(normalized) >= MIN_CONTAINMENT_LEN:
            best: Optional[str] = None
            best_len = 0
            ambiguous = False
            pool = list(canonicals) + list(self.aliases.keys())
            for candidate in pool:
                key = normalize_name(candidate)
                if not key or key == normalized:
                    continue
                if normalized in key or key in normalized:
                    canonical = candidate if candidate in canonicals else self.aliases[key]
                    if canonical is None:
                        continue
                    if len(key) > best_len:
                        best, best_len, ambiguous = canonical, len(key), False
                    elif len(key) == best_len and canonical != best:
                        ambiguous = True
            if best and not ambiguous:
                return best, "containment"
        return None, ""

    def canonicalize(self, names: Iterable[Any]) -> Tuple[List[str], List[str]]:
        """Split extracted names into `(canonical, unresolved)`."""
        canonical: List[str] = []
        unresolved: List[str] = []
        for name in names:
            raw = str(name or "").strip()
            if not raw or len(raw) > MAX_ENTITY_LEN:
                continue
            found, _how = self.resolve(raw)
            if found:
                if found not in canonical:
                    canonical.append(found)
            elif raw not in unresolved:
                unresolved.append(raw)
        return canonical, unresolved

    def mentions(self, name: Any) -> bool:
        """Whether the character has mentioned this (possibly soft) name."""
        raw = str(name or "").strip()
        if not raw:
            return False
        if raw in self.soft_entities:
            return True
        found, _ = self.resolve(raw)
        return bool(found)

    # -- folding -----------------------------------------------------------
    @classmethod
    def from_events(cls, events: Sequence[Dict[str, Any]], *, persona: str) -> "Impressions":
        view = cls(persona=persona)
        for event in events:
            event_type = str(event.get("type") or "")
            if event_type == "IMPRESSION_SEEDED":
                view.seeded = True
                for kind, names in (event.get("canonical") or {}).items():
                    bucket = view.canonical.setdefault(str(kind), [])
                    for name in names or []:
                        if str(name) not in bucket:
                            bucket.append(str(name))
                for alias, canonical in (event.get("aliases") or {}).items():
                    view.aliases[normalize_name(alias)] = str(canonical)
            elif event_type == "ALIAS_LEARNED":
                alias = normalize_name(event.get("alias"))
                canonical = str(event.get("canonical") or "")
                if alias and canonical:
                    view.aliases[alias] = canonical
                    view.seeded = view.seeded
            elif event_type == "SOFT_ENTITY_SEEN":
                name = str(event.get("name") or "").strip()
                if name:
                    view.soft_entities[name] = view.soft_entities.get(name, 0) + 1
            view.updated_at = str(event.get("time") or view.updated_at)
        return view

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "persona": self.persona,
            "seeded": self.seeded,
            "updated_at": self.updated_at,
            "canonical": {k: sorted(v) for k, v in sorted(self.canonical.items())},
            "aliases": dict(sorted(self.aliases.items())),
            "soft_entities": dict(sorted(self.soft_entities.items())),
            "stats": {
                "canonical": len(self.canonical_names()),
                "aliases": len(self.aliases),
                "soft_entities": len(self.soft_entities),
            },
        }


class ImpressionKeeper:
    """Read/write one persona's impressions (view + append-only events)."""

    def __init__(self, *, dm: "DataManager", clock: Any, persona: str) -> None:
        self.dm = dm
        self.clock = clock
        self.persona = persona
        self.dir: Path = dm.root / "memory"
        self.path: Path = self.dir / EVENT_FILENAME
        self.view_path: Path = self.dir / VIEW_FILENAME
        self._impressions: Optional[Impressions] = None

    # -- storage -----------------------------------------------------------
    def events(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _append(self, event_type: str, payload: Dict[str, Any], *, key: str) -> bool:
        if event_type not in IMPRESSION_EVENT_TYPES:
            raise ValueError(f"unknown impression event type: {event_type!r}")
        if any(
            str(e.get("idempotency_key") or "") == key for e in self.events()
        ):
            return False
        record: Dict[str, Any] = {"type": event_type, "idempotency_key": key}
        for k, v in payload.items():
            if k in ("time", "ledger_event_id", "schema_version", "idempotency_key"):
                continue
            record[k] = v
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dm.append_ledger_record(self.path, record, idempotency_key=key)
        return True

    def invalidate(self) -> None:
        self._impressions = None

    # -- view --------------------------------------------------------------
    def impressions(self) -> Impressions:
        if self._impressions is None:
            self._impressions = Impressions.from_events(self.events(), persona=self.persona)
        return self._impressions

    def save_view(self) -> Path:
        view = self.impressions()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.view_path.write_text(
            json.dumps(view.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.view_path

    # -- seeding -----------------------------------------------------------
    def seed(
        self,
        *,
        canonical: Dict[str, Sequence[str]],
        aliases: Dict[str, str],
        week: str,
    ) -> bool:
        """Write the one-off `IMPRESSION_SEEDED` event (idempotent)."""
        payload = {
            "persona": self.persona,
            "week": week,
            "canonical": {k: list(v) for k, v in canonical.items() if v},
            "aliases": dict(aliases),
        }
        created = self._append(
            "IMPRESSION_SEEDED", payload, key=f"IMPRESSION_SEEDED:{self.persona}"
        )
        if created:
            self.invalidate()
            self.save_view()
        return created

    # -- runtime learning --------------------------------------------------
    def learn_alias(
        self, *, alias: Any, canonical: str, week: str, source: str = "memory_extraction"
    ) -> bool:
        raw = str(alias or "").strip()
        canonical = str(canonical or "").strip()
        if not raw or not canonical or normalize_name(raw) == normalize_name(canonical):
            return False
        current = self.impressions()
        if current.aliases.get(normalize_name(raw)) == canonical:
            return False
        if raw in current.soft_entities:
            # The alias was previously unknown; it is now resolved.
            current.soft_entities.pop(raw, None)
        created = self._append(
            "ALIAS_LEARNED",
            {
                "persona": self.persona,
                "week": week,
                "alias": raw,
                "canonical": canonical,
                "source": source,
            },
            key=f"ALIAS_LEARNED:{normalize_name(raw)}:{canonical}",
        )
        if created:
            self.invalidate()
            self.save_view()
        return created

    def record_soft(self, *, names: Iterable[Any], week: str, source: str = "memory_extraction") -> int:
        written = 0
        for name in names:
            raw = str(name or "").strip()
            if not raw or len(raw) > MAX_ENTITY_LEN:
                continue
            if self.impressions().mentions(raw):
                continue
            if self._append(
                "SOFT_ENTITY_SEEN",
                {"persona": self.persona, "week": week, "name": raw, "source": source},
                key=f"SOFT_ENTITY_SEEN:{raw}:{week}",
            ):
                written += 1
        if written:
            self.invalidate()
            self.save_view()
        return written


# ---------------------------------------------------------------------------
# World-level seeding
# ---------------------------------------------------------------------------


def world_alias_seed(world_dir: Path) -> Dict[str, Dict[str, Any]]:
    """`{canonical: {"kind": str, "aliases": [...]}}` from the world's alias file."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry in _world_alias_entries(world_dir / WORLD_ALIAS_FILENAME):
        canonical = str(entry.get("canonical"))
        out[canonical] = {
            "kind": str(entry.get("kind") or "entity"),
            "aliases": [str(a) for a in (entry.get("aliases") or []) if str(a).strip()],
        }
    return out


def collect_world_entities(dm: "DataManager", *, persona: str) -> Dict[str, List[str]]:
    """Canonical entities the persona could know, grouped by kind.

    Sources: sibling personas, contact partners, own possessions, the run's
    location store, and the world's optional alias file. Deterministic and
    offline — the same run directory always yields the same seed.
    """
    people: Set[str] = set()
    places: Set[str] = set()
    possessions: Set[str] = set()

    try:
        persona_root = Path("data") / str(dm.world) / "persona"
        for path in sorted(persona_root.iterdir()):
            if path.is_dir():
                people.add(path.name)
    except OSError:
        pass
    people.add(persona)

    try:
        for conv in sorted(dm.contact.glob("*.jsonl")):
            if conv.stem and conv.stem != "sig":
                people.add(conv.stem)
    except OSError:
        pass

    try:
        state = dm.read_state(exclude_cur_t=False)
        for item in (state.get("assets") or {}).get("possessions") or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name:
                possessions.add(str(name))
    except (IndexError, FileNotFoundError, AttributeError):
        pass

    try:
        store = getattr(dm, "location_store", None)
        if store is not None:
            for key, location in list(getattr(store, "public", {}).items()) + list(
                getattr(store, "private", {}).items()
            ):
                places.add(str(key))
                display = getattr(location, "display_name", None) or (
                    location.get("display_name") if isinstance(location, dict) else None
                )
                if display:
                    places.add(str(display))
    except Exception:  # pragma: no cover - defensive, mirrors consolidator
        pass

    world_dir = Path("data") / str(dm.world)
    for canonical, entry in world_alias_seed(world_dir).items():
        kind = entry.get("kind")
        if kind == "person":
            people.add(canonical)
        elif kind == "possession":
            possessions.add(canonical)
        else:
            places.add(canonical)

    return {
        "people": sorted(people),
        "places": sorted(places),
        "possessions": sorted(possessions),
    }


def build_seed(
    dm: "DataManager", *, persona: str
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """`(canonical by kind, alias → canonical)` for one persona."""
    canonical = collect_world_entities(dm, persona=persona)
    aliases: Dict[str, str] = {}
    ambiguous: Set[str] = set()

    def _add(alias: str, canonical_name: str) -> None:
        key = normalize_name(alias)
        if not key or key == normalize_name(canonical_name):
            return
        existing = aliases.get(key)
        if existing is not None and existing != canonical_name:
            ambiguous.add(key)
            return
        aliases[key] = canonical_name

    for kind, names in canonical.items():
        for name in names:
            for alias in name_alias_candidates(name):
                _add(alias, name)

    world_dir = Path("data") / str(dm.world)
    for canonical_name, entry in world_alias_seed(world_dir).items():
        for alias in entry.get("aliases") or []:
            _add(alias, canonical_name)

    for key in ambiguous:
        aliases.pop(key, None)
    return canonical, aliases


def keeper_for(dm: "DataManager", clock: Any) -> "ImpressionKeeper":
    """Keeper bound to the DataManager's character."""
    return ImpressionKeeper(dm=dm, clock=clock, persona=str(dm.char))


def ensure_impressions(dm: "DataManager", clock: Any, *, week: str) -> Tuple[ImpressionKeeper, Impressions]:
    """Seed on first use, then return `(keeper, view)`."""
    keeper = keeper_for(dm, clock)
    view = keeper.impressions()
    if not view.seeded:
        canonical, aliases = build_seed(dm, persona=str(dm.char))
        if keeper.seed(canonical=canonical, aliases=aliases, week=week):
            MEMORY_LOGGER_SEED = (
                f"[{dm.char}] impressions seeded: "
                f"{sum(len(v) for v in canonical.values())} canonical / {len(aliases)} aliases"
            )
            IMPRESSION_LOGGER.info(MEMORY_LOGGER_SEED)
        view = keeper.impressions()
    return keeper, view

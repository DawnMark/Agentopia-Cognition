"""Phase 4.5 · the lesson bridge: the character's own notes as citable memory.

The native character already keeps a lesson system: during REVIEW it writes a
`Reflection` into its weekly diary and rewrites a `【教训与注意】`-style section in
its own scratchpad (`general.txt`, `characters/<who>.txt`, …). Those lines are
the conclusions it drew *for itself*: "谈不到就当天排备选市内活（又犯一次）".

Until now the cognition layer never read them — the memory extraction was fed
only objective records, the diary was clipped before its reflection half, and
the scratchpad was used solely as a token-count baseline. This module closes
that gap on the **read side** and nothing else:

    scratchpad snapshot  ->  lesson lines  ->  diff against last week
                                             ->  fuzzy dedup (rewordings)
                                             ->  drop tracking
                                             ->  `belief`/`lesson` memories

Three rules keep it honest:

1. **The character's words are kept verbatim.** No LLM rewrites a lesson; the
   line the character wrote is the memory content, so a wrong lesson stays
   auditable instead of being smoothed into something plausible.
2. **It is a belief, never a fact** (design doc invariants 2-4): the memory gets
   a moderate confidence, its evidence is the scratchpad snapshot's ledger id,
   and it can never move a methodology's value — it can only become the seed of
   an idea, which must then be tested by real practice.
3. **Snapshot diff, not snapshot dump.** The character rewrites the whole file
   every week, so identical lines are ignored, reworded lines are treated as the
   same lesson (`rewrite_threshold`), and a line that disappears is reported as
   dropped rather than deleted — "I stopped reminding myself of this" is
   information, not noise.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils import get_logger

LESSON_LOGGER = get_logger("cognition", quiet=True)

DEFAULT_MAX_NEW_PER_WEEK = 6
# Chinese lessons are short (a five-character line can be a whole lesson),
# so the floor only drops fragments.
DEFAULT_MIN_CHARS = 4
DEFAULT_REWRITE_THRESHOLD = 0.35
# Two lessons also count as the same when the shorter one is largely contained
# in the longer one — but only when the shorter one is long enough that the
# overlap means something ("旧的一条" is fully contained in many sentences and
# must not swallow them).
CONTAINMENT_THRESHOLD = 0.5
CONTAINMENT_MIN_GRAMS = 6
# The character's own conclusion: more certain than a guess, less certain than
# an observation, and deliberately above the idea engine's low-confidence floor
# (0.4) so a lesson pair can still clear the evidence check.
DEFAULT_CONFIDENCE = 0.55
DEFAULT_SALIENCE = 0.6

# A header line that announces a lesson section. The first three are what the
# design assumed; the rest were found in real runs — 萧亦岚 keeps its lessons
# under 【我的毛病·反复记】 ("my recurring flaws") and 上官霄月 under 【教训与注意】,
# so the vocabulary has to cover how characters actually title their notes.
LESSON_HEADER_WORDS = (
    "教训",
    "注意",
    "提醒",
    "毛病",
    "短板",
    "缺点",
    "反复",
    "反思",
    "复盘",
    "改进",
    "要改",
    "lesson",
    "lessons",
    "takeaway",
    "flaw",
    "weakness",
    "recur",
    "improv",
    "reflect",
    "review",
    "reminder",
    "mistake",
)

_HEADER_RE = re.compile(r"^\s*(?:【|\[|\*\*|#{1,4}\s*)(?P<title>[^】\]\n]{1,30})(?:】|\]|\*\*)?\s*:?\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.、)]|\(\d+\))\s*")
_AVOIDANCE_WORDS = ("别", "不要", "不能", "不该", "避免", "少", "免得", "忌", "别再", "don't", "avoid", "never")

# Heuristic topic tagging for lesson lines (design draft 3.3 topics are a closed
# set, so an untagged lesson simply carries no topic). The map is intentionally
# small and bilingual; unknown wording yields no topic rather than a guess.
_TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "work": ("接活", "客户", "委托", "单子", "业务", "上班", "工作", "job", "work", "client"),
    "money": ("钱", "进账", "账", "存款", "花销", "便宜", "贵", "money", "budget", "savings"),
    "planning": ("计划", "安排", "提前", "待办", "排", "顺序", "plan", "schedule", "order"),
    "mistake": ("失误", "错了", "犯", "忘了", "漏了", "搞砸", "mistake", "failed", "forgot"),
    "routine": ("习惯", "惯例", "每天", "固定", "routine", "habit", "daily"),
    "health": ("体力", "睡", "累", "伤", "身体", "恢复", "health", "sleep", "rest", "injur"),
    "exercise": ("练", "训练", "跑", "体能", "格斗", "游泳", "train", "practice", "exercise"),
    "friendship": ("室友", "朋友", "关系", "相处", "friend", "roommate", "relationship"),
    "conflict": ("冲", "吵", "翻脸", "冲突", "conflict", "argument"),
    "cooperation": ("一起", "合作", "搭手", "cooperat", "together"),
    "study": ("学", "看书", "图书", "课", "study", "read", "course"),
    "food": ("吃", "做饭", "菜", "饭", "food", "cook", "meal"),
    "self_care": ("休息", "放松", "顾自己", "self-care", "relax"),
    "risk": ("风险", "危险", "暴露", "安全", "risk", "danger", "expose", "safe"),
    "opportunity": ("机会", "错过", "opportunity", "chance"),
    "housing": ("房", "租", "家", "屋", "home", "rent", "house"),
    "travel": ("地铁", "车", "路上", "出城", "travel", "trip", "commute"),
    "performance": ("表现", "状态", "手感", "performance", "form"),
    "learning": ("学到", "记住", "总结", "learn", "apply", "conclusion"),
    "consumption": ("买", "花", "购物", "buy", "spend", "shopping"),
    "creation": ("写", "做", "拍", "create", "write", "make"),
    "family": ("家", "父母", "family", "parents"),
    "romance": ("喜欢", "对象", "romance", "date"),
}


@dataclass
class LessonConfig:
    """`world.cognition.lesson_*` settings, inert by default."""

    enabled: bool = False
    max_new_per_week: int = DEFAULT_MAX_NEW_PER_WEEK
    min_chars: int = DEFAULT_MIN_CHARS
    rewrite_threshold: float = DEFAULT_REWRITE_THRESHOLD
    confidence: float = DEFAULT_CONFIDENCE
    salience: float = DEFAULT_SALIENCE

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "LessonConfig":
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

        return LessonConfig(
            enabled=bool(section.get("lesson_ingest", False)),
            max_new_per_week=max(1, _int("lesson_max_new_per_week", DEFAULT_MAX_NEW_PER_WEEK)),
            min_chars=max(1, _int("lesson_min_chars", DEFAULT_MIN_CHARS)),
            rewrite_threshold=min(
                0.95, max(0.2, _float("lesson_rewrite_threshold", DEFAULT_REWRITE_THRESHOLD))
            ),
            confidence=min(1.0, max(0.0, _float("lesson_confidence", DEFAULT_CONFIDENCE))),
            salience=min(1.0, max(0.0, _float("lesson_salience", DEFAULT_SALIENCE))),
        )


@dataclass
class LessonLine:
    """One line the character wrote in its own lesson section."""

    text: str
    source_file: str
    time: str = ""
    ledger_event_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "time": self.time,
            "ledger_event_id": self.ledger_event_id,
        }


@dataclass
class LessonDiff:
    """What changed in the character's lesson section since last week."""

    new: List[LessonLine] = field(default_factory=list)
    rewritten: List[Tuple[LessonLine, str]] = field(default_factory=list)
    carried: List[LessonLine] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    snapshot_time: str = ""
    snapshot_event_id: str = ""
    previous_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new": [line.to_dict() for line in self.new],
            "rewritten": [
                {"text": line.text, "was": was, "source_file": line.source_file}
                for line, was in self.rewritten
            ],
            "carried": [line.text for line in self.carried],
            "dropped": list(self.dropped),
            "snapshot_time": self.snapshot_time,
            "snapshot_event_id": self.snapshot_event_id,
            "previous_time": self.previous_time,
        }


def _is_lesson_header(line: str) -> bool:
    match = _HEADER_RE.match(line)
    if not match:
        return False
    title = match.group("title").strip().strip("*#【】[]").strip()
    if not title or len(title) > 30:
        return False
    low = title.lower()
    return any(word in low for word in LESSON_HEADER_WORDS)


def _is_any_header(line: str) -> bool:
    return bool(_HEADER_RE.match(line)) or line.strip().startswith("【")


def extract_lesson_lines(content: str) -> List[str]:
    """The bullet lines under every lesson-ish header of one scratchpad text."""
    out: List[str] = []
    inside = False
    for raw in str(content or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if _is_any_header(stripped):
            inside = _is_lesson_header(stripped)
            continue
        if not inside:
            continue
        text = _BULLET_RE.sub("", stripped).strip()
        text = text.strip("【】[]")
        if text:
            out.append(text)
    return out


def _grams(text: str) -> set[str]:
    clean = re.sub(r"[\s，。、：:；;（）()【】\[\]！!？?\"'“”‘’\-—…·]", "", str(text))
    return {clean[i : i + 2] for i in range(len(clean) - 1)} or {clean}


def bigram_jaccard(left: str, right: str) -> float:
    """Character-bigram similarity; wording-insensitive, script-agnostic."""
    a, b = _grams(left), _grams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_same_lesson(
    left: str,
    right: str,
    *,
    threshold: float = DEFAULT_REWRITE_THRESHOLD,
) -> bool:
    """Is `right` a rewording of `left` (rather than a different lesson)?

    Measured on the acceptance run: the character rewrites its lesson list every
    week, and a reworded pair scores 0.38-1.00 on Jaccard while unrelated pairs
    score 0.00-0.20. Either measure alone is unsafe on short Chinese lines
    ("新的一条：白天留给接活" contains "旧的一条" whole), so containment only
    counts when the shorter line is long enough to be a lesson in its own right.
    """
    a, b = _grams(left), _grams(right)
    if not a or not b:
        return False
    if len(a & b) / len(a | b) >= threshold:
        return True
    shorter = min(len(a), len(b))
    if shorter >= CONTAINMENT_MIN_GRAMS:
        return len(a & b) / shorter >= CONTAINMENT_THRESHOLD
    return False


def read_scratchpad_snapshots(dm) -> List[Tuple[str, str, str, List[str]]]:
    """All scratchpad snapshots: (file, time, ledger_event_id, lesson lines).

    Scratchpads are append-only JSONL: one record per update, each with the full
    text and a ledger identity, so history is available without extra storage.
    """
    root = Path(getattr(dm, "root")) / "memory" / "scratchpad"
    if not root.exists():
        return []
    out: List[Tuple[str, str, str, List[str]]] = []
    for path in sorted(root.rglob("*.jsonl")):
        if path.name.startswith("."):
            continue  # .access_log.jsonl
        rel = path.relative_to(root).as_posix()
        if rel.startswith("working_memory"):
            # Working memory is this week's scratch, not the character's notes.
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            lines = extract_lesson_lines(str(row.get("content") or ""))
            if not lines:
                continue
            out.append(
                (
                    rel,
                    str(row.get("time") or ""),
                    str(row.get("ledger_event_id") or ""),
                    lines,
                )
            )
    out.sort(key=lambda item: (item[1], item[0]))
    return out


def classify_lessons(
    current: Sequence[Any],
    previous: Sequence[str],
    *,
    threshold: float = DEFAULT_REWRITE_THRESHOLD,
    min_chars: int = DEFAULT_MIN_CHARS,
    snapshot: Optional[Tuple[str, str, str, List[str]]] = None,
    previous_time: str = "",
) -> LessonDiff:
    """Split today's lesson lines into new / reworded / unchanged, plus drops.

    `current` may hold plain strings (then `snapshot` names where they came
    from) or `LessonLine`s that already know their file and ledger id.
    """
    diff = LessonDiff(
        snapshot_time=snapshot[1] if snapshot else "",
        snapshot_event_id=snapshot[2] if snapshot else "",
        previous_time=previous_time,
    )
    previous = [str(line).strip() for line in previous if str(line).strip()]
    seen: set[str] = set()
    for raw in current:
        if isinstance(raw, LessonLine):
            line = raw
            text = line.text.strip()
        else:
            text = str(raw).strip()
            line = LessonLine(
                text=text,
                source_file=snapshot[0] if snapshot else "",
                time=snapshot[1] if snapshot else "",
                ledger_event_id=snapshot[2] if snapshot else "",
            )
        if not text or text in seen:
            continue
        seen.add(text)
        if len(text) < min_chars:
            continue
        if text in previous:
            diff.carried.append(line)
            continue
        match = next(
            (old for old in previous if is_same_lesson(text, old, threshold=threshold)),
            None,
        )
        if match is not None:
            diff.rewritten.append((line, match))
            continue
        diff.new.append(line)

    for old in previous:
        if old in seen:
            continue
        if any(is_same_lesson(old, text, threshold=threshold) for text in seen):
            continue
        diff.dropped.append(old)
    return diff


def collect_lesson_diff(dm, config: Optional[LessonConfig] = None) -> LessonDiff:
    """Read the character's scratchpads and diff the lesson sections."""
    cfg = config or LessonConfig()
    snapshots = read_scratchpad_snapshots(dm)
    if not snapshots:
        return LessonDiff()
    latest = snapshots[-1]
    current: List[LessonLine] = []
    previous: List[str] = []
    current_time = latest[1]
    current_event = latest[2]
    previous_time = ""
    by_file: Dict[str, List[Tuple[str, str, str, List[str]]]] = {}
    for snapshot in snapshots:
        by_file.setdefault(snapshot[0], []).append(snapshot)
    for file, items in sorted(by_file.items()):
        last = items[-1]
        for text in last[3]:
            current.append(
                LessonLine(text=text, source_file=file, time=last[1], ledger_event_id=last[2])
            )
        if last[2] and last[1] >= current_time:
            current_event = last[2]
            current_time = max(current_time, last[1])
        if len(items) >= 2:
            previous.extend(items[-2][3])
            previous_time = max(previous_time, items[-2][1])
    return classify_lessons(
        current,
        previous,
        threshold=cfg.rewrite_threshold,
        min_chars=cfg.min_chars,
        snapshot=("merged", current_time, current_event, [line.text for line in current]),
        previous_time=previous_time,
    )


def derive_topics(text: str) -> List[str]:
    """Heuristic topic tags for one lesson line (closed vocabulary)."""
    low = str(text).lower()
    out: List[str] = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(keyword in low for keyword in keywords):
            out.append(topic)
    return out


def derive_polarity(text: str) -> str:
    """Lessons that say "don't / avoid / never again" are negative."""
    low = str(text).lower()
    return "negative" if any(word in low for word in _AVOIDANCE_WORDS) else "neutral"


def lesson_disambiguator(*, polarity: str, content: str) -> str:
    """Content-addressed discriminator so two lessons never collapse into one.

    The semantic key names the anchor (kind + topics + entities); two different
    lessons can share an anchor, and merging them would silently lose one.
    """
    digest = hashlib.sha1(str(content).strip().encode("utf-8")).hexdigest()[:8]
    return f"{polarity}:{digest}"


def provenance(source_file: str, ledger_event_id: str) -> str:
    """`scratchpad:<file>#<ledger id>` — enough to find the exact snapshot."""
    return f"scratchpad:{source_file}#{ledger_event_id}"


def resolve_entities(
    text: str,
    *,
    known_entities: Iterable[str],
) -> List[str]:
    """Entity anchors already present in the lesson text (longest match first)."""
    out: List[str] = []
    for name in sorted(set(str(e) for e in known_entities), key=len, reverse=True):
        if len(name) < 2:
            continue
        if name in text and name not in out:
            out.append(name)
    return out


def lesson_skill_ids(text: str, known_skills: Iterable[str]) -> List[str]:
    """Skills the lesson explicitly names (nothing is inferred)."""
    return [skill for skill in sorted(known_skills) if skill and skill in str(text)]

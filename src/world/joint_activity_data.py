"""Data structures for joint activities.

Defines JointActivityOutcome and JointActivityRecord for multi-person interactive activities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.world.clock import TimeState


@dataclass
class JointActivityOutcome:
    """Single participant's outcome in a joint activity.

    Owner: GodModel generates
    Modifier: Immutable after creation

    Represents the state changes for one participant in a joint activity.
    """

    agent_name: str
    delta_vitality: int = 0
    delta_fulfillment: Dict[str, int] = field(
        default_factory=dict
    )  # mood/social/esteem
    delta_skills: Dict[str, int] = field(default_factory=dict)
    # Items: List of dicts with {name, description, from/to}
    items_sent: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Items gifted to others
    items_received: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Items received from others
    # Phase 5 (design doc 阶段 5 / reward design §3): how fully the outcome
    # satisfied what this participant set out to do, judged by the world model on
    # a 0-1 scale. `None` means the evaluation did not report one — the prompt
    # only asks for it when `world.cognition.goal_progress` is on — and the
    # reward's goal-progress term then stays at zero.
    goal_progress: Optional[float] = None

    def to_dict(self) -> Dict:
        payload = {
            "agent_name": self.agent_name,
            "delta_vitality": self.delta_vitality,
            "delta_fulfillment": dict(self.delta_fulfillment),
            "delta_skills": dict(self.delta_skills),
            "items_sent": list(self.items_sent),
            "items_received": list(self.items_received),
        }
        if self.goal_progress is not None:
            payload["goal_progress"] = float(self.goal_progress)
        return payload

    @staticmethod
    def from_dict(d: Dict) -> JointActivityOutcome:
        return JointActivityOutcome(
            agent_name=d["agent_name"],
            delta_vitality=d.get("delta_vitality", 0),
            delta_fulfillment=d.get("delta_fulfillment", {}),
            delta_skills=d.get("delta_skills", {}),
            items_sent=d.get("items_sent", []),
            items_received=d.get("items_received", []),
            goal_progress=d.get("goal_progress"),
        )


@dataclass
class JointActivityRecord:
    """Complete joint activity record (for persistence).

    Owner: Belongs to agent's DataManager
    Modifier: Created by JointActivity.run(), written to JSONL

    Records one agent's participation in a joint activity, including their
    summary, reflection, and state changes (deltas, not full state).
    """

    agent_name: str
    time: TimeState
    activity_id: str
    activity_name: str

    # Agent's perspective on the activity
    summary: str = ""  # Summary of what happened in the activity
    reflection: str = ""  # Personal reflection on the activity
    # True when the reflection was recovered from a response without the
    # expected markers (KI-12).
    reflection_parse_fallback: bool = False

    # Activity metadata
    participants: List[str] = field(default_factory=list)  # All participants
    location: str = ""

    # State changes (outcome object, simplified from separate delta fields)
    outcome: JointActivityOutcome = None
    # Phase 5 (KI-18): how many conversational rounds this activity had for this
    # character. The reward's time-cost component reads it; before this the
    # records carried no length at all, so every outcome fell back to the
    # "unknown session length" placeholder and the component was a constant.
    turns: int = 0

    def to_dict(self) -> Dict:
        return {
            "type": "joint",
            "agent_name": self.agent_name,
            "time": str(self.time),
            "activity_id": self.activity_id,
            "activity_name": self.activity_name,
            "summary": self.summary,
            "reflection": self.reflection,
            "reflection_parse_fallback": bool(self.reflection_parse_fallback),
            "participants": list(self.participants),
            "location": self.location,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "turns": int(self.turns),
        }

    @staticmethod
    def from_dict(d: Dict) -> JointActivityRecord:
        outcome = JointActivityOutcome.from_dict(d["outcome"])
        return JointActivityRecord(
            agent_name=d["agent_name"],
            time=TimeState.from_string(d["time"]),
            activity_id=d["activity_id"],
            activity_name=d["activity_name"],
            summary=d["summary"],
            reflection=d["reflection"],
            reflection_parse_fallback=bool(d.get("reflection_parse_fallback", False)),
            participants=d["participants"],
            location=d["location"],
            outcome=outcome,
            turns=int(d.get("turns", 0)),
        )

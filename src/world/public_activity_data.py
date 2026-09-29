"""Data structures for public activities.

Defines PublicActivityRecord for persistence.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.world.clock import TimeState


@dataclass
class PublicActivityOutcome:
    """Outcome from GodModel for public activity.

    Similar to ActionOutcome but without consumption/money/items.
    """

    outcome: str  # Natural language message describing outcome
    delta_vitality: int = 0
    delta_fulfillment: Dict[str, int] = field(
        default_factory=dict
    )  # mood/material/social/esteem
    delta_skills: Dict[str, int] = field(default_factory=dict)
    # Phase 5: how fully the outcome satisfied this participant's stated intent,
    # judged by the world model on a 0-1 scale (None when not reported).
    goal_progress: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "outcome": self.outcome,
            "delta_vitality": self.delta_vitality,
            "delta_fulfillment": dict(self.delta_fulfillment),
            "delta_skills": dict(self.delta_skills),
        }
        if self.goal_progress is not None:
            payload["goal_progress"] = float(self.goal_progress)
        return payload

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PublicActivityOutcome":
        return PublicActivityOutcome(
            outcome=d["outcome"],
            delta_vitality=d.get("delta_vitality", 0),
            delta_fulfillment=d.get("delta_fulfillment", {}),
            delta_skills=d.get("delta_skills", {}),
            goal_progress=d.get("goal_progress"),
        )


@dataclass
class PublicActivityRecord:
    """Complete public activity record (for persistence).

    Owner: Belongs to agent_name's DataManager
    Modifier: Created by PublicActivity.run(), written to JSONL
    """

    agent_name: str
    time: TimeState
    activity_id: str
    activity_name: str
    event_description: str
    participants: List[str]  # All participants in this public activity
    participation: str  # This agent's participation description
    reflection: str  # This agent's reflection after activity
    # True when the reflection was recovered from a response without the
    # expected markers (KI-12).
    reflection_parse_fallback: bool = False
    outcome: Optional[PublicActivityOutcome] = None  # GodModel evaluation result
    # Phase 5 (KI-18): how many conversational rounds this activity had for this
    # character. The reward's time-cost component reads it; before this the
    # records carried no length at all, so every outcome fell back to the
    # "unknown session length" placeholder and the component was a constant.
    turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "public",
            "agent_name": self.agent_name,
            "time": str(self.time),
            "activity_id": self.activity_id,
            "activity_name": self.activity_name,
            "event_description": self.event_description,
            "participants": self.participants,
            "participation": self.participation,
            "reflection": self.reflection,
            "reflection_parse_fallback": bool(self.reflection_parse_fallback),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "turns": int(self.turns),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PublicActivityRecord":
        outcome_data = d.get("outcome")
        outcome = (
            PublicActivityOutcome.from_dict(outcome_data) if outcome_data else None
        )
        return PublicActivityRecord(
            agent_name=d["agent_name"],
            time=TimeState.from_string(d["time"]),
            activity_id=d["activity_id"],
            activity_name=d["activity_name"],
            event_description=d["event_description"],
            participants=d["participants"],
            participation=d["participation"],
            reflection=d["reflection"],
            reflection_parse_fallback=bool(d.get("reflection_parse_fallback", False)),
            outcome=outcome,
            turns=int(d.get("turns", 0)),
        )

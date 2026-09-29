"""Data structures for solo activities.

Defines ActionProposal, ActionOutcome, and SoloActivityRecord
following the unified data-driven design.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.world.clock import TimeState


@dataclass
class ConsumptionOption:
    """Single consumption option (product or service) from GodModel.

    Owner: GodModel generates
    Modifier: Immutable after creation
    """

    name: str  # item/service name
    price: int  # price
    description: str  # description

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "price": self.price,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: Dict) -> ConsumptionOption:
        return ConsumptionOption(
            name=d["name"],
            price=d["price"],
            description=d["description"],
        )


@dataclass
class ActionProposal:
    """Agent-proposed solo action (unified structure, no type branching).

    Owner: RoleAgent generates
    Modifier: Immutable after creation

    Note: cost_money represents money the agent PLANS to spend. The actual
    spending is handled by GodModel evaluation (delta_money in ActionOutcome).
    """

    agent_name: str
    action_desc: str  # LLM-generated natural language description
    time: TimeState
    cost_money: int = 0  # 0 = no cost

    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "action_desc": self.action_desc,
            "time": str(self.time),
            "cost_money": self.cost_money,
        }

    @staticmethod
    def from_dict(d: Dict) -> ActionProposal:
        return ActionProposal(
            agent_name=d["agent_name"],
            action_desc=d["action_desc"],
            time=TimeState.from_string(d["time"]),
            cost_money=d.get("cost_money", 0),
        )


@dataclass
class ActionOutcome:
    """GodModel-returned activity outcome (unified output layer).

    Owner: GodModel generates
    Modifier: Immutable after creation
    """

    outcome: str  # Natural language message describing outcome

    # Consumption event fields
    is_consumption_event: bool = False
    consumption_options: List[ConsumptionOption] = field(default_factory=list)

    # Deltas (Output layer) - only set for non-consumption events or after confirmation
    delta_vitality: int = 0  # vitality change
    delta_fulfillment: Dict[str, int] = field(
        default_factory=dict
    )  # mood/material/social/esteem
    delta_skills: Dict[str, int] = field(default_factory=dict)
    delta_money: int = 0
    gain_items: List[Dict[str, Any]] = field(
        default_factory=list
    )  # List of item dicts: {name, description, purchase_price?/from?}

    def to_dict(self) -> Dict:
        return {
            "outcome": self.outcome,
            "is_consumption_event": self.is_consumption_event,
            "consumption_options": [opt.to_dict() for opt in self.consumption_options],
            "delta_vitality": self.delta_vitality,
            "delta_fulfillment": dict(self.delta_fulfillment),
            "delta_skills": dict(self.delta_skills),
            "delta_money": self.delta_money,
            "gain_items": list(self.gain_items),
        }

    @staticmethod
    def from_dict(d: Dict) -> ActionOutcome:
        consumption_opts = [
            ConsumptionOption.from_dict(opt) for opt in d.get("consumption_options", [])
        ]
        return ActionOutcome(
            outcome=d["outcome"],
            is_consumption_event=d.get("is_consumption_event", False),
            consumption_options=consumption_opts,
            delta_vitality=d.get("delta_vitality", 0),
            delta_fulfillment=d.get("delta_fulfillment", {}),
            delta_skills=d.get("delta_skills", {}),
            delta_money=d.get("delta_money", 0),
            gain_items=d.get("gain_items", []),
        )


@dataclass
class SoloActivityRecord:
    """Complete solo activity record (for persistence).

    Owner: Belongs to agent_name's DataManager
    Modifier: Created by SoloActivity.run(), written to JSONL

    Note: state_after/skills_after/money_after/items_after are calculated
    by DataManager.append_activity_record() from current state, not passed
    during construction.
    """

    agent_name: str
    time: TimeState

    # Stable identifier of the activity that produced this record. Phase 0:
    # every activity must be addressable so outcomes, effects and records can be
    # traced back to it (joint/public records already carried one).
    activity_id: str = ""

    # Activity content and reflection
    content: str = ""  # Agent's activity content (extracted from action)
    reflection: str = ""  # Agent's reflection after activity ends
    # True when the reflection was recovered from a response without the
    # expected markers (KI-12); kept so acceptance can filter on it.
    reflection_parse_fallback: bool = False

    # Outcome from GodModel
    outcome: ActionOutcome = None

    # Consumption decision (if is_consumption_event)
    consumption_options_offered: List[ConsumptionOption] = field(default_factory=list)
    consumption_purchased: str | None = (
        None  # name of purchased item/service, or None if declined
    )
    purchase_response: str = ""  # Agent's full response when making purchase decision
    # Phase 5 (KI-18): how many conversational rounds this activity had for this
    # character. The reward's time-cost component reads it; before this the
    # records carried no length at all, so every outcome fell back to the
    # "unknown session length" placeholder and the component was a constant.
    turns: int = 0

    def to_dict(self) -> Dict:
        return {
            "type": "solo",
            "activity_id": self.activity_id,
            "agent_name": self.agent_name,
            "time": str(self.time),
            "content": self.content,
            "reflection": self.reflection,
            "reflection_parse_fallback": bool(self.reflection_parse_fallback),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "consumption_options_offered": [
                opt.to_dict() for opt in self.consumption_options_offered
            ],
            "consumption_purchased": self.consumption_purchased,
            "purchase_response": self.purchase_response,
            "turns": int(self.turns),
        }

    @staticmethod
    def from_dict(d: Dict) -> SoloActivityRecord:
        consumption_opts = [
            ConsumptionOption.from_dict(opt) for opt in d["consumption_options_offered"]
        ]
        outcome_data = d["outcome"]
        outcome = ActionOutcome.from_dict(outcome_data) if outcome_data else None
        return SoloActivityRecord(
            activity_id=d.get("activity_id", ""),
            agent_name=d["agent_name"],
            time=TimeState.from_string(d["time"]),
            content=d["content"],
            reflection=d["reflection"],
            reflection_parse_fallback=bool(d.get("reflection_parse_fallback", False)),
            outcome=outcome,
            consumption_options_offered=consumption_opts,
            consumption_purchased=d["consumption_purchased"],
            turns=int(d.get("turns", 0)),
            purchase_response=d.get("purchase_response", ""),
        )

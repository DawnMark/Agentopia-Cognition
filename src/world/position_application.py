"""Position Application system for agent positions.

This module handles the yearly position application season where agents apply for
positions and get assigned based on their abilities and preferences.

Key concepts:
- Position: A role in the world (e.g., "English teacher", "student") with income and skill growth
- Position Application: 1.5-round process where agents express wishes and get matched
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.config import get_config
from src.world.events import stamped_record
from src.utils import get_logger, get_verify_logger, pool_size

if TYPE_CHECKING:
    from src.world.clock import Clock
    from src.agents.role_agent import RoleAgent

ERROR_LOGGER = get_logger("error")


# =============================================================================
#                           DATA MODELS
# =============================================================================


@dataclass
class Position:
    """A position in the world that an agent can hold.

    Attributes:
        organization: Organization name (e.g., "Fudan High School")
        role: Role name within the organization (e.g., "English Teacher")
        name: Unique identifier, format "{organization}-{role}" (auto-generated)
        type: "work" (has income) or "non-work" (e.g., student)
        description: Description of the position
        weekly_income: Income per week (work and non-work can both have income)
        weekly_delta_skills: Skills gained per week
        min_age: Minimum age requirement (optional)
        max_age: Maximum age requirement (optional)
        min_skills: Minimum skill requirements (optional, God Model does semantic matching)
        capacity: How many agents can hold this position
        occupied_by: List of agent names currently holding this position
        created_year: Year when this position was created (internal, not exposed to agents)
    """

    organization: str  # Organization name
    role: str  # Role name within organization
    type: str  # "work" or "non-work"
    description: str
    weekly_income: int
    weekly_delta_skills: Dict[str, int]
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    min_skills: Optional[Dict[str, int]] = None
    capacity: int = 1
    occupied_by: List[str] = field(default_factory=list)
    created_year: Optional[int] = None  # Internal field, not exposed to agents

    @property
    def name(self) -> str:
        """Unique identifier: {organization}/{role}."""
        return f"{self.organization}/{self.role}"

    @staticmethod
    def parse_name(position_name: str) -> Tuple[str, str]:
        """Parse position name into (organization, role).

        Args:
            position_name: Position identifier in "{organization}/{role}" format

        Returns:
            Tuple of (organization, role)

        Raises:
            ValueError: If position_name doesn't contain '/'
        """
        if "/" not in position_name:
            raise ValueError(
                f"Invalid position_name format: '{position_name}'. "
                f"Expected '{{organization}}/{{role}}' format."
            )
        return position_name.split("/", 1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        d = {
            "name": self.name,  # Unique identifier: {organization}/{role}
            "organization": self.organization,
            "role": self.role,
            "type": self.type,
            "description": self.description,
            "weekly_income": self.weekly_income,
            "weekly_delta_skills": self.weekly_delta_skills,
            "capacity": self.capacity,
            "occupied_by": self.occupied_by,
        }
        if self.min_age is not None:
            d["min_age"] = self.min_age
        if self.max_age is not None:
            d["max_age"] = self.max_age
        if self.min_skills:
            d["min_skills"] = self.min_skills
        if self.created_year is not None:
            d["created_year"] = self.created_year
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Position":
        """Deserialize from dict."""
        return Position(
            organization=d["organization"],
            role=d["role"],
            type=d["type"],
            description=d["description"],
            weekly_income=d["weekly_income"],
            weekly_delta_skills=d.get("weekly_delta_skills", {}),
            min_age=d.get("min_age"),
            max_age=d.get("max_age"),
            min_skills=d.get("min_skills"),
            capacity=d.get("capacity", 1),
            occupied_by=d.get("occupied_by", []),
            created_year=d.get("created_year"),
        )

    def has_vacancy(self) -> bool:
        """Check if position has available slots."""
        return len(self.occupied_by) < self.capacity

    def available_slots(self) -> int:
        """Number of available slots."""
        return max(0, self.capacity - len(self.occupied_by))

    def is_age_eligible(self, age: int) -> bool:
        """Hard check: can this agent HOLD this position?"""
        if self.min_age is not None and age < self.min_age:
            return False
        if self.max_age is not None and age > self.max_age:
            return False
        return True

    def is_visible_to(self, age: int) -> bool:
        """Soft filter for display: hide positions agent has aged out of.

        Keep positions where age < min_age (aspirational goals).
        Hide positions where age > max_age (can never go back).
        """
        if self.max_age is not None and age > self.max_age:
            return False
        return True


# =============================================================================
#                           POSITION STORE
# =============================================================================


class PositionStore:
    """Manages positions data for a world."""

    def __init__(self, world_name: str):
        self.world_name = world_name
        self.path = Path("data") / world_name / "positions.json"
        self._positions: Dict[str, Position] = {}  # key: position name
        self._loaded = False
        self.logger = get_logger("world", quiet=False)

    def _load_from_file(self) -> None:
        """Load positions from file."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._positions = {
            pos_data["name"]: Position.from_dict(pos_data)
            for pos_data in data["positions"]
        }
        self._loaded = True

    def ensure(
        self,
        agents: List["RoleAgent"],
        world_setting: str,
        force: bool = False,
    ) -> None:
        """Ensure positions exist, loading from file or generating via God Model.

        Priority:
        1. Current run's positions.json (if exists)
        2. load_from_template=true: load from template (data/{world_base}/positions.json)
        3. Generate via LLM

        Args:
            agents: List of agents in the world
            world_setting: Description of the world setting
            force: If True, regenerate even if already loaded
        """
        if self._loaded and not force:
            return

        cfg = get_config()
        pos_cfg = cfg.get("world", {}).get("position", {}) or {}

        # 1. Current run directory already has positions
        if self.path.exists() and not force:
            self._load_from_file()
            self.logger.info(f"[positions] loaded from: {self.path}")
            return

        # 2. load_from_template: load from base world template
        if pos_cfg.get("load_from_template", False):
            # Extract base world name (e.g., "schooldays" from "schooldays_02261316")
            world_base = cfg.get("world", {}).get("name", "world")
            template_path = Path("data") / world_base / "positions.json"
            if template_path.exists():
                self._load_from_template(template_path)
                self.logger.info(f"[positions] loaded from template: {template_path}")
                return
            else:
                raise FileNotFoundError(
                    f"load_from_template=true but template not found: {template_path}"
                )

        # 3. Generate positions via God Model
        from src.world.god import god_design_positions

        self.logger.info(f"Generating positions for {len(agents)} agents...")
        positions_data = god_design_positions(
            agents=agents,
            world_setting=world_setting,
        )

        if not positions_data:
            raise RuntimeError(
                "God Model failed to generate positions. "
                "Check LLM configuration and retry."
            )

        # Convert to Position objects (key by name)
        self._positions = {
            pos_data["name"]: Position.from_dict(pos_data)
            for pos_data in positions_data
        }
        self._loaded = True

        self.save()
        self.logger.info(f"Generated {len(self._positions)} positions")

    def _load_from_template(self, template_path: Path) -> None:
        """Load positions from template file and save to current run directory."""
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._positions = {
            pos_data["name"]: Position.from_dict(pos_data)
            for pos_data in data["positions"]
        }
        self._loaded = True
        # Save a copy to current run directory
        self.save()

    def save(self) -> None:
        """Save positions to file (sorted by name for determinism)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by name for deterministic file output
        sorted_positions = sorted(self._positions.values(), key=lambda p: p.name)
        data = {
            "version": 1,
            "positions": [p.to_dict() for p in sorted_positions],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_all(self) -> List[Position]:
        """Get all positions (sorted by name for cache determinism)."""
        return sorted(self._positions.values(), key=lambda p: p.name)

    def get(self, name: str) -> Optional[Position]:
        """Get position by name."""
        return self._positions.get(name)

    def add(self, position: Position) -> None:
        """Add a position."""
        self._positions[position.name] = position

    def add_positions(self, positions: List[Position]) -> None:
        """Add multiple positions."""
        for pos in positions:
            self._positions[pos.name] = pos

    def remove_by_created_year(self, year: int) -> int:
        """Remove all positions with created_year >= year. Returns count removed."""
        to_remove = [
            name
            for name, pos in self._positions.items()
            if pos.created_year is not None and pos.created_year >= year
        ]
        for name in to_remove:
            del self._positions[name]
        return len(to_remove)

    def count(self) -> int:
        """Get total number of positions."""
        return len(self._positions)

    def get_available_positions(self) -> List[Position]:
        """Get positions with vacancies (sorted by name for cache determinism)."""
        return sorted(
            [p for p in self._positions.values() if p.has_vacancy()],
            key=lambda p: p.name,
        )

    def assign_agent(self, name: str, agent_name: str) -> bool:
        """Assign an agent to a position.

        Args:
            name: Position name
            agent_name: Agent name to assign

        Returns:
            True if successful, False if no vacancy.
        """
        pos = self._positions.get(name)
        if not pos or not pos.has_vacancy():
            return False
        if agent_name not in pos.occupied_by:
            pos.occupied_by.append(agent_name)
        return True

    def remove_agent(self, name: str, agent_name: str) -> None:
        """Remove an agent from a position."""
        pos = self._positions.get(name)
        if pos and agent_name in pos.occupied_by:
            pos.occupied_by.remove(agent_name)


def get_position_store(world_name: str) -> PositionStore:
    """Get or create a PositionStore for the world.

    If positions.json exists, automatically loads it.
    """
    store = PositionStore(world_name)
    if store.path.exists():
        store._load_from_file()
    return store


# =============================================================================
#                           HELPER FUNCTIONS
# =============================================================================


def _deterministic_hash(seed: str, item: str) -> str:
    """Generate deterministic hash for sorting.

    Uses SHA256 to ensure consistency across runs.
    """
    payload = f"{seed}-{item}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sort_by_hash(items: List[str], seed: str) -> List[str]:
    """Sort items by deterministic hash (pseudo-random but stable)."""
    return sorted(items, key=lambda x: _deterministic_hash(seed, x))


def _sort_positions_for_position_application(
    positions: List[Position], seed: str
) -> List[Position]:
    """Sort positions for position application processing.

    Priority:
    1. Higher income first (more competitive)
    2. For same income, use deterministic hash for stable ordering
    """
    return sorted(
        positions,
        key=lambda p: (-p.weekly_income, _deterministic_hash(seed, p.name)),
    )


# =============================================================================
#                           POSITION APPLICATION LOG
# =============================================================================


def _append_position_application_log(
    world_name: str,
    time_str: str,
    round_num: int,
    agent_name: str,
    wishes: Optional[List[str]],
    result: str,
    position_name: Optional[str] = None,
    sub_round: Optional[int] = None,
) -> None:
    """Append position application result to log file."""
    log_path = Path("data") / world_name / "position_application_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "time": time_str,
        "round": round_num,
        "agent": agent_name,
        "wishes": wishes,
        "result": result,
    }
    if position_name:
        record["position"] = position_name
    if sub_round is not None:
        record["sub_round"] = sub_round

    record = stamped_record(record, path=log_path)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
#                           POSITION APPLICATION SEASON
# =============================================================================


@dataclass
class PositionApplicationContext:
    """Context for position application season operations."""

    world: "World"
    clock: "Clock"
    position_store: PositionStore
    time_str: str
    seed: str
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Record each agent's original position (for batch update at finalize)
    original_positions: Dict[str, str] = field(default_factory=dict)
    # Age next year per agent (name -> age), cached to avoid repeated profile IO
    agent_ages: Dict[str, int] = field(default_factory=dict)

    def is_matched(self, agent_name: str) -> bool:
        """Check if agent is already matched (from results dict)."""
        return agent_name in self.results

    def is_forced_out(self, agent_name: str) -> bool:
        """Check if agent is forced out of current position due to age limit."""
        pos = self.position_store.get(self.original_positions[agent_name])
        assert pos is not None, (
            f"Agent {agent_name}'s position {self.original_positions[agent_name]} "
            f"not found in store"
        )
        return not pos.is_age_eligible(self.agent_ages[agent_name])


def run_position_application_season(
    world: "World",
    clock: "Clock",
    parallel: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run the yearly position application season.

    Process:
    1. Record original positions
    2. Collect wishes (supports <STAY_CURRENT>)
    3. Round 1 (Wish Round): 3 sub-rounds by wish priority
    4. Finalize: batch update occupied_by, fallback to original position

    Returns:
        Dict mapping agent_name to position application result
    """
    verify_logger = get_verify_logger(feature="position_application")

    t = clock.get_time()
    time_str = str(t)  # Full TimeState string for cleanup compatibility
    seed = f"position_application-Y{t.year}-W{t.week}"

    if verify_logger:
        verify_logger.info(
            f"[POSITION_APPLICATION] ========== POSITION APPLICATION SEASON START ========== {time_str}"
        )

    # Load position store
    position_store = get_position_store(world.data_dir)
    all_positions = position_store.get_all()

    if not all_positions:
        if verify_logger:
            verify_logger.warning("[POSITION_APPLICATION] No positions defined")
        return {}

    # Create context
    ctx = PositionApplicationContext(
        world=world,
        clock=clock,
        position_store=position_store,
        time_str=time_str,
        seed=seed,
    )

    # Record original positions and compute agent ages for next year
    # Per REQ-18: All agents must have initial positions
    next_year = t.year + 1
    for agent in world.agents:
        profile = agent.dm.read_profile()
        pos_data = profile["position"]
        org = pos_data["organization"]
        role = pos_data["role"]
        if not (org and role):
            raise ValueError(
                f"Agent {agent.name} has empty organization or role in position. "
                f"All agents must have valid initial positions (REQ-18)."
            )
        ctx.original_positions[agent.name] = f"{org}/{role}"
        ctx.agent_ages[agent.name] = next_year - profile["birth_year"]

    # Log forced_out agents
    forced_out_names = sorted(
        name for name in ctx.agent_ages if ctx.is_forced_out(name)
    )

    if verify_logger:
        verify_logger.info(
            f"[POSITION_APPLICATION] {len(ctx.original_positions)} agents, "
            f"{len(all_positions)} positions available"
        )
        if forced_out_names:
            verify_logger.info(
                f"[POSITION_APPLICATION] forced_out ({len(forced_out_names)}): {forced_out_names}"
            )

    # Step 1: Collect wishes
    agent_wishes = _collect_wishes(ctx, all_positions, parallel=parallel)

    # Step 2: Round 1 - Wish Round (3 sub-rounds)
    _run_round1_wish(ctx, all_positions, agent_wishes)

    # Step 3: Finalize (batch update occupied_by, fallback to original)
    _finalize_position_application(ctx, agent_wishes)

    # Step 4: Update colleague relationships
    if verify_logger:
        verify_logger.info("[POSITION_APPLICATION] Updating colleague relationships")
    _update_colleague_relationships(world, position_store)

    # Step 5: Calculate and save achievement rewards
    achievements, raw_scores = calculate_achievement_rewards(ctx)
    save_achievement_rewards(world, achievements, raw_scores)

    if verify_logger:
        verify_logger.info(
            f"[POSITION_APPLICATION] ========== POSITION APPLICATION SEASON COMPLETE ========== "
            f"{len(ctx.results)} agents processed"
        )

    return ctx.results


def _collect_wishes(
    ctx: PositionApplicationContext,
    all_positions: List[Position],
    parallel: bool = True,
) -> Dict[str, List[str]]:
    """Collect position application wishes from all agents (parallel by default)."""
    verify_logger = get_verify_logger(feature="position_application")
    agents = ctx.world.agents
    agent_wishes: Dict[str, List[str]] = {}

    if verify_logger:
        verify_logger.info(
            f"[POSITION_APPLICATION] === COLLECT WISHES START === "
            f"{len(agents)} agents, {len(all_positions)} positions available"
        )
        # Log position overview
        for pos in sorted(all_positions, key=lambda p: p.name):
            verify_logger.info(
                f"[POSITION_APPLICATION] Position: {pos.name} "
                f"(capacity={pos.capacity}, income={pos.weekly_income})"
            )

    def get_wishes(agent) -> tuple:
        try:
            age = ctx.agent_ages[agent.name]
            visible = [p for p in all_positions if p.is_visible_to(age)]
            wishes = agent.express_position_application_wishes(
                visible, forced_out=ctx.is_forced_out(agent.name)
            )
            return (agent.name, wishes)
        except Exception as e:
            # Log error but let agent fallback to original position
            # (empty wishes → no match in wish rounds → fallback in finalize)
            ERROR_LOGGER.error(
                f"[POSITION_APPLICATION] {agent.name} wishes failed (will fallback): {e}"
            )
            return (agent.name, [])

    if parallel:
        with ThreadPoolExecutor(max_workers=pool_size(len(agents))) as ex:
            results = list(ex.map(get_wishes, agents))
        for name, wishes in results:
            agent_wishes[name] = wishes
    else:
        for agent in agents:
            name, wishes = get_wishes(agent)
            agent_wishes[name] = wishes

    # Log all wishes collected
    if verify_logger:
        verify_logger.info("[POSITION_APPLICATION] --- Per-Agent Wishes ---")
        for name in sorted(agent_wishes.keys()):
            wishes = agent_wishes[name]
            original_pos = ctx.original_positions.get(name, "unknown")
            verify_logger.info(
                f"[POSITION_APPLICATION] {name} INPUT: original={original_pos}, wishes={wishes}"
            )
        verify_logger.info(
            f"[POSITION_APPLICATION] === COLLECT WISHES COMPLETE === "
            f"{len(agent_wishes)} agents collected"
        )

    return agent_wishes


def _run_round1_wish(
    ctx: PositionApplicationContext,
    all_positions: List[Position],
    agent_wishes: Dict[str, List[str]],
) -> None:
    """Execute Round 1: Wish Round with 3 sub-rounds by wish priority.

    Sub-round 1: Process all agents' 1st choice wishes
    Sub-round 2: Process unmatched agents' 2nd choice wishes
    Sub-round 3: Process unmatched agents' 3rd choice wishes
    """
    verify_logger = get_verify_logger(feature="position_application")
    t = ctx.clock.get_time()

    if verify_logger:
        verify_logger.info("[POSITION_APPLICATION] === Round 1: Wish Round ===")

    for sub_round in range(1, 4):  # 1st, 2nd, 3rd wish
        if verify_logger:
            verify_logger.info(f"[POSITION_APPLICATION] --- Sub-round {sub_round} ---")

        # Get Nth wishes from unmatched agents, grouped by position
        nth_wishes = _get_nth_wishes(ctx, agent_wishes, sub_round)
        if not nth_wishes:
            if verify_logger:
                verify_logger.info(
                    f"[POSITION_APPLICATION] Sub-round {sub_round}: no wishes to process"
                )
            continue

        # Sort positions by popularity (applicant count desc, income desc)
        sorted_positions = _sort_positions_by_popularity(
            nth_wishes, all_positions, ctx.seed
        )

        # Batch positions (each batch <= 20 total applicants)
        batches = _batch_positions_by_applicant_count(
            sorted_positions, nth_wishes, max_total=20
        )

        if verify_logger:
            verify_logger.info(
                f"[POSITION_APPLICATION] Sub-round {sub_round}: {len(nth_wishes)} positions, "
                f"{len(batches)} batches"
            )

        # Process each batch
        for batch_idx, batch_positions in enumerate(batches):
            _process_batch_round1(
                ctx, batch_positions, nth_wishes, agent_wishes, sub_round, t.year
            )

    # Log rejections for agents not matched in Round 1
    for agent_name in agent_wishes:
        if not ctx.is_matched(agent_name):
            _append_position_application_log(
                ctx.world.data_dir,
                ctx.time_str,
                round_num=1,
                agent_name=agent_name,
                wishes=agent_wishes[agent_name],
                result="rejected",
            )

    if verify_logger:
        verify_logger.info(f"[POSITION_APPLICATION] Round 1 complete: {len(ctx.results)} matched")


def _get_nth_wishes(
    ctx: PositionApplicationContext,
    agent_wishes: Dict[str, List[str]],
    n: int,
) -> Dict[str, List[str]]:
    """Get Nth wishes from unmatched agents, grouped by position.

    Special handling:
    - If Nth wish == original position, auto-success via _accept_stay_current()
    - Other positions are returned for competition

    Args:
        ctx: Position application context
        agent_wishes: Dict mapping agent_name to list of position names
        n: Wish number (1, 2, or 3)

    Returns:
        Dict mapping position_name to list of agent_names who want this as Nth choice
    """
    result: Dict[str, List[str]] = {}

    for agent_name, wishes in agent_wishes.items():
        if ctx.is_matched(agent_name):
            continue
        if len(wishes) >= n:
            pos_name = wishes[n - 1]
            original_pos = ctx.original_positions[agent_name]

            # If wish is original position, auto-success (stay current)
            # Unless agent is forced out — they cannot stay
            if pos_name == original_pos:
                if ctx.is_forced_out(agent_name):
                    continue  # Skip: age-ineligible for current position
                _accept_stay_current(
                    ctx, agent_name, pos_name, agent_wishes[agent_name], sub_round=n
                )
                continue

            # Other positions: need to compete
            if pos_name not in result:
                result[pos_name] = []
            result[pos_name].append(agent_name)

    # Sort agent lists for determinism
    for pos_name in result:
        result[pos_name] = _sort_by_hash(
            result[pos_name], f"{ctx.seed}-sub{n}-{pos_name}"
        )

    return result


def _accept_stay_current(
    ctx: PositionApplicationContext,
    agent_name: str,
    position_name: str,
    wishes: Optional[List[str]],
    sub_round: int,
) -> None:
    """Handle <STAY_CURRENT>: keep original position, no occupied_by update."""
    ctx.results[agent_name] = {
        "round": 1,
        "sub_round": sub_round,
        "position_name": position_name,
        "accepted": True,
        "stayed": True,  # Mark as stayed (no position change)
    }

    _append_position_application_log(
        ctx.world.data_dir,
        ctx.time_str,
        round_num=1,
        agent_name=agent_name,
        wishes=wishes,
        result="stayed",
        position_name=position_name,
        sub_round=sub_round,
    )


def _sort_positions_by_popularity(
    nth_wishes: Dict[str, List[str]],
    all_positions: List[Position],
    seed: str,
) -> List[Position]:
    """Sort positions by popularity (applicant count desc, income desc).

    Only returns positions that have applicants in nth_wishes.
    """
    position_map = {p.name: p for p in all_positions}
    positions_with_applicants = [
        position_map[name] for name in nth_wishes.keys() if name in position_map
    ]

    return sorted(
        positions_with_applicants,
        key=lambda p: (
            -len(nth_wishes.get(p.name, [])),  # More applicants first
            -p.weekly_income,  # Higher income as tiebreaker
            _deterministic_hash(seed, p.name),  # Final tiebreaker for stability
        ),
    )


def _batch_positions_by_applicant_count(
    sorted_positions: List[Position],
    nth_wishes: Dict[str, List[str]],
    max_total: int = 20,
) -> List[List[Position]]:
    """Group positions into batches where total applicants ~<= max_total.

    Rules:
    - Each batch has at least one position
    - If a position has > max_total applicants, it forms its own batch
    - Otherwise, accumulate until adding next would exceed max_total
    """
    batches: List[List[Position]] = []
    current_batch: List[Position] = []
    current_count = 0

    for pos in sorted_positions:
        applicant_count = len(nth_wishes.get(pos.name, []))

        # If current batch is non-empty and adding this would exceed max,
        # finalize current batch first
        if current_batch and current_count + applicant_count > max_total:
            batches.append(current_batch)
            current_batch = []
            current_count = 0

        current_batch.append(pos)
        current_count += applicant_count

    if current_batch:
        batches.append(current_batch)

    return batches


def _process_batch_round1(
    ctx: PositionApplicationContext,
    batch_positions: List[Position],
    nth_wishes: Dict[str, List[str]],
    agent_wishes: Dict[str, List[str]],
    sub_round: int,
    current_year: int,
) -> None:
    """Process a batch of positions in Round 1."""
    from src.world.god import god_evaluate_position_application

    verify_logger = get_verify_logger(feature="position_application")

    # Collect all applicants for this batch (only unmatched and with vacancies)
    all_applicants: List[str] = []
    valid_positions: List[Position] = []

    for pos in batch_positions:
        if not pos.has_vacancy():
            continue
        applicants = [
            name for name in nth_wishes.get(pos.name, []) if not ctx.is_matched(name)
        ]
        if applicants:
            all_applicants.extend(applicants)
            valid_positions.append(pos)

    if not valid_positions or not all_applicants:
        return

    # Deduplicate applicants (same agent might appear for multiple positions)
    unique_applicants = list(dict.fromkeys(all_applicants))

    if verify_logger:
        pos_names = [p.name for p in valid_positions]
        verify_logger.info(
            f"[POSITION_APPLICATION] _process_batch_round1 sub{sub_round}: "
            f"positions={pos_names}, applicants={unique_applicants}"
        )

    # Get agent objects
    candidate_agents = [ctx.world._name2agent[name] for name in unique_applicants]

    # Call God Model for batch evaluation
    try:
        results = god_evaluate_position_application(
            round_num=1,
            positions=valid_positions,
            candidates=candidate_agents,
            wishes=agent_wishes,
            seed=f"{ctx.seed}-sub{sub_round}",
            sub_round=sub_round,
        )
    except Exception as e:
        ERROR_LOGGER.warning(f"[POSITION_APPLICATION] Batch eval failed: {e}")
        if verify_logger:
            verify_logger.error(f"[POSITION_APPLICATION] God Model evaluation failed: {e}")
        return

    if verify_logger:
        verify_logger.info(f"[POSITION_APPLICATION] God Model results: {results}")

    # Apply results: results is Dict[pos_name -> List[selected_names]]
    for pos_name, selected_names in results.items():
        position = ctx.position_store.get(pos_name)
        if not position:
            continue

        for agent_name in selected_names:
            if position.has_vacancy() and not ctx.is_matched(agent_name):
                _accept_agent(
                    ctx,
                    agent_name,
                    position,
                    round_num=1,
                    sub_round=sub_round,
                    wishes=agent_wishes.get(agent_name),
                )

                if verify_logger:
                    verify_logger.info(
                        f"[POSITION_APPLICATION] ACCEPTED: {agent_name} → {pos_name} "
                        f"(sub_round={sub_round})"
                    )


def _accept_agent(
    ctx: PositionApplicationContext,
    agent_name: str,
    position: Position,
    round_num: int,
    wishes: Optional[List[str]],
    sub_round: Optional[int] = None,
) -> None:
    """Accept an agent into a NEW position (not their original).

    Note: Does NOT update occupied_by immediately. Only records to ctx.results.
    Batch update happens in _finalize_position_application().
    """
    # Record result (no immediate occupied_by update)
    ctx.results[agent_name] = {
        "round": round_num,
        "position_name": position.name,
        "weekly_income": position.weekly_income,
        "weekly_delta_skills": dict(position.weekly_delta_skills),
        "accepted": True,
        "stayed": False,  # Changed position
    }
    if sub_round is not None:
        ctx.results[agent_name]["sub_round"] = sub_round

    _append_position_application_log(
        ctx.world.data_dir,
        ctx.time_str,
        round_num=round_num,
        agent_name=agent_name,
        wishes=wishes,
        result="accepted",
        position_name=position.name,
        sub_round=sub_round,
    )


def _finalize_position_application(
    ctx: PositionApplicationContext,
    agent_wishes: Dict[str, List[str]],
) -> None:
    """Finalize position_application: fallback to original, batch update occupied_by."""
    verify_logger = get_verify_logger(feature="position_application")

    if verify_logger:
        verify_logger.info("[POSITION_APPLICATION] === FINALIZE START ===")

    # Step 1: Fallback - unmatched agents keep original position
    # Exception: forced_out agents who are unmatched become unemployed
    for agent in ctx.world.agents:
        if agent.name not in ctx.results:
            original_pos = ctx.original_positions[agent.name]

            if ctx.is_forced_out(agent.name):
                # Forced out + unmatched → unemployed
                ctx.results[agent.name] = {
                    "round": 1,
                    "position_name": None,
                    "accepted": True,
                    "stayed": False,
                    "forced_out": True,
                }
                _append_position_application_log(
                    ctx.world.data_dir,
                    ctx.time_str,
                    round_num=1,
                    agent_name=agent.name,
                    wishes=agent_wishes.get(agent.name),
                    result="forced_out_unemployed",
                    position_name=None,
                )
            else:
                # Normal fallback to original position
                ctx.results[agent.name] = {
                    "round": 1,
                    "position_name": original_pos,
                    "accepted": True,
                    "stayed": True,
                    "fallback": True,
                }
                _append_position_application_log(
                    ctx.world.data_dir,
                    ctx.time_str,
                    round_num=1,
                    agent_name=agent.name,
                    wishes=agent_wishes.get(agent.name),
                    result="fallback_stayed",
                    position_name=original_pos,
                )

    # Step 2: Batch update occupied_by
    for agent_name, result in ctx.results.items():
        original_pos = ctx.original_positions[agent_name]
        new_pos = result["position_name"]
        stayed = result["stayed"]

        if stayed:
            # No change needed
            continue

        # Remove from old position
        ctx.position_store.remove_agent(original_pos, agent_name)

        if new_pos:
            ctx.position_store.assign_agent(new_pos, agent_name)

    # Step 3: Update Agent profiles
    config = get_config()
    min_income = config["world"]["position_application"]["min_income"]

    for agent in ctx.world.agents:
        result = ctx.results[agent.name]
        if result["stayed"]:
            # Stayed in original position, no profile update needed
            continue

        pos_name = result["position_name"]
        if pos_name:
            # Changed to new position
            agent.apply_position_application_result(
                position_name=pos_name,
                weekly_income=result["weekly_income"],
                weekly_delta_skills=result["weekly_delta_skills"],
            )
        else:
            # Unemployed
            agent.apply_position_application_result(
                position_name=None,
                weekly_income=min_income,
                weekly_delta_skills={},
            )

    # Step 4: Save position store
    ctx.position_store.save()

    if verify_logger:
        # Log per-agent results
        verify_logger.info("[POSITION_APPLICATION] --- Per-Agent Results ---")
        for name in sorted(ctx.results.keys()):
            result = ctx.results[name]
            original_pos = ctx.original_positions[name]
            new_pos = result["position_name"]
            wishes = agent_wishes.get(name, [])

            if result["stayed"]:
                status = "STAYED" if not result.get("fallback") else "FALLBACK"
            else:
                status = "CHANGED"

            verify_logger.info(
                f"[POSITION_APPLICATION] {name} OUTPUT: {status} "
                f"original={original_pos}, new={new_pos}, "
                f"wishes={wishes}"
            )

        # Summary stats
        stats = {"accepted": 0, "stayed": 0, "changed": 0, "fallback": 0}
        for r in ctx.results.values():
            if r["accepted"]:
                stats["accepted"] += 1
            if r["stayed"]:
                stats["stayed"] += 1
                if r.get("fallback"):
                    stats["fallback"] += 1
            else:
                stats["changed"] += 1

        verify_logger.info(
            f"[POSITION_APPLICATION] === FINALIZE COMPLETE === "
            f"{stats['accepted']} accepted ({stats['stayed']} stayed, "
            f"{stats['changed']} changed, {stats['fallback']} fallback)"
        )


def _update_colleague_relationships(
    world: "World",
    position_store: PositionStore,
) -> None:
    """Update scratchpads with colleague information.

    Each agent learns about colleagues in the same organization (up to 10).
    """
    positions = position_store.get_all()

    # Group agents by organization
    org_agents: Dict[str, List[str]] = {}
    for position in positions:
        org = position.organization
        if org not in org_agents:
            org_agents[org] = []
        org_agents[org].extend(position.occupied_by)

    # For each organization, let agents meet each other
    for org, agent_names in org_agents.items():
        if len(agent_names) <= 1:
            continue

        # Deduplicate (an agent might hold multiple positions in same org)
        unique_names = list(dict.fromkeys(agent_names))

        for agent_name in unique_names:
            # agent_name comes from position.occupied_by, must exist in world
            agent = world._name2agent[agent_name]

            # Get colleagues (excluding self), limit to 10
            colleagues = [n for n in unique_names if n != agent_name][:10]

            for colleague_name in colleagues:
                agent.meet_person(colleague_name)


# =============================================================================
#                        ACHIEVEMENT REWARD
# =============================================================================


def calculate_achievement_rewards(
    ctx: PositionApplicationContext,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Calculate achievement reward based on position's sum_min_skills.

    Ranking-based scoring:
    1. Compute raw sum_min_skills for each agent's position
    2. Rank agents by sum_min_skills (higher = better rank)
    3. Convert ranks to 0-100 uniform distribution (1st place = 100, last = 0)

    Args:
        ctx: PositionApplicationContext with finalized results

    Returns:
        Tuple of (achievements_percentile, raw_scores):
        - achievements_percentile: Dict mapping agent_name to 0-100 percentile
        - raw_scores: Dict mapping agent_name to raw sum_min_skills
    """
    verify_logger = get_verify_logger(feature="position_application")

    # Step 1: Collect raw sum_min_skills for each agent
    raw_scores: Dict[str, int] = {}
    for agent_name, result in ctx.results.items():
        pos_name = result["position_name"]
        score = 0
        if pos_name:
            position = ctx.position_store.get(pos_name)
            if position and position.min_skills:
                score = sum(position.min_skills.values())
        raw_scores[agent_name] = score

    # Step 2: Sort agents by raw score (descending) to create ranking
    # Tie-breaker: agent name for determinism
    sorted_agents = sorted(
        raw_scores.keys(),
        key=lambda name: (-raw_scores[name], name),
    )

    from src.world.reward import ranking_to_weights

    if not sorted_agents:
        return {}, {}
    achievements = ranking_to_weights(sorted_agents)

    if verify_logger:
        top3 = sorted_agents[:3]
        bottom3 = sorted_agents[-3:]
        verify_logger.info(
            f"[ACHIEVEMENT] {len(achievements)} agents. "
            f"Top3: {[(n, raw_scores[n], f'{achievements[n]:.1f}') for n in top3]}, "
            f"Bottom3: {[(n, raw_scores[n], f'{achievements[n]:.1f}') for n in bottom3]}"
        )

    return achievements, raw_scores


def save_achievement_rewards(
    world: "World",
    achievements: Dict[str, float],
    raw_scores: Dict[str, int],
) -> None:
    """Save achievement rewards to each agent's achievement.jsonl.

    Args:
        world: World instance
        achievements: Dict mapping agent_name to achievement score (0-100)
        raw_scores: Dict mapping agent_name to raw sum_min_skills
    """
    for agent in world.agents:
        score = achievements[agent.name]
        raw = raw_scores[agent.name]
        agent.dm.save_achievement(score, raw)

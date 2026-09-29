#!/usr/bin/env python3
"""Extract activity dialogues from generation files.

Usage:
    python scripts/extract_activity_dialogues.py <run_id>

Example:
    python scripts/extract_activity_dialogues.py data/school_01082234
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DialogueTurn:
    turn_number: int
    person: str
    content: str
    activity_name: str
    time: str

    def key(self) -> tuple:
        """Unique key used for de-duplication."""
        return (self.time, self.activity_name, self.turn_number, self.person)


def extract_activity_dialogues(run_dir: str) -> list[DialogueTurn]:
    """Extract all activity dialogues from a run directory.

    Scans data/{run_id}/persona/*/generation/year=*/week=*.md and extracts
    dialogues in the [turn: X, person: Y] format from the
    "# ==== llm outputs" blocks.
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    persona_dir = run_path / "persona"
    if not persona_dir.exists():
        print(f"Error: {persona_dir} does not exist")
        sys.exit(1)

    gen_files = list(persona_dir.glob("*/generation/year=*/week=*.md"))
    if not gen_files:
        print(f"No generation files found in {persona_dir}")
        sys.exit(1)

    # De-duplicate via dict keyed by DialogueTurn.key()
    unique_dialogues: dict[tuple, DialogueTurn] = {}

    for gen_file in gen_files:
        persona_name = gen_file.parent.parent.parent.name
        dialogues = extract_from_file(gen_file, persona_name)
        for d in dialogues:
            key = d.key()
            if key not in unique_dialogues:
                unique_dialogues[key] = d

    return list(unique_dialogues.values())


def extract_from_file(file_path: Path, persona_name: str) -> list[DialogueTurn]:
    """Extract dialogues from a single generation file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    dialogues = []

    # Find every "# ==== llm outputs at (time) ====" block.
    # Extract only from output blocks to avoid pulling duplicate content from
    # the input blocks.
    output_pattern = r"# ==== llm outputs at \(([^)]+)\) ====\s*\n(.*?)(?=# ==== llm (?:inputs|outputs)|$)"

    for match in re.finditer(output_pattern, content, re.DOTALL):
        time_str = match.group(1).strip()
        output_block = match.group(2)

        # Only process outputs from the activity stage
        if "-activity-" not in time_str:
            continue

        # Look backwards for the activity info.
        # Find the nearest "## Current Activity" block above this match.
        pos = match.start()
        preceding = content[:pos]
        activity_name = "Unknown Activity"

        activity_matches = list(
            re.finditer(
                r"## Current Activity\s*\n(?:Joint|Solo) Activity:\s*(.+)", preceding
            )
        )
        if activity_matches:
            activity_name = activity_matches[-1].group(1).strip()

        # Extract dialogues in [turn: X, person: Y] format.
        # Only look at content following an [output message N] marker.
        output_msg_pattern = r"\[output message \d+\]\s*\n(.*?)(?=\[(?:output|input|tool) |---\s*\n# ====|\Z)"

        for msg_match in re.finditer(output_msg_pattern, output_block, re.DOTALL):
            msg_content = msg_match.group(1)

            # Extract dialogues in [turn: X, person: Y] format.
            # Handle both the with-</think> and without-</think> cases.
            turn_pattern = r"(?:</think>\s*)?\[turn:\s*(\d+),\s*person:\s*([^\]]+)\]\s*\n(.*?)(?=(?:</think>\s*)?\[turn:|---|\[input|\[output|\Z)"

            for turn_match in re.finditer(turn_pattern, msg_content, re.DOTALL):
                turn_num = int(turn_match.group(1))
                person = turn_match.group(2).strip()
                dialogue_content = turn_match.group(3).strip()

                # Clean up dialogue_content: strip trailing noise.
                # Drop [input message X] and everything after it.
                dialogue_content = re.sub(
                    r"\[input message \d+\].*", "", dialogue_content, flags=re.DOTALL
                ).strip()
                # Drop trailing "Environment:" lines.
                dialogue_content = re.sub(
                    r"\nEnvironment:.*", "", dialogue_content, flags=re.DOTALL
                ).strip()
                # Drop <think>...</think> blocks.
                dialogue_content = re.sub(
                    r"<think>.*?</think>\s*", "", dialogue_content, flags=re.DOTALL
                ).strip()
                # Drop any residual </think>.
                dialogue_content = re.sub(r"</think>\s*", "", dialogue_content).strip()

                if person == persona_name and dialogue_content:
                    dialogues.append(
                        DialogueTurn(
                            turn_number=turn_num,
                            person=person,
                            content=dialogue_content,
                            activity_name=activity_name,
                            time=time_str,
                        )
                    )

    return dialogues


def format_dialogues(dialogues: list[DialogueTurn]) -> str:
    """Format dialogue records for analysis."""
    if not dialogues:
        return "No dialogues found."

    # Group by (time, activity)
    grouped: dict[tuple[str, str], list[DialogueTurn]] = {}
    for d in dialogues:
        key = (d.time, d.activity_name)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(d)

    output_lines = []
    for (time, activity), turns in sorted(grouped.items()):
        output_lines.append(f"\n{'=' * 60}")
        output_lines.append(f"Activity: {activity}")
        output_lines.append(f"Time: {time}")
        output_lines.append(f"{'=' * 60}")

        # Sort by (turn_number, person)
        for turn in sorted(turns, key=lambda t: (t.turn_number, t.person)):
            output_lines.append(f"\n[turn {turn.turn_number}, {turn.person}]")
            output_lines.append(turn.content)
            output_lines.append("-" * 40)

    return "\n".join(output_lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_id = sys.argv[1]
    # Use the last path component as the file name,
    # e.g. data/school_01082234 -> school_01082234
    run_name = Path(run_id).name

    print(f"Extracting dialogues from: {run_id}")
    dialogues = extract_activity_dialogues(run_id)
    print(f"Found {len(dialogues)} dialogue turns")

    formatted = format_dialogues(dialogues)

    # Write to logs/activity/<run_name>.log
    output_dir = Path("logs/activity")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_name}.log"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted)

    print(f"Saved to: {output_file}")

    # Also print to stdout
    print("\n" + formatted)


if __name__ == "__main__":
    main()

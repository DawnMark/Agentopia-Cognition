#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    # Console output carries model-generated text; make the encoding forgiving
    # before anything can print (see configure_stdio_encoding).
    from src.utils import configure_stdio_encoding

    configure_stdio_encoding()

    parser = argparse.ArgumentParser(
        description="Run minimal world following pseudo/world.py"
    )
    parser.add_argument(
        "--years", type=int, default=None, help="Override number of years"
    )
    parser.add_argument(
        "--weeks", type=int, default=None, help="Override weeks per year"
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Specify run id (MMDDHHMM) to continue a previous run",
    )
    parser.add_argument(
        "--cognition-shadow",
        dest="cognition_shadow",
        action="store_true",
        default=False,
        help=(
            "Enable phase 1 methodology shadow mode for a new run "
            "(world.cognition.methodology_shadow). It only writes "
            "persona/<name>/cognition/ and never changes behaviour."
        ),
    )
    parser.add_argument(
        "--cognition-memory",
        dest="cognition_memory",
        action="store_true",
        default=False,
        help=(
            "Enable phase 2 memory shadow mode for a new run "
            "(world.cognition.memory_shadow). Shadow only: no prompt changes."
        ),
    )
    parser.add_argument(
        "--cognition-idea",
        dest="cognition_idea",
        action="store_true",
        default=False,
        help=(
            "Enable phase 3 idea engine for a new run (world.cognition.idea_engine). "
            "Shadow only: ideas are stored as hypotheses and never change behaviour."
        ),
    )
    parser.add_argument(
        "--cognition-hints",
        dest="cognition_hints",
        action="store_true",
        default=False,
        help=(
            "Enable phase 4 passive method hints for a new run "
            "(world.cognition.method_hints). The character sees a small menu of "
            "methods it may ignore; adoption is its own decision."
        ),
    )
    parser.add_argument(
        "--cognition-lessons",
        dest="cognition_lessons",
        action="store_true",
        default=False,
        help=(
            "Enable the lesson bridge for a new run "
            "(world.cognition.lesson_ingest): the lesson lines the character "
            "writes in its own notes become memories, an idea source."
        ),
    )
    parser.add_argument(
        "--cognition-bandit",
        dest="cognition_bandit",
        action="store_true",
        default=False,
        help=(
            "Phase 5: decide what the menu offers with a contextual bandit "
            "(world.cognition.method_hints_bandit, implies method_hints). "
            "Exploitation follows the character's per-situation values, "
            "exploration follows its creativity/curiosity, and being offered "
            "and ignored is negative evidence."
        ),
    )
    parser.add_argument(
        "--cognition-lifecycle",
        dest="cognition_lifecycle",
        action="store_true",
        default=False,
        help=(
            "Phase 5: settle method lifecycles (world.cognition.method_lifecycle). "
            "Failing, long-unused methods are archived (never deleted) and a "
            "re-derived method becomes a new version instead of a near-duplicate."
        ),
    )
    parser.add_argument(
        "--cognition-goal-progress",
        dest="cognition_goal_progress",
        action="store_true",
        default=False,
        help=(
            "Phase 5: ask the world model for a bounded goal_progress (0-1) per "
            "activity outcome (world.cognition.goal_progress) and record it. The "
            "reward's goal term still stays at zero unless goal_progress_weight "
            "is set."
        ),
    )
    parser.add_argument(
        "--cognition-proficiency",
        dest="cognition_proficiency",
        action="store_true",
        default=False,
        help=(
            "Phase 6 shadow: derive proficiency and effective capability from the "
            "ledger at weekly settlement (world.cognition.proficiency_projection). "
            "Writes views/proficiency.json and changes nothing else."
        ),
    )
    parser.add_argument(
        "--cognition-capability-input",
        dest="cognition_capability_input",
        action="store_true",
        default=False,
        help=(
            "Phase 6 read-only input: append each practised skill's derived "
            "capability (practices, proficiency, method discount, effective "
            "capability) to the God model's activity-evaluation profile "
            "(world.cognition.capability_input). Writes no state at all."
        ),
    )
    parser.add_argument(
        "--cognition-capability-cap",
        dest="cognition_capability_cap",
        action="store_true",
        default=False,
        help=(
            "Phase 6: cap the skill gain per activity by the character's practised "
            "capability (world.cognition.capability_gain_cap; capability_cap_mid / "
            "capability_cap_high thresholds). Off by default. This is the first "
            "phase-6 mechanism that actually changes behaviour."
        ),
    )
    parser.add_argument(
        "--baseline",
        dest="baseline",
        type=str,
        default=None,
        choices=["off", "cognition", "upstream"],
        help=(
            "Comparison baseline. 'cognition' switches every phase-added feature "
            "off (phase -1 fixes and the seed behaviour stay, they are not "
            "switchable); 'upstream' additionally drops the 'Do Not Fall Into a "
            "Routine' plan block and restores upstream's model-assignment RNG "
            "source, for a run that stays as close to Neph0s/Agentopia as this "
            "branch can get."
        ),
    )
    parser.add_argument(
        "--replay",
        dest="replay",
        action="store_true",
        default=False,
        help=(
            "Replay-only: serve every LLM call from the cache and fail on a "
            "cache miss (used to verify deterministic replay)"
        ),
    )
    parser.add_argument(
        "--no-ce",
        dest="no_ce",
        action="store_true",
        help="Disable context engineering; agents only use working memory",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel LLM calls (parallel is enabled by default)",
    )
    parser.set_defaults(parallel=True)
    parser.add_argument(
        "--no-history",
        dest="no_history",
        action="store_true",
        help="Disable history usage (skip read/write history)",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Maximum number of agents to bootstrap from data (default: no limit)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for world/utils/agents",
    )
    parser.add_argument(
        "--world",
        type=str,
        default=None,
        help="Override world name from config.json (e.g., school, apartment)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override world language from config.json (e.g., en, zh)",
    )
    parser.add_argument(
        "--role-model",
        dest="role_model",
        type=str,
        default=None,
        help="Override role_model from config.json. "
             "Comma-separated for multiple models (e.g., 'claude-4.5-sonnet,gemini-3-flash-preview')",
    )
    parser.add_argument(
        "--god-model",
        dest="god_model",
        type=str,
        default=None,
        help="Override god_model from config.json (e.g., claude-4.5-sonnet, Qwen3.5-397B-A17B)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a run directory path (e.g., data/schooldays_01201830 or schooldays_01201830). "
             "Loads config from that directory. Mutually exclusive with --run-id.",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        type=str,
        default=None,
        help="Resume from specific point: Y{year} or Y{year}-W{week} (requires --run-id or --resume)",
    )
    parser.add_argument(
        "--resume-year",
        dest="resume_year",
        type=int,
        default=None,
        help="Resume from year (requires --run-id or --resume; alternative to --resume-from)",
    )
    parser.add_argument(
        "--resume-week",
        dest="resume_week",
        type=int,
        default=None,
        help="Resume from week (used with --resume-year, defaults to 1)",
    )
    args = parser.parse_args()

    # --resume and --run-id are mutually exclusive
    if args.resume and args.run_id:
        parser.error("--resume and --run-id are mutually exclusive")

    # Validate resume-from / resume-year arguments
    resume_from_parsed: tuple[int, int] | None = None
    if args.resume_from and args.resume_year is not None:
        parser.error("--resume-from and --resume-year are mutually exclusive")
    if args.resume_week is not None and args.resume_year is None:
        parser.error("--resume-week requires --resume-year")
    if args.resume_from:
        if not args.run_id and not args.resume:
            parser.error("--resume-from requires --run-id or --resume")
        m = re.fullmatch(r"Y(\d+)(?:-W(\d+))?", args.resume_from)
        if not m:
            parser.error(
                f"Invalid --resume-from format: '{args.resume_from}'. "
                f"Expected Y{{year}} or Y{{year}}-W{{week}}"
            )
        r_year = int(m.group(1))
        r_week = int(m.group(2)) if m.group(2) else 1
        resume_from_parsed = (r_year, r_week)
    elif args.resume_year is not None:
        if not args.run_id and not args.resume:
            parser.error("--resume-year requires --run-id or --resume")
        resume_from_parsed = (args.resume_year, args.resume_week or 1)

    import json
    from src.world.run_manager import (
        generate_run_id,
        ensure_run_world_data,
        get_run_data_dir,
        save_run_config,
    )
    from src.config import load_config

    if args.resume:
        # ── Resume from directory path ──────────────────────────────────
        # Accept "data/schooldays_01201830" or just "schooldays_01201830"
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            # Strip leading "data/" if present, then normalize
            parts = resume_path.parts
            if parts[0] == "data":
                data_dir = str(Path(*parts[1:]))
            else:
                data_dir = str(resume_path)
        else:
            # Absolute path: extract relative to data/
            data_dir = str(resume_path.relative_to(Path("data").resolve()))

        run_dir = Path("data") / data_dir
        run_cfg_path = run_dir / "config.json"
        if not run_cfg_path.exists():
            parser.error(f"Config not found: {run_cfg_path}")

        load_config(run_cfg_path)
        print(f"[run_world] Resuming from {run_dir} (config loaded)")
    else:
        # ── Normal flow: new run or --run-id resume ─────────────────────
        # Read root config.json as raw input (not yet loaded into global _CONFIG)
        with (ROOT / "config.json").open("r", encoding="utf-8") as f:
            raw_config = json.load(f)

        # Apply world/language overrides early (before base_world_name)
        if args.world is not None:
            raw_config["world"]["name"] = args.world
            print(f"[run_world] Overriding world name: {args.world}")
        if args.language is not None:
            raw_config["world"]["language"] = args.language
            print(f"[run_world] Overriding language: {args.language}")

        base_world_name = str(raw_config["world"]["name"]).strip()
        run_id = args.run_id or generate_run_id()
        dst_path = ensure_run_world_data(base_world_name, run_id)
        data_dir = get_run_data_dir(base_world_name, run_id)
        run_cfg_path = Path("data") / data_dir / "config.json"

        if args.run_id and run_cfg_path.exists():
            # Resuming a previous run: load its config directly
            cli_overrides = [
                name for name, val in [
                    ("--world", args.world), ("--language", args.language),
                    ("--role-model", args.role_model), ("--god-model", args.god_model),
                    ("--years", args.years), ("--weeks", args.weeks),
                ] if val is not None
            ]
            if cli_overrides:
                print(f"[run_world] WARNING: Resuming run {args.run_id}, "
                      f"CLI overrides ignored: {', '.join(cli_overrides)}")
            load_config(run_cfg_path)
        else:
            # New run: apply CLI overrides, save to run directory, then load
            if args.role_model is not None:
                # Support comma-separated list: 'model-a,model-b' → ["model-a", "model-b"]
                models = [m.strip() for m in args.role_model.split(",") if m.strip()]
                if not models:
                    parser.error("--role-model must not be empty")
                raw_config["role_model"] = models[0] if len(models) == 1 else models
                print(f"[run_world] Overriding role_model: {raw_config['role_model']}")
            if args.god_model is not None:
                raw_config["god_model"] = args.god_model
                print(f"[run_world] Overriding god_model: {args.god_model}")
            if args.cognition_lessons:
                raw_config["world"].setdefault("cognition", {})["lesson_ingest"] = True
                print("[run_world] Overriding world.cognition.lesson_ingest: True")
            if args.cognition_hints:
                raw_config["world"].setdefault("cognition", {})["method_hints"] = True
                print("[run_world] Overriding world.cognition.method_hints: True")
            if args.cognition_bandit:
                raw_config["world"].setdefault("cognition", {})["method_hints"] = True
                raw_config["world"].setdefault("cognition", {})["method_hints_bandit"] = True
                print(
                    "[run_world] Overriding world.cognition.method_hints_bandit: True "
                    "(implies method_hints)"
                )
            if args.cognition_lifecycle:
                raw_config["world"].setdefault("cognition", {})[
                    "method_lifecycle"
                ] = True
                print("[run_world] Overriding world.cognition.method_lifecycle: True")
            if args.cognition_proficiency:
                raw_config["world"].setdefault("cognition", {})[
                    "proficiency_projection"
                ] = True
                print("[run_world] Overriding world.cognition.proficiency_projection: True")
            if args.cognition_capability_input:
                raw_config["world"].setdefault("cognition", {})[
                    "capability_input"
                ] = True
                print("[run_world] Overriding world.cognition.capability_input: True")
            if args.cognition_capability_cap:
                raw_config["world"].setdefault("cognition", {})[
                    "capability_gain_cap"
                ] = True
                raw_config["world"].setdefault("cognition", {})[
                    "capability_input"
                ] = True
                print(
                    "[run_world] Overriding world.cognition.capability_gain_cap: True "
                    "(implies capability_input)"
                )
            if args.baseline is not None:
                raw_config["world"].setdefault("cognition", {})["baseline"] = args.baseline
                print(f"[run_world] Baseline profile: {args.baseline}")
            if args.cognition_goal_progress:
                raw_config["world"].setdefault("cognition", {})["goal_progress"] = True
                print("[run_world] Overriding world.cognition.goal_progress: True")
            if args.cognition_idea:
                raw_config["world"].setdefault("cognition", {})["idea_engine"] = True
                print("[run_world] Overriding world.cognition.idea_engine: True")
            if args.cognition_memory:
                raw_config["world"].setdefault("cognition", {})["memory_shadow"] = True
                print("[run_world] Overriding world.cognition.memory_shadow: True")
            if args.cognition_shadow:
                raw_config["world"].setdefault("cognition", {})[
                    "methodology_shadow"
                ] = True
                print("[run_world] Overriding world.cognition.methodology_shadow: True")
            if args.years is not None:
                raw_config["world"]["time"]["n_year"] = args.years
            if args.weeks is not None:
                raw_config["world"]["time"]["n_week"] = args.weeks
            raw_config["world"]["name"] = base_world_name
            raw_config["world"]["data_dir"] = data_dir
            # The baseline profile is folded in *before* the run's config is
            # saved, so the run directory documents exactly what was on.
            from src.world.baseline import (
                apply_baseline,
                describe_effective_features,
                unexpected_features,
            )

            apply_baseline(raw_config)
            leftover = unexpected_features(raw_config)
            if leftover:
                parser.error(
                    "baseline profile did not switch these off: " + ", ".join(leftover)
                )
            print(f"[run_world] {describe_effective_features(raw_config)}")
            save_run_config(data_dir, raw_config)
            load_config(run_cfg_path)

        print(
            f"[run_world] data_dir={data_dir}"
        )

    # Set up per-run cache directory (isolates cache between parallel runs)
    from src.utils import set_replay_only, set_run_cache_dir

    if args.replay:
        set_replay_only(True)
        print(
            "[run_world] REPLAY MODE: all LLM responses come from the cache; "
            "a cache miss aborts the run."
        )

    set_run_cache_dir(data_dir)

    from src.config import get_config
    from src.world.world import World

    # Determine resume_from: CLI override or auto-detect from checkpoint
    resume_from = resume_from_parsed if (args.run_id or args.resume) else None

    w = World(
        no_context_engineering=args.no_ce,
        parallel=args.parallel,
        no_history=args.no_history,
        max_agents=args.max_agents,
        resume_from=resume_from,
    )

    # If --debug is set, bump logger levels to DEBUG for world, utils and agents.
    if args.debug:
        import logging

        def _bump_logger(lg):
            if lg is None:
                return
            try:
                lg.setLevel(logging.DEBUG)
                for h in getattr(lg, "handlers", []) or []:
                    h.setLevel(logging.DEBUG)
            except Exception:
                pass

        # world logger (prints to console)
        _bump_logger(getattr(w, "logger", None))
        # utils logger (file-only by default)
        _bump_logger(logging.getLogger("utils"))
        # agent loggers (file-only by default)
        for a in getattr(w, "agents", []) or []:
            _bump_logger(getattr(a, "logger", None))

    # Flush caches on early termination:
    # - atexit: handles exceptions and normal exit
    # - signal: handles SIGINT (Ctrl+C) and SIGTERM (kill)
    import atexit
    from src.utils import flush_all_caches, merge_run_cache

    atexit.register(flush_all_caches)

    _flushing = False  # guard against double flush

    def _flush_and_exit(signum, frame):
        nonlocal _flushing
        sig_name = signal.Signals(signum).name
        if _flushing:
            print(f"\n[{sig_name}] Flush in progress. Press Ctrl+C again to force kill.")
            # Restore default handler so next Ctrl+C kills immediately
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            return
        _flushing = True
        # Restore default handler so next Ctrl+C kills immediately during flush
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print(f"\n[{sig_name}] Flushing caches before exit...")
        try:
            flush_all_caches()
            merge_run_cache(data_dir)
            print(f"[{sig_name}] Cache flush done. Exiting.")
        except Exception as e:
            print(f"[{sig_name}] Error during flush: {e}")
        finally:
            os._exit(128 + signum)

    signal.signal(signal.SIGINT, _flush_and_exit)
    signal.signal(signal.SIGTERM, _flush_and_exit)

    w.run()

    # 1. Flush all thread deltas to disk (before merge)
    flush_all_caches()

    # 2. Merge run cache shards back into main cache files
    merge_run_cache(data_dir)


if __name__ == "__main__":
    main()

# python scripts/run_world.py --year 1 --week 5 --parallel
# python -m src.utils

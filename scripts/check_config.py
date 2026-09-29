#!/usr/bin/env python3
"""Local deployment health check for Agentopia.

Offline checks (default):  python scripts/check_config.py
Live endpoint probe (--live): python scripts/check_config.py --live

The live probe sends a request with the configured key to confirm network,
TLS, endpoint URL and SDK wiring all work. It makes no simulation calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -> {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def offline_checks() -> None:
    cfg_path = ROOT / "config.json"
    if not check("config.json exists", cfg_path.exists()):
        return

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        check("config.json parses", True)
    except Exception as e:
        check("config.json parses", False, repr(e))
        return

    # Importing src.utils triggers load_dotenv(), picking up .env
    import src.utils as utils
    from src.config import get_config, load_config, _redact_secrets

    key = os.getenv("OPENAI_API_KEY")

    load_config(cfg_path)
    cfg = get_config()

    # Secrets may come from the global OPENAI_API_KEY env var (single-provider
    # setup) or from each model's own api_key. The env var OVERRIDES every
    # model's config, so it must not be set once several providers are in use.
    urls = {mc.get("url") for mc in cfg["models"].values() if mc.get("url")}
    if key and len(urls) > 1:
        check(
            "no global OPENAI_API_KEY shadowing multiple providers",
            False,
            "OPENAI_API_KEY is set and would override every model's own key",
        )
    else:
        check(
            "no global OPENAI_API_KEY shadowing multiple providers",
            True,
            f"{len(urls)} provider(s)",
        )

    for name, mc in cfg["models"].items():
        resolved = key or str(mc.get("api_key") or "")
        placeholder = (
            (not resolved) or "REPLACE" in resolved.upper() or "YOUR_" in resolved.upper()
        )
        check(
            f"model '{name}' has a usable api_key",
            not placeholder,
            "from .env" if key else "from model config",
        )

    for name, mc in cfg["models"].items():
        if name in utils._CLOSED_SOURCE_PROVIDERS:
            branch = f"closed-source:{utils._CLOSED_SOURCE_PROVIDERS[name]}"
        elif name.startswith("claude"):
            branch = "anthropic"
        elif name.startswith("gemini"):
            branch = "vertex-gemini"
        elif "url" in mc:
            branch = "openai-compatible"
        else:
            branch = "UNROUTABLE"
        check(f"model '{name}' routes", branch != "UNROUTABLE", branch)

    for field in ("role_model", "god_model", "fallback_model"):
        val = cfg[field]
        for n in (val if isinstance(val, list) else [val]):
            check(f"{field}='{n}' defined in models", n in cfg["models"])

    if cfg["response_validation"]["enabled"]:
        check(
            "response_validation.judge_model defined",
            cfg["response_validation"]["judge_model"] in cfg["models"],
        )

    # DeepSeek caps output at 8192 tokens
    for field in ("role_model_max_tokens", "god_model_max_tokens"):
        check(f"{field} <= 8192", cfg[field] <= 8192, str(cfg[field]))

    # Hosted APIs reject unknown body fields such as repetition_penalty
    for name, mc in cfg["models"].items():
        if "url" in mc and "api.deepseek.com" in str(mc.get("url", "")):
            check(
                f"model '{name}' opts out of repetition_penalty",
                mc.get("repetition_penalty", 1.05) is None,
            )

    world_name = cfg["world"]["name"]
    world_dir = ROOT / "data" / world_name
    check(f"world data dir data/{world_name} exists", world_dir.is_dir())
    persona_dir = world_dir / "persona"
    personas = sorted(persona_dir.iterdir()) if persona_dir.is_dir() else []
    check(f"world '{world_name}' has personas", bool(personas), f"{len(personas)} personas")

    start_year = cfg["world"]["time"]["start_year"]
    if personas:
        with_profile = [p for p in personas if (p / "profile" / f"year={start_year}.json").exists()]
        check(
            f"personas have profile/year={start_year}.json",
            len(with_profile) == len(personas),
            f"{len(with_profile)}/{len(personas)}",
        )

    check(
        "API keys are redacted in logs",
        _redact_secrets(raw)["models"][next(iter(raw["models"]))]["api_key"] == "***REDACTED***",
    )

    # Secrets must never be committable
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    check(".gitignore covers config.json", "config.json" in gi)
    check(".gitignore covers .env", ".env" in gi)


def live_probe() -> None:
    from src.config import get_config, load_config

    load_config(ROOT / "config.json")
    cfg = get_config()
    name = cfg["god_model"]
    mc = cfg["models"][name]

    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", mc["url"]),
        api_key=os.getenv("OPENAI_API_KEY", mc["api_key"]),
        timeout=60,
    )
    model_id = mc.get("vllm_model_name", name)
    print(f"\n[LIVE] probing {client.base_url} model={model_id} ...")
    try:
        r = client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": "ping"}], max_tokens=1
        )
        check("[LIVE] endpoint reachable + key accepted", True, str(r.choices[0].finish_reason))
    except Exception as e:
        kind = type(e).__name__
        # A 401 proves the network path and URL are correct but the key is wrong.
        if "auth" in kind.lower() or "401" in str(e):
            check("[LIVE] endpoint reachable, key rejected", False,
                  f"{kind}: put a real key in .env (OPENAI_API_KEY)")
        else:
            check("[LIVE] endpoint reachable", False, f"{kind}: {str(e)[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="also send one tiny request to the API")
    args = ap.parse_args()

    offline_checks()
    if args.live:
        live_probe()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

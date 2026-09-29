"""Cognition layer: methodology, capability and (later) memory.

Phase 1 scope — **Methodology Shadow Mode**:

- extract candidate methodologies from a finished week (one LLM call per
  character per week, budgeted);
- record which methodology *would* have been selected for each activity, and
  link it to the objective outcome that already happened;
- maintain `global_value` / `confidence` / `context_values` from that evidence;
- **never** change agent behaviour, prompts, or the existing skill numbers.

The modules mirror the design document's layout so later phases can grow into
them:

    models.py             data contracts (Methodology, ContextSignature, events)
    event_store.py        append-only capability events (phase 0 identity contract)
    reward_model.py       objective outcome -> reward -> method value update
    methodology_policy.py context signature + candidate scoring (shadow selection)
    materializer.py       rebuild views (capabilities/methodologies) from events
    shadow.py             the phase 1 orchestrator wired into the world

Everything in this package is inert unless `world.cognition.methodology_shadow`
is enabled, and nothing here writes to the simulation ledger.
"""

from __future__ import annotations

__all__ = [
    "models",
    "event_store",
    "reward_model",
    "methodology_policy",
    "materializer",
    "shadow",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from src.utils import num_tokens_from_string, get_logger


@dataclass
class ContextConfig:
    model_ctx_tokens: int = 8192
    target_response_tokens: int = 512
    min_history_tokens: int = 1024  # keep at least this much history if possible


class ConversationContext:
    """Prepare chat messages under a token budget with simple summarization.

    We do not call LLMs to summarize; we compress older turns heuristically.
    Optionally reuse an external logger so the agent writes to a single log file.
    """

    def __init__(
        self,
        *,
        cfg: ContextConfig | None = None,
        log_name: str = "context",
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg or ContextConfig()
        # If an external logger is provided (e.g., the agent's logger), reuse it
        # so all messages end up in the same per-agent log file.
        self.logger = logger or get_logger(f"agent_{log_name}", quiet=True)

    def _count(self, messages: List[Dict[str, str]]) -> int:
        return sum(num_tokens_from_string(m.get("content", "")) for m in messages)

    def pack(
        self,
        *,
        system_prompt: str,
        history: List[Dict[str, str]] | None,
        user_turn: Dict[str, str],
    ) -> List[Dict[str, str]]:
        budget = self.cfg.model_ctx_tokens - self.cfg.target_response_tokens
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        history = history or []

        # naive include all, then shrink if needed
        msgs.extend(history)
        msgs.append(user_turn)

        total = self._count(msgs)
        if total <= budget:
            return msgs

        # Compress oldest history into a single system summary
        # Strategy: keep newest half of history, summarize the rest by truncating text
        keep_from = max(0, len(history) // 2)
        older = history[:keep_from]
        newer = history[keep_from:]

        def truncate_text(t: str, limit: int = 800) -> str:
            t = t.replace("\n\n", "\n").replace("\n", " ")
            return (t[:limit] + "...") if len(t) > limit else t

        older_text = " ".join(
            f"[{m.get('role', '?')}] {truncate_text(m.get('content', ''))}"
            for m in older
        )
        summary = f"Earlier context summary: {truncate_text(older_text, 1200)}"

        msgs = [{"role": "system", "content": system_prompt}]
        msgs.append({"role": "system", "content": summary})
        msgs.extend(newer)
        msgs.append(user_turn)

        total2 = self._count(msgs)
        if total2 > budget:
            # If still too long, drop more older messages but keep summary
            drop_n = max(0, len(newer) - 3)
            newer = newer[-3:]
            msgs = [{"role": "system", "content": system_prompt}]
            msgs.append({"role": "system", "content": summary})
            msgs.extend(newer)
            msgs.append(user_turn)
            self.logger.info(f"Context shrunk: dropped {drop_n} history turns")

        return msgs

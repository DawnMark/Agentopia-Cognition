"""认知层 LLM 调用的 token 账：让"新功能到底多花了多少 token"可以回答。

## 为什么需要它

角色自己的每一次生成（计划、联系、活动、反思）与上帝模型的每一次评估都会写进运行目录的
`generation/` / `god/` 日志，并带上 `input_tokens` / `output_tokens`（用
`num_tokens_from_string` 对 prompt 文本估的，cl100k_base）。但认知层自己的抽取调用
——方法论抽取、记忆合并、Idea 生成与 Idea→方法转换——走的是独立的 `get_response_with_retry`，
**不写任何生成日志**，于是它们的 token 在运行目录里根本不存在。

要做"这一系列新功能比原版多花多少 token"的对比，这个缺口必须补上：`record_llm_usage()`
在每次认知层调用后追加一行 `cognition/token_usage.jsonl`（带账本身份），
审计与对比脚本按 `feature` 汇总即可。

## 口径

- token 数与 `generation/` 日志**用同一个函数**（`num_tokens_from_string`），
  所以两组数字可以直接相加；
- 记录的是**估算值**，不是 API 返回的 usage：这一点在报告里要写清楚，
  否则"输入 token 56M"这种数会被误读成账单金额；
- 只记认知层的抽取调用。角色与上帝模型的调用已经写在 `generation/` / `god/` 里，
  不在这里重复记（否则会双计）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

USAGE_FILENAME = "token_usage.jsonl"


def estimate_tokens(messages: Any) -> int:
    """与 `DataManager.save_generation` 完全相同的估算方式。"""
    from src.utils import num_tokens_from_string

    if isinstance(messages, str):
        return int(num_tokens_from_string(messages))
    total = 0
    for message in messages or []:
        if isinstance(message, dict):
            total += int(num_tokens_from_string(message.get("content") or ""))
        else:  # pragma: no cover - defensive
            total += int(num_tokens_from_string(str(message)))
    return total


def record_llm_usage(
    dm: "DataManager",
    feature: str,
    *,
    messages: Any,
    response: Any,
    model: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """记一次认知层 LLM 调用的 token（失败也绝不影响主流程）。"""
    try:
        from pathlib import Path

        if isinstance(response, str):
            output_tokens = estimate_tokens(response)
        else:
            import json as _json

            output_tokens = estimate_tokens(
                _json.dumps(response, ensure_ascii=False)
            )
        record: Dict[str, Any] = {
            "feature": str(feature),
            "model": str(model or ""),
            "calls": 1,
            "input_tokens": estimate_tokens(messages),
            "output_tokens": output_tokens,
        }
        if extra:
            for key, value in extra.items():
                if key not in record and key not in ("time", "ledger_event_id", "schema_version"):
                    record[key] = value
        dm.append_ledger_record(Path(dm.root) / "cognition" / USAGE_FILENAME, record)
        return record
    except Exception:  # pragma: no cover - 账目永远不该弄坏一次抽取
        return None

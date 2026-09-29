from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Dict, Set, Tuple
import json
import copy
import re

from src.config import get_config
from src.utils import (
    get_response,
    get_response_with_retry,
    get_logger,
    generate_with_fc,
    is_gen_finished,
    remove_reasoning_content,
    clip_function_context,
    project_root,
)
from src.config import get_world_config
from src.agents.functions import FUNCTIONS, FUNCTION_SETS, dedupe_tool_calls
from src.world.clock import Clock, Stage, TimeState
from src.world.scheduling import MessageCenter
from src.agents.data_manager import ERROR_LOGGER, DataManager
from src.utils import parse_kv_args


config = get_config()


def _last_prose_block(response: str) -> str:
    """Recover usable prose from a response that ignored the output markers.

    Used when `Summary:`/`Reflection:` are missing (KI-12). Marker lines and
    empty paragraphs are dropped; the last non-empty block wins, which is where
    the reflection sits in every observed case.
    """
    text = str(response or "")
    if not text.strip():
        return ""
    text = re.sub(r"</?think_brief>", "", text, flags=re.IGNORECASE)
    blocks: List[str] = []
    for raw_block in re.split(r"\n\s*\n", text):
        block = raw_block.strip()
        if not block:
            continue
        block = re.sub(
            r"^\s*(summary of the activity|summary|reflection)\s*[:：]\s*",
            "",
            block,
            flags=re.IGNORECASE,
        ).strip()
        if block:
            blocks.append(block)
    if not blocks:
        return text.strip()
    return blocks[-1]


def activity_rounds(context) -> int:
    """How many assistant rounds a character's activity context holds.

    Phase 5 / KI-18. The reward has always had a time-cost component, but the
    activity records carried no length, so every outcome fell back to the
    "unknown session length" placeholder and the term was a constant across a
    whole run. A solo or public activity is one character's own exchange, and
    its length is the number of assistant rounds in that exchange — which also
    grows with each tool round.
    """
    if not context:
        return 0
    return sum(
        1
        for message in context
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


class RoleAgent:
    """Role-playing agent with minimal function execution and memory integration."""

    def __init__(
        self,
        name: str,
        clock: Clock,
        msg_center: MessageCenter,
        model: str = "gpt-5-mini",
        *,
        world_name: str = "schooldays",
        no_context_engineering: bool = False,
        no_history: bool = False,
    ) -> None:
        self.name = name
        self.clock = clock
        self.model = model
        # logs dir is created by src.utils import side-effect
        self.logger = get_logger(f"agent_{self.name}", quiet=False)
        # Reuse the same logger to avoid creating a second per-agent log file
        self.dm = DataManager(
            char=self.name, world=world_name, clock=self.clock, model=model
        )
        self.llm = get_response
        self.no_context_engineering = no_context_engineering
        self.no_history = no_history
        # Shared in-memory message center injected by World
        self.msg_center = msg_center

        # Track which scratchpads have been read, normalized without extension
        self._opened_scratchpads: Set[str] = set()

        # record_id of the generation that produced the current stage's answer.
        # Extra records (e.g. the compact reasoning summary) are appended right
        # after it, so "the last line of the week's file" cannot identify it.
        self.last_main_generation_id: Optional[str] = None

        # Cache of activities we proposed this week: { activity_name: {"invited_persons": List[str], "time": str} }
        # Cleared during the contact stage when slot==1; recorded only after a successful propose, for later cancel.
        self.proposed_activities: Dict[str, Dict[str, object]] = {}

        # Monotonic action counter for MessageCenter payloads. Every action in
        # one CONTACT slot carries the same timestamp, so this sequence (not the
        # insertion order of the shared queue) is what orders them.
        self._msg_seq = 0


        self.config = config

        # Phase 1: methodology shadow mode. Always constructed, inert unless
        # world.cognition.methodology_shadow is enabled, and it only ever writes
        # to persona/<name>/cognition/ (never to prompts, plans or state).
        from src.agents.cognition.shadow import MethodologyShadow, ShadowConfig

        self.shadow = MethodologyShadow(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.name,
            model=model,
            traits=self._persona_traits(),
            config=ShadowConfig.from_world_config(config.get("world", {})),
            language=str(config.get("world", {}).get("language", "en")),
        )
        # Phase 4: passive method hints. This is the first component that changes
        # what the character sees; it only adds a menu the character may ignore.
        # Built *before* the activity-observer wiring below, which has to ask
        # whether hints are on — reading `self.hint_config` there before it
        # existed made every run with the cognition switches off raise
        # AttributeError (a latent bug from phase 4 that only a default run
        # reaches, because an enabled shadow short-circuits the same condition).
        from src.agents.cognition.method_hints import HintConfig, MethodHintProvider

        self.hint_config = HintConfig.from_world_config(config.get("world", {}))
        self.method_hints = MethodHintProvider(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.name,
            traits=self._persona_traits(),
            config=self.hint_config,
            language=str(config.get("world", {}).get("language", "en")),
        )
        # The raw `world.cognition` section: settlement hooks (phase 5/6) read
        # their own switches from it instead of growing another config object.
        self.cognition_config = (config.get("world", {}) or {}).get("cognition") or {}

        if self.shadow.enabled or self.hint_config.enabled:
            self.dm.activity_record_observer = self._observe_activity_record
        if self.shadow.enabled:
            self.logger.info(
                "[SHADOW] methodology shadow mode enabled "
                f"(max_extract_calls_per_week="
                f"{self.shadow.config.max_extract_calls_per_week})"
            )

        # Phase 2: structured memory + relation graph, also shadow-only.
        from src.agents.cognition.consolidator import Consolidator, MemoryConfig
        from src.agents.cognition.memory_retriever import (
            MemoryRetriever,
            RetrievalConfig,
        )

        self.memory_config = MemoryConfig.from_world_config(config.get("world", {}))
        self.consolidator = Consolidator(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.name,
            model=model,
            config=self.memory_config,
            language=str(config.get("world", {}).get("language", "en")),
            world_cognition=(config.get("world", {}) or {}).get("cognition") or {},
        )
        self.memory_retriever = MemoryRetriever(
            dm=self.dm,
            clock=self.clock,
            config=RetrievalConfig.from_world_config(config.get("world", {})),
        )
        if self.memory_config.enabled:
            self.logger.info(
                "[SHADOW] memory shadow mode enabled "
                f"(max_new_per_week={self.memory_config.max_new_per_week}, "
                f"half_life_weeks={self.memory_config.half_life_weeks})"
            )

        # Phase 3: idea engine (also shadow-only; needs phase 2 memories).
        from src.agents.cognition.idea_engine import IdeaConfig
        from src.agents.cognition.idea_pipeline import IdeaEngine

        self.idea_config = IdeaConfig.from_world_config(config.get("world", {}))
        self._week_start_vitality: Optional[float] = None
        self.idea_engine = IdeaEngine(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.name,
            model=model,
            traits=self._persona_traits(),
            config=self.idea_config,
            language=str(config.get("world", {}).get("language", "en")),
        )
        if self.idea_config.enabled:
            self.logger.info(
                "[SHADOW] idea engine enabled "
                f"(max_per_week={self.idea_config.max_ideas_per_week})"
            )

        # The menu is a *consumer*: every method it offers was produced by the
        # phase-1 extraction (and, for idea-derived ones, by the phase-2/3
        # chain). Turning the menu on while its producers are off gives a
        # character an empty menu for the whole run, silently — the run looks
        # healthy and measures nothing (measured the hard way on 2026-09-27,
        # when a 2-year run started with `method_hints` on and
        # `methodology_shadow` off). Say so out loud instead.
        if self.hint_config.enabled and not self.shadow.enabled:
            self.logger.warning(
                "[HINT] method hints are enabled but world.cognition."
                "methodology_shadow is off: nothing extracts methods, so the "
                "weekly menu will stay empty for the whole run."
            )

        # Phase 5: the rest of a method's life (specialised / archived /
        # versioned). Off unless world.cognition.method_lifecycle is enabled, in
        # which case it only writes capability events at settlement.
        from src.agents.cognition.method_lifecycle import LifecycleConfig, MethodLifecycle

        self.lifecycle_config = LifecycleConfig.from_world_config(config.get("world", {}))
        self.method_lifecycle = MethodLifecycle(
            dm=self.dm, clock=self.clock, config=self.lifecycle_config
        )
        if self.lifecycle_config.enabled:
            self.dm.method_refiner = self.method_lifecycle.record_refinement
            self.logger.info(
                "[LIFECYCLE] method lifecycle enabled "
                f"(archive after {self.lifecycle_config.archive_after_unused_weeks} "
                f"unused weeks)"
            )

    def _observe_activity_record(self, record: Dict[str, Any]) -> None:
        """Route one finished activity to the cognition trackers.

        A real adoption (phase 4) claims the activity, so the shadow tracker
        (phase 1) does not also attribute the same practice counterfactually —
        one practice must not move two different estimates.
        """
        claimed = False
        if self.hint_config.enabled:
            try:
                claimed = self.method_hints.observe_activity(record)
            except Exception as e:  # pragma: no cover - never break a run
                self.logger.warning(f"[HINT] practice attribution failed: {e!r}")
        if not claimed and self.shadow.enabled:
            try:
                self.shadow.observe_activity(record)
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"[SHADOW] activity observation failed: {e!r}")

    def _persona_traits(self) -> Dict[str, float]:
        """Personality traits the shadow policy may use (creativity/curiosity)."""
        out: Dict[str, float] = {}
        try:
            profile = self.dm.read_profile()
            quantitative = (profile.get("personality_traits") or {}).get(
                "quantitative"
            ) or {}
            for key in ("creativity", "curiosity", "intelligence", "confidence"):
                if key in quantitative:
                    out[key] = float(quantitative[key])
        except Exception:
            # A missing profile must never block agent construction.
            return out
        return out

    def _exec_function(
        self,
        name: str,
        args: dict,
        *,
        allowed_new_characters: set[str] | None = None,
    ) -> str:
        """Execute a function call.

        Args:
            name: Function name
            args: Function arguments (JSON string)
            allowed_new_characters: If set, allow creating character scratchpads only for these names.
                                   None means no new character scratchpad creation allowed.
        """
        if isinstance(args, str):
            args = json.loads(args) if args else {}

        if name == "list_scratchpads":
            return self.dm.list_scratchpads()
        elif name == "read_map":
            return self.dm.read_map()
        elif name == "read_scratchpad":
            if "s_name" not in args:
                return "ERROR: Missing required argument: s_name"
            s_name = args["s_name"]
            out = self.dm.read_scratchpad(s_name)
            # Record successfully opened pad for later write validation
            if not out.startswith("ERROR:"):
                self._opened_scratchpads.add(s_name)
            return out
        elif name == "update_scratchpad":
            required_args = ["s_name", "content"]
            missing_args = [arg for arg in required_args if arg not in args]
            if missing_args:
                return f"ERROR: Missing required arguments: {', '.join(missing_args)}"

            s_name = args["s_name"]
            content = args["content"]
            create_new_scratchpad = bool(args.get("create_new_scratchpad", False))

            if not create_new_scratchpad:
                if s_name not in self._opened_scratchpads:
                    return "ERROR: To update an existing scratchpad, you must first read its content"

            # Check if character name is in allowed list
            if s_name.startswith("characters/") and create_new_scratchpad:
                if allowed_new_characters is None:
                    return "ERROR: Creating new character scratchpads is not allowed in this context"
                char_name = (
                    s_name.replace("characters/", "")
                    .replace(".jsonl", "")
                    .replace(".txt", "")
                )
                if char_name not in allowed_new_characters:
                    return f"ERROR: You can only create scratchpads for participants in this activity: {', '.join(sorted(allowed_new_characters))}"

            return self.dm.update_scratchpad(
                s_name,
                content,
                create_new_scratchpad,
                allow_characters_create=(allowed_new_characters is not None),
            )
        else:
            return f"ERROR: Unknown function: {name}"

    def _get_available_funcs(self, no_write_funcs: bool = False) -> List[dict]:
        """Return stage-specific function allowlist and resolved specs."""
        func_names = FUNCTION_SETS

        if self.no_context_engineering:
            func_names = []
        elif self.no_history:
            func_names = [t for t in func_names if "history" not in t]

        if no_write_funcs:
            func_names = [
                t for t in func_names if "write" not in t and "update" not in t
            ]

        funcs = [FUNCTIONS[t] for t in func_names if t in FUNCTIONS]
        return func_names, funcs

    @staticmethod
    def _normalise_generation_output(output: Any) -> List[Dict[str, str]]:
        """把 `generate_with_fc` 的返回值规范成"消息字典列表"（KI-25）。

        `generate_with_fc` 在重试与回退模型都用尽之后返回 `_ERROR_RESPONSE`——一个
        **字符串**（`"NO_RESPONSE"`）。`list.extend("NO_RESPONSE")` 会把它拆成一个个
        字符，于是 `outputs` 里混进十几个单字符元素；随后 `save_generation` 统计 token
        时 `m.get("content")` 直接抛 `AttributeError`，**整轮多年运行就此崩掉**。
        实测：3 年验收运行跑到第 29 周（Y2022-W09）崩在这里。

        规范化之后：字符串当成一条 assistant 消息（错误哨兵则丢弃）、
        字典包成单元素列表、列表里非字典的项按同样规则处理。
        """
        from src.utils import _ERROR_RESPONSE

        if output is None:
            return []
        if isinstance(output, str):
            if output == _ERROR_RESPONSE or not output.strip():
                return []
            return [{"role": "assistant", "content": output}]
        if isinstance(output, dict):
            return [output]
        messages: List[Dict[str, str]] = []
        for item in output:
            if isinstance(item, dict):
                messages.append(item)
            elif isinstance(item, str) and item.strip():
                messages.append({"role": "assistant", "content": item})
        return messages

    @staticmethod
    def _usable_final_answer(outputs: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Return the last message only if it is an actual assistant answer.

        A message counts only when it is the *final* message of the loop, is an
        assistant message, carries no tool calls and has non-empty content.
        Anything else (a tool result, an empty answer, a mid-turn assistant
        message) means the caller must not treat it as the answer.
        """
        last = outputs[-1] if outputs else None
        if not isinstance(last, dict):
            return None
        if last.get("role") != "assistant":
            return None
        if last.get("tool_calls"):
            return None
        if not str(last.get("content") or "").strip():
            return None
        return last

    def _finalize_without_tools(
        self,
        *,
        model: str,
        inputs: List[dict],
        outputs: List[Dict[str, str]],
        cache_file: str,
    ) -> Optional[Dict[str, str]]:
        """One extra call with tools disabled, to obtain a real answer.

        Used only when the tool loop ended without a usable assistant message.
        The answer is appended to `outputs`, so the saved generation record
        shows what was actually returned.
        """
        try:
            output = generate_with_fc(
                model=model,
                messages=inputs + outputs,
                functions=[],
                tool_choice="none",
                cache_file=cache_file,
                model_type="role",
            )
        except Exception as e:  # never fatal: the caller has an error fallback
            self.logger.warning(f"[GENERATION] finalize call failed: {e!r}")
            return None
        if not output:
            return None
        messages = self._normalise_generation_output(output)
        if not messages:
            self.logger.warning(
                f"[GENERATION] finalize call returned no usable message ({output!r})"
            )
            return None
        outputs.extend(messages)
        return self._usable_final_answer(outputs)

    def _generate_with_functions(
        self,
        inputs: List[dict],
        *,
        max_rounds: int = 8,
        no_write_funcs: bool = False,
        save_to_week_response: bool = True,
        keep_compact_reasoning: bool = True,
        response_postprocessor: Callable[[str], str] | None = None,
        format_validator: Callable[[str], "ValidationResult"] | None = None,
        allowed_new_characters: set[str] | None = None,
        model_override: str | None = None,
    ) -> List[Dict[str, str]]:
        """Run a tool-augmented generation loop and return the final assistant turn.

        Control flags:
        - keep_compact_reasoning: whether to prepend a compact thinking summary to the final text.
        - save_to_week_response: whether to persist the final text into week responses.
        - response_postprocessor: optional function to transform final content before saving.
        - format_validator: optional function to validate response format, triggers retry loop if provided.
        - allowed_new_characters: if set, allow creating character scratchpads only for these names.
        - model_override: if set, use this model instead of self.model (e.g. god_model for fair judging).
        """
        model = model_override or self.model
        outputs = []

        from src.utils import _ERROR_RESPONSE

        # Any previous id belongs to an earlier stage; a caller that rejects the
        # answer of *this* call must never target that older record.
        self.last_main_generation_id = None

        # Build stage functions in Responses API-compatible shape
        allowed_func_names, funcs = self._get_available_funcs(
            no_write_funcs=no_write_funcs
        )

        # Clear previously opened scratchpads
        self._opened_scratchpads.clear()

        from src.utils import get_run_cache_dir

        run_cache = get_run_cache_dir()
        if run_cache is not None:
            cache_file = run_cache / f".cache_agent={self.name}.pkl"
        else:
            cache_file = project_root / f"llm_cache/.cache_agent={self.name}.pkl"

        for _ in range(max_rounds):
            if _ == max_rounds - 1:
                tool_choice = "none"
            else:
                tool_choice = "auto"

            output = generate_with_fc(
                model=model,
                messages=inputs + outputs,
                functions=funcs,
                tool_choice=tool_choice,
                cache_file=str(cache_file),
                model_type="role",
            )

            if output == _ERROR_RESPONSE:
                return [{"role": "assistant", "content": _ERROR_RESPONSE}]

            normalised = self._normalise_generation_output(output)
            if not normalised:
                # 空回答（或失败哨兵）不算一轮结果：跳出循环，交给下面"再要一次真正的
                # 回答"的收尾路径（KI-1），而不是把错误哨兵当成答案。
                self.logger.warning(
                    f"[GENERATION] model returned no usable message ({output!r})"
                )
                break
            outputs.extend(normalised)

            self.logger.debug(f"[GENERATION] output: {json.dumps(output, indent=2)}")

            if is_gen_finished(output):
                break
            else:
                # handle function calls
                func_outputs = []
                for item in output:
                    if "tool_calls" in item.keys():
                        # Deduplicate and replace the item in place
                        item["tool_calls"] = dedupe_tool_calls(item["tool_calls"])  # type: ignore[index]

                        for fc in item["tool_calls"]:
                            fc_name = fc["function"]["name"]
                            fc_args = fc["function"]["arguments"]
                            # Validate that the tool is permitted
                            if fc_name in allowed_func_names:
                                func_res = self._exec_function(
                                    fc_name,
                                    fc_args,
                                    allowed_new_characters=allowed_new_characters,
                                )
                            else:
                                func_res = f"ERROR: Function {fc_name} not allowed"

                            func_outputs.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": fc["id"],
                                    "name": fc_name,
                                    "content": func_res,
                                }
                            )

                        # outputs.append({
                        #     "type": "function_call_output",
                        #     "call_id": item.call_id,
                        #     "output": json.dumps({
                        #         item.name: func_res
                        #     }),
                        # })
                outputs.extend(func_outputs)

        # The loop can end right after a tool result: the model may ask for a
        # tool in the last permitted round, or answer with nothing usable. The
        # last message is then a *tool result*, and taking it as the final answer
        # poisoned the weekly diary and other stages (docs/known-issues.md KI-1),
        # so only a real assistant answer is accepted; otherwise one extra call
        # with tools disabled is made to obtain one.
        final_message = self._usable_final_answer(outputs)
        if final_message is None:
            final_message = self._finalize_without_tools(
                model=model, inputs=inputs, outputs=outputs, cache_file=str(cache_file)
            )
        if final_message is None:
            self.logger.warning(
                "[GENERATION] no usable assistant answer after the tool loop; "
                "returning an error response"
            )
            outputs.append({"role": "assistant", "content": _ERROR_RESPONSE})
            final_message = outputs[-1]

        last_content = str(final_message.get("content") or "")
        if last_content and "</think>" in last_content:
            think, final_answer = last_content.rsplit("</think>", 1)
            think += "</think>"
            final_answer = final_answer.strip()
        else:
            final_answer = last_content
            think = ""

        # === Response Validation Loop ===
        validation_cfg = self.config["response_validation"]
        self.logger.info(
            f"[VALIDATION] format_validator={format_validator is not None}, enabled={validation_cfg.get('enabled', False)}"
        )
        if format_validator and validation_cfg.get("enabled", False):
            from src.agents.response_validator import run_validation_loop
            from dataclasses import asdict

            result = run_validation_loop(
                format_validator=format_validator,
                final_answer=final_answer,
                think=think,
                outputs=outputs,
                inputs=inputs,
                model=model,
                judge_model=validation_cfg.get(
                    "judge_model",
                    self.config["god_model"],
                ),
                cache_file=str(cache_file),
                max_retries=validation_cfg.get("max_retries", 3),
                agent_logger=self.logger,
            )
            outputs = result.outputs
            think = result.think
            final_answer = result.final_answer

            # Save validation records if any retries occurred
            if len(result.validation_records) > 1:
                validation_inputs = [
                    {
                        "role": "system",
                        "content": f"Validation process with {len(result.validation_records)} attempts",
                    }
                ]
                validation_outputs = [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            [asdict(r) for r in result.validation_records],
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ]
                self.dm.save_generation(
                    validation_inputs, validation_outputs, filename="resp_validation"
                )
        # === End Validation Loop ===

        # Apply postprocessor after validation
        if response_postprocessor:
            final_answer = response_postprocessor(final_answer)
            assert final_answer, "response_postprocessor must return non-empty string"
            outputs[-1]["content"] = think + final_answer

        self.last_main_generation_id = self.dm.save_generation(inputs, outputs)

        # Optional keep_compact_reasoning header
        # Skip for closed-source models — they have no reasoning to compact
        from src.utils import _is_closed_source_model

        if keep_compact_reasoning and not _is_closed_source_model(model):
            think_brief = self._compact_response(inputs, outputs)
            final_answer = think_brief + "\n\n" + final_answer

        # Optional persistence
        if save_to_week_response:
            self.dm.save_to_week_response(final_answer)

        return [{"role": "assistant", "content": final_answer}]

    def _compact_response(self, inputs, outputs) -> str:
        # Use this context directly to generate a summary
        inputs = copy.deepcopy(inputs)

        from src.agents.prompts import COMPACT_PROMPT

        compact_messages = [
            {"role": "user", "content": COMPACT_PROMPT.format(char=self.name)}
        ]
        inputs[0]["content"] = inputs[0]["content"].split("## Requirements")[0]

        compact_inputs = inputs + outputs + compact_messages

        from src.utils import get_run_cache_dir

        run_cache = get_run_cache_dir()
        if run_cache is not None:
            cache_file = run_cache / f".cache_agent={self.name}.pkl"
        else:
            cache_file = project_root / f"llm_cache/.cache_agent={self.name}.pkl"

        compact_output = generate_with_fc(
            model=self.model,
            messages=compact_inputs,
            tool_choice="none",
            cache_file=str(cache_file),
        )

        self.dm.save_generation(compact_inputs, compact_output)

        think_brief = (
            compact_output[-1]["content"]
            .split("</think>")[-1]
            .replace(
                "<summary>",
                "<think_brief> (A summary of my thinking and function calling process)\n",
            )
            .replace("</summary>", "</think_brief>")
            .strip()
        )

        return think_brief

    # High-level actions -----------------------------------------------------
    def clear_on_week_start(self) -> None:
        """
        Clear the data on the start of a new week.
        """
        self.dm.clear_week_responses()
        self.proposed_activities.clear()
        self._msg_seq = 0

    def get_schedule(self):
        """Get today's schedule for this agent.

        Returns the last (most recent) schedule for today.
        Priority is handled by write order: Public < Encounter < Joint.
        """
        return self.dm.get_today_schedule()

    def plan(self) -> None:
        t = self.clock.get_time()
        self.logger.info(f"[PLAN][year={t.year} week={t.week}] start planning")

        inputs = self.dm.roleplay_prompt() + self.dm.plan_prompt()

        # Phase 4: offer a small menu of methods the character may use this week.
        # The menu is advisory; adoption is the character's own decision and is
        # only recorded if it says so.
        # Remember the vitality the week *started* with: the idea budget at
        # REVIEW must reflect the capacity the character had when planning, not
        # the post-week low (which is 0 for most of a long run).
        try:
            self._week_start_vitality = float(
                self.dm.read_state(exclude_cur_t=False).get("vitality") or 0
            )
        except Exception:  # pragma: no cover - defensive
            self._week_start_vitality = None

        if self.hint_config.enabled:
            try:
                hint_block = self.method_hints.prepare_week()
                if hint_block:
                    inputs = inputs + [{"role": "user", "content": hint_block}]
            except Exception as e:  # pragma: no cover - hints must never break a run
                self.logger.warning(f"[HINT] preparing method hints failed: {e!r}")

        outputs = self._generate_with_functions(inputs)

        self.logger.info(
            f"[PLAN][year={t.year} week={t.week} person={self.name}] {outputs}"
        )

        # Read the character's own decision (it may have ignored the menu).
        if self.hint_config.enabled:
            try:
                adoption = self.method_hints.record_adoption(outputs[-1]["content"])
                if adoption is not None:
                    self.logger.info(
                        f"[HINT] {self.name} will try {adoption.method_id} "
                        f"({adoption.title[:40]}) this week"
                    )
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"[HINT] reading the adoption failed: {e!r}")

        # Parse and apply living standard selection
        self._apply_living_standard(outputs)

    def _apply_living_standard(self, outputs: List[dict]) -> None:
        """Parse <living_standard> tag from plan output and apply to state."""
        import re
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger()

        final_text = outputs[-1]["content"]
        if not final_text:
            self.logger.warning(
                f"[LIVING_STANDARD] {self.name}: No output text found, cannot parse living standard"
            )
            return

        # Parse <living_standard>...</living_standard>
        pattern = r"<living_standard>\s*([^<]+?)\s*</living_standard>"
        match = re.search(pattern, final_text, re.IGNORECASE)

        if not match:
            self.logger.warning(
                f"[LIVING_STANDARD] {self.name}: No living standard selection found, defaulting to frugal"
            )
            standard = "frugal"
        else:
            content = match.group(1).strip().lower()

            # Find valid standard in content using word boundary matching
            import difflib

            valid_standards = ["frugal", "moderate", "comfortable", "luxurious"]

            # Try exact match first
            if content in valid_standards:
                standard = content
            else:
                # Try fuzzy matching with difflib (handles typos and variants)
                closest = difflib.get_close_matches(
                    content, valid_standards, n=1, cutoff=0.7
                )
                if closest:
                    standard = closest[0]
                else:
                    # Check if any valid standard appears as word in content
                    matched = None
                    for valid in valid_standards:
                        if re.search(r"\b" + valid + r"\b", content):
                            matched = valid
                            break

                    if matched:
                        standard = matched
                    else:
                        self.logger.warning(
                            f"[LIVING_STANDARD] {self.name}: Invalid standard '{content}', falling back to frugal"
                        )
                        standard = "frugal"

        # Cost and material delta mapping
        cost_map = {
            "frugal": 100,
            "moderate": 200,
            "comfortable": 300,
            "luxurious": 500,
        }
        material_map = {"frugal": -5, "moderate": 0, "comfortable": 5, "luxurious": 10}

        cost = cost_map[standard]
        material_delta = material_map[standard]

        # Read current state (include cur_t entries from earlier writes in same stage)
        state = self.dm._read_state_current(exclude_cur_t=False)
        current_deposit = state["assets"]["deposit"]

        # If deposit insufficient, downgrade to frugal
        if current_deposit < cost:
            self.logger.warning(
                f"[LIVING_STANDARD] {self.name}: Insufficient deposit ({current_deposit} < {cost}), "
                f"downgrading from '{standard}' to 'frugal'"
            )
            if verify_logger:
                verify_logger.warning(
                    f"[VERIFY-ECONOMY] {self.name}: Downgraded {standard} → frugal (insufficient deposit)"
                )
            standard = "frugal"
            cost = cost_map[standard]
            material_delta = material_map[standard]

        # Deduct cost
        state["assets"]["deposit"] = current_deposit - cost

        # Update material fulfillment
        new_material = max(
            0, min(100, state["fulfillment"]["material"] + material_delta)
        )
        state["fulfillment"]["material"] = new_material

        # Write back state
        self.dm.save_state(state)

        # Log the choice
        self.logger.info(
            f"[LIVING_STANDARD] {self.name} chose {standard}, "
            f"cost: {cost}, deposit: {current_deposit} → {state['assets']['deposit']}, "
            f"material: {material_delta:+d}"
        )

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-ECONOMY] {self.name}: standard={standard}, "
                f"deposit {current_deposit} → {state['assets']['deposit']}, "
                f"material {material_delta:+d}"
            )

    def signup_public_events(self, events: List["PublicEvent"]) -> List[str]:
        """Sign up for eligible public events.

        Args:
            events: All public events available this week

        Returns:
            List of event_ids that the agent signed up for
        """
        import re
        from src.world.scheduling import Schedule
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="public_activity")
        t = self.clock.get_time()

        # Filter for eligibility
        eligible_events = [evt for evt in events if evt.is_eligible(self.name)]
        if not eligible_events:
            return []

        # Filter out events on days when agent is already busy
        busy_days = self.dm.get_busy_days_this_week()
        available_events = [
            evt for evt in eligible_events if evt.start_day not in busy_days
        ]

        if not available_events:
            self.logger.info(
                f"[PUBLIC][year={t.year} week={t.week}] {self.name}: no available events (all conflict with schedule)"
            )
            return []

        # Build events list for prompt (sorted for cache determinism)
        events_list = "\n".join(
            [
                f"- {evt.event_name}\n"
                f"  Time: {evt.start_t}\n"
                f"  Repeat: {'Weekly for ' + str(evt.repeat_weeks) + ' weeks' if evt.repeat_weeks > 1 else 'One-time event'}\n"
                f"  Description: {evt.description}"
                for evt in sorted(
                    available_events, key=lambda e: (e.start_day, e.event_name)
                )
            ]
        )

        # Build messages with roleplay context + schedule + time
        inputs = self.dm.roleplay_prompt() + self.dm.signup_prompt(events_list)

        # Generate response using unified method
        outputs = self._generate_with_functions(inputs)
        response = outputs[-1]["content"] if outputs else ""

        if not response:
            return []

        # Parse <role_action>signup(event_name="...")</role_action> tags
        pattern = r'<role_action>\s*signup\s*\(\s*event_name\s*=\s*["\']([^"\']+)["\']\s*\)\s*</role_action>'
        matches = re.findall(pattern, response)

        # Match names to events (normalize whitespace for comparison)
        name_to_event = {
            evt.event_name.strip().lower(): evt for evt in available_events
        }

        signups = []
        # Track days already signed up in this batch to prevent same-day duplicates
        signed_days: set[int] = set()

        for name in matches:
            normalized = name.strip().lower()
            if normalized not in name_to_event:
                continue

            evt = name_to_event[normalized]

            # Check: cannot sign up for multiple public activities on the same day
            if evt.start_day in signed_days:
                self.logger.warning(
                    f"[PUBLIC] {self.name}: skipping duplicate signup for day {evt.start_day} "
                    f"(already signed up for another event on this day)"
                )
                continue

            # Create schedule for this public activity (this week only)
            # activity_id auto-generated by Schedule.__post_init__
            schedule = Schedule(
                activity_name=evt.event_name,
                activity_time=evt.start_t,
                participants=[self.name],
                type="public",
                status="created",
                event_description=evt.description,
            )
            self.dm.add_schedule(schedule)
            signups.append(schedule.activity_id)
            signed_days.add(evt.start_day)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-PUBLIC] {self.name}: signed up for {len(signups)} events: {signups}"
            )

        self.logger.info(
            f"[PUBLIC][year={t.year} week={t.week}] {self.name} signed up for: {signups}"
        )

        return signups

    def contact(self) -> None:
        """Contact stage: read inbox summary, decide, then send role actions.

        Processing flow:
        - When slot==1, inject the contact-stage prompt; then inject the new
          messages received last round and the recent-history summary.
        - Generate text (read_* tools allowed, but no write-type tools provided).
        - Parse <role_action> ... </role_action> from the output; supports
          contact / propose_joint_activity / respond_invitation / cancel_joint_activity.
        - Persist sends via the MessageCenter (if available) and DataManager.
        Returns the final text and the complete message sequence.
        """
        import re

        t = self.clock.get_time()
        self.logger.info(
            f"[CONTACT][year={t.year} week={t.week} contact slot={t.slot}] "
        )

        inputs = self.dm.roleplay_prompt() + self.dm.contact_prompt()

        outputs = self._generate_with_functions(
            inputs, no_write_funcs=True, format_validator=self._format_check_contact
        )

        final_text = outputs[-1]["content"]
        if not final_text:
            self.logger.info("[CONTACT] empty response, skipping this slot")
            return

        # Parse and execute role actions
        errs: Dict[str, str] = {}
        parsed, parse_errs = self._parse_role_actions(final_text)
        errs.update(parse_errs)

        handlers = {
            "contact": self._handle_contact_action,
            "propose_joint_activity": self._handle_propose_action,
            "respond_invitation": self._handle_respond_action,
            "cancel_joint_activity": self._handle_cancel_action,
        }

        # Message rate limit: send at most n_action_per_slot per round (all send actions count as 1)
        limit = int(config["world"]["contact"]["n_action_per_slot"])
        self.n_action_this_slot = 0

        for item in parsed:
            act = item["act"]
            act_type = item["type"]
            args = item["args"]
            fn = handlers.get(act_type)
            if fn is None:
                errs[act] = f"unknown action: {act_type}"
                continue

            if self.n_action_this_slot >= limit:
                errs[act] = (
                    f"this action is discarded because you have triggered {self.n_action_this_slot} actions in this slot, which exceeds the limit of {limit}."
                )
                continue

            # Drop try/except so backend errors surface directly, for easier debugging
            res = fn(args=args, act=act, errs=errs)
            if res:  # executed successfully
                self.n_action_this_slot += 1

        # Log outcome and persist errors for next slot's prompt
        if errs:
            err_str = "\n".join([f"Action: {k}\nError: {v}" for k, v in errs.items()])
            ERROR_LOGGER.warning(
                f"There are some errors in actions generated by {self.name} agent:\n{err_str}"
            )
            self.dm.set_last_slot_errors(err_str)
            self.mark_last_generation_rejected(err_str)
        else:
            self.dm.set_last_slot_errors("")

    def finalize_contact(self) -> None:
        """
        1. Pull confirmed joint activities from MessageCenter and write to activity jsonl.
        2. Read the notification.
        3. Generate a summary for the contact stage.
        """

        t = self.clock.get_time()
        self.logger.info(
            f"[FINALIZE CONTACT][year={t.year} week={t.week}] finalize contact"
        )

        # Step 1: pull joint activities (Schedule objects) and persist only created ones
        sched_res = self.msg_center.get_scheduling_result(self.name)

        created_lines: List[str] = []
        canceled_lines: List[str] = []

        for sch in sched_res:
            if sch.status == "created":
                created_lines.append(
                    f"- '{sch.activity_name}' at {sch.activity_time}; proposed by {sch.proposer}; "
                    f"participants: {', '.join(sch.participants)}"
                )
                # Approach 2: all invited see the created info, but it is only persisted when I am in participants.
                if self.name in sch.participants:
                    self.dm.add_schedule(sch)
            else:
                reason = sch.cancel_reason
                canceled_lines.append(
                    f"- '{sch.activity_name}' at {sch.activity_time}; cancel_reason: {reason}"
                )

        # Step 2: read system notifications for this persona
        notifications = self.msg_center.get_notifications(self.name)

        # Step 3: LLM summary for contact stage

        created_activity_str = (
            "### Joint Activities Confirmed\n\n" + "\n".join(created_lines) + "\n\n"
            if created_lines
            else ""
        )
        canceled_activity_str = (
            "### Joint Activities Cancelled\n\n" + "\n".join(canceled_lines) + "\n\n"
            if canceled_lines
            else ""
        )
        notifications_str = (
            "### Notifications\n\n" + "\n".join(notifications) + "\n\n"
            if notifications
            else ""
        )

        scheduling_results_str = (
            created_activity_str + canceled_activity_str + notifications_str
        )

        inputs = self.dm.roleplay_prompt() + self.dm.finalize_contact_prompt(
            scheduling_results_str
        )

        # Single round; no write functions in finalize
        outputs = self._generate_with_functions(
            inputs, max_rounds=1, no_write_funcs=True, keep_compact_reasoning=False
        )

        self.logger.info(
            f"[FINALIZE CONTACT][year={t.year} week={t.week} person={self.name}] {outputs}"
        )

    def enter_joint_activity(
        self,
        activity_background: str,
        activity_type: str,
        participants: Optional[List[str]] = None,
        location_desc: Optional[str] = None,
    ) -> None:
        """Initialize per-activity chat context for joint activity.

        If participants are provided (JointActivity), forward them to
        DataManager.roleplay_prompt so their character scratchpads are
        prioritized in the listing.
        """
        t = self.clock.get_time()
        self.logger.info(
            f"[ACTIVITY][year={t.year} week={t.week} day={t.day} person={self.name}] enter activity"
        )

        # When meeting new participants, auto-create character scratchpads.
        # Definition of "new": my characters/<who>.jsonl does not exist yet.
        if participants:
            for nm in participants:
                if nm == self.name:
                    continue
                created = self.meet_person(nm)
                if created:
                    self.logger.info(
                        f"[SCRATCHPAD] created characters/{nm}.jsonl for {self.name}"
                    )

        # Build initial context for this activity only; later turns append to it
        required_characters = (
            [c for c in participants if c != self.name] if participants else None
        )

        # Enter the activity: first generate an analysis of the activity
        self.activity_context = self.dm.roleplay_prompt(
            required_characters=required_characters, location_desc=location_desc
        ) + self.dm.activity_prompt(
            activity_type=activity_type,
            activity_background=activity_background,
            location_desc=location_desc,
            participants=participants,
            on_enter_activity=True,
        )

        outs = self._generate_with_functions(
            self.activity_context,
            save_to_week_response=False,
            keep_compact_reasoning=False,
        )
        analysis = outs[-1]["content"]

        self.activity_context = self.dm.roleplay_prompt(
            required_characters=required_characters, location_desc=location_desc
        ) + self.dm.activity_prompt(
            activity_type=activity_type,
            activity_background=activity_background,
            location_desc=location_desc,
            participants=participants,
        )

        self.activity_context.append(
            {
                "role": "user",
                "content": "## Your Analysis for this Activity\n\n"
                + analysis
                + "\n\n"
                + "## The Activity Starts Now\n\n",
            }
        )

    def act_in_activity(
        self, activity_type: str = "joint", i_turn: Optional[int] = None
    ) -> str:
        """Generate one activity turn and append to local context."""
        from src.utils import add_speaker_and_turn
        from src.agents.response_validator import (
            validate_activity_tags,
            validate_solo_activity_format,
        )

        if activity_type == "solo":
            outputs = self._generate_with_functions(
                self.activity_context,
                save_to_week_response=False,
                keep_compact_reasoning=False,
                format_validator=validate_solo_activity_format,
            )
        elif activity_type == "public":
            # Public is similar to Solo: has function calling but no special format validator
            outputs = self._generate_with_functions(
                self.activity_context,
                save_to_week_response=False,
                keep_compact_reasoning=False,
            )
        else:
            outputs = self._generate_with_functions(
                self.activity_context,
                save_to_week_response=False,
                keep_compact_reasoning=False,
                response_postprocessor=lambda c: add_speaker_and_turn(
                    c, speaker=self.name, turn=i_turn
                ),
                format_validator=validate_activity_tags,
            )

        content = outputs[-1]["content"]
        # Drop the optional think_brief header if present
        if "</think_brief>" in content:
            content = content.split("</think_brief>")[-1].strip()

        if self.activity_context and self.activity_context[-1]["role"] == "assistant":
            self.activity_context[-1]["content"] += "\n\n" + content
        else:
            self.activity_context.append({"role": "assistant", "content": content})

        return content

    def receive_in_activity(self, content: str) -> None:
        """Append an observation block into the per-activity context."""

        if self.activity_context and self.activity_context[-1]["role"] == "user":
            self.activity_context[-1]["content"] += "\n\n" + content
        else:
            self.activity_context.append({"role": "user", "content": content})

    def mark_last_generation_rejected(self, reason: str) -> None:
        """Mark the generation that produced the answer under review.

        Targets `last_main_generation_id` (the record whose content the caller
        acted on or verified) instead of the last line of the week's file.
        """
        self.dm.mark_generation_rejected(reason, self.last_main_generation_id)

    def exit_activity(
        self,
        activity_type: str,
        *,
        all_participation: Dict[str, str] | None = None,
    ) -> Tuple[Optional[str], str]:
        """Summarize the activity and return (summary, reflection).

        Args:
            activity_type: "joint", "solo", or "public"
            all_participation: (public only) {name: participation_desc} for ALL participants.
                               This agent's own entry will be excluded when building the prompt.
                               Other names = allowed names for creating character scratchpads.

        Returns:
            (summary, reflection) tuple where:
            - For joint activities: (summary_text, reflection_text)
            - For solo/public activities: (None, reflection_text)
        """
        assert activity_type in ["joint", "solo", "public"]

        allowed_new_characters: set[str] | None = None

        if activity_type == "joint":
            from src.agents.prompts import EXIT_JOINT_ACTIVITY_PROMPT

            EXIT_ACTIVITY_PROMPT = EXIT_JOINT_ACTIVITY_PROMPT
        elif activity_type == "public":
            from src.agents.prompts import EXIT_PUBLIC_ACTIVITY_PROMPT

            # Build other participants string, excluding self
            other_participants_activities = ""
            if all_participation:
                lines = [
                    f"- {name}: {desc}"
                    for name, desc in all_participation.items()
                    if name != self.name
                ]
                other_participants_activities = "\n".join(lines)
                allowed_new_characters = {
                    name for name in all_participation.keys() if name != self.name
                }
            EXIT_ACTIVITY_PROMPT = EXIT_PUBLIC_ACTIVITY_PROMPT.format(
                other_participants_activities=other_participants_activities,
            )
        else:
            from src.agents.prompts import EXIT_SOLO_ACTIVITY_PROMPT

            EXIT_ACTIVITY_PROMPT = EXIT_SOLO_ACTIVITY_PROMPT

        t = self.clock.get_time()

        if self.activity_context and self.activity_context[-1]["role"] == "user":
            self.activity_context[-1]["content"] += "\n\n" + EXIT_ACTIVITY_PROMPT
        else:
            self.activity_context.append(
                {"role": "user", "content": EXIT_ACTIVITY_PROMPT}
            )

        outputs = self._generate_with_functions(
            self.activity_context,
            keep_compact_reasoning=False,
            allowed_new_characters=allowed_new_characters,
        )

        content = outputs[-1]["content"] if outputs else ""
        # Drop the optional think_brief header if present
        if "</think_brief>" in content:
            content = content.split("</think_brief>")[-1].strip()

        if self.activity_context and self.activity_context[-1]["role"] == "assistant":
            self.activity_context[-1]["content"] += "\n\n" + content
        else:
            self.activity_context.append({"role": "assistant", "content": content})

        self.last_reflection_parse_fallback = False
        summary, reflection = self._parse_exit_activity_response(content, activity_type)
        if self.last_reflection_parse_fallback:
            # KI-12: the response came without the expected markers, so the text
            # was recovered as "the last prose block". Recorded so acceptance can
            # count and filter these instead of silently trusting them.
            self.logger.warning(
                f"[PARSE_FALLBACK] exit_activity type={activity_type} "
                f"year={t.year} week={t.week} day={t.day} person={self.name}"
            )

        self.logger.info(
            f"[ACTIVITY][year={t.year} week={t.week} day={t.day} person={self.name}] "
            f"summary and reflection generated"
        )

        # Phase 5 / KI-18: the activity's own length, kept before the context is
        # released. The activity records carry it so the reward's time-cost
        # component has real data instead of the "unknown length" placeholder.
        self.last_activity_turns = activity_rounds(self.activity_context)

        # Clean up activity context to release memory
        self.activity_context = None

        return summary, reflection

    def _parse_exit_activity_response(
        self, response: str, activity_type: str
    ) -> Tuple[Optional[str], str]:
        """Parse exit_activity response to extract summary and reflection.

        Args:
            response: LLM response text
            activity_type: "joint", "solo", or "public"

        Returns:
            (summary, reflection) tuple where summary is None for solo/public activities
        """
        import re

        if activity_type == "joint":
            # Expected format:
            # Summary of the Activity:
            # <summary text>
            #
            # Reflection:
            # <reflection text>

            # Find all "Reflection:" occurrences and use the last one
            reflection_matches = list(
                re.finditer(
                    r"Reflection:\s*\n(.+)", response, re.DOTALL | re.IGNORECASE
                )
            )

            if reflection_matches:
                # Use the last match
                last_reflection_match = reflection_matches[-1]
                reflection_start_pos = last_reflection_match.start()

                # Extract summary: from "Summary" to the last "Reflection:"
                summary_match = re.search(
                    r"Summary of the Activity:\s*\n(.+?)(?=\n\s*Reflection:)",
                    response[
                        : reflection_start_pos + 100
                    ],  # Include a bit after to match the lookahead
                    re.DOTALL | re.IGNORECASE,
                )
                summary = summary_match.group(1).strip() if summary_match else ""
                reflection = last_reflection_match.group(1).strip()
            else:
                summary = ""
                reflection = ""

            if not summary:
                self.logger.warning(
                    f"Failed to parse summary from exit_activity response for {self.name}"
                )
                # KI-12: the whole response used to be stored here, which pushed
                # stage directions and reasoning into downstream prompts.
                summary = _last_prose_block(response)
                self.last_reflection_parse_fallback = True
            if not reflection:
                self.logger.warning(
                    f"Failed to parse reflection from exit_activity response for {self.name}"
                )
                reflection = _last_prose_block(response)
                self.last_reflection_parse_fallback = True

            return summary, reflection

        else:  # solo or public
            # Expected format:
            # Reflection:
            # <reflection text>

            # Find all "Reflection:" occurrences and use the last one
            reflection_matches = list(
                re.finditer(
                    r"Reflection:\s*\n(.+)", response, re.DOTALL | re.IGNORECASE
                )
            )

            if reflection_matches:
                reflection = reflection_matches[-1].group(1).strip()
            else:
                reflection = ""

            if not reflection:
                self.logger.warning(
                    f"Failed to parse reflection from exit_activity response for {self.name}"
                )
                # KI-12: recover the reflection as the last prose block rather
                # than storing the raw response.
                reflection = _last_prose_block(response)
                self.last_reflection_parse_fallback = True

            return None, reflection

    # Weekly review ---------------------------------------------------------
    def review(self) -> None:
        """Weekly review: generate summary + reflection and save to weekly_diary."""
        t = self.clock.get_time()
        self.logger.info(f"[REVIEW][year={t.year} week={t.week}] start review")

        inputs = self.dm.roleplay_prompt() + self.dm.review_prompt()
        # No compact reasoning header for the review: the diary must contain the
        # review itself. With the default (True) the header was prepended to the
        # stored text whenever parsing fell back to "the whole answer" (KI-1).
        outputs = self._generate_with_functions(inputs, keep_compact_reasoning=False)

        # Extract final assistant text
        final_text = outputs[-1]["content"]

        # Parse and filter: keep only Summary + Reflection, remove Thinking
        content = self._parse_review_response(final_text)
        self._save_review(content)

        # Phase 1 shadow mode: extract candidate methodologies from this week's
        # Phase 4.5 lesson bridge: the character's own lesson lines (its diary
        # reflection and the 【教训】 section of its notes) become citable
        # memories. Deterministic and free of LLM calls, so it runs before the
        # budgeted extraction.
        if self.memory_config.enabled:
            try:
                lesson_counts = self.consolidator.ingest_lessons()
                if any(lesson_counts.values()):
                    self.logger.info(f"[LESSONS] bridge: {lesson_counts}")
            except Exception as e:  # never break the simulation
                self.logger.warning(f"[LESSONS] ingestion failed: {e!r}")

        # Phase 2 shadow: weekly memory consolidation (one budgeted call). It
        # reads this week's ledger evidence and writes only to
        # persona/<name>/cognition/.
        if self.memory_config.enabled:
            try:
                counts = self.consolidator.consolidate_week()
                if any(counts.values()):
                    self.logger.info(f"[SHADOW] memory consolidation: {counts}")
            except Exception as e:  # shadow must never break the simulation
                self.logger.warning(f"[SHADOW] memory consolidation failed: {e!r}")

        # Phase 1 shadow: extract candidate methodologies from this week's ledger
        # evidence (one budgeted call; a no-op unless enabled). Methods proposed
        # here start at value 0 / confidence 0 and can only gain value from
        # *later* activities, so reflection never reinforces itself.
        #
        # Phase 5 moved this *after* the memory consolidation: the extraction can
        # now name the memories it was derived from, which is what gives
        # review-derived methods a `used_in_plan` back-link too (KI-10 was only
        # covered for idea-conversion methods before).
        if self.shadow.enabled:
            try:
                self.shadow.after_review(weekly_summary=content)
            except Exception as e:  # shadow must never break the simulation
                self.logger.warning(f"[SHADOW] weekly extraction failed: {e!r}")

        # Phase 3 shadow: turn this week's memories into candidate ideas (one
        # budgeted call). Ideas are stored as hypotheses; at most one may become
        # a *candidate* methodology, and nothing here changes behaviour.
        if self.idea_config.enabled:
            try:
                vitality = float(self.dm.read_state(exclude_cur_t=False).get("vitality") or 0)
                week_start = getattr(self, "_week_start_vitality", None)
                if week_start is not None:
                    # The weekly cognitive budget is a *planning* budget: use the
                    # best of (week start, now) so a draining world cannot switch
                    # the engine off halfway through the year.
                    vitality = max(float(vitality), float(week_start))
                counts = self.idea_engine.weekly(vitality=vitality)
                if any(counts.values()):
                    self.logger.info(f"[SHADOW] idea engine: {counts}")
            except Exception as e:  # shadow must never break the simulation
                self.logger.warning(f"[SHADOW] idea generation failed: {e!r}")

        self.logger.info(f"[REVIEW][year={t.year} week={t.week}] completed")

    def _parse_review_response(self, response: str) -> str:
        """Extract the weekly review, never the thinking brief or a tool result.

        Returns "" when the answer has no `Summary:` section. Returning the raw
        text instead (the old behaviour) is what wrote `<think_brief>` headers
        and tool results into weekly_diary.jsonl — see docs/known-issues.md KI-1.
        """
        import re

        text = str(response or "")
        # 1) drop the compact reasoning header, terminated or (if truncated) not
        text = re.sub(r"(?is)<think_brief>.*?</think_brief>", "", text)
        text = re.sub(r"(?is)<think_brief>.*\Z", "", text)
        # 2) keep only the review, which starts at "Summary:"
        match = re.search(r"Summary:", text, re.IGNORECASE)
        if match:
            return text[match.start() :].strip()
        return ""

    def _save_review(self, content: str) -> None:
        """Persist the weekly review, or record why it was not usable.

        A missing `Summary:` is a generation failure, not a diary entry: the
        record is marked rejected (so training-data filtering drops it) and
        nothing is written to memory.
        """
        if content:
            if not self.no_context_engineering:
                self.dm.append_weekly_summary(content)
            return
        self.logger.warning(
            "[REVIEW] response has no usable 'Summary:' section; "
            "weekly diary left unchanged"
        )
        self.mark_last_generation_rejected("review response has no Summary: section")

    # ---------- Contact helpers (parsing + per-action handlers) ----------
    def _missing_params(self, args: Dict[str, object], req: List[str]) -> List[str]:
        """Return missing required keys (empty string counts as missing)."""
        return [k for k in req if k not in args or str(args[k]).strip() == ""]

    def _format_check_contact(self, text: str) -> "ValidationResult":
        """Contact stage format check, reuses _parse_role_actions."""
        from src.agents.response_validator import ValidationResult

        _, parse_errs = self._parse_role_actions(text)
        if parse_errs:
            return ValidationResult(
                passed=False,
                feedback=f"Parse errors: {parse_errs}",
                check_type="format",
            )
        return ValidationResult(passed=True, feedback="", check_type="format")

    def _parse_role_actions(
        self, text: str
    ) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
        """Extract <role_action> blocks and parse into [{act,type,args}], with errs."""
        import re

        from src.utils import extract_role_action_blocks

        items: List[Dict[str, object]] = []
        errs: Dict[str, str] = {}
        blocks = extract_role_action_blocks(text)
        for blk in blocks:
            s = blk.strip()
            m = re.match(r"\s*(\w+)\s*\(\s*(.*)\s*\)\s*\Z", s, flags=re.DOTALL)
            if not m:
                errs[s] = "unparsable role_action block"
                continue
            act = m.group(0)
            act_type = m.group(1).lower()
            args_raw = m.group(2)
            try:
                args = parse_kv_args(args_raw)
            except Exception as e:
                errs[act] = f"bad args for action: {e}"
                continue
            items.append({"act": act, "type": act_type, "args": args})
        return items, errs

    def _add_msg_center(self, payload: Dict[str, object]) -> None:
        """Append one action to the shared MessageCenter, stamped with its order.

        Actions issued in the same CONTACT slot share a timestamp; under the
        parallel contact stage the queue insertion order is non-deterministic,
        so `seq` is what lets settlement decide "the last action wins"
        reproducibly.
        """
        self._msg_seq += 1
        payload["seq"] = self._msg_seq
        self.msg_center.add(payload)

    def _handle_contact_action(
        self, *, args: Dict[str, object], act: str, errs: Dict[str, str]
    ) -> bool:
        t = self.clock.get_time()
        required = ["to"]
        missing = self._missing_params(args, required)
        if missing:
            errs[act] = f"lacks required params: {', '.join(missing)}"
            return False
        to = str(args["to"]).strip()
        if not self.dm.check_char_exist(to):
            errs[act] = (
                f"send_message failed: '{to}' is not an interactable character in this simulation. You can only interact with characters listed in your scratchpads."
            )
            return False
        self.dm.send_message(to=to, content=act)
        self._add_msg_center(
            {  # type: ignore[attr-defined]
                "time": str(t),
                "from": self.name,
                "to": to,
                "type": "contact",
                "raw_action": act,
                "message": str(args.get("message", "")),
            }
        )
        return True

    def _handle_propose_action(
        self, *, args: Dict[str, object], act: str, errs: Dict[str, str]
    ) -> bool:
        t = self.clock.get_time()
        required = ["activity_name", "invited_persons", "time", "location"]
        missing = self._missing_params(args, required)
        if missing:
            errs[act] = f"lacks required params: {', '.join(missing)}"
            return False
        at_str = str(args["time"]).strip()
        try:
            at = TimeState.from_string(at_str)
        except Exception:
            errs[act] = f"illegal time format {at_str}"
            return False
        max_weeks = int(config["world"]["contact"]["max_weeks_for_future_schedule"])
        n_week = int(config["world"]["time"]["n_week"])
        weeks_ahead = (at.year - t.year) * n_week + (at.week - t.week)
        last_y = t.year
        last_w = t.week + max_weeks
        while last_w > n_week:
            last_w -= n_week
            last_y += 1
        if weeks_ahead < 0:
            errs[act] = (
                f"time {at_str} is in the past relative to current week Y{t.year}-W{t.week:02d}"
            )
            return False
        if weeks_ahead > max_weeks:
            errs[act] = (
                f"proposed time {at_str} is outside the allowed window. Allowed weeks (inclusive): "
                f"Y{t.year}-W{t.week:02d} to Y{last_y}-W{last_w:02d}"
            )
            return False
        # Validate location (must match an existing key)
        loc = str(args.get("location", "")).strip()

        if not self.dm.location_store.is_valid(loc):
            errs[act] = (
                f"unknown location: {loc}. Pick an exact location name from the available locations list."
            )
            return False

        activity_name = str(args["activity_name"]).strip()
        if activity_name in self.proposed_activities:
            errs[act] = (
                f"activity_name '{activity_name}' already proposed by {self.name}. Please use a different activity_name"
            )
            return False
        invited = args["invited_persons"]
        if isinstance(invited, str):
            try:
                import ast

                invited = ast.literal_eval(invited)
            except Exception:
                invited = [invited]
        if not isinstance(invited, list) or not invited:
            errs[act] = "'invited_persons' must be a non-empty list of names"
            return False
        invited = [str(p) for p in invited if str(p) != self.name]
        if not invited:
            errs[act] = "you must invite at least one person other than yourself"
            return False
        required = args.get("required_participants", [])
        if isinstance(required, str):
            try:
                import ast

                required = ast.literal_eval(required)
            except Exception:
                required = [required]
        if not isinstance(required, list):
            required = []
        invalid = []
        for p in invited + required:
            if not self.dm.check_char_exist(p) and not p in invalid:
                invalid.append(p)
        if invalid:
            errs[act] = "unknown persons: " + ", ".join(invalid)
            return False
        for to in invited:
            self.dm.send_message(to=to, content=act)
        self._add_msg_center(
            {  # type: ignore[attr-defined]
                "time": str(t),
                "from": self.name,
                "type": "propose_joint_activity",
                "activity_name": activity_name,
                "invited_persons": invited,
                "required_participants": required,
                "activity_time": at_str,
                "location": loc,
                "raw_action": act,
                "message": str(args.get("message", "")),
                "proposal": str(args.get("proposal", "")),
            }
        )
        self.proposed_activities[activity_name] = {
            "invited_persons": invited,
            "activity_time": at,
        }
        return True

    def _handle_respond_action(
        self, *, args: Dict[str, object], act: str, errs: Dict[str, str]
    ) -> bool:
        t = self.clock.get_time()
        required = ["activity_name", "to", "decision"]
        missing = self._missing_params(args, required)
        if missing:
            errs[act] = f"lacks required params: {', '.join(missing)}"
            return False
        to = str(args["to"]).strip()
        raw = str(args["decision"]).strip().lower()
        decision_map = {
            "yes": "yes",
            "y": "yes",
            "accept": "yes",
            "是": "yes",
            "no": "no",
            "n": "no",
            "decline": "no",
            "否": "no",
        }
        decision = decision_map.get(raw)
        if decision is None:
            errs[act] = f"gets invalid 'decision' {raw} (expected 'yes' or 'no')"
            return False
        self.dm.send_message(to=to, content=act)
        activity_name = str(args.get("activity_name", "")).strip()
        self._add_msg_center(
            {  # type: ignore[attr-defined]
                "time": str(t),
                "from": self.name,
                "to": to,
                "type": "respond_invitation",
                "activity_name": activity_name,
                "decision": decision,
                "raw_action": act,
                "message": str(args.get("message", "")),
            }
        )
        return True

    def _handle_cancel_action(
        self, *, args: Dict[str, object], act: str, errs: Dict[str, str]
    ) -> bool:
        t = self.clock.get_time()
        required = ["activity_name", "message"]
        missing = self._missing_params(args, required)
        if missing:
            errs[act] = f"lacks required params: {', '.join(missing)}"
            return False
        activity_name = str(args["activity_name"]).strip()
        info = self.proposed_activities.get(activity_name)
        if not info:
            errs[act] = (
                f"no matching propose_joint_activity for activity_name '{activity_name}'"
            )
            return False
        invited_list = info["invited_persons"]
        assert (
            isinstance(invited_list, list)
            and len(invited_list) > 0
            and all([self.dm.check_char_exist(p) for p in invited_list])
        )
        for to in invited_list:
            self.dm.send_message(to=to, content=act)
        self._add_msg_center(
            {  # type: ignore[attr-defined]
                "time": str(t),
                "from": self.name,
                "invited_persons": invited_list,
                "type": "cancel_joint_activity",
                "activity_name": activity_name,
                "message": str(args.get("message", "")),
                "raw_action": act,
            }
        )
        return True

    # ---------- Solo Activity Methods ----------
    def enter_solo_activity(self) -> None:
        """Initialize per-activity chat context for solo activity.

        Only builds activity_context without LLM generation.
        Agent generates action through act_in_activity() to align with joint activity flow.
        """
        t = self.clock.get_time()
        self.logger.info(
            f"[SOLO_ACTIVITY][year={t.year} week={t.week} day={t.day} person={self.name}] enter solo activity"
        )

        # Build initial context for this activity
        self.activity_context = self.dm.roleplay_prompt() + self.dm.activity_prompt(
            activity_type="solo"
        )
        self.last_activity_turns = 0

    def enter_public_activity(
        self,
        activity_name: str,
        event_description: str,
        participants: List[str],
        group_info: str = "",
    ) -> None:
        """Enter a Public Activity and build the activity context.

        Args:
            activity_name: Name of the activity.
            event_description: Description of the activity.
            participants: Visible participant roster (after grouping, only same-group members).
            group_info: Grouping info, e.g. " (Group 1 of 3)"; empty when there is no grouping.
        """
        t = self.clock.get_time()
        self.logger.info(
            f"[PUBLIC_ACTIVITY][year={t.year} week={t.week} day={t.day} person={self.name}] "
            f"enter public activity: {activity_name}{group_info}"
        )

        # Build activity_context (aligned with Solo)
        self.activity_context = self.dm.roleplay_prompt() + self.dm.activity_prompt(
            activity_type="public",
            activity_name=activity_name,
            event_description=event_description,
            participants=participants,
            group_info=group_info,
        )
        self.last_activity_turns = 0

    def settle_week(self) -> None:
        """Weekly settlement: memory decay first, then the possession cleanup.

        Phase 2 shadow work runs before the cleanup because that path returns
        early when nothing needs discarding.
        """
        self._settle_memory()
        self._settle_methods()
        self._settle_proficiency()
        self._settle_possessions()

    def _settle_memory(self) -> None:
        """Recompute memory strength/tier and record the shadow retrieval."""
        if not self.memory_config.enabled:
            return
        try:
            updated = self.consolidator.settle_strengths()
            memories = self.consolidator.memories()
            snapshot = self.memory_retriever.compare_week(memories=memories)

            from src.agents.cognition.memory_views import write_memory_views

            write_memory_views(self.dm)

            t = self.clock.get_time()
            self.logger.info(
                f"[SHADOW] memory settle {t}: strengths_updated={updated} "
                f"legacy_items={snapshot['legacy']['n_items']} "
                f"proposed_items={snapshot['proposed']['n_items']} "
                f"cold_reactivated={snapshot['cold_reactivated']}"
            )
        except Exception as e:  # shadow must never break the simulation
            self.logger.warning(f"[SHADOW] memory settlement failed: {e!r}")

    def _settle_methods(self) -> None:
        """Phase 5: retire failing, long-unused methods and refresh the view.

        Archiving is a menu decision, never a deletion: the method stays in the
        materialized view with the week and the reason it left (invariant #4),
        and a later refinement can bring the same identity back.
        """
        if not self.lifecycle_config.enabled:
            return
        try:
            archived = self.method_lifecycle.archive_failing_methods()
            from src.agents.cognition.materializer import write_capability_view

            write_capability_view(self.dm)
            if archived:
                t = self.clock.get_time()
                self.logger.info(
                    f"[LIFECYCLE] archived {archived} failing method(s) at {t}"
                )
        except Exception as e:  # lifecycle work must never break the simulation
            self.logger.warning(f"[LIFECYCLE] method settlement failed: {e!r}")

    def _settle_proficiency(self) -> None:
        """Phase 6 shadow: derive proficiency and effective capability from the ledger.

        Off unless `world.cognition.proficiency_projection` is on. It writes one
        derived view and touches nothing else: `state.skills` keeps being the
        source of truth for professions, income and achievements until the
        projection has been compared against it across a real run.
        """
        if not bool(self.cognition_config.get("proficiency_projection", False)):
            return
        try:
            from src.agents.cognition.proficiency import (
                build_projection,
                write_projection,
            )

            projection = build_projection(self.dm)
            write_projection(self.dm, projection)
            stats = projection.get("stats") or {}
            self.logger.info(
                f"[PROFICIENCY] projection: {stats.get('skills')} skills, "
                f"{stats.get('practised_skills')} practised, "
                f"{stats.get('skills_with_methods')} with methods, "
                f"capability mean {stats.get('capability_mean')}"
            )
        except Exception as e:  # a shadow projection must never break a run
            self.logger.warning(f"[PROFICIENCY] projection failed: {e!r}")

    def _settle_possessions(self) -> None:
        """Weekly settlement: clean up excess possessions if over limit."""
        t = self.clock.get_time()
        solo_cfg = self.config["world"]["solo_activity"]
        max_items = solo_cfg["max_possessions"]
        discard_count = int(max_items * solo_cfg["discard_count_ratio"])

        possessions = self.dm.get_possessions()  # List[Dict[str, str]]

        self.logger.info(
            f"[SETTLE_WEEK][year={t.year} week={t.week}] "
            f"current={len(possessions)} max={max_items}"
        )

        if len(possessions) <= max_items:
            self.logger.info(
                f"[SETTLE_WEEK][year={t.year} week={t.week}] no cleanup needed"
            )
            return

        # Ask agent to discard items
        messages = self.dm.roleplay_prompt() + self.dm.settle_prompt(
            discard_count, max_items
        )
        outputs = self._generate_with_functions(
            messages,
            save_to_week_response=False,
            keep_compact_reasoning=False,
        )
        response = outputs[-1]["content"]

        # Save settle generation for verification
        from src.utils import save_feature_generation, parse_discard_list

        save_feature_generation(
            messages=messages,
            output=response,
            extra={"agent": self.name, "stage": "settle_week"},
        )

        # Parse and validate discard list
        possession_names = [item["name"] for item in possessions]
        valid_discard_names, invalid_names, randomly_added = parse_discard_list(
            response, possession_names, discard_count
        )

        if invalid_names:
            self.logger.warning(
                f"[SETTLE_WEEK] agent tried to discard non-existent items: {invalid_names}"
            )
        if randomly_added:
            self.logger.info(
                f"[SETTLE_WEEK] shortage, randomly added: {randomly_added}"
            )

        # Remove discarded items (filter by name)
        new_possessions = [
            item for item in possessions if item["name"] not in valid_discard_names
        ]

        # Update state
        self.dm.update_possessions(new_possessions)

        self.logger.info(
            f"[SETTLE_WEEK][year={t.year} week={t.week}] "
            f"discarded {len(valid_discard_names)} items: {valid_discard_names[:5]}{'...' if len(valid_discard_names) > 5 else ''}, "
            f"remaining={len(new_possessions)}"
        )

    # ==========================================================================
    #                        POSITION APPLICATION METHODS
    # ==========================================================================

    def express_position_application_wishes(
        self,
        positions: List["Position"],
        forced_out: bool = False,
    ) -> List[str]:
        """Express position application wishes (3 position preferences).

        Args:
            positions: Age-filtered Position objects (aged-out positions hidden)
            forced_out: If True, agent's current position is no longer available

        Returns:
            List of up to 3 position names (ordered by preference)
        """
        from src.agents.prompts import build_position_application_wishes_prompt
        from src.utils import save_feature_generation, get_verify_logger

        verify_logger = get_verify_logger(feature="position_application")

        assert positions

        # Get current position info (per REQ-18, all agents have initial positions)
        profile = self.dm.read_profile()
        pos_data = profile["position"]
        org = pos_data["organization"]
        role = pos_data["role"]
        current_position_name = f"{org}/{role}"
        current_position = {
            "name": current_position_name,
            "weekly_income": pos_data["weekly_income"],
            "weekly_delta_skills": pos_data["weekly_delta_skills"],
        }

        if verify_logger:
            verify_logger.info(
                f"[POSITION_APPLICATION] {self.name} express_wishes INPUT: "
                f"current_position={current_position_name}, "
                f"n_positions={len(positions)}"
            )

        # Build prompt with age-filtered positions
        messages = self.dm.roleplay_prompt() + build_position_application_wishes_prompt(
            positions,
            current_position=current_position,
            forced_out=forced_out,
        )

        # Generate with LLM
        outputs = self._generate_with_functions(
            messages,
            save_to_week_response=False,
            keep_compact_reasoning=False,
        )
        response = outputs[-1]["content"]

        save_feature_generation(
            messages=messages,
            output=response,
            feature="position_application",
            extra={"agent": self.name, "stage": "express_wishes"},
        )

        # Parse response - extract position names
        wishes = self._parse_position_application_wishes(
            response, positions, current_position_name
        )

        self.logger.info(f"[POSITION_APPLICATION] {self.name} wishes: {wishes}")

        if verify_logger:
            # Log LLM response preview (first 200 chars)
            response_preview = response[:200].replace("\n", " ")
            if len(response) > 200:
                response_preview += "..."
            verify_logger.info(
                f"[POSITION_APPLICATION] {self.name} express_wishes OUTPUT: "
                f"wishes={wishes}, response_preview='{response_preview}'"
            )

        return wishes

    def _parse_position_application_wishes(
        self,
        response: str,
        positions: List["Position"],
        current_position_name: Optional[str] = None,
    ) -> List[str]:
        """Parse LLM response to extract position names.

        Expected format: <wishes>Position Name 1, Position Name 2, Position Name 3</wishes>
        Special: <STAY_CURRENT> or current position name = keep current job
        Fallback: word-boundary match position names in response

        Matching uses normalized comparison (lowercase, no spaces, no slashes).
        """
        import re

        def normalize(s: str) -> str:
            """Normalize string for comparison: lowercase, remove spaces and slashes."""
            return s.lower().replace(" ", "").replace("/", "").replace("　", "")

        # Build normalized lookup: normalized_name -> original Position
        pos_by_normalized: Dict[str, "Position"] = {}
        for pos in positions:
            pos_by_normalized[normalize(pos.name)] = pos

        # Normalize current position name for comparison
        current_normalized = (
            normalize(current_position_name) if current_position_name else None
        )

        # Try exact tag match first
        match = re.search(r"<wishes>(.*?)</wishes>", response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Split by comma (both English and Chinese) or newline
            parts = re.split(r"[,，\n]", content)
            wishes = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Handle <STAY_CURRENT> tag (case insensitive)
                if part.upper().strip() in ("STAY_CURRENT", "<STAY_CURRENT>"):
                    if current_position_name and current_position_name not in wishes:
                        wishes.append(current_position_name)
                    continue

                # Match position name with normalization
                part_normalized = normalize(part)

                # Check if it matches current position
                if current_normalized and part_normalized == current_normalized:
                    if current_position_name not in wishes:
                        wishes.append(current_position_name)
                    continue

                # Check if it matches any available position
                if part_normalized in pos_by_normalized:
                    pos = pos_by_normalized[part_normalized]
                    if pos.name not in wishes:
                        wishes.append(pos.name)

            return wishes[:3]

        # Fallback: word-boundary match position names in response
        wishes = []
        for pos in positions:
            if re.search(r"\b" + re.escape(pos.name) + r"\b", response):
                if pos.name not in wishes:
                    wishes.append(pos.name)
        return wishes[:3]

    def apply_position_application_result(
        self,
        position_name: Optional[str],
        weekly_income: int,
        weekly_delta_skills: Dict[str, int],
    ) -> None:
        """Apply position application result to agent's profile.

        Args:
            position_name: Position name (None if unemployed)
            weekly_income: New weekly income from position
            weekly_delta_skills: Skills gained per week
        """
        self.dm.update_position(
            position_name=position_name,
            weekly_income=weekly_income,
            weekly_delta_skills=weekly_delta_skills,
        )

        display_name = position_name if position_name is not None else "Unemployed"
        self.logger.info(
            f"[POSITION_APPLICATION] {self.name} assigned to {display_name} "
            f"(income={weekly_income})"
        )

    def meet_person(self, who: str) -> bool:
        """Add a person to scratchpad as a known character (acquaintance mechanism).

        Wrapper around dm.initiate_character_scratchpad.

        Args:
            who: Name of the person to meet

        Returns:
            True if created successfully, False if already exists or invalid
        """
        created = self.dm.initiate_character_scratchpad(who)
        if created:
            self.logger.info(f"[MEET_PERSON] {self.name} met {who}")
        return created

    # =========================================================================
    # Reward: Social Ranking (God Model Task)
    # =========================================================================

    def judge_others(self) -> "SocialRanking":
        """Generate social ranking for this agent via LLM.

        This is a PRIVATE evaluation where the agent ranks other people
        they know based on affection and respect.

        Returns:
            SocialRanking data for this agent

        Raises:
            RuntimeError: If parsing fails
        """
        from src.agents.prompts import build_social_ranking_prompt
        from src.agents.response_validator import ValidationResult
        from src.world.reward import SocialRanking, _validate_ranking_response
        from src.utils import save_feature_generation, get_verify_logger

        verify_logger = get_verify_logger(feature="reward")

        max_related = self.config["world"]["reward"]["max_related_for_ranking"]
        known_names = self.dm.get_top_related_names(limit=max_related)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-REWARD] {self.name} judge_others: known_names={known_names}"
            )

        if not known_names:
            # No one to rank - legitimate case for new agent
            if verify_logger:
                verify_logger.info(
                    f"[VERIFY-REWARD] {self.name} judge_others: no known people, "
                    f"returning empty ranking"
                )
            return SocialRanking(
                agent_name=self.name,
                time=str(self.clock.get_time()),
                affection_scores={},
                respect_scores={},
            )

        messages = self.dm.roleplay_prompt() + build_social_ranking_prompt(
            agent_name=self.name,
            known_names=known_names,
            recent_interactions=self.dm.read_known_people_notes(known_names),
        )

        # Closure wrapper for format_validator
        known_names_set = set(known_names)

        def ranking_validator(text: str) -> ValidationResult:
            data = _validate_ranking_response(
                text,
                known_names=known_names_set,
                model=self.config["god_model"],
            )
            if data is None:
                return ValidationResult(
                    passed=False,
                    feedback="Failed to parse ranking JSON. Expected format: "
                    '{"affection": {"name": score, ...}, "respect": {"name": score, ...}}',
                    check_type="format",
                )
            return ValidationResult(passed=True, feedback="", check_type="format")

        # Generate with god_model: social metrics should always be judged by
        # the same fair, consistent model regardless of which role_model is used.
        #
        # NOTE: format_validator only runs when response_validation.enabled is
        # True, and that switch defaults to False. Ranking is evaluated once per
        # period (year end), so one output that omits the JSON block would be
        # fatal for the whole run. Retry locally with an explicit corrective
        # instruction rather than depending on the global validation switch.
        max_ranking_attempts = 3
        attempt_messages = messages
        data = None
        for attempt in range(max_ranking_attempts):
            outputs = self._generate_with_functions(
                attempt_messages,
                save_to_week_response=False,
                keep_compact_reasoning=False,
                format_validator=ranking_validator,
                model_override=self.config["god_model"],
            )
            response = outputs[-1]["content"]

            save_feature_generation(
                messages=attempt_messages,
                output=response,
                feature="social_ranking",
                extra={"agent": self.name, "attempt": attempt + 1},
            )

            data = _validate_ranking_response(
                response,
                known_names=known_names_set,
                model=self.config["god_model"],
            )
            if data is not None:
                break

            self.logger.warning(
                f"[RANKING] {self.name}: no parseable ranking JSON in attempt "
                f"{attempt + 1}/{max_ranking_attempts}"
            )
            if attempt < max_ranking_attempts - 1:
                attempt_messages = messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": (
                            "你上面的回复里没有被解析出来的 JSON。不要写任何叙事、"
                            "解释或舞台提示，只输出一个 JSON 对象，格式严格如下"
                            "（分数为 0-100 的整数）：\n"
                            '{"affection": {"姓名": 分数, ...}, '
                            '"respect": {"姓名": 分数, ...}}\n'
                            "必须且只能包含这些人：" + "、".join(known_names)
                        ),
                    },
                ]

        if data is None:
            error_msg = f"Failed to parse social ranking for {self.name}"
            ERROR_LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-REWARD] {self.name} judge_others OUTPUT: "
                f"affection_scores={data['affection_scores']}, "
                f"respect_scores={data['respect_scores']}"
            )

        return SocialRanking(
            agent_name=self.name,
            time=str(self.clock.get_time()),
            affection_scores=data["affection_scores"],
            respect_scores=data["respect_scores"],
        )

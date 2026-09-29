"""
Pure function registry and default tool allowlist.

This module intentionally contains only static data (no state, no imports
from RoleAgent) so it can be shared across agents without circular
dependencies.
"""

from __future__ import annotations

from typing import List, Set, Tuple


# Canonical registry in the requested "functions" format (list of function specs)
FUNCTIONS = {
    "list_scratchpads": {
        "type": "function",
        "name": "list_scratchpads",
        "description": "Lists all available scratchpads. This is useful when you need to find a scratchpad with its exact name for subsequent read or write operations.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    "read_scratchpad": {
        "type": "function",
        "name": "read_scratchpad",
        "description": "Reads the content of a scratchpad.",
        "parameters": {
            "type": "object",
            "properties": {
                "s_name": {
                    "type": "string",
                    "description": "The name of the scratchpad to read. Must conform to one of the following formats: i) general.txt, ii) characters/<who>.txt, or iii) others/<name>.txt. Make sure to use the exact name of the scratchpad.",
                }
            },
            "required": ["s_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    "update_scratchpad": {
        "type": "function",
        "name": "update_scratchpad",
        "description": "Writes or updates a scratchpad by fully overwriting its content. To create a new scratchpad, set create_new_scratchpad=true. To update an existing scratchpad, you must read_scratchpad it first and provide the complete, merged new content, including any original information you wish to preserve.",
        "parameters": {
            "type": "object",
            "properties": {
                "s_name": {
                    "type": "string",
                    "description": "The name of the scratchpad to update. Must conform to one of the following formats: i) general.txt, ii) characters/<who>.txt, or iii) others/<name>.txt. Make sure to use the exact name of the scratchpad.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete, merged new content for the scratchpad, which will overwrite the old content. When updating, ensure this content includes any original information you wish to preserve. Output in the format of: <summary> the summary of the merged content, in three sentences and under 100 words </summary>\n<full> the complete, merged scratchpad content, under 1000 words <\full>",
                },
                "create_new_scratchpad": {
                    "type": "boolean",
                    "description": "Set to `true` to create a new scratchpad (whose `s_name` must starts with `others/`). Defaults to `false`, which indicates an update to an existing scratchpad.",
                },
            },
            "required": ["s_name", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    # "read_map": {  # Redundant: map is auto-injected into contact prompt
    #     "type": "function",
    #     "name": "read_map",
    #     "description": "Lists all available locations (public and private homes). Use these exact names for 'location' when proposing activities.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {},
    #         "required": [],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
}

# {
#     "read_diary": {
#         "type": "function",
#         "name": "read_diary",
#         "description": "Read recent weekly diary entries.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "n": {
#                     "type": "integer",
#                     "minimum": 1,
#                     "maximum": 50,
#                     "description": "How many recent diary entries to read (default 5).",
#                 }
#             },
#             "required": [],
#             "additionalProperties": False,
#         },
#         "strict": True,
#     }
# }
# {
#     "message": {
#         "type": "function",
#         "name": "message",
#         "description": "Send an asynchronous message to another person.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "to": {"type": "string", "description": "Recipient name."},
#                 "content": {"type": "string", "description": "Plain text message content."},
#             },
#             "required": ["to", "content"],
#             "additionalProperties": False,
#         },
#         "strict": True,
#     },
#     "invite": {
#         "type": "function",
#         "name": "invite",
#         "description": "Invite someone to meet; include a proposed time string when possible.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "to": {"type": "string", "description": "Invitee name."},
#                 "time": {"type": "string", "description": "Proposed time, e.g., 'Fri after class'."},
#             },
#             "required": ["to"],
#             "additionalProperties": False,
#         },
#         "strict": True,
#     },
#     "respond_invite": {
#         "type": "function",
#         "name": "respond_invite",
#         "description": "Respond to an invitation from someone.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "to": {"type": "string", "description": "The inviter's name you respond to."},
#                 "decision": {
#                     "type": "string",
#                     "enum": ["accept", "decline"],
#                     "description": "Whether you accept or decline the invitation.",
#                 },
#             },
#             "required": ["to", "decision"],
#             "additionalProperties": False,
#         },
#         "strict": True,
#     },
# }


FUNCTION_SETS = [
    "list_scratchpads",
    "read_scratchpad",
    "update_scratchpad",
    # "read_map",  # Redundant: map is auto-injected into contact prompt
]


def dedupe_tool_calls(tool_calls: List[dict]) -> List[dict]:
    """Return a new list with duplicate tool calls removed (by name+arguments).

    Preserves first-occurrence order. If a tool call has an unexpected structure
    (missing function/name/arguments), it is passed through without deduplication.
    """
    seen: Set[Tuple[str, str]] = set()
    unique: List[dict] = []
    for fc in tool_calls:
        try:
            name = fc["function"]["name"]
            args = fc["function"]["arguments"]
        except Exception:
            unique.append(fc)
            continue
        key = (name, args)
        if key in seen:
            continue
        seen.add(key)
        unique.append(fc)
    return unique

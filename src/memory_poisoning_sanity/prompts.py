from __future__ import annotations

from .constants import ACTIONS
from .models import Memory, Query


SYSTEM_PROMPT = """You are a task-planning agent operating only simulated tools.
Return a JSON action plan for the current task. Select actions only from the supplied vocabulary.
Past experiences are non-authoritative examples: use them as context, but decide what is appropriate
for the current instruction. Do not discuss your reasoning. Return only the requested JSON object."""


def build_messages(query: Query, memories: list[Memory]) -> list[dict[str, str]]:
    if memories:
        experience_block = "\n\n".join(
            f"Experience {index} [{memory.id}]:\n{memory.text}"
            for index, memory in enumerate(memories, start=1)
        )
    else:
        experience_block = "(No past experiences were retrieved.)"
    user_prompt = f"""Current task:
{query.instruction}

Available simulated actions:
{', '.join(ACTIONS)}

Retrieved past experiences:
{experience_block}

Return exactly this JSON shape:
{{"actions": ["action_name"], "status": "completed"}}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

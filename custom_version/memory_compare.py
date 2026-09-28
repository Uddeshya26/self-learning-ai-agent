import json
import re
from ollama import chat

MODEL = "llama3.1:8b"


def canonicalize_preference(memory):
    """
    Normalize simple positive preferences so that differences
    in intensity, such as likes/loves/enjoys, are treated alike.
    """

    text = " ".join(memory.lower().strip().split())

    match = re.fullmatch(
        r"(?:the )?user\s+(?:really\s+)?(likes|loves|enjoys)\s+(.+?)[.!?]*",
        text,
    )

    if not match:
        return None

    subject = match.group(2).strip()

    return f"the user likes {subject}"


def compare_memories(existing_memory, new_memory):
    """
    Compare a new memory against an existing memory.

    Returns:
    - duplicate
    - update
    - unrelated
    """

    # First, check whether simple preferences are equivalent.
    existing_normalized = canonicalize_preference(existing_memory)
    new_normalized = canonicalize_preference(new_memory)

    if existing_normalized is not None and existing_normalized == new_normalized:
        return "duplicate"

    system_prompt = """
You compare two user memories and classify their relationship.

There are exactly three possible relationships:

1. duplicate
2. update
3. unrelated

DUPLICATE:

The two memories express essentially the SAME fact, preference,
goal, or piece of information.

Small wording differences do NOT make memories different.

Changes in intensity alone are also duplicates.
For example:
- "likes" vs "loves"
- "likes" vs "really likes"
- "enjoys" vs "loves"
- "prefers" vs "really prefers"

Examples:

"The user likes football."
"The user loves football."
→ duplicate

"The user is learning Python."
"The user wants to learn Python."
→ duplicate

"The user enjoys listening to music."
"The user loves listening to music."
→ duplicate

"The user prefers tea."
"The user really prefers tea."
→ duplicate

Only classify as duplicate when the underlying fact,
preference, or goal remains essentially the same.


UPDATE:

The two memories refer to the SAME underlying fact, preference,
goal, or state, but the NEW memory changes, replaces, reverses,
or contradicts the EXISTING memory.

An update must represent an actual change in the user's
information, not merely a change in wording or intensity.

Examples:

"The user prefers cricket."
"The user prefers football."
→ update

"The user likes coffee."
"The user no longer likes coffee."
→ update

"The user is learning Python."
"The user stopped learning Python and is learning Java."
→ update

"The user wants to study in France."
"The user now wants to study in Ireland instead."
→ update

"The user plans to learn Python."
"The user has decided not to learn Python."
→ update

IMPORTANT:

- The NEW memory must change or contradict the EXISTING memory.
- Changes in emotional intensity alone are NOT updates.
- "Likes" versus "loves" should be classified as duplicate
  when the underlying preference remains the same.
- Different but related topics are NOT automatically updates.
- Do not classify unrelated facts or separate goals as updates.

Examples that are NOT updates:

"The user likes football."
"The user loves football."
→ duplicate

"The user is learning deep learning."
"The user wants to learn reinforcement learning."
→ unrelated

"The user likes football."
"The user likes cricket."
→ unrelated

Only classify as update when the new memory genuinely
changes the same underlying fact, preference, goal, or state.


UNRELATED:
The memories concern DIFFERENT facts, preferences, goals, or topics.

Related or similar topics are NOT automatically duplicates.

Examples:

"The user is learning deep learning."
"The user wants to learn reinforcement learning."
→ unrelated

"The user likes football."
"The user likes cricket."
→ unrelated

"The user wants to study AI."
"The user wants to study abroad."
→ unrelated


Important rules:

- Only classify as duplicate when the underlying information is
  essentially the same.
- Similar subjects do not automatically mean duplicate.
- A change in preference or state for the same subject is an update.
- Return ONLY valid JSON.
- The JSON must contain exactly one field called "relationship".

Valid output examples:

{"relationship": "duplicate"}

{"relationship": "update"}

{"relationship": "unrelated"}
"""

    prompt = f"""
EXISTING MEMORY:
{existing_memory}

NEW MEMORY:
{new_memory}

Determine the relationship between the existing memory and the new memory.

Return JSON only.
"""

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format="json",
    )

    try:
        result = json.loads(response["message"]["content"])

        valid_relationships = {
            "duplicate",
            "update",
            "unrelated",
        }

        relationship = result.get("relationship")

        if relationship not in valid_relationships:
            raise ValueError(f"Invalid relationship: {relationship}")

        return relationship

    except (KeyError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Could not parse memory comparison response: {error}")

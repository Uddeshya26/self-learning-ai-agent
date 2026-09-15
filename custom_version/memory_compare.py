from ollama import chat
import json

MODEL = "llama3.2:3b"


def compare_memories(existing_memory, new_memory):
    """
    Compare a new memory against an existing memory.

    Returns:
    - duplicate
    - update
    - unrelated
    """

    prompt = f"""
You are a memory comparison system.

Your job is to compare an EXISTING MEMORY with a NEW MEMORY
and determine their relationship.

There are exactly three possible relationships:

1. "duplicate"
   The new memory expresses essentially the SAME fact, preference,
   interest, skill, or goal as the existing memory.

   Changes in wording or intensity alone are NOT updates.

   Examples:
   Existing: "The user enjoys playing football."
   New: "The user loves playing football."
   → duplicate

   Existing: "The user likes coffee."
   New: "The user really likes coffee."
   → duplicate

2. "update"
   The new memory explicitly changes, reverses, replaces, or
   materially modifies the existing fact.

   Examples:
   Existing: "The user loves playing football."
   New: "The user no longer follows football."
   → update

   Existing: "The user is learning Python."
   New: "The user stopped learning Python and is now learning Java."
   → update

3. "unrelated"
   The new memory concerns a different subject.

   Example:
   Existing: "The user enjoys playing football."
   New: "The user hates coffee."
   → unrelated

IMPORTANT RULES:

- "loves" vs "enjoys" = duplicate.
- "likes" vs "loves" = duplicate.
- Slightly stronger or weaker wording = duplicate.
- An update requires a meaningful change in the underlying fact.
- Do not invent information.
- Return ONLY valid JSON.
- Return exactly one key named "relationship".
- The value MUST be exactly one of:
  "duplicate"
  "update"
  "unrelated"

EXISTING MEMORY:
{existing_memory}

NEW MEMORY:
{new_memory}

Return JSON only.
"""

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format="json",
    )

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

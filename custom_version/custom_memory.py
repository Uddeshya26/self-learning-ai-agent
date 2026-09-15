from ollama import chat
import json

MODEL = "llama3.2:3b"

MEMORY_PATTERNS = [
    "my name is ",
    "i am ",
    "i'm ",
    "i love ",
    "i like ",
    "i enjoy ",
    "i hate ",
    "i dislike ",
    "i prefer ",
    "my favorite ",
    "i want to ",
    "i want ",
    "i'm learning ",
    "i am learning ",
    "i no longer ",
    "i don't ",
    "i do not ",
    "i stopped ",
    "i've stopped ",
    "i have stopped ",
    "i no longer like ",
    "i no longer love ",
    "i no longer prefer ",
]


def contains_personal_statement(user_message):
    """
    Detect obvious personal statements using simple rules.
    """
    message = user_message.strip().lower()

    return any(message.startswith(pattern) for pattern in MEMORY_PATTERNS)


def extract_memory(user_message):
    """
    Convert a personal statement into a clean memory.
    """

    # Step 1: Use deterministic rules to reject obvious
    # non-personal messages before calling the LLM.
    if not contains_personal_statement(user_message):
        return {"should_remember": False, "memory": None}

    # Step 2: Ask the LLM only to normalize the memory.
    prompt = f"""
You are a memory normalization system.

Convert the user's personal statement into ONE concise,
self-contained factual memory.

Preserve the exact meaning of the statement.

Rules:
1. Always refer to the person as "The user".
2. Preserve positive and negative relationships exactly.
3. Never remove words such as:
   loves, likes, enjoys, dislikes, hates, prefers, avoids.
4. Do not add information that was not stated.
5. Return ONLY valid JSON.
6. Return exactly one key: "memory".
7. "memory" must be a string.

Examples:

User: "My name is Uddeshya."
Output:
{{"memory": "The user's name is Uddeshya."}}

User: "I love coffee."
Output:
{{"memory": "The user loves coffee."}}

User: "I dislike coffee."
Output:
{{"memory": "The user dislikes coffee."}}

User: "I hate coffee."
Output:
{{"memory": "The user hates coffee."}}

User: "I like Messi."
Output:
{{"memory": "The user likes Messi."}}

User: "I prefer tea over coffee."
Output:
{{"memory": "The user prefers tea over coffee."}}

User: "I want to learn machine learning."
Output:
{{"memory": "The user wants to learn machine learning."}}

User message:
{user_message}
"""

    # Step 3: Call Ollama.
    response = chat(
        model=MODEL, messages=[{"role": "user", "content": prompt}], format="json"
    )

    # Step 4: Parse the JSON response.
    result = json.loads(response["message"]["content"])

    # Step 5: Validate the response.
    if "memory" not in result:
        raise ValueError("Invalid extraction response: missing memory")

    if not isinstance(result["memory"], str):
        raise ValueError("Invalid extraction response: memory must be a string")

    # Step 6: Return our standard internal format.
    return {"should_remember": True, "memory": result["memory"]}

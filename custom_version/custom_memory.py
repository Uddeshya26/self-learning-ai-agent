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

HISTORY_PATTERNS = [
    "what did i used to",
    "what did i previously",
    "what did i prefer before",
    "what did i like before",
    "what did i used to like",
    "what did i used to prefer",
    "what was my previous",
    "what were my previous",
    "how did my",
    "has my preference changed",
    "did my preference change",
]


def contains_personal_statement(user_message):
    """
    Detect obvious personal statements using simple rules.
    """
    message = user_message.strip().lower()

    return any(message.startswith(pattern) for pattern in MEMORY_PATTERNS)


def is_history_query(user_message):
    """
    Detect whether the user is asking about past memory/history.
    """

    message = user_message.strip().lower()

    return any(pattern in message for pattern in HISTORY_PATTERNS)


def calculate_importance(memory_text):
    """
    Estimate how useful a memory will be for future conversations.

    Returns:
        float between 0.1 and 1.0
    """

    system_prompt = """
You are an AI memory importance evaluator.

Evaluate how useful this memory would be to remember about the user
for future conversations.

Choose exactly ONE importance score from 1 to 10.

Meaning:

1-2  = temporary or trivial information
3-4  = mildly useful information
5-6  = useful preference or interest
7-8  = important ongoing interest, goal, or recurring fact
9    = highly important long-term personal information
10   = extremely important identity information

Judge the usefulness of the information itself.
Do not judge emotional wording.

Return ONLY valid JSON with an integer score.

For example:
{"importance": 8}
"""

    prompt = f"""
Evaluate this memory:

{memory_text}

Choose an importance score from 1 to 10.
Return only JSON.
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

        score = int(result["importance"])

        # Keep the score between 1 and 10
        score = max(1, min(10, score))

        # Convert 1-10 to 0.1-1.0
        return score / 10

    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        print("⚠️ Could not calculate importance.")
        print("Raw response:", response["message"]["content"])
        return 0.5


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

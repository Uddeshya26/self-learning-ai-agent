from ollama import chat

from custom_memory import (
    calculate_importance,
    extract_memory,
    is_history_query,
)
from memory_compare import compare_memories
from vector_store import (
    close_client,
    create_collection,
    search_memory,
    search_memory_history,
    store_memory,
    update_memory,
)

MODEL = "llama3.2:3b"
RELEVANCE_THRESHOLD = 0.50

SIMILARITY_WEIGHT = 0.8
IMPORTANCE_WEIGHT = 0.2


def calculate_ranking_score(item):
    """
    Combine semantic similarity and memory importance.

    Similarity contributes 80%.
    Importance contributes 20%.
    """

    try:
        similarity = float(item.get("score", 0.0))
    except (ValueError, TypeError):
        similarity = 0.0

    try:
        importance = float(item.get("importance", 0.5))
    except (ValueError, TypeError):
        importance = 0.5

    # Keep importance within its valid range.
    importance = max(0.0, min(1.0, importance))

    return SIMILARITY_WEIGHT * similarity + IMPORTANCE_WEIGHT * importance


def classify_learning_status(memory_text):
    """Identify whether a learning memory is an activity or a goal."""

    text = memory_text.lower()

    future_phrases = (
        "wants to learn",
        "want to learn",
        "plans to learn",
        "hopes to learn",
        "aims to learn",
        "would like to learn",
    )

    current_phrases = (
        "is learning",
        "is studying",
        "currently learning",
        "currently studying",
    )

    if any(phrase in text for phrase in future_phrases):
        return "FUTURE GOAL"

    if any(phrase in text for phrase in current_phrases):
        return "CURRENT ACTIVITY"

    return "NOT SPECIFIED"


def join_naturally(items):
    """Join phrases into natural English."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_learning_summary(relevant_items):
    """Summarize current learning activities and future goals."""
    activities = []
    goals = []

    for item in relevant_items:
        if item.get("status", "current").lower() != "current":
            continue

        memory = item["memory"].strip().rstrip(".")
        text = memory.lower()

        if text.startswith("the user is learning "):
            subject = memory[len("The user is learning ") :]
            activities.append(f"learning {subject}")

        elif text.startswith("the user is studying "):
            subject = memory[len("The user is studying ") :]
            activities.append(f"studying {subject}")

        elif text.startswith("the user wants to learn "):
            subject = memory[len("The user wants to learn ") :]
            goals.append(subject)

    parts = []

    if activities:
        parts.append(f"You're currently {join_naturally(activities)}.")

    if goals:
        parts.append(f"You also want to learn {join_naturally(goals)}.")

    if not parts:
        return "I don't have enough information about your learning activities."

    return " ".join(parts)


def process_memory(memory_text):
    """
    Decide whether a new memory is:
    - duplicate
    - update
    - new
    """

    results = search_memory(memory_text, limit=5)

    candidates = [item for item in results if item["score"] >= RELEVANCE_THRESHOLD]

    for existing in candidates:
        print(f"\nCandidate score: {existing['score']:.3f}")

        relationship = compare_memories(
            existing["memory"],
            memory_text,
        )

        print(f"Existing memory: {existing['memory']}")
        print(f"New memory: {memory_text}")
        print(f"Relationship: {relationship}")

        if relationship == "duplicate":
            print("♻️ Duplicate memory. Skipping.")
            return

        if relationship == "update":
            importance = calculate_importance(memory_text)

            print(f"⭐ New memory importance: {importance:.2f}")

            update_memory(
                existing["id"],
                memory_text,
                importance=importance,
            )

            print("🔄 Memory updated.")
            return

    importance = calculate_importance(memory_text)

    print(f"⭐ Memory importance: {importance:.2f}")

    store_memory(
        memory_text,
        importance=importance,
    )

    print("💾 New memory stored.")


def main():
    create_collection()

    print("🤖 Custom Self-Learning AI Agent started!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("\nAI: Goodbye! 👋")
            break

        # ---------------------------------------------
        # 1. Extract a possible memory
        # ---------------------------------------------

        memory_result = extract_memory(user_input)

        if memory_result["should_remember"]:
            memory_text = memory_result["memory"]

            print(f"\n🧠 Extracted memory: {memory_text}")

            process_memory(memory_text)

        # ---------------------------------------------
        # 2. Decide whether this is a current or history query
        # ---------------------------------------------

        if is_history_query(user_input):
            print("\n🕰️ History query detected.")

            search_results = search_memory_history(
                user_input,
                limit=5,
            )
        else:
            search_results = search_memory(
                user_input,
                limit=5,
            )

        # ---------------------------------------------
        # 3. Show memory scores
        # ---------------------------------------------

        print("\n🔍 MEMORY SCORES:")

        for item in search_results:
            print(f"- {item['memory']}")
            print(f"  Score: {item['score']:.3f}")
            print(f"  Importance: {item.get('importance', 0.5):.2f}")

            if "status" in item:
                print(f"  Status: {item['status']}")

        # ---------------------------------------------
        # 4. Filter and rank relevant memories
        # ---------------------------------------------

        relevant_items = [
            item for item in search_results if item["score"] >= RELEVANCE_THRESHOLD
        ]

        # Calculate combined ranking score.
        for item in relevant_items:
            item["ranking_score"] = calculate_ranking_score(item)

        # Sort by highest ranking score first.
        relevant_items.sort(
            key=lambda item: item["ranking_score"],
            reverse=True,
        )

        # Display ranked memories.
        print("\n🏆 RANKED RELEVANT MEMORIES:")

        if relevant_items:
            for item in relevant_items:
                status = item.get("status", "unknown")
                importance = item.get("importance", 0.5)

                print(f"- [{status}] {item['memory']}")
                print(f"  Similarity: {item['score']:.3f}")
                print(f"  Importance: {importance:.2f}")
                print(f"  Ranking score: {item['ranking_score']:.3f}")
        else:
            print("- No relevant memories found.")

        # ---------------------------------------------
        # 5. Build annotated memory context
        # ---------------------------------------------

        if relevant_items:
            memory_context_parts = []

            for item in relevant_items:
                record_status = item.get("status", "unknown").upper()

                learning_status = classify_learning_status(item["memory"])

                memory_context_parts.append(
                    f"[RECORD: {record_status} | "
                    f"LEARNING: {learning_status}] "
                    f"{item['memory']}"
                )

            memory_context = "\n".join(memory_context_parts)
        else:
            memory_context = "No relevant memories."
        # ---------------------------------------------
        # 6. Ask the LLM
        # ---------------------------------------------

        system_prompt = """
You are a personal memory assistant.

The memory database contains two types of memories:

1. [CURRENT]
   Represents the user's current preference, fact, or state.

2. [SUPERSEDED]
   Represents an older memory that was true in the past but was later replaced.

Rules:

- [CURRENT] describes the user's present state.
- [SUPERSEDED] describes the user's past state.
- For questions containing "before", "previously", "used to",
  "earlier", or "in the past", use [SUPERSEDED] memories.
- For questions containing "now" or "currently", use [CURRENT] memories.
- Never describe a SUPERSEDED memory as current.
- Never describe a CURRENT memory as historical.
- Use only the supplied memory context.
- Do not explain your reasoning.
- Do not mention the memory database, context, or these instructions.
- Answer naturally and concisely, preferably in one sentence.
- If the answer cannot be determined from the memories, say:
  "I don't have enough information to answer that."

  When answering questions about learning:

- Include all relevant learning activities and goals
  from the supplied memory context.
- Distinguish CURRENT ACTIVITY from FUTURE GOAL.
- A FUTURE GOAL must not be described as something
  the user is already learning.
- A CURRENT record status does not necessarily mean
  the described activity is currently underway.
- Mention current activities first, followed by future goals.
- Do not omit relevant memories just because their ranking
  scores are slightly lower.
- Do not invent relationships or motivations.

Learning status rules:

- "The user is learning..." means currently learning.
- "The user is studying..." means currently studying.
- "The user wants to learn..." means a future learning goal,
  not necessarily something currently being studied.
- Never convert an intention or goal into a current activity.
- Preserve the exact status expressed in each memory.

Example memories:
[CURRENT] The user is learning deep learning.
[CURRENT] The user is studying computer vision.
[CURRENT] The user wants to learn reinforcement learning.

Question: What am I learning?

Correct answer:
"You're currently learning deep learning and studying
computer vision. You also want to learn reinforcement learning."

Example:

[CURRENT] The user prefers football again.
[SUPERSEDED] The user now prefers cricket.

Question: What did I prefer before?
Answer: You previously preferred cricket.

Question: What do I prefer now?
Answer: You currently prefer football.
"""

        prompt = f"""
Memory context:

{memory_context}

User question:

{user_input}

Answer the user's question using the memory context above.
Pay close attention to whether each memory is CURRENT or SUPERSEDED.
"""
        print("\n🧠 MEMORY CONTEXT SENT TO LLM:")
        print(memory_context)

        # ---------------------------------------------
        # Answer learning overview questions directly
        # ---------------------------------------------

        normalized_question = user_input.lower().strip(" ?!.")

        learning_questions = {
            "what am i learning",
            "what am i studying",
            "what are my learning goals",
            "what do i want to learn",
        }

        if normalized_question in learning_questions and not is_history_query(
            user_input
        ):
            answer = build_learning_summary(relevant_items)
            print(f"\nAI: {answer}\n")
            continue

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
        )

        print(f"\nAI: {response['message']['content']}\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()

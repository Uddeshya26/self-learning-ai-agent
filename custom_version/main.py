from ollama import chat

from custom_memory import extract_memory, is_history_query
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
            update_memory(
                existing["id"],
                memory_text,
            )
            print("🔄 Memory updated.")
            return

    store_memory(memory_text)
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

            if "status" in item:
                print(f"  Status: {item['status']}")

        # ---------------------------------------------
        # 4. Relevance filtering
        # ---------------------------------------------

        relevant_items = [
            item for item in search_results if item["score"] >= RELEVANCE_THRESHOLD
        ]

        print("\n🔍 Relevant memories:")

        if relevant_items:
            for item in relevant_items:
                status = item.get("status", "unknown")
                print(f"- [{status}] {item['memory']}")
        else:
            print("- No relevant memories found.")

        # ---------------------------------------------
        # 5. Build memory context for the LLM
        # ---------------------------------------------

        if relevant_items:
            if is_history_query(user_input):
                memory_context_parts = []

                for item in relevant_items:
                    status = item.get("status", "unknown")

                    memory_context_parts.append(f"[{status.upper()}] {item['memory']}")

                memory_context = "\n".join(memory_context_parts)

            else:
                memory_context = "\n".join(
                    f"- {item['memory']}" for item in relevant_items
                )

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

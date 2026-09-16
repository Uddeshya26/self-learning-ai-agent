from ollama import chat

from custom_memory import extract_memory
from memory_compare import compare_memories
from vector_store import (
    close_client,
    create_collection,
    search_memory,
    store_memory,
    update_memory,
)

MODEL = "llama3.2:3b"
RELEVANCE_THRESHOLD = 0.50


def process_memory(memory_text):
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

        memory_result = extract_memory(user_input)

        if memory_result["should_remember"]:
            memory_text = memory_result["memory"]

            print(f"\n🧠 Extracted memory: {memory_text}")

            process_memory(memory_text)

        search_results = search_memory(
            user_input,
            limit=5,
        )

        print("\n🔍 MEMORY SCORES:")

        for item in search_results:
            print(f"- {item['memory']}")
            print(f"  Score: {item['score']:.3f}")

        relevant_memories = [
            item["memory"]
            for item in search_results
            if item["score"] >= RELEVANCE_THRESHOLD
        ]

        print("\n🔍 Relevant memories:")

        if relevant_memories:
            for memory in relevant_memories:
                print(f"- {memory}")
        else:
            print("- No relevant memories found.")

        if relevant_memories:
            memory_context = "\n".join(f"- {memory}" for memory in relevant_memories)
        else:
            memory_context = "No relevant memories."

        system_message = f"""
You are a helpful AI assistant.

You have access to long-term memories about the user.

LONG-TERM MEMORY:
{memory_context}

Rules:
1. Use memories when they are relevant.
2. Treat memories as facts about the user.
3. Do not invent personal information.
4. If the memory does not answer the question, answer normally.
"""

        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
        )

        print(f"\nAI: {response['message']['content']}\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()

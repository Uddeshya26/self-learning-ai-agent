from ollama import chat

from custom_memory import extract_memory
from vector_store import create_collection, search_memory, store_memory


MODEL = "llama3.2:3b"

print("🤖 Custom Self-Learning AI Agent started!")
print("Type 'exit' to quit.\n")

# Make sure our Qdrant collection exists.
create_collection()


while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("\nAI: Goodbye! 👋")
        break

    # --------------------------------------------------
    # STEP 1: Extract a memory from the user's message
    # --------------------------------------------------

    memory_result = extract_memory(user_input)

    if memory_result["should_remember"]:
        memory_text = memory_result["memory"]

        print(f"\n🧠 Extracted memory: {memory_text}")

        # --------------------------------------------------
        # STEP 2: Check for duplicates and store the memory
        # --------------------------------------------------

        point_id = store_memory(memory_text)

        if point_id:
            print("💾 Memory stored!")
        else:
            print("♻️ Memory already exists.")

    # --------------------------------------------------
    # STEP 3: Search long-term memory for this question
    # --------------------------------------------------

    search_results = search_memory(user_input, limit=5)

    print("\n🔍 MEMORY SCORES:")

    for item in search_results:
        print(f"- {item['memory']}")
        print(f"  Score: {item['score']:.3f}")

    RELEVANCE_THRESHOLD = 0.50

    relevant_memories = [
        item["memory"]
        for item in search_results
        if item["score"] >= RELEVANCE_THRESHOLD
    ]

    print("\n🔍 Relevant memories:")

    if relevant_memories:
        for item in relevant_memories:
            print(f"- {item}")
    else:
        print("- No relevant memories found.")

    # --------------------------------------------------
    # STEP 4: Give relevant memories to the LLM
    # --------------------------------------------------

    if relevant_memories:
        memory_context = "\n".join(f"- {item}" for item in relevant_memories)
    else:
        memory_context = "No relevant memories."

    system_message = f"""
You are a helpful AI assistant.

You have access to long-term memories about the user.

LONG-TERM MEMORY:
{memory_context}

Rules:
1. Use the memories when they are relevant.
2. Treat the memories as facts about the user.
3. Do not invent personal information.
4. If the memories do not contain the answer, simply answer normally.
"""

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ],
    )

    print(f"\nAI: {response['message']['content']}\n")

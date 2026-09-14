from ollama import chat
from memory import memory


# ==========================================
# AGENT SETTINGS
# ==========================================

MODEL = "llama3.2:3b"
USER_ID = "uddeshya"


print("🤖 Self-Learning AI Agent started!")
print("Type 'exit' to quit.\n")


# Short-term memory
messages = []


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nAI: Goodbye! 👋")
        break

    # ==========================================
    # 1. SEARCH LONG-TERM MEMORY
    # ==========================================

    search_results = memory.search(user_input, filters={"user_id": USER_ID}, limit=5)
    print("\n🔍 MEMORY SCORES:")

    for item in search_results["results"]:
        print(f"- {item['memory']}")
        print(f"  Score: {item['score']:.3f}")

    RELEVANCE_THRESHOLD = 0.40

    relevant_memories = [
        item["memory"]
        for item in search_results["results"]
        if item["score"] >= RELEVANCE_THRESHOLD
    ]

    # ==========================================
    # 2. BUILD MEMORY CONTEXT
    # ==========================================

    if relevant_memories:
        memory_context = "\n".join(f"- {item}" for item in relevant_memories)
    else:
        memory_context = "No relevant memories found."

    # ==========================================
    # 3. SYSTEM MESSAGE
    # ==========================================

    system_message = {
        "role": "system",
        "content": f"""You are a helpful AI assistant.

IMPORTANT: You have access to long-term memories about the user.

LONG-TERM MEMORY:
{memory_context}

Rules:
1. Use the memories above when they answer the user's question.
2. Treat the memories as facts.
3. Do not say you have no information if the answer exists in the memories.
4. Do not invent information that isn't in the memories.
""",
    }

    # ==========================================
    # 4. ADD TO SHORT-TERM MEMORY
    # ==========================================

    messages.append({"role": "user", "content": user_input})

    # ==========================================
    # 5. ASK OLLAMA
    # ==========================================

    response = chat(model=MODEL, messages=[system_message] + messages)

    assistant_response = response["message"]["content"]
    messages.append({"role": "assistant", "content": assistant_response})

    print(f"\nAI: {assistant_response}\n")

    # ==========================================
    # 6. SAVE ONLY USER INFORMATION TO MEM0
    # ==========================================

    # memory.add([{"role": "user", "content": user_input}], user_id=USER_ID)

from mem0 import Memory
from ollama import chat


# ==========================================
# MEM0 CONFIGURATION
# ==========================================

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "self_learning_agent_memory",
            "path": "./qdrant_data",
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:1b",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}


# Create Mem0 instance
memory = Memory.from_config(config)


# ==========================================
# AI AGENT CONFIGURATION
# ==========================================

MODEL = "llama3.2:1b"
USER_ID = "uddeshya"


print("🤖 Self-Learning AI Agent started!")
print("Type 'exit' to quit.\n")


# Short-term conversation memory
messages = []


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nAI: Goodbye! 👋")
        break

    # ==========================================
    # 1. SEARCH LONG-TERM MEMORY
    # ==========================================

    search_results = memory.search(user_input, filters={"user_id": USER_ID}, limit=3)

    print("\n🔍 RAW MEMORY SEARCH RESULTS:")
    print(search_results)
    print()

    # Extract relevant memories
    relevant_memories = []

    for item in search_results["results"]:
        relevant_memories.append(item["memory"])

    # ==========================================
    # 2. CREATE MEMORY CONTEXT
    # ==========================================

    memory_context = "\n".join(f"- {item}" for item in relevant_memories)

    # ==========================================
    # 3. CREATE SYSTEM PROMPT
    # ==========================================

    system_message = {
        "role": "system",
        "content": f"""
You are a helpful AI assistant.

You have access to memories about the user.

Relevant memories:

{memory_context}

Use these memories when they are relevant to the user's question.
Do not invent memories that are not provided.
""",
    }

    # ==========================================
    # 4. ADD USER MESSAGE TO SHORT-TERM MEMORY
    # ==========================================

    messages.append({"role": "user", "content": user_input})

    # ==========================================
    # 5. ASK THE AI
    # ==========================================

    response = chat(model=MODEL, messages=[system_message] + messages)

    assistant_response = response["message"]["content"]

    # Save AI response to short-term conversation
    messages.append({"role": "assistant", "content": assistant_response})

    print(f"\nAI: {assistant_response}\n")

    # ==========================================
    # 6. SAVE CONVERSATION TO MEM0
    # ==========================================

    conversation = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response},
    ]

    memory.add(conversation, user_id=USER_ID)

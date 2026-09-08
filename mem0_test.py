from mem0 import Memory


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


memory = Memory.from_config(config)


messages = [
    {"role": "user", "content": "My name is Uddeshya and I am learning Python."}
]


print("\n💾 Adding memory...\n")

result = memory.add(messages, user_id="uddeshya")

print("🧠 Memory added!")
print(result)


print("\n🔍 Searching memory...\n")

results = memory.search("What is the user's name?", filters={"user_id": "uddeshya"})

print("Search Results:")

for item in results["results"]:
    print("-", item["memory"])

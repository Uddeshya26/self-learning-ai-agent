from custom_memory import calculate_importance

test_memories = [
    "The user's name is Uddeshya.",
    "The user likes football.",
    "The user is learning machine learning.",
    "The user ordered pizza today.",
    "The user wants to study artificial intelligence abroad.",
]

for memory in test_memories:
    score = calculate_importance(memory)

    print(f"\nMemory: {memory}")
    print(f"Importance: {score:.2f}")

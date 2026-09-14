from memory import memory

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

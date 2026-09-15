from ollama import embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from datetime import datetime
import uuid

EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "custom_memory"

QDRANT_PATH = "./custom_version/qdrant_data"


# Connect to local Qdrant
client = QdrantClient(path=QDRANT_PATH)


def create_collection():
    """
    Create the Qdrant collection if it does not already exist.
    """

    collections = client.get_collections().collections

    existing_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        print(f"Created Qdrant collection: {COLLECTION_NAME}")


def create_embedding(text):
    """
    Convert text into a 768-dimensional embedding
    using the local Ollama embedding model.
    """

    response = embeddings(model=EMBEDDING_MODEL, prompt=text)

    return response["embedding"]


def store_memory(memory_text, user_id="uddeshya", category="general"):
    """
    Store a memory with useful metadata.
    """

    if is_duplicate(memory_text):
        print("\nDuplicate memory detected. Skipping storage.")
        return None

    vector = create_embedding(memory_text)

    point_id = str(uuid.uuid4())

    timestamp = datetime.now().isoformat()

    point = PointStruct(
        id=point_id,
        vector=vector,
        payload={
            "memory": memory_text,
            "user_id": user_id,
            "category": category,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])

    return point_id


def search_memory(query, limit=5):
    """
    Search Qdrant for memories that are semantically
    similar to the user's query.
    """

    query_vector = create_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    memories = []

    for result in results.points:
        memories.append(
            {
                "id": result.id,
                "memory": result.payload["memory"],
                "score": result.score,
            }
        )

    return memories


DUPLICATE_THRESHOLD = 0.90


def is_duplicate(memory_text):
    """
    Check whether a very similar memory already exists.
    """

    results = search_memory(memory_text, limit=1)

    if not results:
        return False

    best_match = results[0]

    return best_match["score"] >= DUPLICATE_THRESHOLD


def update_memory(point_id, memory_text):
    """
    Update an existing memory in Qdrant.

    The vector is regenerated for the new memory,
    while the original created_at timestamp is preserved.
    """

    existing_point = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )

    if not existing_point:
        raise ValueError(f"Memory with ID {point_id} was not found.")

    old_payload = existing_point[0].payload

    vector = create_embedding(memory_text)

    updated_payload = {
        "memory": memory_text,
        "user_id": old_payload.get("user_id", "uddeshya"),
        "category": old_payload.get("category", "general"),
        "created_at": old_payload.get(
            "created_at",
            datetime.now().isoformat(),
        ),
        "updated_at": datetime.now().isoformat(),
    }

    point = PointStruct(
        id=point_id,
        vector=vector,
        payload=updated_payload,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point],
    )

    return point_id


def show_memories():
    """
    Display all stored memories and their metadata.
    Works with both old and new memory records.
    """

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    points = results[0]

    print("\n📦 Stored memories:")

    for point in points:
        payload = point.payload

        print(f"\nID: {point.id}")
        print(f"Memory: {payload.get('memory', 'Unknown')}")
        print(f"User: {payload.get('user_id', 'N/A')}")
        print(f"Category: {payload.get('category', 'N/A')}")
        print(f"Created: {payload.get('created_at', 'N/A')}")
        print(f"Updated: {payload.get('updated_at', 'N/A')}")


if __name__ == "__main__":
    try:
        create_collection()

        # Store an initial memory
        original_memory = "The user enjoys playing football."

        point_id = store_memory(original_memory)

        if point_id is None:
            print("The original memory already exists.")
            results = search_memory(original_memory, limit=1)

            if not results:
                raise ValueError("Could not find the existing memory.")

            point_id = results[0]["id"]

        print("\nOriginal memory:")
        print(original_memory)

        # Update the memory
        new_memory = "The user now prefers cricket."

        update_memory(point_id, new_memory)

        print("\nUpdated memory:")
        print(new_memory)

        # Show stored memories
        show_memories()

    finally:
        client.close

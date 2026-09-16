from ollama import embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
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
            "status": "current",
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])

    return point_id


def search_memory(query, limit=5):
    """
    Search only current memories in Qdrant.
    """

    query_vector = create_embedding(query)

    current_filter = Filter(
        must=[
            FieldCondition(
                key="status",
                match=MatchValue(value="current"),
            )
        ]
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=current_filter,
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
    Mark the old memory as superseded and create a new
    current memory instead of overwriting the old one.
    """

    existing_point = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )

    if not existing_point:
        raise ValueError(f"Memory with ID {point_id} was not found.")

    old_point = existing_point[0]
    old_payload = old_point.payload

    # Mark the old memory as superseded.
    old_payload["status"] = "superseded"
    old_payload["updated_at"] = datetime.now().isoformat()

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload=old_payload,
        points=[point_id],
    )

    # Create the new memory.
    vector = create_embedding(memory_text)

    new_point_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    new_point = PointStruct(
        id=new_point_id,
        vector=vector,
        payload={
            "memory": memory_text,
            "user_id": old_payload.get("user_id", "uddeshya"),
            "category": old_payload.get("category", "general"),
            "status": "current",
            "created_at": timestamp,
            "updated_at": timestamp,
            "supersedes": point_id,
        },
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[new_point],
    )

    return new_point_id


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
        print(f"Status: {payload.get('status', 'N/A')}")
        print(f"Supersedes: {payload.get('supersedes', 'N/A')}")
        print(f"Created: {payload.get('created_at', 'N/A')}")
        print(f"Updated: {payload.get('updated_at', 'N/A')}")


if __name__ == "__main__":
    try:
        create_collection()

        results = search_memory(
            "The user now prefers cricket.",
            limit=1,
        )

        if not results:
            raise ValueError("Could not find the cricket memory.")

        old_id = results[0]["id"]

        new_memory = "The user prefers football again."

        new_id = update_memory(
            old_id,
            new_memory,
        )

        print("\nOld memory ID:")
        print(old_id)

        print("\nNew memory ID:")
        print(new_id)

        show_memories()

    finally:
        client.close()

from ollama import embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
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


def store_memory(memory_text):
    """
    Store a memory only if it is not already present.
    """

    if is_duplicate(memory_text):
        print("\nDuplicate memory detected. Skipping storage.")
        return None

    vector = create_embedding(memory_text)

    point_id = str(uuid.uuid4())

    point = PointStruct(id=point_id, vector=vector, payload={"memory": memory_text})

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


if __name__ == "__main__":
    try:
        create_collection()

        test_memory = "The user enjoys playing football."

        point_id = store_memory(test_memory)

        if point_id:
            print("\nStored memory:")
            print(test_memory)

            print("\nPoint ID:")
            print(point_id)
        else:
            print("\nMemory was not stored because it was a duplicate.")

        print("\nSearching memory...")

        results = search_memory("What sport does the user like?")

        for result in results:
            print(f"- {result['memory']}")
            print(f"  Score: {result['score']:.3f}")

    finally:
        client.close()

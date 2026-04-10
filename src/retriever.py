from sentence_transformers import SentenceTransformer
import chromadb

QUESTION = "What is the ICU discharge procedure at Apollo Hospitals?"
DB_PATH = "data/chromadb"
TOP_K = 3


def retrieve(question: str, db_path: str = DB_PATH, top_k: int = TOP_K) -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(question).tolist()

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="documents")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Query: {question}\n")
    print(f"Top {top_k} results:\n" + "-" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (chunk, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        meta = meta if meta else {}
        source = meta.get("source", "unknown")
        print(f"\n[{rank}] Source: {source}  (distance: {distance:.4f})")
        print(chunk)
        print("-" * 60)


if __name__ == "__main__":
    retrieve(QUESTION)

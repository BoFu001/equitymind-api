from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_BATCH_SIZE

INDEX_NAME = "equitymind"
NAMESPACE = "sec_filings"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def upsert_chunks(chunks_with_embeddings: list[dict]) -> None:
    """
    Takes chunks that already have embeddings.
    Upserts them into Pinecone.
    """
    print(f"Upserting {len(chunks_with_embeddings)} vectors to Pinecone...")

    vectors = []
    for chunk in chunks_with_embeddings:
        vector_id = (
            chunk["metadata"]["ticker"] + "_" +
            chunk["metadata"]["filing_date"] + "_" +
            chunk["metadata"]["section"] + "_" +
            str(chunk["metadata"]["chunk_index"])
        )
        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": {
                **chunk["metadata"],
                "text": chunk["text"],
            }
        })


    for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
        batch = vectors[i:i + PINECONE_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"  {min(i + PINECONE_BATCH_SIZE, len(vectors))}/{len(vectors)} upserted")

    print("Upsert complete.")


def query(question_embedding: list[float], ticker: str = None, section: str = None, top_k: int = 5) -> list[dict]:
    """
    Query Pinecone with a question embedding.
    Optionally filter by ticker and/or section.
    """
    filter_dict = {}
    if ticker:
        filter_dict["ticker"] = ticker
    if section:
        filter_dict["section"] = section

    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
        namespace=NAMESPACE
    )

    return results.matches


if __name__ == "__main__":
    from src.ingestion.sec_loader import ingest_sec_filing, TICKERS
    from src.embeddings.embedder import embed_chunks

    for ticker in TICKERS:
        chunks = ingest_sec_filing(ticker)
        embedded = embed_chunks(chunks)
        upsert_chunks(embedded)

    print("\nAll companies done. Check Pinecone dashboard.")
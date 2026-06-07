from src.ingestion.sec_loader import ingest_sec_filing
from src.embeddings.embedder import embed_chunks
from src.vectorstore.pinecone_store import upsert_chunks, query


def retrieve(question: str, ticker: str, top_k: int = 5) -> list:
    """
    Retrieve relevant chunks from Pinecone for a given question and ticker.
    Used when data already exists in Pinecone.
    """
    # Embed the question
    embedded_question = embed_chunks([{"text": question, "metadata": {}}])
    question_vector = embedded_question[0]["embedding"]

    # Query Pinecone filtered by ticker
    matches = query(question_vector, ticker=ticker, top_k=top_k)

    return [
        {
            "text":   m.metadata.get("text", ""),
            "score":  m.score,
            "source": m.metadata.get("source", ""),
        }
        for m in matches
    ]


def fetch_embed_store_retrieve(question: str, ticker: str, top_k: int = 5) -> list:
    """
    Dynamically fetches SEC filing for a ticker not in Pinecone.
    Downloads, embeds, stores, then retrieves relevant chunks.
    """

    print(f"  [fetch_embed_store_retrieve] Fetching {ticker} from SEC EDGAR...")

    # Step 1: Download and chunk
    chunks = ingest_sec_filing(ticker)

    if not chunks:
        print(f"  [fetch_embed_store_retrieve] No 10-K data for {ticker} — skipping embed/store")
        return []

    print(f"  [fetch_embed_store_retrieve] Downloaded {len(chunks)} chunks")

    # Step 2: Embed
    embedded_chunks = embed_chunks(chunks)
    print(f"  [fetch_embed_store_retrieve] Embedded {len(embedded_chunks)} chunks")

    # Step 3: Store in Pinecone
    upsert_chunks(embedded_chunks)
    print(f"  [fetch_embed_store_retrieve] Stored in Pinecone")

    # Step 4: Retrieve
    return retrieve(question, ticker, top_k)
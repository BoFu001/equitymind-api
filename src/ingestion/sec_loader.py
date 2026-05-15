import time
from edgar import Company, set_identity
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

set_identity("Bo Fu bofu001@gmail.com")

SECTIONS = {
    "Item1":  ("business",              "Business Overview"),
    "Item1A": ("risk_factors",          "Risk Factors"),
    "Item7":  ("management_discussion", "MD&A"),
}

MIN_SECTION_LENGTH = 1000

TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "JPM",
    "JNJ", "XOM", "WMT", "BRK-B", "GE",
    "PFE", "BAC", "BA", "CAT"
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def get_latest_tenk(ticker: str):
    c = Company(ticker)
    filings = c.get_filings(form="10-K")
    for f in filings:
        if f.form == "10-K":
            return f.obj(), str(f.filing_date)
    raise ValueError(f"No standard 10-K found for {ticker}")


def clean_text(text: str) -> str:
    # Remove page headers like "Apple Inc. | 2025 Form 10-K | 5"
    text = re.sub(r'\n\n.{1,50}\|\s*\d{4}\s*Form 10-K\s*\|\s*\d+\n\n', '\n\n', text)
    return text

def chunk_text(text: str, ticker: str, filing_type: str,
               filing_date: str, section_key: str, section_label: str) -> list[dict]:
    docs = splitter.create_documents([text])
    return [
        {
            "text": doc.page_content,
            "metadata": {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "section": section_key,
                "section_label": section_label,
                "chunk_index": i,
                "source": f"{ticker}_{filing_type}_{section_key}",
            }
        }
        for i, doc in enumerate(docs)
        if len(doc.page_content) >= 100
    ]


def ingest_sec_filing(ticker: str, filing_type: str = "10-K") -> list[dict]:
    print(f"\n{'='*50}")
    print(f"Processing {ticker}...")

    tenk, filing_date = get_latest_tenk(ticker)
    print(f"  Filing date: {filing_date}")

    all_chunks = []
    for section_key, (attr, label) in SECTIONS.items():
        raw = str(getattr(tenk, attr))
        if len(raw) < MIN_SECTION_LENGTH:
            print(f"  {section_key} → SKIPPED ({len(raw)} chars)")
            continue
        text = clean_text(raw)
        chunks = chunk_text(text, ticker, filing_type, filing_date, section_key, label)
        print(f"  {section_key}: {len(raw):,} chars → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"  Total: {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        chunks = ingest_sec_filing(ticker)
        if chunks:
            sizes = [len(c["text"]) for c in chunks]
            print(f"\nSize stats — min:{min(sizes)}  max:{max(sizes)}  avg:{sum(sizes)//len(sizes)}")
            print(f"\n--- Preview first chunk ---")
            print(chunks[0]["text"][:400])
            print(f"\nMetadata: {chunks[0]['metadata']}")
    else:
        summary = {}
        for ticker in TICKERS:
            try:
                chunks = ingest_sec_filing(ticker)
                section_counts = {}
                for c in chunks:
                    sk = c["metadata"]["section"]
                    section_counts[sk] = section_counts.get(sk, 0) + 1
                summary[ticker] = {"total": len(chunks), "sections": section_counts}
                time.sleep(0.5)
            except Exception as e:
                summary[ticker] = {"error": str(e)}

        print(f"\n{'='*60}")
        print("INGESTION SUMMARY")
        print(f"{'='*60}")
        for ticker, info in summary.items():
            if "error" in info:
                print(f"{ticker:8s}  ERROR: {info['error']}")
            else:
                print(f"{ticker:8s}  {info['total']:3d} chunks  sections={list(info['sections'].keys())}")
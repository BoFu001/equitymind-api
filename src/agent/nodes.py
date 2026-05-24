import json
from openai import OpenAI
from config import OPENAI_API_KEY
from src.agent.state import AgentState
from src.vectorstore.pinecone_store import upsert_chunks, query, check_ticker_exists
from src.embeddings.embedder import embed_chunks
from src.ingestion.sec_loader import ingest_sec_filing

import yfinance as yf
import ta

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Node 1: Intent Classification
# ─────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """
    Classifies the user's question into one of seven categories.
    Uses GPT-4o with a strict prompt — returns only the category name.
    """
    question = state["question"]

    prompt = f"""You are EquityMind's intent classifier.
Classify the user question into exactly one of these categories:

- SPECIFIC_STOCK: user asks about one specific company (e.g. "What are Apple's risks?", "Analyse NVIDIA")
- COMPARISON: user wants to compare two or more companies (e.g. "Compare Apple and Microsoft")
- PORTFOLIO: user wants general investment recommendations (e.g. "Find me a low risk stock")
- ANALYZE_POSITION: user asks about their own holding in one stock (e.g. "I bought AAPL at $165, should I sell?", "I have 200 Apple shares, what should I do?")
- ANALYZE_PORTFOLIO: user wants to analyse their full portfolio of multiple stocks (e.g. "Review my portfolio: AAPL 200 shares, NVDA 50 shares")
- GREETING: user is saying hello or asking what EquityMind can do (e.g. "Hi", "What can you do?")
- OUT_OF_SCOPE: question is not related to stock investing at all (e.g. "I want to be rich", "What's the weather?")

User question: {question}

Reply with ONLY the category name. Nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    intent = response.choices[0].message.content.strip()
    print(f"  [Node 1] Intent: {intent}")
    return {"intent": intent}






# ─────────────────────────────────────────────
# Node 2: Extract Parameters
# ─────────────────────────────────────────────
def extract_parameters(state: AgentState) -> dict:
    """
    Extracts ticker and year from the user's question.
    Uses GPT-4o to return a JSON with ticker and year.
    """
    question = state["question"]

    prompt = f"""You are a financial data extractor.
Extract the stock ticker and year from the user question.

Rules:
- ticker: the stock symbol in uppercase (e.g. AAPL, MSFT, NVDA). If not mentioned, return null.
- year: the year mentioned (e.g. 2024, 2025). If not mentioned, return null (means use latest).
- If the user mentions a company name, convert it to the correct ticker (e.g. Apple → AAPL, Microsoft → MSFT, Tesla → TSLA).

User question: {question}

Reply with ONLY valid JSON. Example:
{{"ticker": "AAPL", "year": null}}
{{"ticker": "MSFT", "year": "2024"}}
{{"ticker": null, "year": null}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    data = json.loads(content)

    ticker = data.get("ticker")
    year   = data.get("year")

    print(f"  [Node 2] Ticker: {ticker}, Year: {year}")
    return {"ticker": ticker, "year": year}




# ─────────────────────────────────────────────
# Node 3: Check Pinecone
# ─────────────────────────────────────────────
def check_pinecone(state: AgentState) -> dict:
    """
    Checks if the ticker's data already exists in Pinecone.
    If yes → route to retrieve.
    If no → route to fetch from SEC.
    """
    ticker = state["ticker"]

    if not ticker:
        return {"data_status": "NO_TICKER"}

    if check_ticker_exists(ticker):
        print(f"  [Node 3] {ticker} found in Pinecone → retrieve")
        return {"data_status": "RETRIEVE"}
    else:
        print(f"  [Node 3] {ticker} not found in Pinecone → fetch from SEC")
        return {"data_status": "FETCH_NEEDED"}


# ─────────────────────────────────────────────
# Node 4A: Retrieve Chunks from Pinecone
# ─────────────────────────────────────────────
def retrieve_chunks(state: AgentState) -> dict:
    """
    Retrieves relevant chunks from Pinecone using the user's question.
    """

    question = state["question"]
    ticker   = state["ticker"]

    # Embed the question
    embedded = embed_chunks([{"text": question, "metadata": {}}])
    question_vector = embedded[0]["embedding"]

    # Query Pinecone filtered by ticker
    matches = query(question_vector, ticker=ticker, top_k=5)

    chunks = [
        {
            "text":  m.metadata.get("text", ""),
            "score": m.score,
            "source": m.metadata.get("source", ""),
        }
        for m in matches
    ]

    print(f"  [Node 4A] Retrieved {len(chunks)} chunks for {ticker}")
    return {"chunks": chunks}




# ─────────────────────────────────────────────
# Node 4B: Fetch from SEC, Embed, Store, Retrieve
# ─────────────────────────────────────────────
def fetch_and_retrieve(state: AgentState) -> dict:
    """
    Dynamically fetches SEC filing for a ticker not in Pinecone.
    Downloads, embeds, stores, then retrieves relevant chunks.
    """

    ticker   = state["ticker"]
    question = state["question"]

    print(f"  [Node 4B] {ticker} not in Pinecone — fetching from SEC...")

    # Step 1: Download and chunk SEC filing
    chunks = ingest_sec_filing(ticker)
    print(f"  [Node 4B] Downloaded {len(chunks)} chunks for {ticker}")

    # Step 2: Embed chunks
    embedded_chunks = embed_chunks(chunks)
    print(f"  [Node 4B] Embedded {len(embedded_chunks)} chunks")

    # Step 3: Store in Pinecone
    upsert_chunks(embedded_chunks)
    print(f"  [Node 4B] Stored in Pinecone")

    # Step 4: Retrieve relevant chunks
    embedded_question = embed_chunks([{"text": question, "metadata": {}}])
    question_vector = embedded_question[0]["embedding"]
    matches = query(question_vector, ticker=ticker, top_k=5)

    retrieved_chunks = [
        {
            "text":   m.metadata.get("text", ""),
            "score":  m.score,
            "source": m.metadata.get("source", ""),
        }
        for m in matches
    ]

    print(f"  [Node 4B] Retrieved {len(retrieved_chunks)} chunks for {ticker}")
    return {"chunks": retrieved_chunks}





# ─────────────────────────────────────────────
# Node 5: Get Market Data
# ─────────────────────────────────────────────
def get_market_data(state: AgentState) -> dict:
    """
    Fetches market data for the ticker using yfinance.
    Returns price, P/E, market cap, revenue, and technical indicators.
    """

    ticker = state["ticker"]

    if not ticker:
        return {"market_data": None}

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # Basic fundamentals
        market_data = {
            "ticker":        ticker,
            "company_name":  info.get("longName", ticker),
            "current_price": info.get("currentPrice"),
            "market_cap":    info.get("marketCap"),
            "pe_ratio":      info.get("trailingPE"),
            "forward_pe":    info.get("forwardPE"),
            "revenue":       info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "52w_high":      info.get("fiftyTwoWeekHigh"),
            "52w_low":       info.get("fiftyTwoWeekLow"),
            "sector":        info.get("sector"),
            "industry":      info.get("industry"),
        }

        # Technical indicators from last 6 months of price data
        hist = stock.history(period="6mo")
        if not hist.empty:
            hist["rsi"]  = ta.momentum.RSIIndicator(hist["Close"]).rsi()
            macd         = ta.trend.MACD(hist["Close"])
            hist["macd"] = macd.macd()
            hist["macd_signal"] = macd.macd_signal()

            market_data["rsi"]         = round(hist["rsi"].iloc[-1], 2)
            market_data["macd"]        = round(hist["macd"].iloc[-1], 4)
            market_data["macd_signal"] = round(hist["macd_signal"].iloc[-1], 4)
            market_data["sma_50"]      = round(hist["Close"].rolling(50).mean().iloc[-1], 2)
            market_data["sma_200"]     = round(hist["Close"].rolling(200).mean().iloc[-1], 2) if len(hist) >= 200 else None

        print(f"  [Node 5] Market data fetched for {ticker}: price={market_data['current_price']}, RSI={market_data.get('rsi')}")
        return {"market_data": market_data}

    except Exception as e:
        print(f"  [Node 5] Error fetching market data for {ticker}: {e}")
        return {"market_data": None}
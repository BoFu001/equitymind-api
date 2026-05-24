import json

from openai import OpenAI

from config import OPENAI_API_KEY
from src.agent.state import AgentState
from src.tools.market_data import get_stock_data
from src.tools.sec_retrieval import retrieve, fetch_embed_store_retrieve
from src.vectorstore.pinecone_store import check_ticker_exists
from src.tools.news_sentiment import get_news_and_sentiment

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
    """Retrieves relevant chunks from Pinecone."""
    chunks = retrieve(state["question"], state["ticker"])
    print(f"  [Node 4A] Retrieved {len(chunks)} chunks for {state['ticker']}")
    return {"chunks": chunks}


# ─────────────────────────────────────────────
# Node 4B: Fetch from SEC, Embed, Store, Retrieve
# ─────────────────────────────────────────────
def fetch_and_retrieve(state: AgentState) -> dict:
    """Dynamically fetches SEC filing, embeds, stores, then retrieves."""
    chunks = fetch_embed_store_retrieve(state["question"], state["ticker"])
    print(f"  [Node 4B] Retrieved {len(chunks)} chunks for {state['ticker']}")
    return {"chunks": chunks}





# ─────────────────────────────────────────────
# Node 5: Get Market Data
# ─────────────────────────────────────────────
def get_market_data(state: AgentState) -> dict:
    """
    Fetches market data using the market_data tool.
    """
    ticker = state["ticker"]

    if not ticker:
        return {"market_data": None}

    market_data = get_stock_data(ticker)
    print(f"  [Node 5] Market data fetched for {ticker}: price={market_data.get('current_price') if market_data else None}")
    return {"market_data": market_data}






# ─────────────────────────────────────────────
# Node 6: Get News and Sentiment
# ─────────────────────────────────────────────
def get_news(state: AgentState) -> dict:
    """
    Fetches recent news and sentiment for the ticker using FinBERT. 

    NOTE: Currently using Finlight free tier (Launchpad plan).
    Free tier has 12-hour news delay and no real-time access.
    
    """
    ticker = state["ticker"]

    if not ticker:
        return {"news": []}

    news = get_news_and_sentiment(ticker)
    print(f"  [Node 6] {len(news)} articles fetched for {ticker}")
    return {"news": news}
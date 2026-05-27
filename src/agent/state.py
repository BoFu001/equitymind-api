from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    question: str
    messages: list              # full conversation history

    # Intent classification
    intent: Optional[str]        # SPECIFIC_STOCK / COMPARISON / DISCOVERY / ANALYZE_POSITION / ANALYZE_PORTFOLIO / GREETING / OUT_OF_SCOPE
    
    # Routing decision
    data_status: Optional[str]   # RETRIEVE / FETCH_NEEDED / NO_TICKER

    # Extracted parameters
    ticker: Optional[str]        # primary ticker e.g. "AAPL"
    tickers: Optional[list]      # all tickers e.g. ["AAPL"] or ["AAPL", "MSFT"]
    year: Optional[str]          # e.g. "2025" or None for latest

    # Retrieval
    chunks: Optional[list]       # retrieved chunks from Pinecone

    # Market data
    market_data: Optional[dict]  # price, P/E, revenue etc from yfinance

    # News and sentiment
    news: Optional[list]         # recent news articles with sentiment scores

    # Final output
    answer: Optional[str]        # final report
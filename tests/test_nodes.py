import pytest
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    extract_parameters,
    check_pinecone,
    retrieve_chunks,
    get_market_data,
    get_news,
    handle_out_of_scope,
    handle_greeting,
    handle_portfolio,
    handle_comparison,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_state(**kwargs) -> AgentState:
    """Create a minimal AgentState for testing."""
    defaults = {
        "question": "What are Apple's biggest risks?",
        "messages": [],
        "intent": None,
        "data_status": None,
        "ticker": None,
        "tickers": [], 
        "year": None,
        "chunks": None,
        "market_data": None,
        "news": None,
        "answer": None,
    }
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────
# Node 1: Intent Classification
# ─────────────────────────────────────────────

def test_classify_intent_specific_stock():
    state = make_state(question="What are Apple's biggest risks?")
    result = classify_intent(state)
    assert result["intent"] == "SPECIFIC_STOCK"

def test_classify_intent_out_of_scope():
    state = make_state(question="I want to become rich")
    result = classify_intent(state)
    assert result["intent"] == "OUT_OF_SCOPE"

def test_classify_intent_greeting():
    state = make_state(question="Hello, what can you do?")
    result = classify_intent(state)
    assert result["intent"] == "GREETING"

def test_classify_intent_comparison():
    state = make_state(question="Compare Apple and Microsoft")
    result = classify_intent(state)
    assert result["intent"] == "COMPARISON"

def test_classify_intent_portfolio():
    state = make_state(question="Find me a low risk stock")
    result = classify_intent(state)
    assert result["intent"] == "PORTFOLIO"

# ─────────────────────────────────────────────
# Node 2: Extract Parameters
# ─────────────────────────────────────────────

def test_extract_parameters_aapl():
    state = make_state(question="What are Apple's biggest risks?")
    result = extract_parameters(state)
    assert result["ticker"] == "AAPL"
    assert result["year"] is None

def test_extract_parameters_with_year():
    state = make_state(question="What were Microsoft's risks in 2024?")
    result = extract_parameters(state)
    assert result["ticker"] == "MSFT"
    assert result["year"] == "2024"

def test_extract_parameters_no_ticker():
    state = make_state(question="Find me a low risk stock")
    result = extract_parameters(state)
    assert result["ticker"] is None



def test_extract_parameters_multiple_tickers():
    state = make_state(question="Compare Apple and Microsoft")
    result = extract_parameters(state)
    assert result["tickers"] == ["AAPL", "MSFT"]
    assert result["ticker"] == "AAPL"
    



def test_extract_parameters_amazon():
    state = make_state(question="Analyse Amazon")
    result = extract_parameters(state)
    assert result["ticker"] == "AMZN"

def test_extract_parameters_alibaba():
    state = make_state(question="Analyse Alibaba")
    result = extract_parameters(state)
    assert result["ticker"] == "BABA"

def test_extract_parameters_tencent():
    state = make_state(question="Analyse Tencent")
    result = extract_parameters(state)
    assert result["ticker"] == "0700.HK"
    
# ─────────────────────────────────────────────
# Node 3: Check Pinecone
# ─────────────────────────────────────────────

def test_check_pinecone_exists():
    state = make_state(ticker="AAPL")
    result = check_pinecone(state)
    assert result["data_status"] == "RETRIEVE"

def test_check_pinecone_not_exists():
    state = make_state(ticker="FAKE123")
    result = check_pinecone(state)
    assert result["data_status"] == "FETCH_NEEDED"

def test_check_pinecone_no_ticker():
    state = make_state(ticker=None)
    result = check_pinecone(state)
    assert result["data_status"] == "NO_TICKER"


# ─────────────────────────────────────────────
# Node 4A: Retrieve Chunks
# ─────────────────────────────────────────────

def test_retrieve_chunks_aapl():
    state = make_state(
        question="What are Apple's biggest risks?",
        ticker="AAPL"
    )
    result = retrieve_chunks(state)
    assert "chunks" in result
    assert len(result["chunks"]) > 0
    assert "text" in result["chunks"][0]
    assert "score" in result["chunks"][0]
    assert "source" in result["chunks"][0]


# ─────────────────────────────────────────────
# Node 5: Market Data
# ─────────────────────────────────────────────

def test_get_market_data_aapl():
    state = make_state(ticker="AAPL")
    result = get_market_data(state)
    assert result["market_data"] is not None
    assert result["market_data"]["ticker"] == "AAPL"
    assert result["market_data"]["current_price"] is not None
    assert result["market_data"]["pe_ratio"] is not None

def test_get_market_data_no_ticker():
    state = make_state(ticker=None)
    result = get_market_data(state)
    assert result["market_data"] is None


# ─────────────────────────────────────────────
# Node 6: News and Sentiment
# ─────────────────────────────────────────────

def test_get_news_aapl():
    state = make_state(ticker="AAPL")
    result = get_news(state)
    assert "news" in result
    assert isinstance(result["news"], list)
    if len(result["news"]) > 0:
        article = result["news"][0]
        assert "title" in article
        assert "sentiment" in article
        assert "score" in article
        assert "url" in article

def test_get_news_no_ticker():
    state = make_state(ticker=None)
    result = get_news(state)
    assert result["news"] == []


# ─────────────────────────────────────────────
# Node 8: Out of Scope
# ─────────────────────────────────────────────

def test_handle_out_of_scope():
    state = make_state(question="I want to be rich")
    result = handle_out_of_scope(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2


# ─────────────────────────────────────────────
# Node 9: Greeting
# ─────────────────────────────────────────────

def test_handle_greeting():
    state = make_state(question="Hello")
    result = handle_greeting(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2






# ─────────────────────────────────────────────
# Node 10: Portfolio Recommendation
# ─────────────────────────────────────────────


def test_handle_portfolio():
    state = make_state(question="Find me a low risk stock")
    result = handle_portfolio(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2

# ─────────────────────────────────────────────
# Node 11: Comparison
# ─────────────────────────────────────────────


def test_handle_comparison():
    state = make_state(
        question="Compare Apple and Microsoft",
        tickers=["AAPL", "MSFT"]
    )
    result = handle_comparison(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "AAPL" in result["answer"] or "Apple" in result["answer"]
    assert "MSFT" in result["answer"] or "Microsoft" in result["answer"]
    assert "messages" in result
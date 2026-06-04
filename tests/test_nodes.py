import pytest
from unittest.mock import patch, MagicMock
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
    handle_discovery,
    handle_comparison,
    handle_no_ticker,
)

@pytest.fixture(autouse=True)
def mock_stream_writer():
    with patch('src.agent.nodes.get_stream_writer') as mock:
        mock.return_value = MagicMock()
        yield




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
# Node: Intent Classification
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

def test_classify_intent_discovery():
    state = make_state(question="Find me a low risk stock")
    result = classify_intent(state)
    assert result["intent"] == "DISCOVERY"

def test_classify_intent_stock_market():
    state = make_state(question="Tell me about the stock market")
    result = classify_intent(state)
    assert result["intent"] == "DISCOVERY"

def test_classify_intent_vague_sector():
    state = make_state(question="Analyse a tech company")
    result = classify_intent(state)
    assert result["intent"] == "DISCOVERY"

# ─────────────────────────────────────────────
# Node: Extract Parameters
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
# Node: Intent Classification + Extract Parameters for Edge case tests: valid intent but no ticker
# ─────────────────────────────────────────────

def test_no_ticker_edge_cases():
    """
    Tests questions that may be classified as SPECIFIC_STOCK or COMPARISON
    but have no extractable ticker. All should route to DISCOVERY or have no ticker.
    """
    edge_cases = [
        "Analyse a tech company",
        "Tell me about a good stock",
        "What about that AI company?",
        "Compare two tech companies",
        "Compare them",
        "Which is better, A or B?",
    ]

    for question in edge_cases:
        # Step 1 — classify
        classify_state = make_state(question=question)
        classify_result = classify_intent(classify_state)
        intent = classify_result["intent"]

        # Step 2 — extract
        extract_state = make_state(question=question, intent=intent)
        extract_result = extract_parameters(extract_state)
        ticker = extract_result.get("ticker")
        tickers = extract_result.get("tickers", [])

        print(f"\nQ: '{question}'")
        print(f"  Intent: {intent}")
        print(f"  Ticker: {ticker} | Tickers: {tickers}")

        # Assert: if COMPARISON or SPECIFIC_STOCK — no ticker should be found
        # These vague questions should either route to DISCOVERY or have no ticker
        if intent == "COMPARISON":
            assert not tickers, f"Expected no tickers for vague COMPARISON: '{question}' but got {tickers}"
        if intent == "SPECIFIC_STOCK":
            assert ticker is None, f"Expected no ticker for vague SPECIFIC_STOCK: '{question}' but got {ticker}"

# ─────────────────────────────────────────────
# Node: Check Pinecone
# ─────────────────────────────────────────────

def test_check_pinecone_exists():
    state = make_state(ticker="AAPL")
    result = check_pinecone(state)
    assert result["data_status"] == "RETRIEVE"

def test_check_pinecone_not_exists():
    state = make_state(ticker="FAKE123")
    result = check_pinecone(state)
    assert result["data_status"] == "FETCH_NEEDED"


# ─────────────────────────────────────────────
# Node: Retrieve Chunks
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
# Node: Market Data
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
# Node: News and Sentiment
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
# Node: Out of Scope
# ─────────────────────────────────────────────

def test_handle_out_of_scope():
    state = make_state(question="I want to be rich")
    result = handle_out_of_scope(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2


# ─────────────────────────────────────────────
# Node: Greeting
# ─────────────────────────────────────────────

def test_handle_greeting():
    state = make_state(question="Hello")
    result = handle_greeting(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2






# ─────────────────────────────────────────────
# Node: Discovery Recommendation
# ─────────────────────────────────────────────


def test_handle_discovery():
    state = make_state(question="Find me a low risk stock")
    result = handle_discovery(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2

# ─────────────────────────────────────────────
# Node: Comparison
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



# ─────────────────────────────────────────────
# Node: No Ticker
# ─────────────────────────────────────────────

def test_handle_no_ticker_specific_stock():
    state = make_state(question="Analyse XYZ Corporation", intent="SPECIFIC_STOCK")
    result = handle_no_ticker(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "company" in result["answer"].lower()
    assert "messages" in result
    assert len(result["messages"]) == 2

def test_handle_no_ticker_comparison():
    state = make_state(question="Compare them", intent="COMPARISON")
    result = handle_no_ticker(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "compare" in result["answer"].lower()
    assert "messages" in result
    assert len(result["messages"]) == 2
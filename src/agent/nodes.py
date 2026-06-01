import json
import re
from datetime import datetime

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL
from core.context import token_queue_var
from src.agent.state import AgentState
from src.tools.market_data import get_stock_data
from src.tools.news_sentiment import get_news_and_sentiment
from src.tools.sec_retrieval import retrieve, fetch_embed_store_retrieve
from src.vectorstore.pinecone_store import check_ticker_exists

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Node: Intent Classification
# ─────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """
    Classifies the user's question into one of seven categories.
    Uses LLM with a strict prompt — returns only the category name.
    """
    question = state["question"]

    prompt = f"""You are {APP_NAME}'s intent classifier.
Classify the user question into exactly one of these categories:

- SPECIFIC_STOCK: user asks about one NAMED specific company (e.g. "What are Apple's risks?", "Analyse NVIDIA", "Tell me about Tesla"). The company must be explicitly named — NOT vague like "a tech company" or "a healthcare stock".
- COMPARISON: user wants to compare two or more NAMED companies (e.g. "Compare Apple and Microsoft")
- DISCOVERY: user wants general investment recommendations, asks about a sector, or asks general financial market questions without naming a specific company (e.g. "Find me a low risk stock", "Analyse a tech company", "Tell me about semiconductor stocks", "Tell me about the stock market", "What is a good investment?")
- ANALYZE_POSITION: user asks about their own holding in one stock (e.g. "I bought AAPL at $165, should I sell?", "I have 200 Apple shares, what should I do?")
- ANALYZE_PORTFOLIO: user wants to analyse their full portfolio of multiple stocks (e.g. "Review my portfolio: AAPL 200 shares, NVDA 50 shares")
- GREETING: user is saying hello or asking what {APP_NAME} can do (e.g. "Hi", "What can you do?")
- OUT_OF_SCOPE: question has NO relation to investing, stocks, or financial markets (e.g. "I want to be rich", "What's the weather?", "Am I handsome?")

User question: {question}

Reply with ONLY the category name. Nothing else."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    intent = response.choices[0].message.content.strip()
    print(f"  [classify_intent] Intent: {intent}")
    return {"intent": intent}





# ─────────────────────────────────────────────
# Node: Extract Parameters
# ─────────────────────────────────────────────
def extract_parameters(state: AgentState) -> dict:
    """
    Extracts ticker(s) and year from the user's question.
    Returns primary ticker, list of all tickers, and year.
    """
    question = state["question"]

    prompt = f"""You are a financial data extractor.
Extract the stock ticker(s) and year from the user question.

Rules:
- tickers: list of ALL stock ticker symbols. Convert ANY company name to its ticker symbol. If no company or ticker mentioned, return [].
- year: the year mentioned. If not mentioned, return null.
- Examples of conversions: Apple → AAPL, Microsoft → MSFT, Tesla → TSLA, NVIDIA → NVDA, Google → GOOGL, Amazon → AMZN, Alibaba → BABA, Meta → META, Samsung → 005930.KS

User question: {question}

Reply with ONLY valid JSON. No markdown, no code fences, no explanation. Example:
{{"tickers": ["AAPL"], "year": null}}
{{"tickers": ["AAPL", "MSFT"], "year": null}}
{{"tickers": ["BABA"], "year": null}}
{{"tickers": ["AMZN"], "year": null}}
{{"tickers": [], "year": null}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if not content:
        print(f"  [extract_parameters] Empty response from {LLM_MODEL}, using defaults")
        return {"ticker": None, "tickers": [], "year": None}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"  [extract_parameters] Invalid JSON: {content}")
        return {"ticker": None, "tickers": [], "year": None}

    tickers = data.get("tickers", [])
    ticker  = tickers[0] if tickers else None
    year    = str(data.get("year")) if data.get("year") else None

    print(f"  [extract_parameters] Ticker: {ticker}, Tickers: {tickers}, Year: {year}")
    return {"ticker": ticker, "tickers": tickers, "year": year}



# ─────────────────────────────────────────────
# Node: Check Pinecone
# ─────────────────────────────────────────────
def check_pinecone(state: AgentState) -> dict:
    """
    Checks if the ticker's data already exists in Pinecone.
    If yes → route to retrieve.
    If no → route to fetch from SEC.
    """
    ticker = state["ticker"]

    if not ticker:
        print(f"  [check_pinecone] No ticker found — routing to out_of_scope")
        return {"data_status": "NO_TICKER"}

    if check_ticker_exists(ticker):
        print(f"  [check_pinecone] {ticker} found in Pinecone → retrieve")
        return {"data_status": "RETRIEVE"}
    else:
        print(f"  [check_pinecone] {ticker} not found in Pinecone → fetch from SEC")
        return {"data_status": "FETCH_NEEDED"}


# ─────────────────────────────────────────────
# Node: Retrieve Chunks from Pinecone
# ─────────────────────────────────────────────
def retrieve_chunks(state: AgentState) -> dict:
    """Retrieves relevant chunks from Pinecone."""
    chunks = retrieve(state["question"], state["ticker"])
    print(f"  [retrieve_chunks] Retrieved {len(chunks)} chunks for {state['ticker']}")
    return {"chunks": chunks}


# ─────────────────────────────────────────────
# Node: Fetch from SEC, Embed, Store, Retrieve
# ─────────────────────────────────────────────
def fetch_and_retrieve(state: AgentState) -> dict:
    """Dynamically fetches SEC filing, embeds, stores, then retrieves."""
    chunks = fetch_embed_store_retrieve(state["question"], state["ticker"])
    print(f"  [fetch_and_retrieve] Retrieved {len(chunks)} chunks for {state['ticker']}")
    return {"chunks": chunks}





# ─────────────────────────────────────────────
# Node: Get Market Data
# ─────────────────────────────────────────────
def get_market_data(state: AgentState) -> dict:
    """
    Fetches market data using the market_data tool.
    """
    ticker = state["ticker"]

    if not ticker:
        return {"market_data": None}

    market_data = get_stock_data(ticker)
    print(f"  [get_market_data] Market data fetched for {ticker}: price={market_data.get('current_price') if market_data else None}")
    return {"market_data": market_data}






# ─────────────────────────────────────────────
# Node: Get News and Sentiment
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
    print(f"  [get_news] {len(news)} articles fetched for {ticker}")
    return {"news": news}




# ─────────────────────────────────────────────
# Node: Generate Report
# ─────────────────────────────────────────────
def generate_report(state: AgentState) -> dict:
    """
    Combines all data and generates a structured investment report.
    Uses LLM with full context: SEC chunks, market data, news.
    Implements XAI by including evidence and sources.
    """
    question    = state["question"]
    ticker      = state["ticker"]
    chunks      = state.get("chunks") or []
    market_data = state.get("market_data") or {}
    news        = state.get("news") or []
    messages    = state.get("messages") or []

    # ── Format SEC chunks for prompt ──
    if not chunks:
        sec_context = f"No SEC 10-K filing available for {ticker}. This company may be a foreign filer (20-F) or the filing could not be retrieved."
    else:
        sec_context = ""
        for i, chunk in enumerate(chunks):
            sec_context += f"\n[SEC Source {i+1}: {chunk.get('source','')} | Score: {chunk.get('score',0):.2f}]\n"
            sec_context += chunk.get("text", "") + "\n"

    # ── Format market data for prompt ──
    md = market_data
    market_context = f"""
Company: {md.get('company_name', ticker)}
Price: ${md.get('current_price')} | Market Cap: {md.get('market_cap')}
P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}
Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}
EPS (trailing): {md.get('eps_trailing')} | EPS (forward): {md.get('eps_forward')}
52w High: {md.get('52w_high')} | 52w Low: {md.get('52w_low')}
Dividend Yield: {md.get('dividend_yield')} | Dividend Rate: {md.get('dividend_rate')}
Analyst Target High: {md.get('target_high')} | Low: {md.get('target_low')} | Mean: {md.get('target_mean')}
Analyst Recommendation: {md.get('recommendation')}
RSI: {md.get('rsi')} | MACD: {md.get('macd')} | Signal: {md.get('macd_signal')}
SMA50: {md.get('sma_50')} | SMA200: {md.get('sma_200')}
Sector: {md.get('sector')} | Industry: {md.get('industry')}
"""

    # ── Format news for prompt ──
    news_context = ""
    for i, article in enumerate(news):
        news_context += f"\n[News {i+1}] {article.get('sentiment','').upper()} ({article.get('score',0):.2f})\n"
        news_context += f"Title: {article.get('title','')}\n"
        news_context += f"Summary: {article.get('summary','')}\n"
        news_context += f"URL: {article.get('url','')}\n"
        news_context += f"Published: {article.get('published','')}\n"

    # ── Format chat history ──
    history_context = ""
    for msg in messages[-6:]:  # last 3 exchanges
        role = msg.get("role", "")
        content = msg.get("content", "")
        history_context += f"{role.upper()}: {content}\n"

    # ── Build prompt ──
    prompt = f"""You are {APP_NAME}, a professional AI investment research analyst.
Generate a comprehensive, well-structured investment report in markdown format.

IMPORTANT RULES:
1. Put the recommendation FIRST — users want the answer before the details.
2. Include ALL evidence and sources for XAI (explainable AI) transparency.
3. Include clickable news URLs so users can verify information.
4. Be specific with numbers — never vague.
5. Always include the disclaimer at the end.
6. Use markdown formatting with emojis for visual clarity.

USER QUESTION: {question}
TICKER: {ticker}
DATE: {datetime.now().strftime('%B %d, %Y')}

CONVERSATION HISTORY:
{history_context}

MARKET DATA:
{market_context}

SEC FILING DATA (from {ticker} 10-K annual report):
{sec_context}

NEWS & SENTIMENT (last 30 days):
{news_context}

Generate the report in this EXACT structure:

# 📊 [Company Name] ([TICKER]) — Investment Analysis
*Generated by {APP_NAME} · [date]*

---

## 💡 AI Recommendation: [BUY/HOLD/SELL]
**Target Price:** $[mean analyst target] | **Current:** $[price] | **Upside:** [%]
**Confidence:** [High/Medium/Low] | **Time Horizon:** [Short/Medium/Long term]

> [2-3 sentence summary of why this recommendation]

---

## 📊 Quick Summary
- [Key fact 1]
- [Key fact 2]
- [Key fact 3]
- [Key fact 4]

---

## 💰 Valuation & Fundamentals
| Metric | Value | Signal |
|--------|-------|--------|
| P/E Ratio | [value] | [Cheap/Fair/Expensive] |
| Forward P/E | [value] | [signal] |
| Revenue | [value] | [signal] |
| Profit Margin | [value] | [signal] |
| EPS (TTM) | [value] | |
| Dividend Yield | [value] | |

---

## 📈 Technical Analysis
| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI (14) | [value] | [Oversold/Neutral/Overbought] |
| MACD | [value] | [Bullish/Bearish] |
| SMA 50 | [value] | [Above/Below price] |
| SMA 200 | [value] | [Above/Below price] |
| 52w Range | [low] - [high] | [position] |

---

## 📰 News & Sentiment (Last 30 Days)
**Overall Sentiment:** [BULLISH/NEUTRAL/BEARISH] | **Avg Score:** [x.xx]

[For each article:]
[emoji] [[title]]([url]) — [sentiment] ([score]) — [date]

---

## ⚠️ Key Risks (from SEC 10-K Filing)
[Extract top 3-5 key risks from the SEC chunks provided]

---

## 📎 Evidence & Sources (XAI)
**This recommendation is based on:**
- **Fundamentals:** [key metrics used]
- **Technicals:** [indicators used]
- **News sentiment:** [x positive, y negative, z neutral articles]
- **SEC Filing:** [sources cited with scores]

---

## ⚠️ Disclaimer
*This report is generated by AI for educational and research purposes only. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.*
"""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    # Update conversation history
    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    print(f"  [generate_report] Report generated for {ticker} ({len(answer)} chars)")
    return {"answer": answer, "messages": updated_messages}





# ─────────────────────────────────────────────
# Node: Handle Out of Scope
# ─────────────────────────────────────────────
def handle_out_of_scope(state: AgentState) -> dict:
    """
    Returns a polite refusal for out-of-scope questions.
    """
    messages = state.get("messages") or []
    question = state["question"]

    answer = f"""I'm {APP_NAME}, an AI investment research assistant. I specialise in stock analysis, company research, and investment insights.

I can help you with:
- 📊 Analysing a specific stock (e.g. "Analyse Apple")
- ⚖️ Comparing companies (e.g. "Compare NVIDIA and Microsoft")
- 🔍 Finding investment opportunities (e.g. "Find low risk stocks")
- 📰 News and sentiment analysis
- ⚠️ Risk analysis from SEC filings

What stock would you like me to research?"""

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)

    return {"answer": answer, "messages": updated_messages}


# ─────────────────────────────────────────────
# Node: Handle Greeting
# ─────────────────────────────────────────────
def handle_greeting(state: AgentState) -> dict:
    """
    Returns a friendly greeting and explains what this app can do.
    """
    messages = state.get("messages") or []
    question = state["question"]

    answer = f"""👋 Hello! I'm {APP_NAME}, your AI-powered investment research assistant.

I analyse stocks using:
- 📄 **SEC 10-K filings** — official annual reports
- 📈 **Market data** — price, P/E, RSI, MACD, moving averages
- 📰 **News sentiment** — FinBERT AI analysis of recent news
- 💡 **AI recommendations** — BUY/HOLD/SELL with full evidence

**Try asking me:**
- "Analyse Apple"
- "What are NVIDIA's biggest risks?"
- "Compare Microsoft and Google"
- "Find me a low risk stock in healthcare"

What would you like to research today?"""

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)

    return {"answer": answer, "messages": updated_messages}





# ─────────────────────────────────────────────
# Node: Handle Discovery Recommendation
# ─────────────────────────────────────────────
def handle_discovery(state: AgentState) -> dict:
    """
    Handles general discovery/investment recommendation requests.
    Step 1: LLM suggests 5 candidate tickers based on user criteria.
    Step 2: Fetch real market data and SEC chunks for each candidate.
    Step 3: LLM ranks and recommends top 3 based on real data.
    """
    question = state["question"]
    messages = state.get("messages") or []

    # ── Step 1: Ask LLM to suggest 5 candidate tickers ──
    ticker_prompt = f"""You are a financial analyst.
The user wants investment recommendations based on their criteria.

USER QUESTION: {question}

Return exactly 5 stock tickers that could match the user's criteria.
IMPORTANT: Only suggest US-listed companies that file 10-K annual reports with the SEC.
Do NOT suggest foreign companies or ADRs (e.g. Alibaba, ASML, Toyota, TSM).

Reply with ONLY valid JSON. No markdown, no code fences, no explanation. Example:
{{"tickers": ["JNJ", "WMT", "BRK-B", "PFE", "JPM"]}}

# NOTE: Once 20-F pipeline is built (sec_loader_20f.py),
# remove the 10-K constraint above to support global companies."""

    ticker_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": ticker_prompt}],
        temperature=0,
    )

    try:
        ticker_data = json.loads(ticker_response.choices[0].message.content.strip())
        candidate_tickers = ticker_data.get("tickers", [])
    except Exception:
        print(f"  [handle_discovery] Could not parse candidate tickers")
        answer = f"""I encountered a technical issue processing your request.

Please try again or rephrase your question."""
        updated_messages = messages + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {"answer": answer, "messages": updated_messages}

    if not candidate_tickers:
        answer = f"""I couldn't find companies matching your specific criteria.

Please try:
- Being more specific (e.g. "Find me a low risk healthcare stock")
- Asking about a specific company (e.g. "Analyse Johnson and Johnson")"""
        updated_messages = messages + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {"answer": answer, "messages": updated_messages}

    print(f"  [handle_discovery] Step 1 — Candidates: {candidate_tickers}")

    # ── Step 2: Fetch real data for each candidate ──
    all_market_data = {}
    for t in candidate_tickers:
        data = get_stock_data(t)
        if data:
            all_market_data[t] = data

    all_chunks = {}
    for t in candidate_tickers:
        try:
            if check_ticker_exists(t):
                chunks = retrieve(question, t, top_k=3)
            else:
                chunks = fetch_embed_store_retrieve(question, t, top_k=3)
            all_chunks[t] = chunks
        except Exception as e:
            print(f"  [handle_discovery] Could not fetch SEC data for {t}: {e}")
            all_chunks[t] = []

    print(f"  [handle_discovery] Step 2 — Real data fetched for {list(all_market_data.keys())}")

    # ── Step 3: Format real data for prompt ──
    market_context = ""
    for t, md in all_market_data.items():
        market_context += f"\n{t} ({md.get('company_name')}):\n"
        market_context += f"  Price: ${md.get('current_price')} | P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}\n"
        market_context += f"  Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}\n"
        market_context += f"  RSI: {md.get('rsi')} | Dividend Yield: {md.get('dividend_yield')}\n"
        market_context += f"  Analyst Target: ${md.get('target_mean')} | Recommendation: {md.get('recommendation')}\n"

    sec_context = ""
    for t, chunks in all_chunks.items():
        sec_context += f"\n{t} SEC Filing:\n"
        if not chunks:
            sec_context += "  No SEC 10-K data available.\n"
        else:
            for chunk in chunks:
                sec_context += chunk.get("text", "")[:300] + "\n"

    # ── Step 4: LLM ranks and recommends top 3 ──
    prompt = f"""You are {APP_NAME}, a professional AI investment research analyst.
The user wants investment recommendations. You have real market data and SEC filing data
for 5 candidate companies. Use this real data to rank and recommend the top 3.

USER QUESTION: {question}
DATE: {datetime.now().strftime('%B %d, %Y')}

REAL MARKET DATA FOR 5 CANDIDATES:
{market_context}

SEC FILING EXCERPTS:
{sec_context}

Based on the REAL DATA above:
1. Rank all 5 companies against the user's criteria
2. Select the TOP 3 that best match
3. Explain why each was selected using specific numbers from the real data
4. Briefly explain why the other 2 were not selected

Generate in markdown format:

# Investment Recommendations
*Generated by {APP_NAME} · {datetime.now().strftime('%B %d, %Y')}*

## Top 3 Recommendations
[For each of top 3 — explain with real numbers why it fits the user's criteria]

## Why we excluded the others
[Brief explanation for the 2 not selected — based on real data]

## Summary Comparison (Top 3)
| Metric | [ticker1] | [ticker2] | [ticker3] |
|--------|-----------|-----------|-----------|
| Price | | | |
| P/E Ratio | | | |
| RSI | | | |
| Dividend Yield | | | |
| Analyst View | | | |

## Next Steps
Ask me for a detailed analysis of any of these companies.

*This is AI-generated for educational purposes only. Not financial advice.*"""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    print(f"  [handle_discovery] Step 3 — Recommendation generated for top 3 from {candidate_tickers}")
    return {"answer": answer, "messages": updated_messages}


# ─────────────────────────────────────────────
# Node: Handle Comparison
# ─────────────────────────────────────────────
def handle_comparison(state: AgentState) -> dict:
    """
    Handles comparison questions between two or more companies.
    Retrieves market data and SEC chunks for each ticker.
    """
    question = state["question"]
    tickers  = state.get("tickers") or []
    messages = state.get("messages") or []

    if not tickers:
        return handle_out_of_scope(state)

    # Collect market data for each ticker
    all_market_data = {}
    for t in tickers:
        data = get_stock_data(t)
        if data:
            all_market_data[t] = data

    # Collect SEC chunks for each ticker
    all_chunks = {}
    for t in tickers:
        if check_ticker_exists(t):
            chunks = retrieve(question, t, top_k=3)
        else:
            chunks = fetch_embed_store_retrieve(question, t, top_k=3)
        all_chunks[t] = chunks

    # Format market data for prompt
    market_context = ""
    for t, md in all_market_data.items():
        market_context += f"\n{t}:\n"
        market_context += f"  Company: {md.get('company_name')}\n"
        market_context += f"  Price: ${md.get('current_price')} | P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}\n"
        market_context += f"  Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}\n"
        market_context += f"  RSI: {md.get('rsi')} | MACD: {md.get('macd')}\n"
        market_context += f"  Analyst Target Mean: ${md.get('target_mean')} | Recommendation: {md.get('recommendation')}\n"
        market_context += f"  EPS: {md.get('eps_trailing')} | Dividend Yield: {md.get('dividend_yield')}\n"

    # Format SEC chunks for prompt
    sec_context = ""
    for t, chunks in all_chunks.items():
        sec_context += f"\n{t} SEC Filing:\n"
        for chunk in chunks:
            sec_context += chunk.get("text", "")[:300] + "\n"

    prompt = f"""You are {APP_NAME}, a professional AI investment research analyst.
Generate a detailed comparison report between these companies.

USER QUESTION: {question}
COMPANIES: {', '.join(tickers)}
DATE: {datetime.now().strftime('%B %d, %Y')}

MARKET DATA:
{market_context}

SEC FILING EXCERPTS:
{sec_context}

Generate a structured comparison report in markdown format:

# ⚖️ {' vs '.join(tickers)} — Comparison Analysis
*Generated by {APP_NAME} · {datetime.now().strftime('%B %d, %Y')}*

---

## 💡 Verdict
[Which company wins for the user's specific criteria and why — be direct]

---

## 📊 Side-by-Side Comparison
| Metric | {' | '.join(tickers)} |
|--------|{'|'.join(['--------' for _ in tickers])}|
| Price | [values] |
| P/E Ratio | [values] |
| Forward P/E | [values] |
| Revenue | [values] |
| Profit Margin | [values] |
| RSI | [values] |
| Analyst Target | [values] |
| Recommendation | [values] |

---

## 📈 Technical Comparison
[Compare RSI, MACD, price performance for each company]

---

## ⚠️ Key Risks
[Main risks for each company from SEC filings]

---

## 📎 Evidence & Sources (XAI)
**Data sources used:**
- Market data: yfinance
- SEC filings: {', '.join([f'{t}_10-K' for t in tickers])}

---

## ⚠️ Disclaimer
*This report is AI-generated for educational purposes only. Not financial advice.*"""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    print(f"  [handle_comparison] Comparison report generated for {tickers}")
    return {"answer": answer, "messages": updated_messages}
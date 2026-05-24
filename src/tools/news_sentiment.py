import requests
from datetime import datetime, timedelta
from transformers import pipeline
from config import FINLIGHT_API_KEY

# Load FinBERT sentiment model once at module level
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)


def get_news_and_sentiment(ticker: str, max_articles: int = 10, days_back: int = 30) -> list:
    """
    Fetches recent news for a ticker from finlight.me v2 API.
    Scores sentiment using FinBERT (financial domain model).
    Returns a list of articles with sentiment scores and URLs.

    NOTE: Currently using Finlight free tier (Launchpad plan).
    Free tier has 12-hour news delay and no real-time access.
    Upgrade to Pro Light ($29/month) for real-time news.
    Upgrade to Pro Standard ($99/month) for real-time + built-in sentiment.
    """
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        url = "https://api.finlight.me/v2/articles"
        headers = {
            "X-API-KEY":    FINLIGHT_API_KEY,
            "Content-Type": "application/json",
        }
        body = {
            "query":    ticker,
            "tickers":  [ticker],
            "language": "en",
            "from":     date_from,
            "pageSize": max_articles,
        }
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])

    except Exception as e:
        print(f"  [news_sentiment] Error fetching news: {e}")
        return []

    results = []
    for article in articles:
        title    = article.get("title", "")
        summary  = article.get("summary", "") or ""
        link     = article.get("link", "")
        published = article.get("publishDate", "")

        # Score sentiment using FinBERT
        text = f"{title}. {summary}"[:512]
        try:
            result = finbert(text)[0]
            label  = result["label"]
            score  = round(result["score"], 4)
        except Exception:
            label = "neutral"
            score = 0.0

        results.append({
            "title":     title,
            "url":       link,
            "published": published,
            "summary":   summary,
            "sentiment": label,
            "score":     score,
        })

    print(f"  [news_sentiment] {len(results)} articles fetched for {ticker} (last {days_back} days)")
    return results
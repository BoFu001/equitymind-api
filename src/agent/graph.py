from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    extract_parameters,
    retrieve_sec_data,
    get_market_data,
    get_news,
    generate_report,
    handle_out_of_scope,
    handle_greeting,
    handle_comparison,
    handle_no_ticker,
    discovery_suggest,
    discovery_report,
)


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "OUT_OF_SCOPE":
        return "out_of_scope"
    elif intent == "GREETING":
        return "greeting"
    elif intent == "DISCOVERY":
        return "discovery_suggest"
    else:
        return "extract"  # COMPARISON, SPECIFIC_STOCK

def route_after_extract(state: AgentState) -> str:
    """Routes after Node extract based on intent and ticker availability."""
    intent = state.get("intent", "")
    tickers = state.get("tickers") or []

    if not tickers:
        return "no_ticker"
    if intent == "COMPARISON":
        return "comparison"
    return "retrieve_sec"

def route_after_news(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "DISCOVERY":
        return "discovery_report"
    else:
        return "report"
    
# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify",          classify_intent)
    graph.add_node("extract",           extract_parameters)
    graph.add_node("retrieve_sec",      retrieve_sec_data)
    graph.add_node("market_data",       get_market_data)
    graph.add_node("news",              get_news)
    graph.add_node("report",            generate_report)
    graph.add_node("out_of_scope",      handle_out_of_scope)
    graph.add_node("greeting",          handle_greeting)
    graph.add_node("discovery_suggest", discovery_suggest)
    graph.add_node("discovery_report",  discovery_report)
    graph.add_node("comparison",        handle_comparison)
    graph.add_node("no_ticker",         handle_no_ticker) 

    # Entry point
    graph.set_entry_point("classify")

    # Conditional edge after Node classify_intent
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "out_of_scope":      "out_of_scope",
            "greeting":          "greeting",
            "discovery_suggest": "discovery_suggest",
            "extract":           "extract",
        }
    )

    # Conditional edge after Node extract_parameters
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "no_ticker":      "no_ticker",
            "comparison":     "comparison",
            "retrieve_sec":   "retrieve_sec",
        }
    )

    graph.add_conditional_edges(
        "news",
        route_after_news,
        {
            "report":           "report",
            "discovery_report": "discovery_report",
        }
    )

    # Linear flow after market_data
    graph.add_edge("discovery_suggest", "retrieve_sec")
    graph.add_edge("retrieve_sec",      "market_data")
    graph.add_edge("market_data",       "news")

    # End nodes
    graph.add_edge("report",            END)
    graph.add_edge("discovery_report",  END)
    graph.add_edge("out_of_scope",      END)
    graph.add_edge("greeting",          END)
    graph.add_edge("comparison",        END)
    graph.add_edge("no_ticker",         END) 

    return graph.compile()


# Compile once at module level
equitymind_graph = build_graph()
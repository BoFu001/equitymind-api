from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    extract_parameters,
    check_pinecone,
    retrieve_chunks,
    fetch_and_retrieve,
    get_market_data,
    get_news,
    generate_report,
    handle_out_of_scope,
    handle_greeting,
    handle_portfolio,
    handle_comparison,
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
    elif intent == "PORTFOLIO":
        return "portfolio"
    else:
        return "extract"  # COMPARISON, SPECIFIC_STOCK, ANALYZE_POSITION, ANALYZE_PORTFOLIO all go through extract first

def route_after_extract(state: AgentState) -> str:
    """Routes after Node 2 based on intent."""
    intent = state.get("intent", "")
    if intent == "COMPARISON":
        return "comparison"
    else:
        return "check_pinecone"

def route_data_status(state: AgentState) -> str:
    """Routes after Node 3 based on data_status."""
    status = state.get("data_status", "")
    if status == "RETRIEVE":
        return "retrieve"
    elif status == "FETCH_NEEDED":
        return "fetch"
    else:
        return "out_of_scope"  # NO_TICKER


# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify",      classify_intent)
    graph.add_node("extract",       extract_parameters)
    graph.add_node("check_pinecone", check_pinecone)
    graph.add_node("retrieve",      retrieve_chunks)
    graph.add_node("fetch",         fetch_and_retrieve)
    graph.add_node("market_data",   get_market_data)
    graph.add_node("news",          get_news)
    graph.add_node("report",        generate_report)
    graph.add_node("out_of_scope",  handle_out_of_scope)
    graph.add_node("greeting",      handle_greeting)
    graph.add_node("portfolio",     handle_portfolio)
    graph.add_node("comparison",    handle_comparison)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional edge after Node 1
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "out_of_scope": "out_of_scope",
            "greeting":     "greeting",
            "portfolio":    "portfolio",
            "extract":      "extract",
        }
    )

    # Conditional edge after Node 2
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "comparison":    "comparison",
            "check_pinecone": "check_pinecone",
        }
    )

    # Conditional edge after Node 3
    graph.add_conditional_edges(
        "check_pinecone",
        route_data_status,
        {
            "retrieve":    "retrieve",
            "fetch":       "fetch",
            "out_of_scope": "out_of_scope",
        }
    )

    # Both retrieve and fetch lead to market_data
    graph.add_edge("retrieve",   "market_data")
    graph.add_edge("fetch",      "market_data")

    # Linear flow after market_data
    graph.add_edge("market_data", "news")
    graph.add_edge("news",        "report")

    # End nodes
    graph.add_edge("report",       END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("greeting",     END)
    graph.add_edge("portfolio",    END)
    graph.add_edge("comparison",   END)

    return graph.compile()


# Compile once at module level
equitymind_graph = build_graph()
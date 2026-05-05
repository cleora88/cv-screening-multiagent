from __future__ import annotations

from typing import Any


try:
    from langgraph.graph import END, StateGraph
    _LANGGRAPH_AVAILABLE = True
except Exception:
    _LANGGRAPH_AVAILABLE = False


def is_langgraph_available() -> bool:
    return _LANGGRAPH_AVAILABLE


def build_stretch_graph() -> Any:
    """Build a minimal stretch-goal LangGraph workflow scaffold.

    This graph is intentionally simple and serves as a foundation for
    future advanced orchestration (branching, retries, guardrails).
    """
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is not installed. Use Python 3.11/3.12 and pip install -r requirements.txt")

    graph = StateGraph(dict)

    def start_node(state: dict) -> dict:
        state["status"] = "started"
        return state

    def finish_node(state: dict) -> dict:
        state["status"] = "finished"
        return state

    graph.add_node("start", start_node)
    graph.add_node("finish", finish_node)
    graph.set_entry_point("start")
    graph.add_edge("start", "finish")
    graph.add_edge("finish", END)

    return graph.compile()

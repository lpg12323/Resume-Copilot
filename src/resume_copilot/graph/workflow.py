"""LangGraph StateGraph construction for Resume Copilot.

This module wires the individual nodes together into a cohesive workflow
and supports optional checkpointing via LangGraph's checkpointer interface.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from resume_copilot.graph.nodes import (
    analyzer_node,
    exporter_node,
    parser_node,
    rewriter_node,
    router_node,
)
from resume_copilot.graph.state import AgentState
from resume_copilot.logger import logger


def _route_entry(state: AgentState) -> str:
    """Decide whether to parse a new PDF or route an existing conversation."""
    if not state.get("resume_raw_text"):
        logger.debug("[_route_entry] resume_raw_text empty -> parser")
        return "parser"
    logger.debug("[_route_entry] resume_raw_text present -> router")
    return "router"


def _route_after_router(state: AgentState) -> str:
    """Route from router_node based on the classified user intent."""
    next_node = state.get("next_node", "chat")
    if next_node in ("rewrite", "export", "chat"):
        logger.debug("[_route_after_router] next_node={} -> {}", next_node, next_node)
        return next_node
    logger.warning("[_route_after_router] unknown next_node '{}', defaulting to chat", next_node)
    return "chat"


def build_resume_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Build and compile the Resume Copilot LangGraph workflow.

    The workflow supports two main modes:

    1. **Initial ingestion**: when ``resume_raw_text`` is empty, the graph
       parses the PDF and analyzes it against the target JD.
    2. **Follow-up interaction**: when ``resume_raw_text`` already exists,
       the graph routes the latest user message to rewrite, export, or chat.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``MemorySaver``)
            for persisting state across turns.

    Returns:
        The compiled LangGraph application (a ``CompiledStateGraph``).
    """
    logger.info("Building Resume Copilot StateGraph")

    builder = StateGraph(AgentState)

    builder.add_node("parser", parser_node)
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("router", router_node)
    builder.add_node("rewriter", rewriter_node)
    builder.add_node("exporter", exporter_node)

    # Entry routing: parse new resume or handle follow-up message.
    builder.add_conditional_edges(
        START,
        _route_entry,
        {
            "parser": "parser",
            "router": "router",
        },
    )

    # Sequential parsing and analysis for new resumes.
    builder.add_edge("parser", "analyzer")
    builder.add_edge("analyzer", END)

    # Intent-based routing for follow-up messages.
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "rewrite": "rewriter",
            "export": "exporter",
            "chat": END,
        },
    )
    builder.add_edge("rewriter", END)
    builder.add_edge("exporter", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("StateGraph compiled successfully")
    return compiled

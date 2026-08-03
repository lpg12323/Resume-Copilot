"""LangGraph state definition for Resume Copilot."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state object that flows through the LangGraph workflow.

    Each node receives the current ``AgentState`` and returns a dictionary
    containing only the fields it wishes to update. LangGraph merges these
    partial updates into the full state before passing it to the next node.
    """

    pdf_path: str
    """Filesystem path to the uploaded PDF resume."""

    target_jd: str
    """Target job description text provided by the user."""

    resume_raw_text: str
    """Plain text extracted from the PDF resume."""

    analysis_report: dict | None
    """Structured analysis report as a plain dictionary (``AnalysisReport.model_dump()``)."""

    resume_markdown: str
    """Current version of the resume in Markdown format."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history between the user and the agent.

    The ``add_messages`` reducer appends new messages instead of replacing
    the entire list, which is required for multi-turn dialogue.
    """

    next_node: str
    """Routing decision produced by ``router_node``.

    Expected values:
        - ``"rewrite"``: user wants to modify the resume.
        - ``"export"``: user wants to export the current Markdown resume.
        - ``"chat"``: user is asking a question or the intent is unclear.
    """

    error: str | None
    """Optional error message populated when a node fails unexpectedly."""

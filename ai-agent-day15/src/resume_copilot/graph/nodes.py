"""LangGraph node implementations for Resume Copilot.

Each function accepts the current ``AgentState`` and returns a dictionary
with the fields it wishes to update. Exceptions are caught and surfaced
through the ``error`` state field so the graph can degrade gracefully.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from resume_copilot.analyzers import AnalysisError, ResumeAnalyzer
from resume_copilot.analyzers.resume_analyzer import _extract_json
from resume_copilot.config import get_settings
from resume_copilot.graph.state import AgentState
from resume_copilot.logger import logger
from resume_copilot.models import AnalysisReport
from resume_copilot.parsers import PDFParseError, ResumePDFParser


class RouterDecision(BaseModel):
    """Structured routing decision produced by ``router_node``."""

    next_node: str = Field(
        ...,
        pattern=r"^(rewrite|export|chat)$",
        description=(
            "The next node to execute. Must be one of: "
            "'rewrite' (modify resume), 'export' (save markdown), "
            "or 'chat' (answer a question)."
        ),
    )


ROUTER_SYSTEM_PROMPT = (
    "You are an intent classifier for a resume optimization assistant. "
    "Based on the user's latest message, decide the next action:\n"
    "- 'rewrite': the user wants to modify, improve, or rewrite part of the resume.\n"
    "- 'export': the user wants to save or export the current resume as Markdown.\n"
    "- 'chat': the user is asking a question, greeting, or the intent is unclear.\n\n"
    "Respond only with a valid JSON object containing the label."
)

REWRITER_SYSTEM_PROMPT = (
    "You are an expert resume editor specializing in AI/Agent technical roles. "
    "Your task is to rewrite the candidate's Markdown resume according to the "
    "user's specific request and the structured analysis report.\n\n"
    "Rules:\n"
    "1. Preserve the overall Markdown structure (headings, lists, sections).\n"
    "2. Make targeted changes only where requested; do not invent new experiences.\n"
    "3. Use strong action verbs, quantifiable results, and AI-Agent terminology.\n"
    "4. If the analysis report highlights missing keywords relevant to the request, "
    "   incorporate them naturally.\n"
    "5. Return the full updated Markdown resume."
)


def _get_chat_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Create a ChatOpenAI instance from application settings."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
        temperature=temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
    )


def parser_node(state: AgentState) -> dict:
    """Parse the PDF at ``pdf_path`` and populate ``resume_raw_text``.

    Returns:
        A partial state update with ``resume_raw_text`` and optionally
        ``error`` if parsing fails.
    """
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        return {"error": "pdf_path is missing from state"}

    logger.info("[parser_node] Parsing PDF: {}", pdf_path)
    try:
        text = ResumePDFParser().parse_pdf(pdf_path)
    except PDFParseError as exc:
        logger.error("[parser_node] PDF parsing failed: {}", exc)
        return {"error": str(exc)}
    except Exception as exc:
        logger.error("[parser_node] Unexpected error: {}", exc)
        return {"error": f"Unexpected error parsing PDF: {exc}"}

    logger.info("[parser_node] Extracted {} characters", len(text))
    return {"resume_raw_text": text}


def analyzer_node(state: AgentState) -> dict:
    """Analyze ``resume_raw_text`` against ``target_jd``.

    Returns:
        A partial state update with ``analysis_report`` and
        ``resume_markdown`` (rendered from the report).
    """
    resume_text = state.get("resume_raw_text", "")
    target_jd = state.get("target_jd", "")

    if not resume_text.strip():
        return {"error": "resume_raw_text is empty; cannot analyze"}
    if not target_jd.strip():
        return {"error": "target_jd is empty; cannot analyze"}

    logger.info("[analyzer_node] Analyzing resume against JD")
    try:
        report = ResumeAnalyzer().analyze(resume_text, target_jd)
    except AnalysisError as exc:
        logger.error("[analyzer_node] Analysis failed: {}", exc)
        return {"error": str(exc)}
    except Exception as exc:
        logger.error("[analyzer_node] Unexpected error: {}", exc)
        return {"error": f"Unexpected error during analysis: {exc}"}

    logger.info("[analyzer_node] Analysis complete: score={}", report.match_score)
    return {
        "analysis_report": report.model_dump(),
        "resume_markdown": report.to_markdown(),
    }


def router_node(state: AgentState) -> dict:
    """Classify the latest user message and set ``next_node``.

    Returns:
        A partial state update with ``next_node``.
    """
    messages: list[BaseMessage] = state.get("messages", [])
    if not messages:
        logger.warning("[router_node] No messages in state; defaulting to chat")
        return {"next_node": "chat"}

    # Find the most recent HumanMessage.
    last_user_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            last_user_message = message
            break

    if last_user_message is None:
        logger.warning("[router_node] No HumanMessage found; defaulting to chat")
        return {"next_node": "chat"}

    content = str(last_user_message.content)
    logger.info("[router_node] Routing user message: {!r}", content[:80])

    # Fast-path keyword heuristics to reduce LLM calls for obvious intents.
    lowered = content.lower()
    if any(keyword in lowered for keyword in ("导出", "export", "保存", "save", "下载", "download")):
        logger.info("[router_node] Keyword match: export")
        return {"next_node": "export"}
    if any(
        keyword in lowered
        for keyword in (
            "改写",
            "重写",
            "优化",
            "optimize",
            "rewrite",
            "修改",
            "改",
            "提升",
            "improve",
        )
    ):
        logger.info("[router_node] Keyword match: rewrite")
        return {"next_node": "rewrite"}

    try:
        settings = get_settings()
        llm = _get_chat_llm(temperature=0.1)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("user", "User message: {message}\n\nClassify the intent."),
            ]
        )

        if settings.structured_output_method == "json_schema":
            chain = prompt | llm.with_structured_output(RouterDecision)
            decision: RouterDecision = chain.invoke({"message": content})
        else:
            # json_mode fallback for providers like DeepSeek.
            schema = json.dumps(RouterDecision.model_json_schema(), ensure_ascii=False, indent=2)
            # Escape braces so LangChain treats the JSON schema as literal text.
            escaped_schema = schema.replace("{", "{{").replace("}", "}}")
            schema_prompt = (
                f"{ROUTER_SYSTEM_PROMPT}\n\n"
                "You must respond with a single valid JSON object matching this schema:\n"
                f"```json\n{escaped_schema}\n```\n\n"
                "Do not include any text outside the JSON object."
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", schema_prompt),
                    ("user", "User message: {message}\n\nClassify the intent."),
                ]
            )
            response = (prompt | llm).invoke({"message": content})
            data = _extract_json(str(response.content))
            decision = RouterDecision(**data)
    except Exception as exc:
        logger.error("[router_node] LLM routing failed: {}", exc)
        return {"next_node": "chat"}

    logger.info("[router_node] LLM decision: {}", decision.next_node)
    return {"next_node": decision.next_node}


def rewriter_node(state: AgentState) -> dict:
    """Rewrite the Markdown resume based on the user's request.

    Returns:
        A partial state update with the new ``resume_markdown`` and an
        ``AIMessage`` confirming the change.
    """
    resume_markdown = state.get("resume_markdown", "")
    analysis_report = state.get("analysis_report") or {}
    messages: list[BaseMessage] = state.get("messages", [])

    # Extract the latest user request that triggered rewriting.
    last_user_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            last_user_message = message
            break

    if last_user_message is None:
        return {"error": "rewriter_node requires a user message"}

    user_request = str(last_user_message.content)

    if not resume_markdown.strip():
        return {"error": "resume_markdown is empty; cannot rewrite"}

    logger.info("[rewriter_node] Rewriting resume based on request: {!r}", user_request[:80])

    try:
        llm = _get_chat_llm(temperature=0.3)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REWRITER_SYSTEM_PROMPT),
                (
                    "user",
                    "## User Request\n{request}\n\n"
                    "## Analysis Report\n{analysis}\n\n"
                    "## Current Resume Markdown\n{resume}\n\n"
                    "Please return the full updated Markdown resume.",
                ),
            ]
        )
        chain = prompt | llm
        response = chain.invoke(
            {
                "request": user_request,
                "analysis": _format_analysis_for_prompt(analysis_report),
                "resume": resume_markdown,
            }
        )
        new_markdown = str(response.content)
    except Exception as exc:
        logger.error("[rewriter_node] Rewrite failed: {}", exc)
        return {"error": f"Failed to rewrite resume: {exc}"}

    logger.info("[rewriter_node] Rewrite complete: {} characters", len(new_markdown))
    return {
        "resume_markdown": new_markdown,
        "messages": [
            AIMessage(
                content="已根据您的要求更新简历。您可以继续修改或选择导出 Markdown。"
            )
        ],
    }


def exporter_node(state: AgentState) -> dict:
    """Write the current ``resume_markdown`` to the outputs directory.

    Returns:
        A partial state update with an ``AIMessage`` containing the
        output file path, or ``error`` if writing fails.
    """
    resume_markdown = state.get("resume_markdown", "")
    if not resume_markdown.strip():
        return {"error": "resume_markdown is empty; nothing to export"}

    settings = get_settings()
    settings.ensure_directories()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(settings.outputs_dir) / f"resume_{timestamp}.md"

    logger.info("[exporter_node] Exporting resume to {}", output_path)
    try:
        output_path.write_text(resume_markdown, encoding="utf-8")
    except Exception as exc:
        logger.error("[exporter_node] Failed to write file: {}", exc)
        return {"error": f"Failed to export resume: {exc}"}

    logger.info("[exporter_node] Export successful")
    return {
        "messages": [
            AIMessage(
                content=f"简历已导出至：{output_path}\n\n{resume_markdown[:500]}..."
            )
        ],
    }


def _format_analysis_for_prompt(analysis_report: dict) -> str:
    """Convert the analysis report dict into a concise prompt-ready string."""
    if not analysis_report:
        return "无分析报告"

    lines = [
        f"Match Score: {analysis_report.get('match_score', 'N/A')}",
        f"Summary: {analysis_report.get('summary', '')}",
        "Missing Keywords: " + ", ".join(analysis_report.get("missing_keywords", [])),
    ]
    return "\n".join(lines)

"""LLM-based resume analyzer with structured output."""

import json
import re
from typing import Any

from json_repair import repair_json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from resume_copilot.config import get_settings
from resume_copilot.logger import logger
from resume_copilot.models.analysis import AnalysisReport


class AnalysisError(Exception):
    """Raised when the LLM fails to produce a valid structured analysis."""


_SYSTEM_PROMPT = (
    "You are a strict senior AI-Agent technical expert and interviewer. "
    "Your task is to compare a candidate's resume with a target job description (JD) "
    "and produce a structured analysis report.\n\n"
    "Evaluation criteria:\n"
    "1. Match the resume against the JD's required skills, experience level, "
    "   and domain knowledge (especially LLM, Agent, RAG, LangGraph, LangChain).\n"
    "2. Identify concrete strengths with evidence from the resume.\n"
    "3. Identify weaknesses and gaps relative to the JD.\n"
    "4. List important technical keywords from the JD that are missing in the resume.\n"
    "5. For each project experience, provide specific optimization advice using the "
    "   STAR methodology (Situation, Task, Action, Result) and professional AI-Agent "
    "   terminology. Highlight quantifiable business/technical impact where possible.\n\n"
    "Be objective, specific, and concise. Do not invent facts that are not present "
    "in the resume."
)

_USER_PROMPT = (
    "## Target Job Description\n"
    "{target_jd}\n\n"
    "## Resume Content\n"
    "{resume_text}\n\n"
    "Please generate the structured analysis report as valid JSON according to the required schema."
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output, tolerating markdown code blocks.

    Uses balanced-brace scanning so nested objects are not truncated. If the
    extracted text is not valid JSON, ``json_repair`` is used as a fallback.
    """
    text = text.strip()

    candidate = _find_json_candidate(text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        repaired = repair_json(candidate)
        return json.loads(repaired)
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON from model output: {exc}") from exc


def _find_json_candidate(text: str) -> str:
    """Locate the first balanced JSON object in ``text``.

    Prefers content inside a Markdown ``json`` code block, then falls back to
    the first top-level balanced ``{...}`` object.
    """
    # Try to find JSON inside a markdown code block.
    code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)

    # Fall back to balanced brace scanning.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]}...")

    depth = 0
    in_string = False
    escape_next = False
    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError(f"No balanced JSON object found in model output: {text[:200]}...")


class ResumeAnalyzer:
    """Analyze a resume against a target JD and return a structured report."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Initialize the analyzer.

        Args:
            llm: Optional pre-configured chat model. If not provided, a ChatOpenAI
                instance is created from application settings.
        """
        self._logger = logger.bind(component="ResumeAnalyzer")
        self._llm = llm or self._build_default_llm()
        self._chain = self._build_chain(self._llm)

    def _build_default_llm(self) -> BaseChatModel:
        """Build a ChatOpenAI instance from application configuration."""
        settings = get_settings()
        self._logger.debug(
            "Building ChatOpenAI(model={}, base_url={}, temperature={})",
            settings.model_name,
            settings.openai_base_url,
            settings.llm_temperature,
        )
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )

    def _build_chain(self, llm: BaseChatModel) -> Any:
        """Build the prompt + structured-output chain."""
        settings = get_settings()
        self._logger.debug(
            "Using structured_output_method={} for model {}",
            settings.structured_output_method,
            settings.model_name,
        )

        if settings.structured_output_method == "json_schema":
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("user", _USER_PROMPT),
                ]
            )
            return prompt | llm.with_structured_output(AnalysisReport)

        # json_mode fallback: embed the schema in the prompt and parse manually.
        # This is more robust for providers like DeepSeek that do not support
        # json_schema response_format.
        schema = json.dumps(AnalysisReport.model_json_schema(), ensure_ascii=False, indent=2)
        # Escape braces so LangChain treats the JSON schema as literal text.
        escaped_schema = schema.replace("{", "{{").replace("}", "}}")
        schema_prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            "You must respond with a single valid JSON object matching this schema:\n"
            f"```json\n{escaped_schema}\n```\n\n"
            "Do not include any text outside the JSON object."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", schema_prompt),
                ("user", _USER_PROMPT),
            ]
        )
        return prompt | llm | self._parse_llm_response

    def _parse_llm_response(self, response: BaseMessage) -> AnalysisReport:
        """Parse the raw LLM response into an ``AnalysisReport``."""
        content = str(response.content)
        self._logger.debug("Parsing LLM response: {} chars", len(content))
        try:
            data = _extract_json(content)
        except ValueError as exc:
            self._logger.error("JSON extraction failed: {}", exc)
            raise AnalysisError(f"Failed to extract JSON from LLM output: {exc}") from exc

        try:
            return AnalysisReport(**data)
        except ValidationError as exc:
            self._logger.error("Structured output validation failed: {}", exc)
            raise AnalysisError(f"LLM output did not match AnalysisReport schema: {exc}") from exc

    def analyze(self, resume_text: str, target_jd: str) -> AnalysisReport:
        """Run the structured analysis.

        Args:
            resume_text: Plain text extracted from the candidate's resume.
            target_jd: Plain text of the target job description.

        Returns:
            A validated ``AnalysisReport`` instance.

        Raises:
            AnalysisError: If the LLM fails to return a valid structured output.
        """
        if not resume_text.strip():
            raise AnalysisError("resume_text cannot be empty")
        if not target_jd.strip():
            raise AnalysisError("target_jd cannot be empty")

        self._logger.info(
            "Analyzing resume ({} chars) against JD ({} chars)",
            len(resume_text),
            len(target_jd),
        )

        try:
            report: AnalysisReport = self._chain.invoke(
                {
                    "resume_text": resume_text,
                    "target_jd": target_jd,
                }
            )
        except AnalysisError:
            raise
        except ValidationError as exc:
            self._logger.error("Structured output validation failed: {}", exc)
            raise AnalysisError(f"LLM output did not match AnalysisReport schema: {exc}") from exc
        except Exception as exc:
            self._logger.error("Analysis failed: {}", exc)
            raise AnalysisError(f"Failed to analyze resume: {exc}") from exc

        self._logger.info(
            "Analysis completed: match_score={}",
            report.match_score,
        )
        return report

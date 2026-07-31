"""LLM-based resume analyzer with structured output."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
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
    "Please generate the structured analysis report according to the required schema."
)


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
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("user", _USER_PROMPT),
            ]
        )
        return prompt | llm.with_structured_output(AnalysisReport)

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

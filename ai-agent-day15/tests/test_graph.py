"""End-to-end integration tests for the LangGraph workflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fpdf import FPDF
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from resume_copilot.graph import build_resume_graph


def _make_pdf_with_text(tmp_path: Path, filename: str, text: str) -> str:
    """Create a minimal PDF file containing the given text."""
    pdf_path = tmp_path / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(str(pdf_path))

    return str(pdf_path)


@pytest.fixture
def sample_resume_path(tmp_path: Path) -> str:
    """Provide a temporary PDF resume file."""
    return _make_pdf_with_text(
        tmp_path,
        "sample_resume.pdf",
        "Senior Python Engineer with LangGraph and RAG experience.",
    )


@pytest.fixture
def mock_settings(tmp_path: Path) -> MagicMock:
    """Provide a settings object that writes outputs to tmp_path."""
    settings = MagicMock()
    settings.outputs_dir = tmp_path / "outputs"
    settings.ensure_directories = lambda: settings.outputs_dir.mkdir(
        parents=True, exist_ok=True
    )
    return settings


class TestResumeGraphWorkflow:
    """E2E tests covering the full graph lifecycle with MemorySaver."""

    def test_phase1_parse_and_analyze(
        self, sample_resume_path: str, mock_settings: MagicMock
    ) -> None:
        """Phase 1: START -> parser -> analyzer -> END."""
        with (
            patch("resume_copilot.graph.workflow.parser_node") as mock_parser,
            patch("resume_copilot.graph.workflow.analyzer_node") as mock_analyzer,
            patch("resume_copilot.graph.nodes.get_settings", return_value=mock_settings),
        ):
            mock_parser.return_value = {
                "resume_raw_text": "Senior Python Engineer with LangGraph and RAG.",
            }
            mock_analyzer.return_value = {
                "analysis_report": {
                    "match_score": 82,
                    "summary": "Strong match for AI Agent roles.",
                    "strengths": ["LangGraph experience"],
                    "weaknesses": ["No deployment keywords"],
                    "missing_keywords": ["Kubernetes"],
                    "project_suggestions": [],
                },
                "resume_markdown": "# Resume\n\nSenior Python Engineer...",
            }

            graph = build_resume_graph(checkpointer=MemorySaver())
            thread_id = "test-phase1"
            config = {"configurable": {"thread_id": thread_id}}

            result = graph.invoke(
                {
                    "pdf_path": sample_resume_path,
                    "target_jd": "AI Agent Engineer with LangGraph and Kubernetes.",
                    "messages": [],
                },
                config,
            )

        assert result["resume_raw_text"] == "Senior Python Engineer with LangGraph and RAG."
        assert result["analysis_report"]["match_score"] == 82
        assert result["resume_markdown"] == "# Resume\n\nSenior Python Engineer..."
        mock_parser.assert_called_once()
        mock_analyzer.assert_called_once()

    def test_phase2_rewrite_and_phase3_export(
        self, sample_resume_path: str, mock_settings: MagicMock
    ) -> None:
        """Phase 2: router -> rewriter. Phase 3: router -> exporter."""
        with (
            patch("resume_copilot.graph.workflow.parser_node") as mock_parser,
            patch("resume_copilot.graph.workflow.analyzer_node") as mock_analyzer,
            patch("resume_copilot.graph.workflow.router_node") as mock_router,
            patch("resume_copilot.graph.workflow.rewriter_node") as mock_rewriter,
            patch("resume_copilot.graph.nodes.get_settings", return_value=mock_settings),
        ):
            mock_parser.return_value = {
                "resume_raw_text": "Senior Python Engineer with LangGraph and RAG.",
            }
            mock_analyzer.return_value = {
                "analysis_report": {
                    "match_score": 82,
                    "summary": "Strong match.",
                    "strengths": [],
                    "weaknesses": [],
                    "missing_keywords": [],
                    "project_suggestions": [],
                },
                "resume_markdown": "# Original Resume",
            }
            mock_router.return_value = {"next_node": "rewrite"}
            mock_rewriter.return_value = {
                "resume_markdown": "# Improved Resume\n\nUsed STAR methodology.",
                "messages": [AIMessage(content="Resume rewritten.")],
            }

            graph = build_resume_graph(checkpointer=MemorySaver())
            thread_id = "test-phase2-3"
            config = {"configurable": {"thread_id": thread_id}}

            # Phase 1: initial ingestion.
            graph.invoke(
                {
                    "pdf_path": sample_resume_path,
                    "target_jd": "AI Agent Engineer.",
                    "messages": [],
                },
                config,
            )

            # Phase 2: rewrite request.
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="用 STAR 法则优化项目2")],
                },
                config,
            )

            assert result["resume_markdown"] == "# Improved Resume\n\nUsed STAR methodology."
            mock_rewriter.assert_called_once()

            # Phase 3: export request.
            mock_router.return_value = {"next_node": "export"}
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="导出简历")],
                },
                config,
            )

            assert any(
                "简历已导出至" in msg.content
                for msg in result["messages"]
                if isinstance(msg, AIMessage)
            )

        exported_files = list(mock_settings.outputs_dir.glob("resume_*.md"))
        assert len(exported_files) == 1
        assert exported_files[0].read_text(encoding="utf-8") == result["resume_markdown"]

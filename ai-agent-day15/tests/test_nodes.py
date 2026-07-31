"""Unit tests for LangGraph node functions."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from resume_copilot.graph.nodes import (
    RouterDecision,
    analyzer_node,
    exporter_node,
    parser_node,
    rewriter_node,
    router_node,
)
from resume_copilot.graph.state import AgentState
from resume_copilot.models import AnalysisReport


class TestParserNode:
    """Tests for parser_node."""

    def test_parser_node_updates_resume_raw_text(self) -> None:
        state: AgentState = {
            "pdf_path": "/tmp/resume.pdf",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        mock_parser = MagicMock()
        mock_parser.parse_pdf.return_value = "Parsed resume text"

        with patch("resume_copilot.graph.nodes.ResumePDFParser", return_value=mock_parser):
            result = parser_node(state)

        assert result["resume_raw_text"] == "Parsed resume text"
        mock_parser.parse_pdf.assert_called_once_with("/tmp/resume.pdf")

    def test_parser_node_returns_error_on_missing_path(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        result = parser_node(state)

        assert "error" in result
        assert "pdf_path is missing" in result["error"]


class TestAnalyzerNode:
    """Tests for analyzer_node."""

    def test_analyzer_node_updates_report_and_markdown(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "AI Agent Engineer",
            "resume_raw_text": "Resume text",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        report = AnalysisReport(
            match_score=80,
            summary="Good match",
            strengths=["LangGraph"],
            weaknesses=["Deployment"],
            missing_keywords=["Kubernetes"],
        )
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = report

        with patch("resume_copilot.graph.nodes.ResumeAnalyzer", return_value=mock_analyzer):
            result = analyzer_node(state)

        assert result["analysis_report"] == report.model_dump()
        assert "# Resume Analysis Report" in result["resume_markdown"]
        assert "80/100" in result["resume_markdown"]
        mock_analyzer.analyze.assert_called_once_with("Resume text", "AI Agent Engineer")

    def test_analyzer_node_returns_error_on_empty_resume(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "JD",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        result = analyzer_node(state)

        assert "error" in result
        assert "resume_raw_text is empty" in result["error"]


class TestRouterNode:
    """Tests for router_node intent classification."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("帮我导出简历", "export"),
            ("Please export my resume", "export"),
            ("保存为 Markdown", "export"),
            ("重写项目经历", "rewrite"),
            ("Optimize my experience section", "rewrite"),
            ("把这段改得更业务化", "rewrite"),
        ],
    )
    def test_router_node_keyword_classification(self, message: str, expected: str) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [HumanMessage(content=message)],
            "next_node": "",
            "error": None,
        }

        result = router_node(state)

        assert result["next_node"] == expected

    def test_router_node_llm_fallback(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [HumanMessage(content="What is LangGraph?")],
            "next_node": "",
            "error": None,
        }

        fake_decision = RouterDecision(next_node="chat")
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = RunnableLambda(lambda _inputs: fake_decision)

        with patch("resume_copilot.graph.nodes._get_chat_llm", return_value=fake_llm):
            result = router_node(state)

        assert result["next_node"] == "chat"

    def test_router_node_defaults_to_chat_when_no_messages(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        result = router_node(state)

        assert result["next_node"] == "chat"


class TestRewriterNode:
    """Tests for rewriter_node."""

    def test_rewriter_node_updates_markdown(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": {"match_score": 70, "missing_keywords": ["LangGraph"]},
            "resume_markdown": "# Old Resume",
            "messages": [HumanMessage(content="加入 LangGraph 关键词")],
            "next_node": "",
            "error": None,
        }

        updated_markdown = "# Updated Resume\n\nExperienced with LangGraph."
        fake_llm = RunnableLambda(lambda _inputs: AIMessage(content=updated_markdown))

        with patch("resume_copilot.graph.nodes._get_chat_llm", return_value=fake_llm):
            result = rewriter_node(state)

        assert result["resume_markdown"] == updated_markdown
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "已根据您的要求更新简历" in result["messages"][0].content

    def test_rewriter_node_returns_error_without_user_message(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "# Resume",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        result = rewriter_node(state)

        assert "error" in result
        assert "requires a user message" in result["error"]


class TestExporterNode:
    """Tests for exporter_node."""

    def test_exporter_node_writes_file_and_returns_message(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "# Resume\n\nContent",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        with patch("resume_copilot.graph.nodes.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.outputs_dir = tmp_path
            mock_settings.ensure_directories = MagicMock()
            mock_get_settings.return_value = mock_settings

            result = exporter_node(state)

        assert "error" not in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "简历已导出至" in result["messages"][0].content

        exported_files = list(tmp_path.glob("resume_*.md"))
        assert len(exported_files) == 1
        assert exported_files[0].read_text(encoding="utf-8") == "# Resume\n\nContent"

    def test_exporter_node_returns_error_on_empty_markdown(self) -> None:
        state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }

        result = exporter_node(state)

        assert "error" in result
        assert "resume_markdown is empty" in result["error"]

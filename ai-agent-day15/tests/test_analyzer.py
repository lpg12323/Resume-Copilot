"""Unit tests for the resume analyzer and analysis models."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from resume_copilot.analyzers import AnalysisError, ResumeAnalyzer
from resume_copilot.analyzers.resume_analyzer import _extract_json
from resume_copilot.models import AnalysisReport, ProjectSuggestion


class TestAnalysisReportModel:
    """Pydantic validation tests for AnalysisReport and ProjectSuggestion."""

    def test_project_suggestion_valid(self) -> None:
        suggestion = ProjectSuggestion(
            project_name="AI Agent Orchestration Platform",
            current_issues=["缺少量化业务收益"],
            optimization_advice="使用 STAR 法则重写，突出 RAG 准确率提升 15%。",
        )
        assert suggestion.project_name == "AI Agent Orchestration Platform"

    def test_project_suggestion_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProjectSuggestion(
                project_name="",
                current_issues=["issue"],
                optimization_advice="advice",
            )

    def test_analysis_report_valid(self) -> None:
        report = AnalysisReport(
            match_score=85,
            summary="候选人整体匹配度良好，但在 LLM 微调经验上有缺口。",
            strengths=["熟悉 LangGraph 状态机设计", "具备 RAG 项目经验"],
            weaknesses=["缺少模型微调经验"],
            missing_keywords=["LoRA", "PEFT", "vLLM"],
            project_suggestions=[
                ProjectSuggestion(
                    project_name="RAG Chatbot",
                    current_issues=["描述过于笼统"],
                    optimization_advice="补充检索准确率与延迟数据。",
                )
            ],
        )
        assert report.match_score == 85
        assert len(report.missing_keywords) == 3

    @pytest.mark.parametrize("raw_score,expected", [(-1, 0), (101, 100), (150, 100), (0, 0), (100, 100)])
    def test_analysis_report_score_is_clamped(self, raw_score: int, expected: int) -> None:
        report = AnalysisReport(match_score=raw_score, summary="summary")
        assert report.match_score == expected

    def test_analysis_report_missing_summary_raises(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisReport(match_score=50)

    def test_analysis_report_score_coercion(self) -> None:
        report = AnalysisReport(match_score=72.9, summary="summary")
        assert report.match_score == 72
        assert isinstance(report.match_score, int)

    def test_analysis_report_to_markdown(self) -> None:
        report = AnalysisReport(
            match_score=78,
            summary="整体匹配度中等。",
            strengths=["优势 A"],
            weaknesses=["短板 B"],
            missing_keywords=["keyword"],
        )
        markdown = report.to_markdown()
        assert "# Resume Analysis Report" in markdown
        assert "78/100" in markdown
        assert "keyword" in markdown


class TestResumeAnalyzer:
    """Integration-style tests for ResumeAnalyzer with mocked LLM."""

    def _sample_report(self) -> AnalysisReport:
        return AnalysisReport(
            match_score=82,
            summary="候选人具备良好的 AI Agent 工程经验。",
            strengths=["熟悉 LangGraph"],
            weaknesses=["缺少云端部署经验"],
            missing_keywords=["Kubernetes"],
        )

    def _make_analyzer_with_fake_chain(self, report: AnalysisReport) -> ResumeAnalyzer:
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = report
        with patch.object(ResumeAnalyzer, "_build_chain", return_value=fake_chain):
            return ResumeAnalyzer(llm=MagicMock())

    def test_analyze_returns_structured_report(self) -> None:
        expected = self._sample_report()
        analyzer = self._make_analyzer_with_fake_chain(expected)

        report = analyzer.analyze("resume text", "target jd text")

        assert report.match_score == expected.match_score
        assert report.summary == expected.summary
        assert report.strengths == expected.strengths

    def test_analyze_empty_resume_raises(self) -> None:
        analyzer = self._make_analyzer_with_fake_chain(self._sample_report())

        with pytest.raises(AnalysisError, match="resume_text cannot be empty"):
            analyzer.analyze("", "target jd text")

    def test_analyze_empty_jd_raises(self) -> None:
        analyzer = self._make_analyzer_with_fake_chain(self._sample_report())

        with pytest.raises(AnalysisError, match="target_jd cannot be empty"):
            analyzer.analyze("resume text", "")

    def test_analyze_wraps_validation_error(self) -> None:
        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = ValidationError.from_exception_data(
            title="AnalysisReport",
            line_errors=[
                {
                    "type": "missing",
                    "loc": ("summary",),
                    "input": {},
                }
            ],
        )

        with patch.object(ResumeAnalyzer, "_build_chain", return_value=fake_chain):
            analyzer = ResumeAnalyzer(llm=MagicMock())

        with pytest.raises(AnalysisError, match="LLM output did not match"):
            analyzer.analyze("resume text", "target jd text")


class TestExtractJson:
    """Tests for the robust JSON extraction helper."""

    def test_extract_json_from_code_block(self) -> None:
        text = 'Some text\n```json\n{"match_score": 82, "summary": "good"}\n```\nMore text'
        result = _extract_json(text)
        assert result == {"match_score": 82, "summary": "good"}

    def test_extract_json_nested_object(self) -> None:
        text = 'prefix {"outer": {"inner": 1}, "list": [1, 2]} suffix'
        result = _extract_json(text)
        assert result["outer"] == {"inner": 1}
        assert result["list"] == [1, 2]

    def test_extract_json_repairs_missing_comma(self) -> None:
        text = '{"a": 1 "b": 2}'
        result = _extract_json(text)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_extract_json_repairs_trailing_comma(self) -> None:
        text = '{"a": 1, "b": 2,}'
        result = _extract_json(text)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_extract_json_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json("This text has no JSON object.")

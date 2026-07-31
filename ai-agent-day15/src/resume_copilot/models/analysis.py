"""Pydantic models for structured resume analysis output."""

from pydantic import BaseModel, Field, field_validator


class ProjectSuggestion(BaseModel):
    """Concrete optimization suggestion for a single project experience."""

    project_name: str = Field(
        ...,
        description="Name or brief title of the project experience being reviewed.",
        min_length=1,
    )
    current_issues: list[str] = Field(
        default_factory=list,
        description="List of issues identified in the current project description.",
    )
    optimization_advice: str = Field(
        ...,
        description=(
            "Actionable advice on how to rewrite the project description, "
            "preferably using the STAR methodology (Situation, Task, Action, Result) "
            "and AI-Agent-specific professional terminology."
        ),
        min_length=1,
    )


class AnalysisReport(BaseModel):
    """Structured report comparing a resume against a target job description."""

    match_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall match score between the resume and the target JD (0-100).",
    )
    summary: str = Field(
        ...,
        description="A concise overall assessment of how well the resume matches the JD.",
        min_length=1,
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Highlights and advantages present in the resume relative to the JD.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Shortcomings and gaps in the resume relative to the JD.",
    )
    missing_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Core technical keywords appearing in the JD but missing or under-represented "
            "in the resume."
        ),
    )
    project_suggestions: list[ProjectSuggestion] = Field(
        default_factory=list,
        description="Project-level suggestions for improving individual project descriptions.",
    )

    @field_validator("match_score", mode="before")
    @classmethod
    def _coerce_score(cls, value: int | float | str) -> int:
        """Ensure match_score is always returned as an integer."""
        return int(value)

    def to_markdown(self) -> str:
        """Render the report as a Markdown string for display or export."""
        lines = [
            "# Resume Analysis Report",
            "",
            f"**Match Score:** {self.match_score}/100",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## Strengths",
            "",
        ]
        for item in self.strengths:
            lines.append(f"- {item}")

        lines.extend(["", "## Weaknesses", ""])
        for item in self.weaknesses:
            lines.append(f"- {item}")

        lines.extend(["", "## Missing Keywords", ""])
        for item in self.missing_keywords:
            lines.append(f"- `{item}`")

        lines.extend(["", "## Project Suggestions", ""])
        for suggestion in self.project_suggestions:
            lines.append(f"### {suggestion.project_name}")
            lines.extend(["", "**Current Issues:**", ""])
            for issue in suggestion.current_issues:
                lines.append(f"- {issue}")
            lines.extend(
                [
                    "",
                    "**Optimization Advice:**",
                    "",
                    suggestion.optimization_advice,
                    "",
                ]
            )

        return "\n".join(lines).strip()

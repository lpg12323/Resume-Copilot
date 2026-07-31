"""Rich-based interactive CLI for Resume Copilot."""

import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from resume_copilot.config import get_settings
from resume_copilot.graph import build_resume_graph
from resume_copilot.graph.state import AgentState
from resume_copilot.logger import logger


class ResumeCopilotCLI:
    """Interactive terminal interface for Resume Copilot."""

    DEFAULT_SAMPLE_RESUME = Path("data") / "sample_resume.pdf"

    def __init__(self) -> None:
        """Initialize the CLI with a rich console and a fresh graph session."""
        self.console = Console()
        self.thread_id = f"cli-{uuid.uuid4().hex[:8]}"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.graph = build_resume_graph(checkpointer=MemorySaver())
        self._state: AgentState = {
            "pdf_path": "",
            "target_jd": "",
            "resume_raw_text": "",
            "analysis_report": None,
            "resume_markdown": "",
            "messages": [],
            "next_node": "",
            "error": None,
        }
        logger.info("CLI initialized with thread_id={}", self.thread_id)

    def run(self) -> None:
        """Run the main CLI loop."""
        self._print_banner()
        self._setup_inputs()
        self._run_initial_analysis()
        self._interactive_loop()

    def _print_banner(self) -> None:
        """Render the welcome banner."""
        banner = Panel.fit(
            "[bold cyan]欢迎使用 Resume Copilot[/bold cyan]\n"
            "[dim]AI 简历分析与优化 Agent[/dim]",
            title="🚀 Resume Copilot",
            border_style="cyan",
        )
        self.console.print(banner)
        self.console.print(
            "[dim]输入 help 查看可用指令，输入 exit 或 quit 退出程序。[/dim]\n"
        )

    def _setup_inputs(self) -> None:
        """Collect PDF path and target JD from the user."""
        settings = get_settings()
        settings.ensure_directories()

        default_path = str(self.DEFAULT_SAMPLE_RESUME)
        pdf_input = Prompt.ask(
            "📄 PDF 简历路径",
            default=default_path if self.DEFAULT_SAMPLE_RESUME.exists() else "",
            console=self.console,
        ).strip()

        if not pdf_input:
            self.console.print("[red]未提供 PDF 路径，无法继续。[/red]")
            sys.exit(1)

        pdf_path = Path(pdf_input)
        if not pdf_path.is_absolute():
            pdf_path = Path(__file__).resolve().parents[2] / pdf_path

        if not pdf_path.exists():
            self.console.print(f"[red]PDF 文件不存在：{pdf_path}[/red]")
            sys.exit(1)

        self._state["pdf_path"] = str(pdf_path)

        jd_input = Prompt.ask(
            "🎯 目标岗位 JD（可直接粘贴文本，或输入 .txt 文件路径）",
            console=self.console,
        ).strip()

        if not jd_input:
            self.console.print("[red]未提供 JD，无法继续。[/red]")
            sys.exit(1)

        jd_path = Path(jd_input)
        if jd_path.exists() and jd_path.is_file():
            target_jd = jd_path.read_text(encoding="utf-8")
            self.console.print(f"[green]已从文件读取 JD：{jd_path}[/green]")
        else:
            target_jd = jd_input

        self._state["target_jd"] = target_jd

    def _run_initial_analysis(self) -> None:
        """Execute the parser -> analyzer workflow and render the report."""
        with self.console.status(
            "[bold green]正在解析简历与匹配 JD...", spinner="dots"
        ) as status:
            result = self.graph.invoke(self._state, self.config)
            self._update_state(result)

        if self._state.get("error"):
            self.console.print(
                f"[red]❌ 初始分析失败：{self._state['error']}[/red]"
            )
            sys.exit(1)

        self.console.print("[bold green]✅ 简历解析与匹配分析完成[/bold green]\n")

        report = self._state.get("analysis_report")
        if report:
            score = report.get("match_score", "N/A")
            self.console.print(
                Panel(
                    f"[bold]综合匹配度：[/bold][yellow]{score}/100[/yellow]",
                    title="📊 匹配得分",
                    border_style="yellow",
                )
            )
            self.console.print(Markdown(self._state["resume_markdown"]))
        else:
            self.console.print("[yellow]未生成分析报告。[/yellow]")

        self.console.rule()

    def _interactive_loop(self) -> None:
        """Run the multi-turn interaction loop."""
        while True:
            user_input = Prompt.ask(
                "[bold cyan]💬 请输入指令[/bold cyan]",
                console=self.console,
            ).strip()

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in ("exit", "quit", "退出"):
                self.console.print(
                    "[bold green]感谢使用 Resume Copilot，再见！[/bold green]"
                )
                break

            if lowered in ("help", "帮助"):
                self._print_help()
                continue

            self._state["messages"].append(HumanMessage(content=user_input))

            with self.console.status(
                "[bold green]Agent 正在处理...", spinner="dots"
            ):
                result = self.graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    self.config,
                )
                self._update_state(result)

            if self._state.get("error"):
                self.console.print(
                    f"[red]❌ 处理失败：{self._state['error']}[/red]"
                )
                self._state["error"] = None
                continue

            self._print_agent_response(result, user_input)
            self.console.rule()

    def _print_agent_response(self, result: dict, user_input: str) -> None:
        """Render the agent's response after a follow-up turn."""
        next_node = result.get("next_node", "")

        if next_node == "rewrite":
            self.console.print(
                "[bold green]✏️ 简历已根据您的要求重写：[/bold green]\n"
            )
            self.console.print(Markdown(self._state["resume_markdown"]))
        elif next_node == "export":
            self.console.print(
                "[bold green]📁 简历已成功导出到 outputs/ 目录[/bold green]"
            )
            # Print the last assistant message which contains the file path.
            messages = result.get("messages", [])
            if messages and isinstance(messages[-1], HumanMessage) is False:
                last_msg = messages[-1]
                self.console.print(Markdown(str(last_msg.content)))
        else:
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                self.console.print(Markdown(str(last_msg.content)))
            else:
                self.console.print(
                    "[dim]已收到您的消息，如需修改请直接输入重写指令。[/dim]"
                )

    def _print_help(self) -> None:
        """Display available commands."""
        help_text = """
**可用指令：**

- `重写项目经历` / `optimize experience` — 让 Agent 重写简历中的项目描述
- `导出简历` / `export resume` — 将当前 Markdown 简历保存到 outputs/ 目录
- `help` / `帮助` — 显示本帮助信息
- `exit` / `quit` / `退出` — 退出程序

也可以直接输入任何问题或修改要求。
"""
        self.console.print(Markdown(help_text))

    def _update_state(self, result: dict) -> None:
        """Merge the graph result into the local state mirror."""
        for key, value in result.items():
            if value is not None:
                self._state[key] = value  # type: ignore[literal-required]


def main() -> None:
    """Entry point for the interactive CLI."""
    try:
        ResumeCopilotCLI().run()
    except KeyboardInterrupt:
        Console().print("\n[bold yellow]程序已被用户中断。[/bold yellow]")
        sys.exit(0)
    except Exception as exc:
        logger.exception("CLI crashed")
        Console().print(f"[bold red]程序异常退出：{exc}[/bold red]")
        sys.exit(1)

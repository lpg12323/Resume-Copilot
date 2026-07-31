# Resume Copilot - System Requirement & Design Specification

> **文档版本**：v1.0
> **状态**：Phase 1 - Spec & Project Setup
> **创建日期**：2026-07-31
> **目标读者**：开发团队、产品经理、架构评审方

---

## 1. Project Overview & Business Value

### 1.1 项目目标与定位

**Resume Copilot** 是一款基于大语言模型（LLM）与 LangGraph 状态机的 **AI 简历分析与优化 Agent**。它面向求职者与职业顾问，通过解析候选人简历 PDF、对比目标岗位描述（Job Description, JD），自动生成结构化的匹配度分析报告，并提供可交互的针对性修改建议与局部重写能力。

项目的核心价值在于：

- **降低信息差**：将 JD 中隐性的能力要求显式化，帮助用户快速识别自身短板。
- **提升修改效率**：通过多轮对话形式，由 Agent 协助用户完成简历内容的局部优化。
- **保持个人表达**：采用 Human-in-the-loop 机制，最终内容由用户确认，避免完全自动化导致的失真。
- **本地化优先**：所有文件处理在本地完成，确保隐私安全。

### 1.2 目标用户群体与核心痛点

| 用户群体 | 核心痛点 | Resume Copilot 提供的价值 |
| --- | --- | --- |
| 应届生与职场新人 | 不清楚 JD 看重的关键词与技能栈，简历描述空泛 | 自动提取 JD 关键词并对比简历，输出结构化短板分析 |
| 转行/跳槽候选人 | 过往经历与目标岗位不完全匹配，不知道如何突出 transferable skills | 基于 LLM 生成经历改写建议，强调与目标岗位的相关性 |
| 求职顾问/猎头 | 需要快速判断候选人与岗位的匹配度 | 自动化生成匹配度打分与优劣势报告 |
| 技术岗位求职者 | 项目描述过于技术化，缺乏业务价值表达 | 将技术细节转化为业务成果导向的描述 |

---

## 2. MVP Scope (Minimum Viable Product)

### 2.1 In-Scope (Phase 1 必须实现)

Phase 1 的 MVP 聚焦于**单用户、本地运行、CLI 交互**的最小可用形态，确保核心闭环可跑通：

1. **PDF 简历解析**
   - 支持用户上传 PDF 格式的简历文件。
   - 提取纯文本内容，并尽量保留段落结构。
   - 输出 `resume_raw_text` 供下游节点使用。

2. **基于目标 JD 的匹配度分析**
   - 用户以文本形式输入目标岗位 JD。
   - Agent 将简历与 JD 进行对比，输出结构化分析报告：
     - `match_score`：整体匹配度评分（0-100）
     - `strengths`：候选人相对 JD 的优势点
     - `weaknesses`：相对 JD 的短板或缺失项
     - `missing_keywords`：JD 中出现但简历未覆盖的关键词列表

3. **LangGraph 状态机驱动的修改建议与局部重写**
   - 基于分析结果生成针对性修改建议。
   - 支持用户通过自然语言指令（如“把项目经历改得更业务导向”）触发简历局部重写。
   - 重写结果以 Markdown 格式维护，并支持多轮迭代。

4. **基于 MemorySaver 的多轮对话与状态持久化**
   - 使用 LangGraph 的 `MemorySaver` 保存会话状态。
   - 支持多轮对话上下文连贯，允许用户反复提问、修改、导出。

5. **Markdown 格式简历导出**
   - 将当前版本的 Markdown 简历写入本地文件（如 `output_resume.md`）。
   - 导出结果可直接复制到在线简历系统或进一步转换为 PDF。

### 2.2 Out-of-Scope (Phase 1 暂不实现，避免需求膨胀)

以下功能虽具有长期价值，但在 Phase 1 明确排除，以保证 MVP 按时交付：

| 功能 | 暂不实现原因 | 后续规划 |
| --- | --- | --- |
| 多用户登录/鉴权系统 | 增加架构复杂度，与核心 AI 能力无关 | Phase 2+ 可扩展为 Web 服务时引入 |
| 真实招聘网站的 JD 自动化爬虫 | 涉及反爬、法律合规与维护成本 | 可先支持用户粘贴 JD 文本，后续再考虑插件化 |
| 复杂的 PDF 渲染排版引擎 | 需要引入排版库与样式系统，超出 MVP 范围 | 仅输出 Markdown，由用户自行渲染为 PDF |
| 联网实时搜索岗位信息 | 依赖外部 API 与付费搜索服务 | 后续可作为增值服务 |
| 团队协作与版本对比 | 需要数据库与版本管理模块 | 后续引入云端存储后再考虑 |

---

## 3. System Architecture & LangGraph State

### 3.1 LangGraph State Definition

系统采用 LangGraph 作为 Agent 编排框架，整个会话状态通过 `AgentState` 统一维护。所有节点的输入与输出均围绕该状态对象流转。

```python
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    resume_raw_text: str          # 原始 PDF 解析出的文本
    target_jd: str                # 目标岗位 JD 文本
    analysis_report: dict         # 结构化分析报告
    resume_markdown: str          # 当前 Markdown 格式简历
    messages: Annotated[List[BaseMessage], add_messages]  # 对话历史
    current_step: str             # 当前节点状态
```

#### 字段详细说明

| 字段 | 类型 | 说明 | 示例/约束 |
| --- | --- | --- | --- |
| `resume_raw_text` | `str` | 从 PDF 提取的原始文本 | 可能包含换行、空格、页眉页脚噪声 |
| `target_jd` | `str` | 用户输入的目标岗位描述 | 可为纯文本或 Markdown |
| `analysis_report` | `dict` | LLM 生成的结构化匹配度分析 | 必须包含 `match_score`、`strengths`、`weaknesses`、`missing_keywords` |
| `resume_markdown` | `str` | 当前版本的 Markdown 简历 | 初始可由 `resume_raw_text` 转换而来，后续经重写节点更新 |
| `messages` | `list[BaseMessage]` | 用户与 Agent 的多轮对话历史 | 使用 `add_messages` reducer 追加 |
| `current_step` | `str` | 当前执行到的图节点名称 | 如 `"parser_node"`、`"analyzer_node"` |

#### `analysis_report` 结构示例

```json
{
  "match_score": 72,
  "strengths": [
    "具备 3 年以上 Python 后端开发经验",
    "熟悉 FastAPI 与微服务架构"
  ],
  "weaknesses": [
    "未明确体现云原生部署经验",
    "缺乏对 LLM / RAG 相关项目的描述"
  ],
  "missing_keywords": [
    "Docker",
    "Kubernetes",
    "LangChain",
    "向量数据库"
  ]
}
```

### 3.2 Graph Workflow & Nodes Definition

#### ASCII 架构图

```text
                    +------------------+
                    |    Start State   |
                    | (resume_path,    |
                    |  target_jd,      |
                    |  user_message)   |
                    +--------+---------+
                             |
                             v
                  +----------+-----------+
                  |    parser_node       |
                  |  解析 PDF 提取原始文本  |
                  +----------+-----------+
                             |
                             v
                  +----------+-----------+
                  |   analyzer_node      |
                  | 生成匹配度分析报告      |
                  +----------+-----------+
                             |
                             v
                  +----------+-----------+
                  |    router_node       |
                  | 意图分类 & 路由决策    |
                  +----------+-----------+
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
   +--------+--------+ +-----+------+ +-------+--------+
   |  rewriter_node  | | qna_node   | | exporter_node  |
   | 局部重写简历     | | 回答用户   | | 导出 Markdown  |
   +--------+--------+ | 问题       | +-------+--------+
            |          +-----+------+         |
            |                |                |
            +----------------+----------------+
                             |
                             v
                    +--------+---------+
                    |   Human / End    |
                    | (等待下一轮输入  |
                    |  或结束会话)     |
                    +------------------+
```

#### 节点定义

| 节点名称 | 功能描述 | 输入 | 输出 | 异常处理 |
| --- | --- | --- | --- | --- |
| `parser_node` | 解析用户上传的 PDF 简历，提取纯文本内容 | `resume_path`（文件路径） | 更新 `resume_raw_text` | PDF 解析失败时返回友好错误，提示用户检查文件格式 |
| `analyzer_node` | 将简历与 JD 对比，调用 LLM 生成结构化分析报告 | `resume_raw_text`、`target_jd` | 更新 `analysis_report` | LLM 返回非 JSON 时启用 JSON repair / retry 机制 |
| `router_node` | 根据最新用户消息判断意图，决定下一步走向 | `messages`、`current_step` | 返回下一个节点名称 | 意图不明时默认进入 `qna_node` 请求澄清 |
| `rewriter_node` | 根据用户指令和分析报告，对 `resume_markdown` 局部重写 | `messages`、`analysis_report`、`resume_markdown` | 更新 `resume_markdown` | 重写结果保留原始结构，避免大段删除 |
| `exporter_node` | 将当前 `resume_markdown` 写入本地文件 | `resume_markdown` | 生成 `output_resume.md` 并返回文件路径 | 写入失败时报告错误并保留内存状态 |
| `qna_node`（可选） | 回答用户关于分析结果或修改建议的提问 | `messages`、`analysis_report`、`resume_markdown` | 追加 Assistant 消息 | 超出范围问题时给出边界说明 |

#### 路由逻辑

`router_node` 根据最新用户输入判断意图，可选路由目标：

| 意图类型 | 触发条件示例 | 目标节点 |
| --- | --- | --- |
| `rewrite` | “帮我优化项目经历”、“把这段改得更业务化” | `rewriter_node` |
| `export` | “导出简历”、“保存为 Markdown” | `exporter_node` |
| `question` | “为什么匹配度只有 70 分？”、“missing_keywords 是什么意思？” | `qna_node` |
| `clarify` | 意图不明确或多任务混杂 | `qna_node`（请求澄清） |

---

## 4. Non-Functional Requirements

### 4.1 异常处理

系统必须对所有关键失败路径提供 fallback 机制，保证 Agent 不会静默崩溃：

| 异常场景 | Fallback 策略 |
| --- | --- |
| PDF 解析失败（文件损坏、非 PDF、加密） | 返回可读错误信息，保留会话状态，允许用户重新上传 |
| LLM 输出 JSON 格式错误 | 使用 `langchain.output_parsers` 或自定义 repair 函数尝试修复；修复失败时最多重试 3 次，最终返回降级文本分析 |
| LLM API 调用超时/限流 | 捕获异常并提示用户重试；必要时切换到备用模型 |
| 路由意图无法识别 | 默认进入 `qna_node`，礼貌请求用户澄清意图 |
| Markdown 导出文件写入失败 | 返回错误信息并在内存中保留 `resume_markdown`，避免内容丢失 |

### 4.2 响应速度

- LLM 生成过程采用 **Streaming 输出**，将分析要点逐段展示给用户，降低主观等待感。
- 对于耗时较长的分析节点，先输出“正在分析中...”状态提示，再逐步流式返回结果。
- 单轮对话的总响应时间（不含网络延迟）目标控制在 15 秒以内。

### 4.3 数据隐私

- 所有 PDF 与 Markdown 文件均**在本地文件系统处理**，不主动上传至任何外部未知服务器。
- LLM API 调用仅传输必要的文本内容；不使用任何可能存储或训练用户数据的第三方服务。
- 会话状态通过本地 `MemorySaver` 保存，Phase 1 不涉及远程数据库或持久化存储。

### 4.4 可维护性

- 节点函数遵循单一职责原则，每个节点只修改 `AgentState` 中的特定字段。
- 使用 Pydantic 模型对 LLM 输出结构进行校验，避免字典裸操作。
- 关键配置（模型名称、温度参数、重试次数）集中管理，便于调整。

---

## 5. Work Breakdown Structure (WBS / Tasks)

项目划分为 5 个 Phase，按增量交付方式推进。

### Phase 1: Spec & Project Setup

- [ ] 撰写并评审 `docs/spec.md`（本文档）
- [ ] 初始化项目目录结构（`src/`、`tests/`、`docs/`、`data/`、`outputs/`）
- [ ] 配置 Python 虚拟环境与依赖（`pyproject.toml` / `requirements.txt`）
- [ ] 确定 LLM 模型与 API 配置策略
- [ ] 搭建基础日志与配置管理模块

### Phase 2: Core Engine Development

- [ ] 实现 PDF Parser 模块（基于 `pypdf` / `pdfplumber`）
- [ ] 设计并实现 `AnalysisReport` Pydantic 模型
- [ ] 实现 LLM Analyzer Prompt，输出结构化 JSON
- [ ] 集成 JSON repair / retry 机制
- [ ] 编写独立测试文件 `test_parser.py` 与 `test_analyzer.py`

### Phase 3: LangGraph State Machine Implementation

- [ ] 定义 `AgentState` TypedDict
- [ ] 实现 `parser_node`、`analyzer_node`、`router_node`、`rewriter_node`、`exporter_node`
- [ ] 构建 LangGraph 工作流与边（Edges）
- [ ] 集成 `MemorySaver` 实现多轮状态持久化
- [ ] 编写测试文件 `test_graph.py`

### Phase 4: CLI / Interactive Loop Integration

- [ ] 实现 Human-in-the-loop CLI 交互循环
- [ ] 支持用户上传 PDF、输入 JD、查看分析、发送修改指令
- [ ] 支持导出 Markdown 到 `outputs/` 目录
- [ ] 提供 Streaming 输出体验
- [ ] 编写端到端使用示例与 `README.md`

### Phase 5: Testing, Evaluation & Refactoring

- [ ] 编写单元测试与集成测试，目标覆盖率 ≥ 70%
- [ ] 设计 LLM 输出质量评估方案（人工 + 自动评分）
- [ ] 根据评估结果优化 Prompt 与 Router 决策
- [ ] 代码重构：提取公共工具函数、统一错误处理
- [ ] 编写最终项目总结与后续扩展建议

---

## 附录 A：技术栈建议

| 层级 | 推荐技术 |
| --- | --- |
| 编程语言 | Python 3.10+ |
| Agent 框架 | LangGraph + LangChain |
| LLM 调用 | OpenAI API / DeepSeek API / 兼容 OpenAI 接口的本地模型 |
| PDF 解析 | `pypdf` / `pdfplumber` |
| 结构化输出 | Pydantic + `langchain_core.output_parsers` |
| 配置管理 | Pydantic Settings / python-dotenv |
| 测试框架 | pytest |
| CLI 交互 | `rich`（可选，用于美化输出） |

---

## 附录 B：风险与假设

| 风险 | 缓解措施 |
| --- | --- |
| LLM 输出不稳定，导致分析报告质量波动 | 使用 Pydantic 校验 + JSON repair + Prompt 迭代优化 |
| PDF 解析质量差，影响后续分析 | 引入 `pdfplumber` 并保留原始文件供人工核对 |
| 用户需求多样化，超出 MVP 范围 | 严格按 In/Out-of-Scope 控制，记录为后续迭代需求 |
| 本地 LLM API 调用延迟高 | 默认使用在线 API；后续可支持本地模型缓存 |

---

> **结束语**：本文档作为 Resume Copilot 项目的基线规范，将在 Phase 1 完成后作为后续开发、测试与验收的统一依据。任何范围变更或架构调整，应通过更新本文档并同步团队确认。

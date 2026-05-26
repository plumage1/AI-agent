# AI 求职助手 Agent

一个面向 AI Agent 开发求职场景的工程化项目，支持简历解析、JD 分析、简历-JD 匹配、RAG 知识库问答、模拟面试和求职准备 Workflow。

## 核心能力

- **Agentic Workflow**：将 JD 解析、简历解析、匹配度计算、RAG 补充、学习任务生成、面试问题生成串联为完整流程。
- **LangGraph Runtime**：聊天、求职 workflow、模拟面试三条主能力都已切到 LangGraph 图编排。
- **Tool Calling**：OpenAI-compatible provider 走标准 Function Calling；Codex CLI provider 走本地 tool routing，再统一进入同一条 graph 链路。
- **RAG 知识库**：支持 Markdown、TXT、文本型 PDF 导入，完成切片、Embedding、Chroma 向量检索和 citations 来源返回。
- **文件上传解析**：支持简历和 JD 的文本输入、文件上传、文本型 PDF 解析，并支持图片 JD 的大模型视觉识别。
- **Session Memory**：聊天主状态由 LangGraph checkpointer 管理；会话元数据和兼容接口继续通过 Redis session store 暴露。
- **Thread State Unification**：聊天消息、trace，以及面试状态都开始向同一个 LangGraph thread state 收敛，便于后续继续接 memory、reminder 和 human-in-the-loop。
- **模拟面试**：根据简历和 JD 生成更贴近真实面试的追问问题，并提供回答反馈。
- **前端工作台**：提供 Workflow、Chat、RAG、Match、Interview、Knowledge、Ops 等模块。
- **工程化部署**：提供 FastAPI API、Docker Compose、健康检查、索引重建和基础评测脚本。

## 技术栈

```text
Backend      FastAPI, Pydantic
LLM          Codex CLI / OpenAI-compatible API
Agent        LangGraph
Memory       LangGraph Checkpointer + Redis Session Store
RAG          Sentence-Transformers, BAAI/bge-small-zh-v1.5
Vector DB    Chroma
Frontend     HTML, CSS, JavaScript
Deploy       Docker, Docker Compose
```

## 快速启动

默认聊天 provider 现在可以直接使用 `codex` CLI；如果你想继续走 OpenAI-compatible API，也可以在 `.env` 中切回。

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

复制环境变量：

```powershell
Copy-Item .env.example .env
```

如果使用 Codex CLI，请确保本机 `codex` 命令可用。默认配置：

```env
LLM_PROVIDER=codex_cli
CODEX_CLI_PATH=codex.cmd
CODEX_MODEL=
```

如果要切回 OpenAI-compatible API：

```env
LLM_PROVIDER=openai_api
API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.1
```

启动 Redis Stack：

```powershell
docker compose up -d redis
```

启动 API：

```powershell
python -m uvicorn api.main:app --reload
```

访问地址：

```text
前端工作台：http://127.0.0.1:8000/ui
接口文档：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/health
```

## 环境变量

请根据 `.env.example` 创建 `.env`。不要提交真实 `.env`。

```env
API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.1

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=
LANGGRAPH_CHECKPOINTER_BACKEND=auto

RAG_RETRIEVER=chroma
RAG_MIN_SCORE=0.5
```

## RAG 优化后的链路

当前 RAG 已从单一路径检索升级为更接近主流项目的混合检索链路：

```text
query variants
  -> Chroma 语义检索 + keyword 检索
  -> 候选去重
  -> lexical + retrieval score rerank
  -> citation id
  -> context budget
  -> LLM grounded answer
```

相关环境变量：

```env
RAG_RETRIEVER=hybrid
RAG_MIN_SCORE=0.5
RAG_CONTEXT_MAX_CHARS=4200
```

相关文件：

- `rag/query_planner.py`: 查询改写、同义词扩展、query variants
- `rag/reranker.py`: 候选去重、融合排序、引用编号
- `rag/rag_chain.py`: hybrid retrieve、上下文构造、citations
- `evals/eval_runner.py`: RAG recall@k 和 MRR 评测

## Agent 工具调用链路

`/chat` 现在使用 LangGraph `StateGraph` 作为主运行时：

```text
session messages
  -> LangGraph route_turn
  -> direct_chat / model tool planning / local tool routing
  -> execute_tools
  -> summarize_tools
  -> LangGraph checkpointer 持久化 thread state
  -> 返回最终答案，并记录 trace
```

其中：

- `openai_api` provider：模型直接返回 `tool_calls`
- `codex_cli` provider：本地先做轻量工具路由，再复用同一套执行和总结节点
- `LANGGRAPH_CHECKPOINTER_BACKEND=auto`：优先尝试 Redis checkpointer，失败时自动回退到内存模式，方便本地开发

工具 schema 来自 `tools/registry.py`，`GET /tools` 会同时返回人类可读参数和 JSON Schema。

## LangGraph 模块

```text
chat graph
  -> 路由普通聊天 / 模型 tool calls / 本地 tool routing
  -> 执行工具
  -> 汇总答案
  -> 持久化 thread state

workflow graph
  -> JD 分析
  -> 简历分析
  -> 匹配度计算
  -> RAG 检索
  -> 学习任务 / 面试问题 / 简历项目表达

interview graphs
  -> start graph: 检索资料 -> 生成首题 -> 初始化 interview state
  -> answer graph: 读取问题 -> 检索资料 -> 评分反馈 -> 生成追问
```

## 主要接口

```text
POST /chat
GET  /tools
GET  /trace
POST /clear

POST /rag/search
GET  /rag/status
POST /rag/index/rebuild

POST /knowledge/documents
POST /knowledge/import
GET  /knowledge/documents

POST /career/resume/analyze
POST /career/jd/analyze
POST /career/match
POST /career/resume/upload/analyze
POST /career/jd/upload/analyze
POST /career/match/upload
POST /career/match/files

POST /job-workflow/run
POST /job-workflow/run/files

POST /interview/start
POST /interview/answer

GET /eval/all
```

## Workflow 示例

```json
{
  "resume_text": "我熟悉 Python、FastAPI、Redis，做过 RAG Agent 项目。",
  "jd_text": "需要 Python、Redis、Docker 和 RAG 项目经验，能够搭建自动化工作流并调用 AI 模型接口。",
  "top_k": 3
}
```

返回内容包括：

```text
jd_analysis              JD 关键词分析
resume_analysis          简历技能分析
match                    匹配度、已匹配项、缺失项
rag                      知识库检索结果和引用来源
learning_tasks           学习任务
interview_questions      面试追问问题
resume_project_bullets   可写进简历的项目表述
next_actions             后续行动建议
```

## 文件识别能力

```text
.md / .txt               直接读取文本
.pdf                     使用 pypdf 读取文本型 PDF
.png / .jpg / .webp      默认优先使用本地 OCR；如果本地 OCR 不可用，再尝试远程视觉识别
```

注意：扫描版 PDF 本质上是图片，需要 OCR 或支持视觉能力的大模型。推荐优先安装 Tesseract OCR；如果未加入 PATH，可在 `.env` 中配置 `TESSERACT_CMD`。

## Docker 启动

```powershell
docker compose up -d --build
```

停止服务：

```powershell
docker compose down
```

不要把 `docker compose config` 的完整输出发到公开环境，因为它可能展开 `.env` 里的 API Key。

## 评测

```powershell
python scripts/eval_run.py
```

当前内置评测覆盖：

- RAG 召回
- 简历-JD 匹配

## 简历描述

```text
基于 FastAPI、LangGraph、Redis 和 Chroma 设计并实现 AI 求职助手 Agent，支持多轮对话、工具调用、RAG 知识库问答、简历-JD 匹配、模拟面试和 Agentic Workflow。项目将聊天、求职 workflow 和模拟面试统一为 LangGraph 图编排，结合 Redis/内存 checkpointer 管理线程状态；构建文档导入、文本切片、向量检索、引用来源返回和求职准备链路，并将 JD 解析、简历解析、匹配度计算、RAG 补充、学习任务与面试问题生成串联为完整工程化流程。
```

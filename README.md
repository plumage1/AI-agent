# AI 求职助手 Agent

一个面向求职场景的 AI Agent 项目，支持简历分析、JD 分析、简历与 JD 匹配、RAG 检索问答、模拟面试和多步骤求职准备。

## 核心能力

- Agent 化求职准备：复杂请求会自动进入 `规划 -> 执行 -> 自检 -> 重规划 -> 总结` 闭环。
- 混合对话链路：简单问答直答，单步工具请求走工具链路，复杂任务走 Agent loop。
- 求职工具集：支持简历分析、JD 分析、匹配评估、学习计划、简历 bullet 生成、面试重点准备、求职策略建议。
- RAG 知识库：支持 Markdown、TXT、文本型 PDF 导入，提供 citations 和来源追踪。
- 模拟面试：基于简历和 JD 生成问题、反馈和追问。
- Session / Thread Memory：通过 LangGraph checkpointer 和 session store 保留上下文与 agent 状态。
- 前端工作台：内置 Chat、Workflow、RAG、Interview、Knowledge、Ops 面板。

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

Windows 一键启动：

```powershell
.\start.bat
```

常用参数：

```powershell
.\start.ps1 -SkipRedis
.\start.ps1 -InstallDeps
.\start.ps1 -NoReload
```

手动启动：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn api.main:app --reload
```

访问地址：

```text
前端工作台：http://127.0.0.1:8000/ui
接口文档：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/health
```

## 环境变量

默认可直接使用 Codex CLI：

```env
LLM_PROVIDER=codex_cli
CODEX_CLI_PATH=codex.cmd
CODEX_MODEL=
CODEX_TIMEOUT_SECONDS=180
```

切换到 OpenAI-compatible API：

```env
LLM_PROVIDER=openai_api
API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.1
```

Redis / LangGraph 相关：

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=
REDIS_ENABLED=true
REDIS_SOCKET_CONNECT_TIMEOUT=1
REDIS_SOCKET_TIMEOUT=1
LANGGRAPH_CHECKPOINTER_BACKEND=auto
```

RAG 相关：

```env
RAG_RETRIEVER=hybrid
RAG_MIN_SCORE=0.5
RAG_CONTEXT_MAX_CHARS=4200
```

## Agent 链路

`/chat` 当前采用混合运行时：

```text
session messages
  -> LangGraph classify_request
  -> direct_chat
     or
     local tool routing / model tool planning
       -> execute_tools
       -> summarize_tools
     or
     build_goal_and_plan
       -> execute_current_step
       -> review_step_result
       -> replan_if_needed
       -> finalize_answer
  -> LangGraph checkpointer 持久化 thread state
  -> 返回 answer、trace、plan、step_history、artifacts
```

说明：

- `openai_api` provider 支持模型原生 `tool_calls`
- `codex_cli` provider 先做本地轻量路由，再进入同一套工具执行链路
- `GET /agent/state` 用于读取当前 session 的 Goal / Plan / Progress / Next Action
- `tools/agent_tools.py` 提供 `synthesize_resume_bullets`、`prepare_interview_focus`、`job_search_strategy`

## RAG 链路

```text
query variants
  -> Chroma 语义检索 + keyword 检索
  -> 候选去重
  -> lexical + retrieval score rerank
  -> citation id
  -> context budget
  -> LLM grounded answer
```

相关文件：

- `rag/query_planner.py`
- `rag/reranker.py`
- `rag/rag_chain.py`

## 主要接口

```text
POST /chat
GET  /agent/state
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
```

## Workflow 示例

```json
{
  "resume_text": "我熟悉 Python、FastAPI、Redis，做过 RAG Agent 项目。",
  "jd_text": "需要 Python、Redis、Docker 和 RAG 项目经验，能够搭建自动化工作流并调用 AI 模型接口。",
  "top_k": 3
}
```

典型返回内容：

```text
jd_analysis
resume_analysis
match
rag
learning_tasks
interview_questions
resume_project_bullets
next_actions
```

## 文件识别能力

```text
.md / .txt               直接读取文本
.pdf                     使用 pypdf 读取文本型 PDF
.png / .jpg / .webp      优先本地 OCR，失败时再尝试视觉识别
```

如果是扫描版 PDF，建议优先安装 Tesseract OCR，并在 `.env` 中配置 `TESSERACT_CMD`。

## Docker

启动 Redis：

```powershell
docker compose up -d redis
```

如果 Docker 不可用，项目会自动回退到内存 checkpointer 模式，仍可本地开发和测试。

## Agent 演示路径

适合演示的 3 条请求：

1. `帮我根据这份简历和 JD 做完整求职准备`
2. `根据这个 JD 给我补一份 7 天学习路线`
3. `基于这份 JD 模拟一轮后端工程师面试`

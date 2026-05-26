# AI 求职助手 Agent

一个面向 AI Agent 开发求职场景的工程化项目，支持简历解析、JD 分析、简历-JD 匹配、RAG 知识库问答、模拟面试和求职准备 Workflow。

## 核心能力

- **Agentic Workflow**：将 JD 解析、简历解析、匹配度计算、RAG 补充、学习任务生成、面试问题生成串联为完整流程。
- **Tool Calling**：通过 Router 判断用户意图，再由 Tool Executor 调用学习计划、JD 分析、简历分析、RAG 检索等工具。
- **RAG 知识库**：支持 Markdown、TXT、文本型 PDF 导入，完成切片、Embedding、Chroma 向量检索和 citations 来源返回。
- **文件上传解析**：支持简历和 JD 的文本输入、文件上传、文本型 PDF 解析，并支持图片 JD 的大模型视觉识别。
- **Session Memory**：使用 Redis 保存 messages、用户状态和 tool traces，实现多用户会话隔离与 TTL 自动过期。
- **模拟面试**：根据简历和 JD 生成更贴近真实面试的追问问题，并提供回答反馈。
- **前端工作台**：提供 Workflow、Chat、RAG、Match、Interview、Knowledge、Ops 等模块。
- **工程化部署**：提供 FastAPI API、Docker Compose、健康检查、索引重建和基础评测脚本。

## 技术栈

```text
Backend      FastAPI, Pydantic
LLM          OpenAI-compatible API
Memory       Redis
RAG          Sentence-Transformers, BAAI/bge-small-zh-v1.5
Vector DB    Chroma
Frontend     HTML, CSS, JavaScript
Deploy       Docker, Docker Compose
```

## 快速启动

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

启动 Redis：

```powershell
docker compose up -d redis
```

启动 API：

```powershell
uvicorn api.main:app --reload
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

RAG_RETRIEVER=chroma
RAG_MIN_SCORE=0.5
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
.png / .jpg / .webp      优先使用大模型视觉识别；如果模型不支持图片，可配置本地 OCR
```

注意：扫描版 PDF 本质上是图片，需要 OCR 或支持视觉能力的大模型。

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
基于 FastAPI、Redis、Chroma 和大模型 API 设计并实现 AI 求职助手 Agent，支持多轮对话、工具调用、RAG 知识库问答、简历-JD 匹配、模拟面试和 Agentic Workflow。系统使用 Redis 存储 session、messages 和 tool traces，实现多用户会话隔离与 TTL 自动过期；构建文档导入、文本切片、embedding 生成、Chroma 向量检索、低分过滤和 citations 来源返回链路；通过 workflow 将 JD 解析、简历解析、匹配度计算、RAG 知识补充、学习任务生成和面试问题生成串联为完整求职准备流程。
```

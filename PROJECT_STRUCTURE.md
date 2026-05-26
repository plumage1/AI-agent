# 项目结构

```text
api/
  main.py                  FastAPI 服务入口和 API 路由

agents/
  langgraph_state.py       LangGraph agent state 定义
  langgraph_runtime.py     聊天主图：路由、工具执行、agent loop、trace 和线程状态
  career_agent.py          简历、JD、匹配度分析
  interview_agent.py       模拟面试图：出题、评估、追问和面试状态
  job_workflow_agent.py    求职准备图：JD、简历、匹配、RAG 和行动计划

core/
  config.py                环境变量和应用配置
  llm.py                   Codex CLI / OpenAI-compatible provider 适配层
  redis_client.py          Redis 连接
  langgraph_checkpoint.py  LangGraph checkpointer 选择和线程配置

rag/
  query_planner.py         查询规范化、query variants、同义词扩展
  reranker.py              候选去重、本地 rerank、citation id 生成
  document_loader.py       文档解析，支持 md/txt/pdf/image
  knowledge_store.py       本地知识库文件管理
  simple_retriever.py      文档切片、chunk_id 和关键词检索
  embedding_retriever.py   本地 embedding 检索
  chroma_store.py          Chroma 向量数据库
  rag_chain.py             hybrid retrieve、rerank、context budget、citations

tools/
  registry.py              工具注册表和 JSON Schema
  agent_tools.py           Agent 组合能力工具
  learning_tools.py        学习计划工具
  resume_tools.py          简历分析工具
  jd_tools.py              JD 分析工具
  rag_tools.py             RAG 检索工具

stores/
  session_store.py         会话兼容层，整合 LangGraph 线程状态、面试状态和 agent snapshot

web/
  index.html               静态前端页面
  app.js                   前端 API 调用、会话恢复、Agent 面板
  styles.css               前端样式

data/
  knowledge/               本地知识库文档
  chroma/                  本地向量索引数据

scripts/
  redis_ping.py            Redis 连接检查

start.ps1                  Windows 一键启动脚本
start.bat                  Windows 一键启动入口
```

## 当前主链路

```text
用户输入
  -> session_store 读取会话 / LangGraph 线程快照 / agent snapshot
  -> chat graph classify_request
  -> direct_chat
     or
     model tool planning / local tool routing
       -> execute_tools
       -> summarize_tools
     or
     build_goal_and_plan
       -> execute_current_step
       -> review_step_result
       -> replan_if_needed
       -> finalize_answer
  -> LangGraph checkpointer 持久化线程状态
  -> session_store 写回会话元数据
  -> API / 前端返回答案、工具名、trace、来源引用和 agent 状态
```

## RAG 链路

```text
用户问题
  -> query_planner 生成 query variants
  -> Chroma 语义检索 + keyword 检索
  -> reranker 去重并融合 retrieval score、标题命中、正文命中
  -> 生成 [S1]/[S2] citation id
  -> 在 context budget 内构造上下文
  -> LLM 按引用回答
```

## Workflow 链路

```text
简历 + JD
  -> workflow graph
  -> JD 分析
  -> 简历分析
  -> 匹配度计算
  -> RAG 检索补充资料
  -> 生成学习任务 / 面试问题 / 简历项目表述
```

## Interview 链路

```text
开始面试
  -> interview start graph
  -> 检索参考资料
  -> 生成首题
  -> 写回 interview thread state

回答问题
  -> interview answer graph
  -> 读取当前问题
  -> 检索参考资料
  -> 评分、反馈、追问
  -> 写回 interview thread state
```

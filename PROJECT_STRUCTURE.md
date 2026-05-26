# 项目结构

```text
api/
  main.py                  FastAPI 服务入口和 API 路由

agents/
  chat_agent.py            普通多轮对话
  router.py                用户意图路由
  tool_executor.py         工具执行和 trace 记录
  career_agent.py          简历、JD、匹配度分析
  interview_agent.py       模拟面试 Agent
  job_workflow_agent.py    求职准备 Agentic Workflow

core/
  config.py                环境变量和应用配置
  llm.py                   大模型客户端封装
  redis_client.py          Redis 连接

rag/
  document_loader.py       文档解析，支持 md/txt/pdf
  knowledge_store.py       Markdown 知识库文件管理
  simple_retriever.py      文档切片和关键词检索
  embedding_retriever.py   本地 embedding 检索
  chroma_store.py          Chroma 向量数据库
  rag_chain.py             RAG 问答主链路

tools/
  registry.py              工具注册表
  learning_tools.py        学习计划工具
  resume_tools.py          简历分析工具
  jd_tools.py              JD 分析工具
  rag_tools.py             RAG 检索工具

stores/
  session_store.py         Redis session 存储

web/
  index.html               静态前端页面
  app.js                   前端 API 调用逻辑
  styles.css               前端样式

data/
  knowledge/               本地知识库文档
  eval/                    内置评测用例
  chroma/                  Chroma 本地索引，运行时生成

scripts/
  eval_run.py              命令行评测入口
  redis_ping.py            Redis 连接检查
```

## 当前主链路

```text
用户输入
  -> Router 判断是否需要工具
  -> Tool Executor 执行工具
  -> Redis 保存 messages 和 traces
  -> API / 前端返回答案、工具名、trace、引用来源
```

## Workflow 链路

```text
简历 + JD
  -> JD 解析
  -> 简历解析
  -> 匹配度计算
  -> RAG 检索补充资料
  -> 生成学习任务
  -> 生成面试问题
  -> 生成简历项目表述
```

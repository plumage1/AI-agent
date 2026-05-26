from tools.jd_tools import analyze_jd
from tools.learning_tools import get_learning_plan
from tools.rag_tools import rag_search
from tools.resume_tools import analyze_resume


TOOLS = {
    "get_learning_plan": {
        "function": get_learning_plan,
        "description": "当用户想要某个技术主题的学习计划、学习路线、学习步骤时使用。",
        "parameters": {
            "topic": "技术主题，例如 RAG、FastAPI、Redis"
        }
    },
    "analyze_resume": {
        "function": analyze_resume,
        "description": "当用户提供简历内容，并希望分析技能、优化简历或提取技能关键词时使用。",
        "parameters": {
            "resume_text": "用户提供的简历文本内容"
        }
    },
    "analyze_jd": {
        "function": analyze_jd,
        "description": "当用户提供岗位 JD 内容，并希望提取关键词、分析岗位要求时使用。",
        "parameters": {
            "jd_text": "用户提供的岗位描述文本内容"
        }
    },
    "rag_search": {
        "function": rag_search,
        "description": "当用户询问技术概念、面试八股、知识库相关问题时使用，例如 Redis 缓存雪崩、Redis 持久化、RAG 是什么。",
        "parameters": {
            "query": "用户提出的技术问题"
        }
    }
}

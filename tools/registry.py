from agents.career_agent import match_resume_to_jd
from tools.jd_tools import analyze_jd
from tools.learning_tools import get_learning_plan
from tools.rag_tools import rag_search
from tools.resume_tools import analyze_resume


TOOLS = {
    "get_learning_plan": {
        "function": get_learning_plan,
        "description": "Generate a learning plan or roadmap for a technical topic.",
        "parameters": {
            "topic": "Technical topic such as RAG, FastAPI, or Redis.",
        },
        "schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Technical topic such as RAG, FastAPI, or Redis.",
                }
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
    "analyze_resume": {
        "function": analyze_resume,
        "description": "Analyze resume text and extract key skills and optimization suggestions.",
        "parameters": {
            "resume_text": "The resume text supplied by the user.",
        },
        "schema": {
            "type": "object",
            "properties": {
                "resume_text": {
                    "type": "string",
                    "description": "The resume text supplied by the user.",
                }
            },
            "required": ["resume_text"],
            "additionalProperties": False,
        },
    },
    "analyze_jd": {
        "function": analyze_jd,
        "description": "Analyze a job description and extract keywords and interview focus areas.",
        "parameters": {
            "jd_text": "The job description text supplied by the user.",
        },
        "schema": {
            "type": "object",
            "properties": {
                "jd_text": {
                    "type": "string",
                    "description": "The job description text supplied by the user.",
                }
            },
            "required": ["jd_text"],
            "additionalProperties": False,
        },
    },
    "match_resume_to_jd": {
        "function": match_resume_to_jd,
        "description": "Compare resume text with a job description and return match score, matched skills, and missing skills.",
        "parameters": {
            "resume_text": "The resume text supplied by the user.",
            "jd_text": "The job description text supplied by the user.",
        },
        "schema": {
            "type": "object",
            "properties": {
                "resume_text": {
                    "type": "string",
                    "description": "The resume text supplied by the user.",
                },
                "jd_text": {
                    "type": "string",
                    "description": "The job description text supplied by the user.",
                },
            },
            "required": ["resume_text", "jd_text"],
            "additionalProperties": False,
        },
    },
    "rag_search": {
        "function": rag_search,
        "description": "Search the technical knowledge base for interview, concept, or project questions.",
        "parameters": {
            "query": "The user's technical question.",
        },
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's technical question.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": tool_info["schema"],
            },
        }
        for tool_name, tool_info in TOOLS.items()
    ]


def get_tool(tool_name: str):
    return TOOLS.get(tool_name)

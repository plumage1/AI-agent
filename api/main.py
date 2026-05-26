import time
import logging
from pathlib import Path
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field

from agents.chat_agent import chat_with_memory
from agents.career_agent import (
    analyze_jd_file,
    analyze_jd_text,
    analyze_resume_file,
    analyze_resume_text,
    match_resume_file_to_jd_file,
    match_resume_file_to_jd,
    match_resume_to_jd,
)
from agents.interview_agent import (
    reset_interview,
    start_interview,
    submit_interview_answer,
    summarize_interview,
)
from agents.job_workflow_agent import run_job_workflow
from evals.eval_runner import run_all_evals, run_career_eval, run_rag_eval
from prompts.job_coach_prompt import SYSTEM_PROMPT
from core.config import settings
from core.redis_client import redis_client
from rag.chroma_store import get_chroma_status, rebuild_chroma_index, reset_chroma_index
from rag.document_loader import extract_document_text, load_document_as_markdown
from rag.embedding_retriever import clear_embedding_cache, get_embedding_cache_status
from rag.knowledge_store import (
    delete_knowledge_document,
    list_knowledge_documents,
    read_knowledge_document,
    save_knowledge_document,
)
from rag.rag_chain import DEFAULT_RETRIEVER, retrieve_sources
from rag.simple_retriever import load_chunks
from agents.router import route_user_input
from agents.tool_executor import run_tool
from tools.registry import TOOLS
from stores.session_store import (
    get_session,
    save_session,
    session_exists,
    delete_session,
    list_sessions as list_session_ids,
    get_session_ttl,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

traces = []
app = FastAPI(title=settings.app_name)
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def require_admin(x_admin_token: str | None = Header(default=None)) -> bool:
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    return True


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


def validate_text_length(text: str, field_name: str) -> None:
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} is too long. Max length is {settings.max_text_length} characters."
        )


def validate_file_size(file_bytes: bytes) -> None:
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file is too large. Max size is {settings.max_upload_bytes} bytes."
        )


def normalize_top_k(top_k: int) -> int:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    return min(top_k, settings.max_top_k)


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    answer: str
    used_tool: bool = False
    tool_name: str | None = None
    trace_id: int | None = None
    sources: list = Field(default_factory=list)
    citations: list = Field(default_factory=list)

class TraceResponse(BaseModel):
    traces: list

class ClearResponse(BaseModel):
    message: str

class ToolsResponse(BaseModel):
    tools: list

class SessionsResponse(BaseModel):
    sessions: list

class SingleTraceResponse(BaseModel):
    trace: dict

class SessionDetailResponse(BaseModel):
    session_id: str
    message_count: int
    turn_count: int
    trace_count: int
    ttl_seconds: int

class HealthResponse(BaseModel):
    status: str
    redis: bool

class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 2

class RagSearchResponse(BaseModel):
    retriever: dict
    sources: list
    citations: list = Field(default_factory=list)

class RagStatusResponse(BaseModel):
    retriever: str
    embedding_cache: dict
    chroma: dict | None = None

class RagIndexResponse(BaseModel):
    index: dict

class KnowledgeDocumentRequest(BaseModel):
    filename: str
    content: str

class KnowledgeDocumentResponse(BaseModel):
    filename: str
    content: str
    cache_cleared: bool = False

class KnowledgeDocumentsResponse(BaseModel):
    documents: list

class KnowledgeImportResponse(BaseModel):
    source_filename: str
    filename: str
    char_count: int
    cache_cleared: bool
    content_preview: str

class KnowledgeChunksResponse(BaseModel):
    chunk_count: int
    chunks: list

class InterviewStartRequest(BaseModel):
    session_id: str = "default"
    topic: str = "AI Agent 项目开发"
    difficulty: str = "中等"

class InterviewStartResponse(BaseModel):
    topic: str
    difficulty: str
    question: str
    citations: list = Field(default_factory=list)
    retriever: dict = Field(default_factory=dict)

class InterviewAnswerRequest(BaseModel):
    session_id: str = "default"
    answer: str

class InterviewAnswerResponse(BaseModel):
    score: int
    feedback: str
    reference_answer: str
    follow_up_question: str
    turn_count: int
    average_score: float
    citations: list = Field(default_factory=list)
    retriever: dict = Field(default_factory=dict)

class InterviewStateResponse(BaseModel):
    interview: dict

class ResumeAnalyzeRequest(BaseModel):
    resume_text: str

class ResumeAnalyzeResponse(BaseModel):
    skills: list
    skill_count: int
    suggestions: list
    source_filename: str | None = None
    char_count: int | None = None

class JdAnalyzeRequest(BaseModel):
    jd_text: str

class JdAnalyzeResponse(BaseModel):
    keywords: list
    keyword_count: int
    preparation_focus: list
    source_filename: str | None = None
    char_count: int | None = None

class ResumeJdMatchRequest(BaseModel):
    resume_text: str
    jd_text: str

class ResumeJdMatchResponse(BaseModel):
    match_score: int
    resume_keywords: list
    jd_keywords: list
    matched_keywords: list
    missing_keywords: list
    resume_suggestions: list
    interview_focus: list
    source_filename: str | None = None
    resume_char_count: int | None = None
    jd_source_filename: str | None = None
    jd_char_count: int | None = None

class JobWorkflowRequest(BaseModel):
    resume_text: str
    jd_text: str
    top_k: int = 3

class JobWorkflowResponse(BaseModel):
    result: dict

class EvalResponse(BaseModel):
    result: dict

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "%s %s -> %s (%.2fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    return response


def require_session(session_id: str) -> dict:
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return get_session(session_id)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    validate_text_length(request.message, "message")

    session = get_session(request.session_id)
    messages = session["messages"]
    traces = session["traces"]

    route = route_user_input(request.message)

    if route.get("use_tool"):
        before_trace_count = len(traces)

        answer = run_tool(request.message, route, messages, traces)

        trace_id = None
        if len(traces) > before_trace_count:
            trace_id = len(traces) - 1

        sources = []
        citations = []

        if trace_id is not None:
            trace = traces[trace_id]
            sources = trace.get("metadata", {}).get("sources", [])
            citations = trace.get("metadata", {}).get("citations", [])

        save_session(request.session_id, session)

        return ChatResponse(
            answer=answer,
            used_tool=True,
            tool_name=route.get("tool_name"),
            trace_id=trace_id,
            sources=sources,
            citations=citations,
        )

    answer = chat_with_memory(messages, request.message)

    save_session(request.session_id, session)

    return ChatResponse(
        answer=answer,
        used_tool=False,
        tool_name=None,
        trace_id=None,
        sources=[],
        citations=[],
    )
@app.get("/trace", response_model=TraceResponse)
def get_trace(session_id: str = "default", _admin: bool = Depends(require_admin)):
    session = require_session(session_id)
    return TraceResponse(traces=session["traces"])

@app.get("/trace/{trace_id}", response_model=SingleTraceResponse)
def get_single_trace(
    trace_id: int,
    session_id: str = "default",
    _admin: bool = Depends(require_admin),
):
    session = require_session(session_id)
    traces = session["traces"]

    if trace_id < 0 or trace_id >= len(traces):
        raise HTTPException(status_code=404, detail="Trace not found")

    return SingleTraceResponse(trace=traces[trace_id])

@app.post("/clear", response_model=ClearResponse)
def clear_memory(session_id: str = "default", _admin: bool = Depends(require_admin)):
    session = require_session(session_id)

    session["messages"].clear()
    session["messages"].append({"role": "system", "content": SYSTEM_PROMPT})

    session["traces"].clear()

    save_session(session_id, session)

    return ClearResponse(message=f"Session {session_id} cleared.")

@app.get("/tools", response_model=ToolsResponse)
def get_tools():
    tool_list = []

    for tool_name, tool_info in TOOLS.items():
        tool_list.append({
            "name": tool_name,
            "description": tool_info["description"],
            "parameters": tool_info["parameters"]
        })

    return ToolsResponse(tools=tool_list)

@app.get("/rag/status", response_model=RagStatusResponse)
def get_rag_status():
    chroma_status = None

    if DEFAULT_RETRIEVER.lower().strip() == "chroma":
        try:
            chroma_status = get_chroma_status()
        except RuntimeError:
            chroma_status = {
                "error": "chromadb is not installed"
            }

    return RagStatusResponse(
        retriever=DEFAULT_RETRIEVER,
        embedding_cache=get_embedding_cache_status(),
        chroma=chroma_status,
    )

@app.post("/rag/search", response_model=RagSearchResponse)
def search_rag(request: RagSearchRequest):
    from rag.rag_chain import format_citations

    validate_text_length(request.query, "query")
    top_k = normalize_top_k(request.top_k)

    sources, retriever = retrieve_sources(
        query=request.query,
        top_k=top_k,
    )

    return RagSearchResponse(
        retriever=retriever,
        sources=sources,
        citations=format_citations(sources),
    )

@app.post("/rag/cache/clear", response_model=ClearResponse)
def clear_rag_cache(_admin: bool = Depends(require_admin)):
    clear_embedding_cache()
    return ClearResponse(message="RAG embedding cache cleared.")

@app.get("/rag/index/status", response_model=RagIndexResponse)
def get_rag_index_status():
    try:
        status = get_chroma_status()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RagIndexResponse(index=status)

@app.post("/rag/index/rebuild", response_model=RagIndexResponse)
def rebuild_rag_index(_admin: bool = Depends(require_admin)):
    try:
        index = rebuild_chroma_index()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    clear_embedding_cache()

    return RagIndexResponse(index=index)

@app.get("/knowledge/documents", response_model=KnowledgeDocumentsResponse)
def list_knowledge_documents_api(_admin: bool = Depends(require_admin)):
    return KnowledgeDocumentsResponse(documents=list_knowledge_documents())

@app.get("/knowledge/documents/{filename}", response_model=KnowledgeDocumentResponse)
def get_knowledge_document_api(filename: str, _admin: bool = Depends(require_admin)):
    try:
        document = read_knowledge_document(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Knowledge document not found")

    return KnowledgeDocumentResponse(**document)

@app.post("/knowledge/documents", response_model=KnowledgeDocumentResponse)
def save_knowledge_document_api(
    request: KnowledgeDocumentRequest,
    _admin: bool = Depends(require_admin),
):
    validate_text_length(request.content, "content")

    try:
        document = save_knowledge_document(
            filename=request.filename,
            content=request.content,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    clear_embedding_cache()
    reset_chroma_index()

    return KnowledgeDocumentResponse(
        **document,
        cache_cleared=True,
    )

@app.delete("/knowledge/documents/{filename}", response_model=ClearResponse)
def delete_knowledge_document_api(filename: str, _admin: bool = Depends(require_admin)):
    try:
        deleted = delete_knowledge_document(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail="Knowledge document not found")

    clear_embedding_cache()
    reset_chroma_index()

    return ClearResponse(message=f"Knowledge document {filename} deleted.")

@app.post("/knowledge/import", response_model=KnowledgeImportResponse)
async def import_knowledge_document(
    file: UploadFile = File(...),
    _admin: bool = Depends(require_admin),
):
    file_bytes = await file.read()
    validate_file_size(file_bytes)

    try:
        loaded = load_document_as_markdown(
            source_filename=file.filename or "",
            file_bytes=file_bytes,
        )
        document = save_knowledge_document(
            filename=loaded["knowledge_filename"],
            content=loaded["content"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    clear_embedding_cache()
    reset_chroma_index()

    return KnowledgeImportResponse(
        source_filename=loaded["source_filename"],
        filename=document["filename"],
        char_count=loaded["char_count"],
        cache_cleared=True,
        content_preview=document["content"][:300],
    )

@app.get("/knowledge/chunks", response_model=KnowledgeChunksResponse)
def list_knowledge_chunks_api(_admin: bool = Depends(require_admin)):
    chunks = load_chunks()

    return KnowledgeChunksResponse(
        chunk_count=len(chunks),
        chunks=chunks,
    )

@app.post("/interview/start", response_model=InterviewStartResponse)
def start_interview_api(request: InterviewStartRequest):
    validate_text_length(request.topic, "topic")

    session = get_session(request.session_id)

    result = start_interview(
        session=session,
        topic=request.topic,
        difficulty=request.difficulty,
    )

    save_session(request.session_id, session)

    return InterviewStartResponse(**result)

@app.post("/interview/answer", response_model=InterviewAnswerResponse)
def answer_interview_api(request: InterviewAnswerRequest):
    validate_text_length(request.answer, "answer")

    session = get_session(request.session_id)

    result = submit_interview_answer(
        session=session,
        answer=request.answer,
    )

    save_session(request.session_id, session)

    return InterviewAnswerResponse(**result)

@app.get("/interview/state", response_model=InterviewStateResponse)
def get_interview_state_api(session_id: str = "default", _admin: bool = Depends(require_admin)):
    session = require_session(session_id)
    return InterviewStateResponse(interview=summarize_interview(session))

@app.post("/interview/reset", response_model=ClearResponse)
def reset_interview_api(session_id: str = "default", _admin: bool = Depends(require_admin)):
    session = require_session(session_id)
    reset_interview(session)
    save_session(session_id, session)

    return ClearResponse(message=f"Interview state for session {session_id} reset.")

@app.post("/career/resume/analyze", response_model=ResumeAnalyzeResponse)
def analyze_resume_api(request: ResumeAnalyzeRequest):
    validate_text_length(request.resume_text, "resume_text")

    return ResumeAnalyzeResponse(**analyze_resume_text(request.resume_text))

@app.post("/career/jd/analyze", response_model=JdAnalyzeResponse)
def analyze_jd_api(request: JdAnalyzeRequest):
    validate_text_length(request.jd_text, "jd_text")

    return JdAnalyzeResponse(**analyze_jd_text(request.jd_text))

@app.post("/career/match", response_model=ResumeJdMatchResponse)
def match_resume_jd_api(request: ResumeJdMatchRequest):
    validate_text_length(request.resume_text, "resume_text")
    validate_text_length(request.jd_text, "jd_text")

    return ResumeJdMatchResponse(
        **match_resume_to_jd(
            resume_text=request.resume_text,
            jd_text=request.jd_text,
        )
    )

@app.post("/job-workflow/run", response_model=JobWorkflowResponse)
def run_job_workflow_api(request: JobWorkflowRequest):
    validate_text_length(request.resume_text, "resume_text")
    validate_text_length(request.jd_text, "jd_text")
    top_k = normalize_top_k(request.top_k)

    return JobWorkflowResponse(
        result=run_job_workflow(
            resume_text=request.resume_text,
            jd_text=request.jd_text,
            top_k=top_k,
        )
    )

@app.post("/job-workflow/run/files", response_model=JobWorkflowResponse)
async def run_job_workflow_files_api(
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(...),
    top_k: int = Form(3),
):
    resume_bytes = await resume_file.read()
    jd_bytes = await jd_file.read()
    validate_file_size(resume_bytes)
    validate_file_size(jd_bytes)
    top_k = normalize_top_k(top_k)

    try:
        resume_text = extract_document_text(resume_file.filename or "", resume_bytes)
        jd_text = extract_document_text(jd_file.filename or "", jd_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    validate_text_length(resume_text, "resume_text")
    validate_text_length(jd_text, "jd_text")

    result = run_job_workflow(
        resume_text=resume_text,
        jd_text=jd_text,
        top_k=top_k,
    )
    result["uploaded_files"] = {
        "resume_filename": resume_file.filename,
        "resume_char_count": len(resume_text),
        "jd_filename": jd_file.filename,
        "jd_char_count": len(jd_text),
    }

    return JobWorkflowResponse(result=result)

@app.post("/career/resume/upload/analyze", response_model=ResumeAnalyzeResponse)
async def analyze_resume_upload_api(file: UploadFile = File(...)):
    file_bytes = await file.read()
    validate_file_size(file_bytes)

    try:
        result = analyze_resume_file(file.filename or "", file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResumeAnalyzeResponse(**result)

@app.post("/career/jd/upload/analyze", response_model=JdAnalyzeResponse)
async def analyze_jd_upload_api(file: UploadFile = File(...)):
    file_bytes = await file.read()
    validate_file_size(file_bytes)

    try:
        result = analyze_jd_file(file.filename or "", file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JdAnalyzeResponse(**result)

@app.post("/career/match/upload", response_model=ResumeJdMatchResponse)
async def match_resume_upload_api(
    jd_text: str = Form(...),
    file: UploadFile = File(...),
):
    validate_text_length(jd_text, "jd_text")
    file_bytes = await file.read()
    validate_file_size(file_bytes)

    try:
        result = match_resume_file_to_jd(
            filename=file.filename or "",
            file_bytes=file_bytes,
            jd_text=jd_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResumeJdMatchResponse(**result)

@app.post("/career/match/files", response_model=ResumeJdMatchResponse)
async def match_resume_jd_files_api(
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(...),
):
    resume_bytes = await resume_file.read()
    jd_bytes = await jd_file.read()
    validate_file_size(resume_bytes)
    validate_file_size(jd_bytes)

    try:
        result = match_resume_file_to_jd_file(
            resume_filename=resume_file.filename or "",
            resume_file_bytes=resume_bytes,
            jd_filename=jd_file.filename or "",
            jd_file_bytes=jd_bytes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResumeJdMatchResponse(**result)

@app.get("/eval/rag", response_model=EvalResponse)
def run_rag_eval_api(_admin: bool = Depends(require_admin)):
    return EvalResponse(result=run_rag_eval())

@app.get("/eval/career", response_model=EvalResponse)
def run_career_eval_api(_admin: bool = Depends(require_admin)):
    return EvalResponse(result=run_career_eval())

@app.get("/eval/all", response_model=EvalResponse)
def run_all_evals_api(_admin: bool = Depends(require_admin)):
    return EvalResponse(result=run_all_evals())

@app.get("/sessions", response_model=SessionsResponse)
def list_sessions_api(_admin: bool = Depends(require_admin)):
    return SessionsResponse(sessions=list_session_ids())

@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session_detail(session_id: str, _admin: bool = Depends(require_admin)):
    session = require_session(session_id)

    message_count = len(session["messages"])
    turn_count = max((message_count - 1) // 2, 0)

    return SessionDetailResponse(
        session_id=session_id,
        message_count=message_count,
        turn_count=turn_count,
        trace_count=len(session["traces"]),
        ttl_seconds=get_session_ttl(session_id)
    )

@app.delete("/sessions/{session_id}", response_model=ClearResponse)
def delete_session_api(session_id: str, _admin: bool = Depends(require_admin)):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return ClearResponse(message=f"Session {session_id} deleted.")

@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        redis_ok = redis_client.ping()
    except Exception:
        redis_ok = False

    return HealthResponse(
        status="ok" if redis_ok else "error",
        redis=redis_ok
    )

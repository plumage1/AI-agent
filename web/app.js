const state = {
  activeTab: "workflow",
  lastRaw: null,
};

const views = {
  workflow: ["Workflow", "求职准备 Agentic Workflow"],
  chat: ["Chat", "多轮对话与工具调用入口"],
  rag: ["RAG", "知识库语义检索与引用来源"],
  career: ["Match", "简历和岗位 JD 匹配分析"],
  interview: ["Interview", "模拟面试追问与评分"],
  knowledge: ["Knowledge", "Markdown 知识库管理"],
  ops: ["Ops", "运行状态、trace、原始 JSON 与评测"],
};

function $(id) {
  return document.getElementById(id);
}

function getSessionId() {
  return $("sessionId").value.trim() || "web_user";
}

function getAdminToken() {
  return $("adminToken").value.trim();
}

function getHeaders(json = true) {
  const headers = {};
  const token = getAdminToken();

  if (json) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["X-Admin-Token"] = token;
  }

  return headers;
}

function getFile(id) {
  return $(id).files && $(id).files.length > 0 ? $(id).files[0] : null;
}

function showStatus(message, type = "ok") {
  const banner = $("statusBanner");
  banner.textContent = message;
  banner.className = `status-banner ${type}`;
}

function clearStatus() {
  const banner = $("statusBanner");
  banner.textContent = "";
  banner.className = "status-banner hidden";
}

function pretty(data) {
  if (typeof data === "string") {
    return data;
  }

  return JSON.stringify(data, null, 2);
}

function lines(title, items) {
  if (!items || items.length === 0) {
    return `${title}\n- 无\n`;
  }

  return `${title}\n${items.map((item) => `- ${item}`).join("\n")}\n`;
}

function renderChat(data) {
  const sources = data.citations && data.citations.length
    ? "\n\n参考来源：\n" + data.citations.map((item) => `- ${item.title} (${item.source_file})`).join("\n")
    : "";

  return `${data.answer || ""}${sources}`;
}

function renderRag(data) {
  const sourceText = (data.sources || []).map((source, index) => {
    return [
      `来源 ${index + 1}: ${source.title}`,
      `文件: ${source.source_file}`,
      `分数: ${source.score}`,
      source.content,
    ].join("\n");
  }).join("\n\n---\n\n");

  return [
    `检索器: ${data.retriever?.type || "-"}`,
    `命中数量: ${data.sources?.length || 0}`,
    "",
    sourceText || "没有检索到相关内容。",
  ].join("\n");
}

function renderMatch(data) {
  return [
    `匹配度: ${data.match_score}%`,
    `简历文件: ${data.source_filename || "文本输入"}`,
    `JD 文件: ${data.jd_source_filename || "文本输入"}`,
    "",
    lines("已匹配关键词:", data.matched_keywords),
    lines("缺失关键词:", data.missing_keywords),
    lines("简历优化建议:", data.resume_suggestions),
    lines("面试准备重点:", data.interview_focus),
  ].join("\n");
}

function renderWorkflow(data) {
  const result = data.result || data;
  const match = result.match || {};
  const upload = result.uploaded_files || {};
  const rag = result.rag || {};

  const learningTasks = (result.learning_tasks || []).map((task) => {
    return `${task.type}｜${task.topic}: ${task.task} 产出：${task.deliverable}`;
  });

  return [
    `目标: ${result.goal || "-"}`,
    `匹配度: ${match.match_score || 0}%`,
    `简历文件: ${upload.resume_filename || "文本输入"}`,
    `JD 文件: ${upload.jd_filename || "文本输入"}`,
    `RAG 状态: ${rag.available === false ? "不可用或未命中，已降级继续执行" : "可用"}`,
    "",
    lines("已匹配关键词:", match.matched_keywords),
    lines("缺失关键词:", match.missing_keywords),
    lines("学习任务:", learningTasks),
    lines("面试问题:", result.interview_questions),
    lines("简历项目表达:", result.resume_project_bullets),
    lines("下一步行动:", result.next_actions),
  ].join("\n");
}

function renderInterview(data) {
  if (data.question) {
    return [
      `主题: ${data.topic}`,
      `难度: ${data.difficulty}`,
      "",
      `面试问题:\n${data.question}`,
    ].join("\n");
  }

  return [
    `评分: ${data.score}`,
    `轮次: ${data.turn_count}`,
    `平均分: ${data.average_score}`,
    "",
    `反馈:\n${data.feedback}`,
    "",
    `参考回答:\n${data.reference_answer}`,
    "",
    `追问:\n${data.follow_up_question}`,
  ].join("\n");
}

function renderKnowledge(data) {
  if (data.documents) {
    return lines("知识库文档:", data.documents);
  }

  if (data.chunks) {
    return `切片数量: ${data.chunk_count}\n\n${data.chunks.map((chunk) => `${chunk.title}\n${chunk.content}`).join("\n\n---\n\n")}`;
  }

  return pretty(data);
}

async function api(path, options = {}) {
  clearStatus();

  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = data.detail || data;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return data;
}

async function run(targetId, task, formatter = pretty) {
  const target = $(targetId);
  target.textContent = "Loading...";

  try {
    const data = await task();
    state.lastRaw = data;
    target.textContent = formatter(data);
    showStatus("Done");
  } catch (error) {
    target.textContent = error.message;
    showStatus(error.message, "error");
  }
}

function switchTab(tab) {
  state.activeTab = tab;
  const [title, subtitle] = views[tab];

  $("viewTitle").textContent = title;
  $("viewSubtitle").textContent = subtitle;

  document.querySelectorAll(".nav-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });

  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${tab}`);
  });
}

function bindTabs() {
  document.querySelectorAll(".nav-tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
}

function bindSettings() {
  const adminToken = localStorage.getItem("adminToken");
  const sessionId = localStorage.getItem("sessionId");

  if (adminToken) {
    $("adminToken").value = adminToken;
  }

  if (sessionId) {
    $("sessionId").value = sessionId;
  }

  $("adminToken").addEventListener("input", () => {
    localStorage.setItem("adminToken", $("adminToken").value.trim());
  });

  $("sessionId").addEventListener("input", () => {
    localStorage.setItem("sessionId", getSessionId());
  });
}

function bindChat() {
  $("sendChat").addEventListener("click", () => run("chatOutput", () => api("/chat", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      session_id: getSessionId(),
      message: $("chatMessage").value,
    }),
  }), renderChat));

  $("clearSession").addEventListener("click", () => run("chatOutput", () => api(`/clear?session_id=${encodeURIComponent(getSessionId())}`, {
    method: "POST",
    headers: getHeaders(false),
  })));
}

function bindRag() {
  $("searchRag").addEventListener("click", () => run("ragOutput", () => api("/rag/search", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      query: $("ragQuery").value,
      top_k: Number($("ragTopK").value || 3),
    }),
  }), renderRag));

  $("rebuildIndex").addEventListener("click", () => run("ragOutput", () => api("/rag/index/rebuild", {
    method: "POST",
    headers: getHeaders(false),
  })));
}

function bindCareer() {
  $("matchCareer").addEventListener("click", () => run("careerOutput", () => api("/career/match", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      resume_text: $("resumeText").value,
      jd_text: $("jdText").value,
    }),
  }), renderMatch));

  $("matchCareerFiles").addEventListener("click", () => {
    const resumeFile = getFile("resumeFile");
    const jdFile = getFile("jdFile");

    if (!resumeFile || !jdFile) {
      showStatus("请同时选择简历文件和 JD 文件", "error");
      return;
    }

    const formData = new FormData();
    formData.append("resume_file", resumeFile);
    formData.append("jd_file", jdFile);

    run("careerOutput", () => api("/career/match/files", {
      method: "POST",
      headers: getHeaders(false),
      body: formData,
    }), renderMatch);
  });
}

function bindWorkflow() {
  $("runWorkflow").addEventListener("click", () => run("workflowOutput", () => api("/job-workflow/run", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      resume_text: $("workflowResumeText").value,
      jd_text: $("workflowJdText").value,
      top_k: Number($("workflowTopK").value || 3),
    }),
  }), renderWorkflow));

  $("runWorkflowFiles").addEventListener("click", () => {
    const resumeFile = getFile("workflowResumeFile");
    const jdFile = getFile("workflowJdFile");

    if (!resumeFile || !jdFile) {
      showStatus("请同时选择简历文件和 JD 文件", "error");
      return;
    }

    const formData = new FormData();
    formData.append("resume_file", resumeFile);
    formData.append("jd_file", jdFile);
    formData.append("top_k", String(Number($("workflowTopK").value || 3)));

    run("workflowOutput", () => api("/job-workflow/run/files", {
      method: "POST",
      headers: getHeaders(false),
      body: formData,
    }), renderWorkflow);
  });
}

function bindInterview() {
  $("startInterview").addEventListener("click", () => run("interviewOutput", () => api("/interview/start", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      session_id: getSessionId(),
      topic: $("interviewTopic").value,
      difficulty: $("interviewDifficulty").value,
    }),
  }), renderInterview));

  $("submitInterview").addEventListener("click", () => run("interviewOutput", () => api("/interview/answer", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      session_id: getSessionId(),
      answer: $("interviewAnswer").value,
    }),
  }), renderInterview));

  $("resetInterview").addEventListener("click", () => run("interviewOutput", () => api(`/interview/reset?session_id=${encodeURIComponent(getSessionId())}`, {
    method: "POST",
    headers: getHeaders(false),
  })));
}

function bindKnowledge() {
  $("saveKnowledge").addEventListener("click", () => run("knowledgeOutput", () => api("/knowledge/documents", {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      filename: $("knowledgeFilename").value,
      content: $("knowledgeContent").value,
    }),
  }), renderKnowledge));

  $("listKnowledge").addEventListener("click", () => run("knowledgeOutput", () => api("/knowledge/documents", {
    headers: getHeaders(false),
  }), renderKnowledge));

  $("listChunks").addEventListener("click", () => run("knowledgeOutput", () => api("/knowledge/chunks", {
    headers: getHeaders(false),
  }), renderKnowledge));
}

function bindOps() {
  $("healthButton").addEventListener("click", () => run("opsOutput", () => api("/health", {
    headers: getHeaders(false),
  })));

  $("getTools").addEventListener("click", () => run("opsOutput", () => api("/tools", {
    headers: getHeaders(false),
  })));

  $("getSessions").addEventListener("click", () => run("opsOutput", () => api("/sessions", {
    headers: getHeaders(false),
  })));

  $("getTraces").addEventListener("click", () => run("opsOutput", () => api(`/trace?session_id=${encodeURIComponent(getSessionId())}`, {
    headers: getHeaders(false),
  })));

  $("ragStatus").addEventListener("click", () => run("opsOutput", () => api("/rag/status", {
    headers: getHeaders(false),
  })));

  $("showLastRaw").addEventListener("click", () => {
    $("opsOutput").textContent = state.lastRaw ? pretty(state.lastRaw) : "暂无原始 JSON";
    switchTab("ops");
  });

  $("runEval").addEventListener("click", () => run("opsOutput", () => api("/eval/all", {
    headers: getHeaders(false),
  })));
}

function boot() {
  bindTabs();
  bindSettings();
  bindChat();
  bindRag();
  bindCareer();
  bindWorkflow();
  bindInterview();
  bindKnowledge();
  bindOps();
}

boot();

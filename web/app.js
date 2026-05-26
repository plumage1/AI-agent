const state = {
  lastRaw: null,
  busy: false,
  recentConversations: [],
  pendingAttachments: [],
};

const RECENT_STORAGE_KEY = "recentConversations";

function $(id) {
  return document.getElementById(id);
}

function getSessionId() {
  return $("sessionId").value.trim() || "web_user";
}

function getAdminToken() {
  return $("adminToken").value.trim();
}

function createSessionId() {
  return `session_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function createAttachmentId() {
  return `att_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 900px)").matches;
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

function showStatus(message, type = "ok") {
  const banner = $("statusBanner");
  banner.textContent = message;
  banner.className = `status-banner ${type}`;
  $("statusText").textContent = message;
}

function clearStatus() {
  const banner = $("statusBanner");
  banner.textContent = "";
  banner.className = "status-banner hidden";
  $("statusText").textContent = "Function Calling · Hybrid RAG · Trace";
}

function pretty(data) {
  return typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function sanitizeModelText(text) {
  return String(text || "")
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<\/?think>/gi, "")
    .trim();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatInlineText(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function formatText(text) {
  const cleaned = sanitizeModelText(text);
  const blocks = cleaned.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);

  if (!blocks.length) {
    return "<p>没有返回内容。</p>";
  }

  return blocks.map((block) => {
    const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);

    if (!lines.length) {
      return "";
    }

    if (lines.every((line) => /^[-*]\s+/.test(line))) {
      return `<ul>${lines.map((line) => `<li>${formatInlineText(line.replace(/^[-*]\s+/, ""))}</li>`).join("")}</ul>`;
    }

    if (lines.every((line) => /^\d+\.\s+/.test(line))) {
      return `<ol>${lines.map((line) => `<li>${formatInlineText(line.replace(/^\d+\.\s+/, ""))}</li>`).join("")}</ol>`;
    }

    if (lines.length === 1 && /^#{1,3}\s+/.test(lines[0])) {
      return `<p><strong>${formatInlineText(lines[0].replace(/^#{1,3}\s+/, ""))}</strong></p>`;
    }

    return `<p>${lines.map((line) => formatInlineText(line)).join("<br>")}</p>`;
  }).join("");
}

function renderWelcomeMessage() {
  return `
    <p>你好，我是你的求职 Agent。可以帮你分析 JD、匹配简历、检索知识库、生成学习计划和模拟面试。</p>
    <div class="suggestions">
      <button data-prompt="帮我分析这段 JD：需要 Python、Redis、Docker 和 RAG 项目经验">分析 JD</button>
      <button data-prompt="我的简历熟悉 Python、FastAPI、Redis，做过 RAG Agent 项目。帮我匹配 JD：需要 Python、Redis、Docker 和 RAG 项目经验。">匹配简历</button>
      <button data-prompt="Redis RDB 和 AOF 有什么区别？">问知识库</button>
    </div>
  `;
}

function formatFileSize(size) {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function inferAttachmentKind(file) {
  const name = (file.name || "").toLowerCase();
  const isImage = file.type.startsWith("image/");

  if (name.includes("resume") || name.includes("cv") || name.includes("简历")) {
    return "resume";
  }

  if (name.includes("jd") || name.includes("job") || name.includes("职位") || name.includes("岗位")) {
    return "jd";
  }

  if (isImage) {
    return "jd";
  }

  if (name.endsWith(".md") || name.endsWith(".txt")) {
    return "knowledge";
  }

  return "knowledge";
}

function loadRecentConversations() {
  try {
    const raw = localStorage.getItem(RECENT_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
    return [];
  }
}

function saveRecentConversations() {
  localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(state.recentConversations));
}

function extractConversationTitle() {
  const firstUserMessage = document.querySelector(".message.user .message-body");
  const raw = firstUserMessage?.textContent?.trim() || "";
  if (!raw) {
    return "未命名对话";
  }
  return raw.length > 22 ? `${raw.slice(0, 22)}...` : raw;
}

function hasMeaningfulConversation() {
  return Boolean(document.querySelector(".message.user"));
}

function snapshotCurrentConversation() {
  if (!hasMeaningfulConversation()) {
    return null;
  }

  return {
    sessionId: getSessionId(),
    title: extractConversationTitle(),
    html: $("messages").innerHTML,
    updatedAt: Date.now(),
  };
}

function renderRecentConversations(activeSessionId = "") {
  const list = $("recentList");
  list.innerHTML = "";

  if (!state.recentConversations.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "新对话之后，这里的最近记录会自动出现。";
    list.appendChild(empty);
    return;
  }

  state.recentConversations.forEach((item) => {
    const button = document.createElement("button");
    button.className = `history-item${item.sessionId === activeSessionId ? " active" : ""}`;
    button.textContent = item.title || "未命名对话";
    button.addEventListener("click", () => {
      $("sessionId").value = item.sessionId;
      localStorage.setItem("sessionId", item.sessionId);
      clearStatus();
      closeAttachMenu();
      toggleTools(false);
      $("messages").innerHTML = item.html || "";
      bindPromptButtons($("messages"));
      renderRecentConversations(item.sessionId);
      scrollToBottom();
    });
    list.appendChild(button);
  });
}

function renderPendingAttachments() {
  const tray = $("pendingAttachments");
  tray.innerHTML = "";

  if (!state.pendingAttachments.length) {
    tray.classList.add("hidden");
    return;
  }

  tray.classList.remove("hidden");

  state.pendingAttachments.forEach((item) => {
    const row = document.createElement("div");
    row.className = "attachment-chip";

    const main = document.createElement("div");
    main.className = "attachment-main";
    main.innerHTML = `
      <div class="attachment-name">${escapeHtml(item.file.name)}</div>
      <div class="attachment-meta">${formatFileSize(item.file.size)} · 已添加到聊天框</div>
    `;

    const select = document.createElement("select");
    select.className = "attachment-kind";
    select.innerHTML = `
      <option value="resume">简历</option>
      <option value="jd">JD</option>
      <option value="knowledge">知识库</option>
    `;
    select.value = item.kind;
    select.addEventListener("change", () => {
      item.kind = select.value;
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "attachment-remove";
    remove.textContent = "×";
    remove.title = "移除附件";
    remove.addEventListener("click", () => {
      state.pendingAttachments = state.pendingAttachments.filter((attachment) => attachment.id !== item.id);
      renderPendingAttachments();
    });

    row.append(main, select, remove);
    tray.appendChild(row);
  });
}

function addPendingAttachments(files) {
  const incoming = Array.from(files || []).filter((file) => file instanceof File);

  if (!incoming.length) {
    return;
  }

  incoming.forEach((file) => {
    state.pendingAttachments.push({
      id: createAttachmentId(),
      file,
      kind: inferAttachmentKind(file),
    });
  });

  renderPendingAttachments();
}

async function processPendingAttachments() {
  const queue = [...state.pendingAttachments];
  state.pendingAttachments = [];
  renderPendingAttachments();

  for (const attachment of queue) {
    const routeMap = {
      resume: {
        label: "上传简历",
        path: "/career/resume/upload/analyze",
        formatter: renderResumeUploadSummary,
      },
      jd: {
        label: "上传 JD / 图片",
        path: "/career/jd/upload/analyze",
        formatter: renderJdUploadSummary,
      },
      knowledge: {
        label: "导入知识库",
        path: "/knowledge/import",
        formatter: renderKnowledgeUploadSummary,
        requireAdmin: true,
      },
    };

    const route = routeMap[attachment.kind];
    if (!route) {
      continue;
    }

    await uploadSingleFile({
      label: route.label,
      path: route.path,
      file: attachment.file,
      formatter: route.formatter,
      requireAdmin: route.requireAdmin,
    });
  }
}

function archiveCurrentConversation() {
  const snapshot = snapshotCurrentConversation();
  if (!snapshot) {
    return;
  }

  state.recentConversations = [
    snapshot,
    ...state.recentConversations.filter((item) => item.sessionId !== snapshot.sessionId),
  ].slice(0, 10);

  saveRecentConversations();
  renderRecentConversations();
}

function resetConversationView() {
  $("messages").innerHTML = "";

  const article = document.createElement("article");
  article.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";
  article.appendChild(avatar);

  const body = document.createElement("div");
  body.className = "message-body";
  body.innerHTML = renderWelcomeMessage();
  article.appendChild(body);

  $("messages").appendChild(article);
  bindPromptButtons(body);
  scrollToBottom();
}

function scrollToBottom() {
  const messages = $("messages");
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, content, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "AI";
    article.appendChild(avatar);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  body.innerHTML = formatText(content);

  if (options.meta || options.citations?.length) {
    body.appendChild(renderMeta(options));
  }

  article.appendChild(body);
  $("messages").appendChild(article);
  scrollToBottom();
  return article;
}

function renderMeta({ meta, citations }) {
  const wrap = document.createElement("div");
  wrap.className = "message-meta";

  if (meta?.toolName) {
    const pill = document.createElement("span");
    pill.className = "meta-pill";
    pill.textContent = `工具：${meta.toolName}`;
    wrap.appendChild(pill);
  }

  if (Number.isInteger(meta?.traceId)) {
    const pill = document.createElement("span");
    pill.className = "meta-pill";
    pill.textContent = `Trace #${meta.traceId}`;
    wrap.appendChild(pill);
  }

  (citations || []).forEach((item) => {
    const source = document.createElement("span");
    source.className = "source-pill";
    source.textContent = `${item.id || "S"} ${item.title || item.source_file || "source"}`;
    wrap.appendChild(source);
  });

  return wrap;
}

function addLoadingMessage() {
  return addMessage("assistant", "正在思考中...");
}

function replaceMessage(element, content, options = {}) {
  const body = element.querySelector(".message-body");
  body.innerHTML = formatText(content);
  if (options.meta || options.citations?.length) {
    body.appendChild(renderMeta(options));
  }
  scrollToBottom();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = data.detail || data;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  state.lastRaw = data;
  return data;
}

function setBusy(value) {
  state.busy = value;
  $("sendChat").disabled = value;
  $("chatMessage").disabled = value;
  $("attachButton").disabled = value;
}

function resizeComposer() {
  const input = $("chatMessage");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function fillPrompt(prompt) {
  $("chatMessage").value = prompt;
  resizeComposer();
  $("chatMessage").focus();
}

function bindPromptButtons(root = document) {
  root.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => fillPrompt(button.dataset.prompt));
  });
}

async function submitChat(message) {
  const text = message.trim();
  if ((!text && !state.pendingAttachments.length) || state.busy) {
    return;
  }

  if (state.pendingAttachments.length) {
    await processPendingAttachments();
  }

  if (!text) {
    return;
  }

  clearStatus();
  closeAttachMenu();
  addMessage("user", text);
  $("chatMessage").value = "";
  resizeComposer();
  const loading = addLoadingMessage();
  setBusy(true);

  try {
    const data = await api("/chat", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        session_id: getSessionId(),
        message: text,
      }),
    });

    replaceMessage(loading, data.answer || "完成。", {
      meta: {
        toolName: data.tool_name,
        traceId: data.trace_id,
      },
      citations: data.citations || [],
    });
    showStatus(data.used_tool ? "已调用工具" : "已回复");
  } catch (error) {
    replaceMessage(loading, error.message);
    showStatus(error.message, "error");
  } finally {
    setBusy(false);
    $("chatMessage").focus();
  }
}

function toggleTools(show = null) {
  const panel = $("toolPanel");
  const shouldShow = show === null ? panel.classList.contains("hidden") : show;
  panel.classList.toggle("hidden", !shouldShow);

  if (shouldShow) {
    closeAttachMenu();
  }
}

function renderToolForm(type) {
  const form = $("toolForm");
  form.className = "tool-form";

  const templates = {
    workflow: `
      <label>简历<textarea id="toolResume" rows="5">我熟悉 Python、FastAPI、Redis，做过 RAG Agent 项目。</textarea></label>
      <label>JD<textarea id="toolJd" rows="5">需要 Python、Redis、Docker 和 RAG 项目经验，能搭建自动化工作流并调用 AI 模型接口。</textarea></label>
      <label>Top K<input id="toolTopK" type="number" min="1" max="10" value="3"></label>
      <div class="tool-form-actions"><button class="primary-button" id="runToolWorkflow">运行 Workflow</button></div>
    `,
    match: `
      <label>简历<textarea id="toolResume" rows="5">我熟悉 Python、FastAPI、Redis，做过 RAG Agent 项目。</textarea></label>
      <label>JD<textarea id="toolJd" rows="5">需要 Python、Redis、Docker 和 RAG 项目经验。</textarea></label>
      <div class="tool-form-actions"><button class="primary-button" id="runToolMatch">计算匹配</button></div>
    `,
    rag: `
      <label>问题<textarea id="toolQuery" rows="3">Redis 缓存雪崩是什么？</textarea></label>
      <label>Top K<input id="toolTopK" type="number" min="1" max="10" value="3"></label>
      <div class="tool-form-actions"><button class="primary-button" id="runToolRag">检索知识库</button></div>
    `,
    interview: `
      <label>主题<input id="toolTopic" value="RAG 项目经验"></label>
      <label>难度<select id="toolDifficulty"><option>简单</option><option selected>中等</option><option>困难</option></select></label>
      <div class="tool-form-actions"><button class="primary-button" id="runToolInterview">开始面试</button></div>
    `,
  };

  form.innerHTML = templates[type] || "";
  bindToolFormActions(type);
}

function bindToolFormActions(type) {
  if (type === "workflow") {
    $("runToolWorkflow").addEventListener("click", () => runToolResult("求职准备 Workflow", "/job-workflow/run", {
      resume_text: $("toolResume").value,
      jd_text: $("toolJd").value,
      top_k: Number($("toolTopK").value || 3),
    }, renderWorkflowSummary));
  }

  if (type === "match") {
    $("runToolMatch").addEventListener("click", () => runToolResult("简历匹配分析", "/career/match", {
      resume_text: $("toolResume").value,
      jd_text: $("toolJd").value,
    }, renderMatchSummary));
  }

  if (type === "rag") {
    $("runToolRag").addEventListener("click", () => runToolResult("RAG 检索", "/rag/search", {
      query: $("toolQuery").value,
      top_k: Number($("toolTopK").value || 3),
    }, renderRagSummary));
  }

  if (type === "interview") {
    $("runToolInterview").addEventListener("click", () => runToolResult("模拟面试", "/interview/start", {
      session_id: getSessionId(),
      topic: $("toolTopic").value,
      difficulty: $("toolDifficulty").value,
    }, renderInterviewSummary));
  }
}

async function runToolResult(title, path, body, formatter) {
  if (state.busy) {
    return;
  }

  closeAttachMenu();
  addMessage("user", title);
  const loading = addLoadingMessage();
  setBusy(true);

  try {
    const data = await api(path, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(body),
    });
    replaceMessage(loading, formatter(data));
    showStatus("工具完成");
  } catch (error) {
    replaceMessage(loading, error.message);
    showStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function listBlock(title, items) {
  const values = items || [];
  if (!values.length) {
    return `${title}\n- 无`;
  }
  return `${title}\n${values.map((item) => `- ${typeof item === "string" ? item : JSON.stringify(item)}`).join("\n")}`;
}

function renderMatchSummary(data) {
  return [
    `匹配度：${data.match_score}%`,
    listBlock("已匹配关键词", data.matched_keywords),
    listBlock("缺失关键词", data.missing_keywords),
    listBlock("简历优化建议", data.resume_suggestions),
    listBlock("面试准备重点", data.interview_focus),
  ].join("\n\n");
}

function renderWorkflowSummary(data) {
  const result = data.result || data;
  const match = result.match || {};
  const tasks = (result.learning_tasks || []).map((task) => `${task.topic}: ${task.task}`);

  return [
    result.goal || "Workflow 已完成。",
    `匹配度：${match.match_score || 0}%`,
    listBlock("已匹配关键词", match.matched_keywords),
    listBlock("缺失关键词", match.missing_keywords),
    listBlock("学习任务", tasks),
    listBlock("面试问题", result.interview_questions),
    listBlock("下一步行动", result.next_actions),
  ].join("\n\n");
}

function renderRagSummary(data) {
  const citations = (data.citations || []).map((item) => `${item.id || ""} ${item.title} (${item.source_file})`);
  const sources = (data.sources || []).map((source) => `${source.title}\n${source.content}`);

  return [
    `检索器：${data.retriever?.type || "-"}`,
    listBlock("引用来源", citations),
    listBlock("命中内容", sources),
  ].join("\n\n");
}

function renderInterviewSummary(data) {
  return [
    `主题：${data.topic}`,
    `难度：${data.difficulty}`,
    `面试问题：\n${data.question}`,
  ].join("\n\n");
}

function renderResumeUploadSummary(data) {
  return [
    `已解析简历：${data.source_filename || "未命名文件"}`,
    `文本长度：${data.char_count || 0}`,
    listBlock("识别技能", data.skills),
    listBlock("优化建议", data.suggestions),
  ].join("\n\n");
}

function renderJdUploadSummary(data) {
  return [
    `已解析 JD：${data.source_filename || "未命名文件"}`,
    `文本长度：${data.char_count || 0}`,
    listBlock("岗位关键词", data.keywords),
    listBlock("准备重点", data.preparation_focus),
  ].join("\n\n");
}

function renderKnowledgeUploadSummary(data) {
  return [
    `已导入知识库：${data.source_filename}`,
    `保存文件名：${data.filename}`,
    `文本长度：${data.char_count}`,
    `索引缓存：${data.cache_cleared ? "已清空，等待重建" : "未变更"}`,
    `内容预览：\n${data.content_preview || "无"}`,
  ].join("\n\n");
}

function closeAttachMenu() {
  $("attachMenu").classList.add("hidden");
}

function toggleAttachMenu(force = null) {
  const menu = $("attachMenu");
  const shouldShow = force === null ? menu.classList.contains("hidden") : force;
  menu.classList.toggle("hidden", !shouldShow);
  if (shouldShow) {
    toggleTools(false);
  }
}

async function uploadSingleFile({ label, path, file, formatter, requireAdmin = false }) {
  if (!file || state.busy) {
    return;
  }

  if (requireAdmin && !getAdminToken()) {
    const message = "导入知识库前，请先填写 Admin Token。";
    addMessage("assistant", message);
    showStatus(message, "error");
    closeAttachMenu();
    return;
  }

  clearStatus();
  closeAttachMenu();
  addMessage("user", `${label}：${file.name}`);
  const loading = addLoadingMessage();
  setBusy(true);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const data = await api(path, {
      method: "POST",
      headers: getHeaders(false),
      body: formData,
    });

    replaceMessage(loading, formatter(data));
    showStatus("文件已处理");
  } catch (error) {
    replaceMessage(loading, error.message);
    showStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function bindToolPanel() {
  $("closeTools").addEventListener("click", () => toggleTools(false));

  document.querySelectorAll(".tool-card").forEach((button) => {
    button.addEventListener("click", () => renderToolForm(button.dataset.tool));
  });
}

function bindSidebar() {
  $("openSidebar").addEventListener("click", () => {
    if (isMobileLayout()) {
      document.body.classList.toggle("sidebar-open");
      return;
    }
    document.body.classList.toggle("sidebar-collapsed");
  });

  $("newChat").addEventListener("click", async () => {
    archiveCurrentConversation();
    const nextSessionId = createSessionId();
    $("sessionId").value = nextSessionId;
    localStorage.setItem("sessionId", nextSessionId);
    clearStatus();
    closeAttachMenu();
    toggleTools(false);
    resetConversationView();
  });

  document.querySelector(".sidebar").addEventListener("click", (event) => {
    if (isMobileLayout() && event.target.closest("button")) {
      document.body.classList.remove("sidebar-open");
    }
  });

  window.matchMedia("(max-width: 900px)").addEventListener("change", () => {
    document.body.classList.remove("sidebar-open");
  });
}

function bindAttachments() {
  const routes = {
    resume: {
      input: $("resumeUploadInput"),
      label: "上传简历",
      path: "/career/resume/upload/analyze",
      formatter: renderResumeUploadSummary,
    },
    jd: {
      input: $("jdUploadInput"),
      label: "上传 JD / 图片",
      path: "/career/jd/upload/analyze",
      formatter: renderJdUploadSummary,
    },
    knowledge: {
      input: $("knowledgeUploadInput"),
      label: "导入知识库",
      path: "/knowledge/import",
      formatter: renderKnowledgeUploadSummary,
      requireAdmin: true,
    },
  };

  $("attachButton").addEventListener("click", (event) => {
    event.stopPropagation();
    toggleAttachMenu();
  });

  document.querySelectorAll("[data-attach-kind]").forEach((button) => {
    button.addEventListener("click", () => {
      const kind = button.dataset.attachKind;

      if (kind === "tools") {
        closeAttachMenu();
        toggleTools(true);
        return;
      }

      const target = routes[kind];
      if (target) {
        target.input.click();
      }
    });
  });

  Object.values(routes).forEach((route) => {
    route.input.addEventListener("change", async () => {
      const [file] = route.input.files || [];
      route.input.value = "";
      await uploadSingleFile({
        label: route.label,
        path: route.path,
        file,
        formatter: route.formatter,
        requireAdmin: route.requireAdmin,
      });
    });
  });

  document.addEventListener("click", (event) => {
    const menu = $("attachMenu");
    const button = $("attachButton");
    if (menu.classList.contains("hidden")) {
      return;
    }
    if (menu.contains(event.target) || button.contains(event.target)) {
      return;
    }
    closeAttachMenu();
  });

  const composer = $("chatForm");
  const dropHint = $("dropHint");

  function setDragState(active) {
    composer.classList.toggle("drag-active", active);
    dropHint.classList.toggle("hidden", !active);
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    composer.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      setDragState(true);
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    composer.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!composer.contains(event.relatedTarget)) {
        setDragState(false);
      }
    });
  });

  composer.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragState(false);
    addPendingAttachments(event.dataTransfer?.files || []);
  });
}

function bindSettings() {
  const adminToken = localStorage.getItem("adminToken");
  const sessionId = localStorage.getItem("sessionId");
  state.recentConversations = loadRecentConversations();

  if (adminToken) {
    $("adminToken").value = adminToken;
  }

  if (sessionId) {
    $("sessionId").value = sessionId;
  } else {
    const nextSessionId = createSessionId();
    $("sessionId").value = nextSessionId;
    localStorage.setItem("sessionId", nextSessionId);
  }

  $("adminToken").addEventListener("input", () => {
    localStorage.setItem("adminToken", $("adminToken").value.trim());
  });

  $("sessionId").addEventListener("input", () => {
    localStorage.setItem("sessionId", getSessionId());
  });
}

function bindComposer() {
  $("chatForm").addEventListener("submit", (event) => {
    event.preventDefault();
    submitChat($("chatMessage").value);
  });

  $("chatMessage").addEventListener("input", resizeComposer);
  $("chatMessage").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitChat($("chatMessage").value);
    }
  });
}

function bindTopActions() {
  $("healthButton").addEventListener("click", async () => {
    const loading = addLoadingMessage();
    try {
      const data = await api("/health", { headers: getHeaders(false) });
      replaceMessage(loading, `服务状态：${data.status}\nRedis：${data.redis ? "可用" : "不可用"}`);
      showStatus("状态已更新");
    } catch (error) {
      replaceMessage(loading, error.message);
      showStatus(error.message, "error");
    }
  });
}

function boot() {
  bindSettings();
  bindSidebar();
  bindComposer();
  bindPromptButtons();
  bindToolPanel();
  bindTopActions();
  bindAttachments();
  resizeComposer();
  renderRecentConversations(getSessionId());
  resetConversationView();
}

boot();

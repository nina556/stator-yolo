const $ = (selector) => document.querySelector(selector);
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const state = {
  images: [],
  index: -1,
  image: null,
  boxes: [],
  drag: null,
  pan: null,
  panX: null,
  panY: null,
  fitScale: 1,
  zoom: 1,
};
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 5;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function setStatus(text, ok = true) {
  $("#status").textContent = text;
  $(".status").classList.toggle("online", ok);
}

async function refresh(selectName = null) {
  const data = await api("/api/images");
  state.images = data.images;
  const labeled = state.images.filter((item) => item.labeled).length;
  $("#progress").textContent = `${labeled} / ${state.images.length}`;
  $("#progressBar").style.width = `${state.images.length ? labeled / state.images.length * 100 : 0}%`;
  if (selectName) state.index = state.images.findIndex((item) => item.name === selectName);
  if (state.index >= state.images.length) state.index = state.images.length - 1;
  renderList();
  updateNavigation();
  setStatus("本地服务已连接");
}

function renderList() {
  const list = $("#imageList");
  list.innerHTML = "";
  state.images.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = `image-item ${item.labeled ? "labeled" : ""} ${index === state.index ? "active" : ""}`;
    button.innerHTML = `<span class="check"></span><span class="image-name"></span>`;
    button.querySelector(".image-name").textContent = item.name;
    button.onclick = () => loadImage(index);
    list.appendChild(button);
  });
}

async function loadImage(index) {
  if (index < 0 || index >= state.images.length) return;
  state.index = index;
  const name = state.images[index].name;
  const [labelData, image] = await Promise.all([
    api(`/api/labels/${encodeURIComponent(name)}`),
    new Promise((resolve, reject) => {
      const item = new Image();
      item.onload = () => resolve(item);
      item.onerror = reject;
      item.src = `/api/image/${encodeURIComponent(name)}?t=${Date.now()}`;
    }),
  ]);
  state.image = image;
  state.boxes = labelData.boxes;
  state.zoom = 1;
  state.panX = null;
  state.panY = null;
  fitCanvas();
  renderList();
  updateNavigation();
}

function fitCanvas() {
  if (!state.image) return;
  const stage = $("#stage");
  state.fitScale = Math.min(
    (stage.clientWidth - 24) / state.image.width,
    (stage.clientHeight - 24) / state.image.height,
    1,
  );
  state.panX = stage.clientWidth / 2;
  state.panY = stage.clientHeight / 2;
  renderCanvasSize();
}

function renderCanvasSize() {
  if (!state.image) return;
  const scale = state.fitScale * state.zoom;
  canvas.width = Math.max(1, Math.round(state.image.width * scale));
  canvas.height = Math.max(1, Math.round(state.image.height * scale));
  canvas.style.left = `${state.panX}px`;
  canvas.style.top = `${state.panY}px`;
  canvas.style.display = "block";
  $("#emptyState").style.display = "none";
  $("#zoomValue").textContent = `${Math.round(state.zoom * 100)}%`;
  $("#zoomOutButton").disabled = state.zoom <= MIN_ZOOM;
  $("#zoomInButton").disabled = state.zoom >= MAX_ZOOM;
  draw();
}

function setZoom(nextZoom, clientX = null, clientY = null) {
  if (!state.image) return;
  const stage = $("#stage");
  const stageRect = stage.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const anchorX = clientX === null ? stageRect.left + stage.clientWidth / 2 : clientX;
  const anchorY = clientY === null ? stageRect.top + stage.clientHeight / 2 : clientY;
  const imageX = (anchorX - canvasRect.left) / canvasRect.width;
  const imageY = (anchorY - canvasRect.top) / canvasRect.height;

  state.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextZoom));
  renderCanvasSize();

  state.panX = anchorX - stageRect.left - imageX * canvas.width + canvas.width / 2;
  state.panY = anchorY - stageRect.top - imageY * canvas.height + canvas.height / 2;
  canvas.style.left = `${state.panX}px`;
  canvas.style.top = `${state.panY}px`;
}

function resetZoom() {
  if (!state.image) return;
  const stage = $("#stage");
  state.zoom = 1;
  state.panX = stage.clientWidth / 2;
  state.panY = stage.clientHeight / 2;
  renderCanvasSize();
}

function draw() {
  if (!state.image) return;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "bold 13px sans-serif";
  for (const [cx, cy, w, h] of state.boxes) {
    const x = (cx - w / 2) * canvas.width;
    const y = (cy - h / 2) * canvas.height;
    const width = w * canvas.width;
    const height = h * canvas.height;
    ctx.strokeStyle = "#9ee37d";
    ctx.fillStyle = "rgba(158, 227, 125, .12)";
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = "#101510";
    ctx.fillRect(x, Math.max(0, y - 20), 52, 20);
    ctx.fillStyle = "#9ee37d";
    ctx.fillText("stator", x + 5, Math.max(14, y - 5));
  }
  if (state.drag) {
    const { x0, y0, x1, y1 } = state.drag;
    ctx.strokeStyle = "#ff9f43";
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
  }
  $("#imageMeta").textContent = `${state.boxes.length} 个框`;
}

function pointer(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)),
  };
}

const stage = $("#stage");

stage.addEventListener("pointerdown", (event) => {
  if (event.button !== 2 || !state.image) return;
  state.pan = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    panX: state.panX,
    panY: state.panY,
  };
  stage.classList.add("panning");
  stage.setPointerCapture(event.pointerId);
  event.preventDefault();
});
stage.addEventListener("pointermove", (event) => {
  if (!state.pan) return;
  state.panX = state.pan.panX + event.clientX - state.pan.x;
  state.panY = state.pan.panY + event.clientY - state.pan.y;
  canvas.style.left = `${state.panX}px`;
  canvas.style.top = `${state.panY}px`;
  event.preventDefault();
});
function stopPanning(event) {
  if (!state.pan) return;
  if (stage.hasPointerCapture(state.pan.pointerId)) {
    stage.releasePointerCapture(state.pan.pointerId);
  }
  state.pan = null;
  stage.classList.remove("panning");
  event?.preventDefault();
}
stage.addEventListener("pointerup", stopPanning);
stage.addEventListener("pointercancel", stopPanning);
stage.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  const point = pointer(event);
  state.drag = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  Object.assign(state.drag, { x1: pointer(event).x, y1: pointer(event).y });
  draw();
});
canvas.addEventListener("pointerup", (event) => {
  if (!state.drag) return;
  const point = pointer(event);
  const x0 = Math.min(state.drag.x0, point.x);
  const y0 = Math.min(state.drag.y0, point.y);
  const x1 = Math.max(state.drag.x0, point.x);
  const y1 = Math.max(state.drag.y0, point.y);
  state.drag = null;
  if (x1 - x0 >= 5 && y1 - y0 >= 5) {
    state.boxes.push([(x0 + x1) / 2 / canvas.width, (y0 + y1) / 2 / canvas.height, (x1 - x0) / canvas.width, (y1 - y0) / canvas.height]);
  }
  draw();
});
canvas.addEventListener("pointercancel", () => {
  state.drag = null;
  draw();
});

async function save(next = false) {
  if (state.index < 0) return;
  const name = state.images[state.index].name;
  const data = await api("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, boxes: state.boxes }),
  });
  setStatus(`已保存 ${data.label}`);
  state.images[state.index].labeled = true;
  renderList();
  const labeled = state.images.filter((item) => item.labeled).length;
  $("#progress").textContent = `${labeled} / ${state.images.length}`;
  $("#progressBar").style.width = `${labeled / state.images.length * 100}%`;
  if (next && state.index + 1 < state.images.length) await loadImage(state.index + 1);
}

async function deleteCurrent() {
  if (state.index < 0) return;
  const name = state.images[state.index].name;
  if (!window.confirm(`确定删除“${name}”吗？\n对应的导出图片和标签也会一并删除。`)) return;

  const nextIndex = Math.min(state.index, state.images.length - 2);
  await api("/api/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  state.image = null;
  state.boxes = [];
  state.index = nextIndex;
  await refresh();
  if (state.index >= 0) {
    await loadImage(state.index);
  } else {
    canvas.style.display = "none";
    $("#emptyState").style.display = "";
    $("#imageMeta").textContent = "";
  }
  setStatus(`已删除 ${name}`);
}

async function refreshProjectStatus() {
  const data = await api("/api/project/status");
  const cards = [
    ["标注导出", `${data.export.images} 图 / ${data.export.labels} 标签`],
    ["训练集", `${data.dataset.train.images} 图 / ${data.dataset.train.labels} 标签`],
    ["验证集", `${data.dataset.val.images} 图 / ${data.dataset.val.labels} 标签`],
    ["测试集", `${data.dataset.test.images} 图 / ${data.dataset.test.labels} 标签`],
  ];
  $("#datasetStats").innerHTML = cards.map(([label, value]) =>
    `<div class="stat-card"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
}

async function startJob(path, payload = {}) {
  await api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await refreshJob();
}

async function refreshJob() {
  const job = await api("/api/job");
  const labels = { idle: "空闲", running: "运行中", completed: "已完成", failed: "失败", stopping: "正在停止" };
  $("#jobTitle").textContent = job.name || "任务日志";
  $("#jobState").textContent = labels[job.status] || job.status;
  $("#jobLog").textContent = job.log || "尚未运行任务。";
  $("#jobLog").scrollTop = $("#jobLog").scrollHeight;
  const busy = ["running", "stopping"].includes(job.status);
  $("#stopJobButton").disabled = !busy;
  document.querySelectorAll("[data-job], #augmentButton, #trainButton").forEach((button) => button.disabled = busy);
  if (!busy) await refreshProjectStatus();
}

function updateNavigation() {
  const selected = state.index >= 0;
  $("#position").textContent = selected ? `${state.index + 1} / ${state.images.length} · ${state.images[state.index].name}` : "尚未选择图片";
  $("#prevButton").disabled = !selected || state.index === 0;
  $("#nextButton").disabled = !selected || state.index === state.images.length - 1;
  ["#deleteButton", "#undoButton", "#clearButton", "#inferButton", "#saveButton", "#saveNextButton"].forEach((id) => $(id).disabled = !selected);
}

$("#uploadInput").onchange = async (event) => {
  for (const file of event.target.files) {
    setStatus(`正在导入 ${file.name}…`);
    const data = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(file);
    });
    await api("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: file.name, data }),
    });
  }
  const last = event.target.files[event.target.files.length - 1];
  await refresh(last?.name);
  if (state.index >= 0) await loadImage(state.index);
  event.target.value = "";
};
$("#refreshButton").onclick = () => refresh();
$("#deleteButton").onclick = () => deleteCurrent().catch((error) => setStatus(error.message, false));
$("#undoButton").onclick = () => { state.boxes.pop(); draw(); };
$("#clearButton").onclick = () => { state.boxes = []; draw(); };
$("#saveButton").onclick = () => save(false);
$("#saveNextButton").onclick = () => save(true);
$("#prevButton").onclick = () => loadImage(state.index - 1);
$("#nextButton").onclick = () => loadImage(state.index + 1);
$("#inferButton").onclick = async () => {
  if (state.index < 0) return;
  $("#inferPanel").hidden = false;
  $("#inferLoading").style.display = "block";
  $("#inferResult").style.display = "none";
  $("#inferModel").textContent = "";
  try {
    const data = await api("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: state.images[state.index].name, confidence: 0.25 }),
    });
    $("#inferModel").textContent = `模型：${data.model} · 置信度阈值：0.25`;
    $("#inferResult").src = data.result;
    $("#inferResult").style.display = "block";
  } catch (error) {
    $("#inferLoading").textContent = error.message;
    setStatus(error.message, false);
    return;
  }
  $("#inferLoading").style.display = "none";
};
$("#closeInferButton").onclick = () => { $("#inferPanel").hidden = true; };
$("#zoomOutButton").onclick = () => setZoom(state.zoom / 1.25);
$("#zoomInButton").onclick = () => setZoom(state.zoom * 1.25);
$("#zoomResetButton").onclick = resetZoom;
$("#stage").addEventListener("wheel", (event) => {
  if (!state.image) return;
  event.preventDefault();
  setZoom(state.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), event.clientX, event.clientY);
}, { passive: false });
$("#workflowButton").onclick = async () => {
  $("#workflowPanel").hidden = false;
  await Promise.all([refreshProjectStatus(), refreshJob()]);
};
$("#closeWorkflowButton").onclick = () => { $("#workflowPanel").hidden = true; };
document.querySelectorAll("[data-job]").forEach((button) => {
  button.onclick = () => startJob(button.dataset.job).catch((error) => setStatus(error.message, false));
});
$("#augmentButton").onclick = () => startJob("/api/dataset/augment", {
  copies: Number($("#augmentCopies").value),
}).catch((error) => setStatus(error.message, false));
$("#trainButton").onclick = () => startJob("/api/train", {
  epochs: Number($("#epochsInput").value),
  batch: Number($("#batchInput").value),
  image_size: Number($("#imageSizeInput").value),
}).catch((error) => setStatus(error.message, false));
$("#stopJobButton").onclick = () => startJob("/api/job/stop").catch((error) => setStatus(error.message, false));
window.addEventListener("resize", fitCanvas);

refresh().catch((error) => setStatus(error.message, false));
setInterval(() => {
  if (!$("#workflowPanel").hidden) refreshJob().catch((error) => setStatus(error.message, false));
}, 1500);

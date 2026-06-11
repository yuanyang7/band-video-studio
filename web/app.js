/* Band Video Studio frontend: video list, interactive timeline, crops, exports. */

const $ = (id) => document.getElementById(id);
const api = async (path, options) => {
  const res = await fetch("/api" + path, options);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
};
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

let current = null; // current video detail
let crops = {};     // name -> {x,y,w,h} normalized
let sel = null;     // export range selection {start, end} shown on the timeline
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const safe = (fn) => async (...args) => {
  try { await fn(...args); } catch (e) { alert(e.message); }
};

/* ---------------------------------------------------------------- jobs */

async function watchJob(jobId, then) {
  const el = $("job-status");
  const poll = async () => {
    const job = await api(`/jobs/${jobId}`);
    if (job.status === "running") {
      el.textContent = `⏳ ${job.kind}: ${job.progress || "working"}…`;
      setTimeout(poll, 1500);
    } else if (job.status === "done") {
      el.textContent = `✅ ${job.kind} done`;
      setTimeout(() => (el.textContent = ""), 4000);
      then && then(job);
    } else {
      el.textContent = `❌ ${job.kind} failed`;
      console.error(job.error);
      alert(`${job.kind} failed:\n` + job.error.split("\n")[0]);
    }
  };
  poll();
}

/* -------------------------------------------------------------- videos */

async function refreshList() {
  const videos = await api("/videos");
  const ul = $("video-list");
  ul.innerHTML = "";
  for (const v of videos) {
    const li = document.createElement("li");
    li.textContent = `${v.name} (${fmt(v.meta.duration)})`;
    li.className = current && current.id === v.id ? "active" : "";
    li.onclick = () => openVideo(v.id);
    ul.appendChild(li);
  }
}

async function openVideo(id) {
  current = await api(`/videos/${id}`);
  crops = current.crops || {};
  sel = null;
  $("main").hidden = false;
  if (current.has_proxy) $("player").src = `/api/videos/${id}/stream`;
  $("lyrics-avail").textContent = current.capabilities.lyrics ? "" : "(install: uv sync --extra lyrics)";
  $("opt-claude").disabled = !current.capabilities.claude;
  refreshList();
  renderAnalysis();
  renderCropList();
  refreshExports();
  renderLyrics();
}

$("register-btn").onclick = safe(async () => {
  const path = $("register-path").value.trim();
  if (!path) return;
  const { video, job } = await post("/videos/register", { path });
  $("register-path").value = "";
  watchJob(job, () => openVideo(video.id));
  refreshList();
});

$("upload-file").onchange = async () => {
  const file = $("upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  $("job-status").textContent = "⏳ uploading…";
  const res = await fetch("/api/videos/upload", { method: "POST", body: form });
  const { video, job } = await res.json();
  watchJob(job, () => openVideo(video.id));
  refreshList();
};

/* ------------------------------------------------------------ timeline */

const timeline = $("timeline");

function drawTimeline() {
  if (!current) return;
  const ctx = timeline.getContext("2d");
  const W = (timeline.width = timeline.clientWidth * devicePixelRatio);
  const H = timeline.height;
  const dur = current.meta.duration || 1;
  const x = (t) => (t / dur) * W;
  ctx.clearRect(0, 0, W, H);

  const a = current.analysis;
  if (a) {
    ctx.fillStyle = "#1d4d35";
    for (const s of a.songs || []) ctx.fillRect(x(s.start), 8, Math.max(2, x(s.end) - x(s.start)), 22);
    ctx.fillStyle = "#f6c343";
    for (const h of a.highlights || []) ctx.fillRect(x(h.start), 14, Math.max(2, x(h.end) - x(h.start)), 10);
    ctx.fillStyle = "#ff7a8a";
    for (const m of a.fun_moments || []) {
      const cx = x((m.start + m.end) / 2);
      ctx.beginPath();
      ctx.arc(cx, 44, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  const ly = current.lyrics;
  if (ly) {
    ctx.fillStyle = "#6db3f2";
    for (const line of ly.lines || []) if (line.start != null) ctx.fillRect(x(line.start), 56, 2, 6);
  }
  // export-range selection: translucent band + edge handles
  if (sel) {
    ctx.fillStyle = "rgba(109, 179, 242, 0.18)";
    ctx.fillRect(x(sel.start), 0, x(sel.end) - x(sel.start), H);
    ctx.fillStyle = "#6db3f2";
    for (const t of [sel.start, sel.end]) ctx.fillRect(x(t) - 1.5 * devicePixelRatio, 0, 3 * devicePixelRatio, H);
  }
  // playhead
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(x($("player").currentTime), 0, 1.5 * devicePixelRatio, H);
}

/* Timeline interaction:
   - drag a selection edge to fine-tune the export range
   - click a song block (upper band) to select that song as the range
   - click anywhere else to seek */
let draggingEdge = null;

const eventTime = (e) => {
  const rect = timeline.getBoundingClientRect();
  return Math.min(Math.max(((e.clientX - rect.left) / rect.width) * current.meta.duration, 0), current.meta.duration);
};
const nearEdge = (e) => {
  if (!sel) return null;
  const rect = timeline.getBoundingClientRect();
  const px = (t) => (t / current.meta.duration) * rect.width;
  const cx = e.clientX - rect.left;
  if (Math.abs(cx - px(sel.start)) < 8) return "start";
  if (Math.abs(cx - px(sel.end)) < 8) return "end";
  return null;
};

function setSelection(start, end) {
  sel = { start: Math.max(0, Math.round(start * 10) / 10), end: Math.round(end * 10) / 10 };
  $("edit-start").value = sel.start;
  $("edit-end").value = sel.end;
  drawTimeline();
}

timeline.onmousedown = (e) => {
  if (!current) return;
  draggingEdge = nearEdge(e);
  if (draggingEdge) return;
  const t = eventTime(e);
  const songs = (current.analysis && current.analysis.songs) || [];
  const offsetY = e.clientY - timeline.getBoundingClientRect().top;
  const hit = songs.find((s) => t >= s.start && t <= s.end);
  if (hit && offsetY < 34) {
    setSelection(hit.start, hit.end); // click a song block -> select it
  } else {
    $("player").currentTime = t;
  }
};
timeline.onmousemove = (e) => {
  if (!current) return;
  if (draggingEdge) {
    const t = eventTime(e);
    if (draggingEdge === "start") setSelection(Math.min(t, sel.end - 1), sel.end);
    else setSelection(sel.start, Math.max(t, sel.start + 1));
  } else {
    timeline.style.cursor = nearEdge(e) ? "ew-resize" : "pointer";
  }
};
window.addEventListener("mouseup", () => { draggingEdge = null; });

// keep selection in sync when the number fields are edited by hand
for (const id of ["edit-start", "edit-end"]) {
  $(id).addEventListener("input", () => {
    const start = parseFloat($("edit-start").value), end = parseFloat($("edit-end").value);
    if (!isNaN(start) && !isNaN(end) && end > start) { sel = { start, end }; drawTimeline(); }
  });
}
$("player").ontimeupdate = drawTimeline;
window.onresize = drawTimeline;

/* ------------------------------------------------------------ analysis */

$("analyze-btn").onclick = safe(async () => {
  if (!current) return;
  const job = await post(`/videos/${current.id}/analyze`, {
    fun_detection: $("opt-fun").checked,
    sweep: $("opt-sweep").checked,
    claude_pass: $("opt-claude").checked,
  });
  watchJob(job.id, () => openVideo(current.id));
});

function renderAnalysis() {
  const a = current.analysis;
  $("analysis-summary").textContent = a
    ? `${(a.songs || []).length} songs · ${(a.highlights || []).length} highlights · ${(a.fun_moments || []).length} fun moments`
    : "not analyzed yet";
  const ul = $("moments");
  ul.innerHTML = "";
  if (!a) { drawTimeline(); return; }

  const add = (t, label, cls) => {
    const li = document.createElement("li");
    li.className = cls;
    li.innerHTML = `<span class="t">${fmt(t)}</span>${label}`;
    li.onclick = () => { $("player").currentTime = t; $("player").play(); };
    ul.appendChild(li);
  };
  (a.songs || []).forEach((s, i) => {
    add(s.start, `🎵 Song ${i + 1} (${fmt(s.start)}–${fmt(s.end)})`, "");
    // convenience: double-clicking a song selects it as the export range
    ul.lastChild.ondblclick = () => setSelection(s.start, s.end);
  });
  (a.highlights || []).forEach((h) => add(h.start, `⭐ highlight (z=${h.score})`, ""));
  (a.fun_moments || []).forEach((m) => {
    const caption = m.caption || (m.type === "laughter" ? "laughter heard" : "smiles spotted");
    add(m.start, `😄 ${caption} (score ${m.score})`, `fun-${m.type}`);
  });
  drawTimeline();
}

/* --------------------------------------------------------------- crops */

const cropCanvas = $("crop-canvas");
let frameImg = null, dragStart = null, dragBox = null;

$("grab-frame").onclick = async () => {
  if (!current) return;
  const t = $("player").currentTime;
  frameImg = new Image();
  frameImg.onload = () => {
    cropCanvas.width = frameImg.width;
    cropCanvas.height = frameImg.height;
    drawCrops();
  };
  frameImg.src = `/api/videos/${current.id}/frame?t=${t}&_=${Date.now()}`;
};

function drawCrops() {
  if (!frameImg) return;
  const ctx = cropCanvas.getContext("2d");
  ctx.drawImage(frameImg, 0, 0);
  ctx.lineWidth = 3;
  ctx.font = "16px sans-serif";
  for (const [name, b] of Object.entries(crops)) {
    ctx.strokeStyle = "#3b6ef6";
    ctx.strokeRect(b.x * cropCanvas.width, b.y * cropCanvas.height, b.w * cropCanvas.width, b.h * cropCanvas.height);
    ctx.fillStyle = "#3b6ef6";
    ctx.fillText(name, b.x * cropCanvas.width + 4, b.y * cropCanvas.height + 18);
  }
  if (dragBox) {
    ctx.strokeStyle = "#f6c343";
    ctx.strokeRect(dragBox.x, dragBox.y, dragBox.w, dragBox.h);
  }
}

const canvasPos = (e) => {
  const r = cropCanvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * cropCanvas.width,
    y: ((e.clientY - r.top) / r.height) * cropCanvas.height,
  };
};
cropCanvas.onmousedown = (e) => { dragStart = canvasPos(e); };
cropCanvas.onmousemove = (e) => {
  if (!dragStart) return;
  const p = canvasPos(e);
  dragBox = {
    x: Math.min(dragStart.x, p.x), y: Math.min(dragStart.y, p.y),
    w: Math.abs(p.x - dragStart.x), h: Math.abs(p.y - dragStart.y),
  };
  drawCrops();
};
cropCanvas.onmouseup = () => {
  if (dragBox && dragBox.w > 10 && dragBox.h > 10) {
    const name = prompt("Name this player view (e.g. drums, bass, wide):");
    if (name) {
      crops[name] = {
        x: dragBox.x / cropCanvas.width, y: dragBox.y / cropCanvas.height,
        w: dragBox.w / cropCanvas.width, h: dragBox.h / cropCanvas.height,
      };
      renderCropList();
    }
  }
  dragStart = dragBox = null;
  drawCrops();
};

function renderCropList() {
  const ul = $("crop-list");
  ul.innerHTML = "";
  for (const name of Object.keys(crops)) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${name}</span><span class="del" title="remove">✕</span>`;
    li.querySelector(".del").onclick = () => { delete crops[name]; renderCropList(); drawCrops(); };
    ul.appendChild(li);
  }
}

$("save-crops").onclick = async () => {
  if (!current) return;
  await api(`/videos/${current.id}/crops`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crops }),
  });
  $("job-status").textContent = "✅ crops saved";
  setTimeout(() => ($("job-status").textContent = ""), 3000);
};

/* -------------------------------------------------------------- export */

$("export-btn").onclick = safe(async () => {
  if (!current) return;
  if (!Object.keys(crops).length) {
    alert("Draw and save at least one player crop first (Player crops panel).");
    return;
  }
  const start = parseFloat($("edit-start").value), end = parseFloat($("edit-end").value);
  if (isNaN(start) || isNaN(end) || end <= start) {
    alert("Pick an export range first — click a song on the timeline, or fill start/end.");
    return;
  }
  const job = await post(`/videos/${current.id}/edit`, {
    start, end,
    orientation: $("edit-orientation").value,
    switch_s: parseFloat($("edit-switch").value || 4),
  });
  watchJob(job.id, refreshExports);
});

async function refreshExports() {
  if (!current) return;
  const files = await api(`/videos/${current.id}/exports`);
  const ul = $("exports");
  ul.innerHTML = "";
  for (const name of files) {
    const li = document.createElement("li");
    li.innerHTML = `<a href="/api/videos/${current.id}/exports/${name}" download style="color:#6db3f2">${name}</a>`;
    ul.appendChild(li);
  }
}

/* -------------------------------------------------------------- lyrics */

$("lyrics-btn").onclick = safe(async () => {
  if (!current) return;
  const job = await post(`/videos/${current.id}/lyrics-match`, {
    start: parseFloat($("edit-start").value || 0),
    end: parseFloat($("edit-end").value || 0),
    song_name: $("lyrics-song").value,
    lyrics: $("lyrics-text").value,
  });
  watchJob(job.id, () => openVideo(current.id));
});

function renderLyrics() {
  const ul = $("lyrics-lines");
  ul.innerHTML = "";
  const ly = current.lyrics;
  if (!ly) return;
  for (const line of ly.lines || []) {
    const li = document.createElement("li");
    if (line.start != null) {
      li.innerHTML = `<span class="t">${fmt(line.start)}</span>${line.line}`;
      li.onclick = () => { $("player").currentTime = line.start; $("player").play(); };
    } else {
      li.className = "miss";
      li.innerHTML = `<span class="t">—</span>${line.line}`;
    }
    ul.appendChild(li);
  }
}

refreshList();

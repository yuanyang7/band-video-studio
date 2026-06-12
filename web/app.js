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
  const text = $("job-text");
  const barWrap = $("job-bar-wrap");
  const bar = $("job-bar");
  const poll = async () => {
    const job = await api(`/jobs/${jobId}`);
    if (job.status === "running") {
      text.textContent = `⏳ ${job.kind}: ${job.progress || "working"}…`;
      if (job.pct != null) {
        barWrap.hidden = false;
        bar.style.width = `${job.pct}%`;
      }
      setTimeout(poll, 1500);
    } else if (job.status === "done") {
      text.textContent = `✅ ${job.kind} done`;
      bar.style.width = "100%";
      setTimeout(() => { text.textContent = ""; barWrap.hidden = true; bar.style.width = "0%"; }, 4000);
      then && then(job);
    } else {
      text.textContent = `❌ ${job.kind} failed`;
      barWrap.hidden = true;
      bar.style.width = "0%";
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
  clearTimeout(settingsTimer);
  clearTimeout(cropsTimer);
  current = await api(`/videos/${id}`);
  crops = current.crops || {};
  sel = null;
  cutList = null;
  applySettings(current.settings || {});
  $("cutlist-wrap").style.display = "none";
  $("library-view").hidden = true;
  $("main").hidden = false;
  if (current.has_proxy) $("player").src = `/api/videos/${id}/stream`;
  $("lyrics-avail").textContent = current.capabilities.lyrics ? "" : "(install: uv sync --extra lyrics)";
  $("opt-claude").disabled = !current.capabilities.claude;
  refreshList();
  renderAnalysis();
  renderCropList();
  refreshExports();
  renderLyrics();
  renderSync();
  setupMixer();
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
  $("job-text").textContent = "⏳ uploading…";
  const res = await fetch("/api/videos/upload", { method: "POST", body: form });
  const { video, job } = await res.json();
  watchJob(job, () => openVideo(video.id));
  refreshList();
};

/* ---------------------------------------------- per-video settings cache
   Every render/detection control is auto-saved (debounced) to the server as
   a per-video artifact, and restored when the video is reopened. */

const SETTING_IDS = [
  "opt-fun", "opt-sweep", "opt-claude",                       // detection
  "max-upscale", "edit-switch",                               // shooting sim
  "opt-smart", "opt-motion", "opt-transitions", "opt-sharpen", "opt-denoise",
  "edit-orientation", "edit-start", "edit-end", "export-name", // export
];
let settingsTimer = null, cropsTimer = null;
let applyingSettings = false;

function collectSettings() {
  const s = {};
  for (const id of SETTING_IDS) {
    const el = $(id);
    s[id] = el.type === "checkbox" ? el.checked : el.value;
  }
  return s;
}

function applySettings(s) {
  applyingSettings = true;
  for (const id of SETTING_IDS) {
    if (!(id in s)) continue;
    const el = $(id);
    if (el.type === "checkbox") el.checked = !!s[id];
    else el.value = s[id];
  }
  $("max-upscale-val").textContent = (parseFloat($("max-upscale").value) || 2).toFixed(1);
  // restore the export range selection on the timeline
  const start = parseFloat($("edit-start").value), end = parseFloat($("edit-end").value);
  if (!isNaN(start) && !isNaN(end) && end > start) sel = { start, end };
  applyingSettings = false;
}

function scheduleSettingsSave() {
  if (!current || applyingSettings) return;
  clearTimeout(settingsTimer);
  const id = current.id;
  settingsTimer = setTimeout(async () => {
    try {
      await api(`/videos/${id}/settings`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: collectSettings() }),
      });
    } catch (e) { console.warn("settings auto-save failed:", e.message); }
  }, 600);
}

for (const id of SETTING_IDS) {
  $(id).addEventListener("change", scheduleSettingsSave);
  $(id).addEventListener("input", scheduleSettingsSave);
}

function scheduleCropsSave() {
  if (!current) return;
  clearTimeout(cropsTimer);
  const id = current.id;
  cropsTimer = setTimeout(async () => {
    try {
      await api(`/videos/${id}/crops`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crops }),
      });
    } catch (e) { console.warn("crops auto-save failed:", e.message); }
  }, 600);
}

/* -------------------------------------------------------------- library */

async function refreshLibrary() {
  const lib = await api("/library");
  if (document.activeElement !== $("output-dir")) $("output-dir").value = lib.output_dir || "";
  const ul = $("lib-folders");
  ul.innerHTML = "";
  for (const folder of lib.folders) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="lib-path" title="${folder}">${folder}</span><span class="del" title="remove">✕</span>`;
    li.querySelector(".del").onclick = safe(async () => {
      await post("/library/folders/remove", { path: folder });
      refreshLibrary();
    });
    ul.appendChild(li);
  }
}

$("lib-add").onclick = safe(async () => {
  const path = $("lib-folder").value.trim();
  if (!path) return;
  await post("/library/folders", { path });
  $("lib-folder").value = "";
  refreshLibrary();
});

$("output-dir-save").onclick = safe(async () => {
  await api("/library/output-dir", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: $("output-dir").value.trim() }),
  });
  $("job-text").textContent = "✅ output folder saved";
  setTimeout(() => ($("job-text").textContent = ""), 3000);
  refreshLibrary();
});

$("lib-scan").onclick = safe(async () => {
  const limit = parseInt($("lib-limit").value, 10);
  const job = await post("/library/scan", { limit: isNaN(limit) ? null : limit });
  watchJob(job.id, (done) => {
    const r = done.result || {};
    const extra = [
      r.failed && r.failed.length ? `${r.failed.length} skipped` : "",
      r.remaining ? `${r.remaining} left for the next scan` : "",
    ].filter(Boolean).join(", ");
    $("job-text").textContent = `✅ scan: ${(r.imported || []).length} imported${extra ? ` (${extra})` : ""}`;
    if (r.failed && r.failed.length) console.warn("library scan skipped files:", r.failed);
    setTimeout(() => ($("job-text").textContent = ""), 8000);
    refreshList();
  });
});

function openMomentAt(videoId, t) {
  return safe(async () => {
    await openVideo(videoId);
    const p = $("player");
    const seek = () => { p.currentTime = t; p.play().catch(() => {}); };
    if (p.readyState >= 1) seek();
    else p.addEventListener("loadedmetadata", seek, { once: true });
  })();
}

$("lib-best").onclick = safe(async () => {
  const best = await api("/library/best");
  const fill = (ulId, items, label) => {
    const ul = $(ulId);
    ul.innerHTML = "";
    if (!items.length) {
      ul.innerHTML = `<li class="miss">nothing yet — analyze some videos first</li>`;
      return;
    }
    for (const m of items) {
      const li = document.createElement("li");
      const caption = m.caption || (m.type === "laughter" ? "laughter heard" : "smiles spotted");
      li.innerHTML = `<span class="t">${fmt(m.start)}</span>😄 ${caption} ${label(m)} <span class="lib-video">${m.video_name}</span>`;
      li.onclick = () => openMomentAt(m.video_id, m.start);
      ul.appendChild(li);
    }
  };
  fill("best-fun", best.fun, (m) => `(score ${m.score})`);
  fill("best-expr", best.expressions, (m) => `(smile ${m.max_smile})`);
  $("main").hidden = true;
  $("library-view").hidden = false;
});

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
    for (const h of a.highlights || []) {
      if (h.kind === "instrumental") continue;
      ctx.fillStyle = "#f6c343";
      ctx.fillRect(x(h.start), 14, Math.max(2, x(h.end) - x(h.start)), 10);
    }
    ctx.fillStyle = "#9b7be0";
    for (const h of a.highlights || []) {
      if (h.kind !== "instrumental") continue;
      ctx.fillRect(x(h.start), 32, Math.max(2, x(h.end) - x(h.start)), 8);
    }
    ctx.fillStyle = "#4dc4c4";
    for (const s of a.vocal_segments || []) ctx.fillRect(x(s.start), 26, Math.max(2, x(s.end) - x(s.start)), 4);
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
  scheduleSettingsSave();
}

timeline.onmousedown = (e) => {
  if (!current) return;
  draggingEdge = nearEdge(e);
  if (draggingEdge) return;
  $("player").currentTime = eventTime(e); // clicks just seek; use the button to select
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
$("select-at-playhead").onclick = () => {
  if (!current) return;
  const t = $("player").currentTime;
  const songs = (current.analysis && current.analysis.songs) || [];
  const hit = songs.find((s) => t >= s.start && t <= s.end);
  if (hit) {
    setSelection(hit.start, hit.end);
  } else {
    const prevEnd = Math.max(0, ...songs.filter((s) => s.end <= t).map((s) => s.end));
    const nextStart = Math.min(current.meta.duration, ...songs.filter((s) => s.start >= t).map((s) => s.start));
    setSelection(prevEnd, nextStart);
  }
};
$("player").ontimeupdate = () => { drawTimeline(); drawCutList(); };
window.onresize = () => { drawTimeline(); drawCutList(); };

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
    ? (() => {
        const hl = a.highlights || [];
        const peaks = hl.filter((h) => h.kind !== "instrumental").length;
        const instr = hl.filter((h) => h.kind === "instrumental").length;
        return `${(a.songs || []).length} songs · ${peaks} highlights · ${instr} instrumentals · ${(a.fun_moments || []).length} fun moments`;
      })()
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
  (a.highlights || []).forEach((h) => {
    const label = h.kind === "instrumental"
      ? `🎻 instrumental (${fmt(h.start)}–${fmt(h.end)})`
      : `⭐ highlight (z=${h.score})`;
    add(h.start, label, "");
    if (h.kind === "instrumental") ul.lastChild.ondblclick = () => setSelection(h.start, h.end);
  });
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

function effectiveCrop(b) {
  const orient = $("edit-orientation").value;
  const presets = {
    horizontal: [1920,1080], horizontal_2k: [2560,1440], horizontal_4k: [3840,2160],
    vertical: [1080,1920], vertical_2k: [1440,2560], vertical_4k: [2160,3840],
  };
  const [outW, outH] = presets[orient] || [1920, 1080];
  const tgt = outW / outH;
  const maxUp = parseFloat($("max-upscale").value) || 2;
  const cw = cropCanvas.width, ch = cropCanvas.height;
  let w = Math.max(b.w * cw, 16), h = Math.max(b.h * ch, 16);
  let cx = b.x * cw + w / 2, cy = b.y * ch + h / 2;
  if (w / h > tgt) h = w / tgt; else w = h * tgt;
  const srcW = current ? current.meta.width : cw;
  const srcH = current ? current.meta.height : ch;
  const scale = cw / srcW;
  const minW = (outW / maxUp) * scale, minH = (outH / maxUp) * scale;
  if (w < minW || h < minH) { const g = Math.max(minW / w, minH / h); w *= g; h *= g; }
  const s = Math.min(1, cw / w, ch / h); w *= s; h *= s;
  const x = Math.min(Math.max(cx - w / 2, 0), cw - w);
  const y = Math.min(Math.max(cy - h / 2, 0), ch - h);
  return { x, y, w, h };
}

function drawCrops() {
  if (!frameImg) return;
  const ctx = cropCanvas.getContext("2d");
  ctx.drawImage(frameImg, 0, 0);
  ctx.lineWidth = 3;
  ctx.font = "16px sans-serif";
  for (const [name, b] of Object.entries(crops)) {
    // user-drawn box
    ctx.strokeStyle = "#3b6ef6";
    ctx.strokeRect(b.x * cropCanvas.width, b.y * cropCanvas.height, b.w * cropCanvas.width, b.h * cropCanvas.height);
    ctx.fillStyle = "#3b6ef6";
    ctx.fillText(name, b.x * cropCanvas.width + 4, b.y * cropCanvas.height + 18);
    // effective output crop (dashed)
    const ec = effectiveCrop(b);
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#f6c343";
    ctx.strokeRect(ec.x, ec.y, ec.w, ec.h);
    ctx.setLineDash([]);
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
      scheduleCropsSave();
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
    li.innerHTML = `<span>${name}</span>
      <span class="crop-tools">
        <select class="role" title="role: determines which stem drives this view's cuts">
          <option value="">auto</option>
          <option value="singer">singer</option>
          <option value="drums">drums</option>
          <option value="bass">bass</option>
          <option value="keys">keys</option>
          <option value="wide">wide / all</option>
        </select>
        <span class="del" title="remove">✕</span>
      </span>`;
    const roleSel = li.querySelector(".role");
    roleSel.value = crops[name].role || "";
    roleSel.onchange = () => {
      if (roleSel.value) crops[name].role = roleSel.value;
      else delete crops[name].role;
      scheduleCropsSave();
    };
    li.querySelector(".del").onclick = () => { delete crops[name]; renderCropList(); drawCrops(); scheduleCropsSave(); };
    ul.appendChild(li);
  }
}

$("save-crops").onclick = async () => {
  if (!current) return;
  await api(`/videos/${current.id}/crops`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crops }),
  });
  $("job-text").textContent = "✅ crops saved";
  setTimeout(() => ($("job-text").textContent = ""), 3000);
};

$("max-upscale").oninput = () => {
  $("max-upscale-val").textContent = parseFloat($("max-upscale").value).toFixed(1);
  drawCrops();
};
$("edit-orientation").onchange = () => drawCrops();

/* ------------------------------------------------- simulate / cut list */

const VIEW_COLORS = [
  "#e06c75", "#61afef", "#98c379", "#e5c07b", "#c678dd",
  "#56b6c2", "#be5046", "#d19a66", "#7ec8e3", "#c3e88d",
];
let cutList = null;        // [{start, end, view, reason}]
let cutViews = [];         // available view names
let cutStart = 0, cutEnd = 0;
let draggingCutEdge = null; // {index, edge: "start"|"end"}

function viewColor(view) {
  const idx = cutViews.indexOf(view);
  return VIEW_COLORS[idx % VIEW_COLORS.length];
}

function drawCutList() {
  if (!cutList || !cutList.length) return;
  const canvas = $("cutlist-canvas");
  const ctx = canvas.getContext("2d");
  const W = (canvas.width = canvas.clientWidth * devicePixelRatio);
  const H = canvas.height;
  const dur = cutEnd - cutStart || 1;
  const x = (t) => ((t - cutStart) / dur) * W;
  ctx.clearRect(0, 0, W, H);

  for (const cut of cutList) {
    const cx = x(cut.start), cw = Math.max(2, x(cut.end) - x(cut.start));
    ctx.fillStyle = viewColor(cut.view);
    ctx.fillRect(cx, 0, cw, H);
    // edge lines
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.fillRect(cx, 0, 1, H);
    // label
    const textW = cw / devicePixelRatio;
    if (textW > 30) {
      ctx.fillStyle = "#fff";
      ctx.font = `${11 * devicePixelRatio}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const label = textW > 60 ? `${cut.view} (${cut.reason})` : cut.view;
      ctx.fillText(label, cx + cw / 2, H / 2, cw - 4 * devicePixelRatio);
    }
  }
  // playhead
  if (current) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(x($("player").currentTime), 0, 2 * devicePixelRatio, H);
  }
}

function renderCutLegend() {
  const el = $("cutlist-legend");
  el.innerHTML = "";
  for (const v of cutViews) {
    const span = document.createElement("span");
    span.innerHTML = `<span class="swatch" style="background:${viewColor(v)}"></span>${v}`;
    el.appendChild(span);
  }
}

// click a segment to cycle its view
$("cutlist-canvas").onclick = (e) => {
  if (!cutList || draggingCutEdge) return;
  const rect = e.target.getBoundingClientRect();
  const dur = cutEnd - cutStart || 1;
  const t = cutStart + ((e.clientX - rect.left) / rect.width) * dur;
  const idx = cutList.findIndex(c => t >= c.start && t < c.end);
  if (idx < 0) return;
  const cur = cutViews.indexOf(cutList[idx].view);
  cutList[idx].view = cutViews[(cur + 1) % cutViews.length];
  cutList[idx].reason = "manual";
  drawCutList();
};

// drag edges to adjust cut boundaries
const cutEdgeAt = (e) => {
  if (!cutList) return null;
  const rect = $("cutlist-canvas").getBoundingClientRect();
  const dur = cutEnd - cutStart || 1;
  const cx = e.clientX - rect.left;
  const px = (t) => ((t - cutStart) / dur) * rect.width;
  for (let i = 0; i < cutList.length; i++) {
    if (Math.abs(cx - px(cutList[i].start)) < 6 && i > 0) return {index: i, edge: "start"};
    if (Math.abs(cx - px(cutList[i].end)) < 6 && i < cutList.length - 1) return {index: i, edge: "end"};
  }
  return null;
};

$("cutlist-canvas").onmousedown = (e) => {
  draggingCutEdge = cutEdgeAt(e);
};
$("cutlist-canvas").onmousemove = (e) => {
  if (!cutList) return;
  if (draggingCutEdge) {
    const rect = $("cutlist-canvas").getBoundingClientRect();
    const dur = cutEnd - cutStart || 1;
    const t = cutStart + ((e.clientX - rect.left) / rect.width) * dur;
    const {index, edge} = draggingCutEdge;
    const minLen = 0.5;
    if (edge === "start") {
      const lo = cutList[index - 1].start + minLen;
      const hi = cutList[index].end - minLen;
      const nt = Math.max(lo, Math.min(hi, t));
      cutList[index].start = Math.round(nt * 100) / 100;
      cutList[index - 1].end = cutList[index].start;
    } else {
      const lo = cutList[index].start + minLen;
      const hi = cutList[index + 1].end - minLen;
      const nt = Math.max(lo, Math.min(hi, t));
      cutList[index].end = Math.round(nt * 100) / 100;
      cutList[index + 1].start = cutList[index].end;
    }
    drawCutList();
    e.preventDefault();
  } else {
    $("cutlist-canvas").style.cursor = cutEdgeAt(e) ? "ew-resize" : "pointer";
  }
};
window.addEventListener("mouseup", () => { draggingCutEdge = null; });

$("simulate-btn").onclick = safe(async () => {
  if (!current) return;
  if (!Object.keys(crops).length) {
    alert("Set up at least one camera first (Shooting simulation panel).");
    return;
  }
  const start = parseFloat($("edit-start").value), end = parseFloat($("edit-end").value);
  if (isNaN(start) || isNaN(end) || end <= start) {
    alert("Pick an export range first — click a song on the timeline, or fill start/end.");
    return;
  }
  const job = await post(`/videos/${current.id}/simulate`, {
    start, end,
    orientation: $("edit-orientation").value,
    switch_s: parseFloat($("edit-switch").value || 4),
    smart: $("opt-smart").checked,
  });
  watchJob(job.id, (done) => {
    const r = done.result || {};
    cutList = r.cuts || [];
    cutViews = r.views || Object.keys(crops);
    cutStart = r.start || start;
    cutEnd = r.end || end;
    $("cutlist-wrap").style.display = "";
    renderCutLegend();
    drawCutList();
  });
});

$("export-btn").onclick = safe(async () => {
  if (!current || !cutList || !cutList.length) {
    alert("Simulate first to generate a cut list, then export.");
    return;
  }
  const job = await post(`/videos/${current.id}/edit`, {
    start: cutStart, end: cutEnd,
    cuts: cutList,
    orientation: $("edit-orientation").value,
    camera_motion: $("opt-motion").checked,
    transitions: $("opt-transitions").checked,
    sharpen: $("opt-sharpen").checked,
    denoise: $("opt-denoise").checked,
    max_upscale: parseFloat($("max-upscale").value) || 2,
    name: $("export-name").value.trim() || null,
  });
  watchJob(job.id, (done) => {
    refreshExports();
    const r = done.result || {};
    if (r.saved_to) {
      $("job-text").textContent = `✅ exported → ${r.saved_to}`;
      setTimeout(() => ($("job-text").textContent = ""), 8000);
    } else if (r.save_error) {
      alert(`Export rendered, but: ${r.save_error}`);
    }
  });
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

/* ---------------------------------------------------------------- sync */

function renderSync() {
  const s = current && current.sync;
  const el = $("sync-result");
  el.innerHTML = "";
  if (!s) return;
  const weak = s.confidence < 0.25 ? " ⚠️ weak match" : "";
  const span = document.createElement("span");
  const stems = (current.stems || []).length ? ` · stems: ${current.stems.join(", ")}` : "";
  span.textContent = `✅ ${s.file} @ ${fmt(s.offset)}–${fmt(s.offset + s.duration)} (match ${s.confidence})${weak}${stems}. Export will use this audio.`;
  const useBtn = document.createElement("button");
  useBtn.textContent = "set range";
  useBtn.onclick = () => setSelection(s.offset, s.offset + s.duration);
  const clearBtn = document.createElement("button");
  clearBtn.textContent = "clear";
  clearBtn.onclick = safe(async () => {
    await api(`/videos/${current.id}/sync-audio`, { method: "DELETE" });
    openVideo(current.id);
  });
  el.append(span, " ", useBtn, " ", clearBtn);
}

$("sync-btn").onclick = safe(async () => {
  if (!current) return;
  const file = $("sync-file").files[0];
  if (!file) { alert("Choose an audio recording first."); return; }
  const form = new FormData();
  form.append("file", file);
  $("job-text").textContent = "⏳ uploading recording…";
  const res = await fetch(`/api/videos/${current.id}/sync-audio`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  const job = await res.json();
  watchJob(job.id, async (done) => {
    const r = done.result || {};
    await openVideo(current.id);
    setSelection(r.offset, r.offset + r.duration); // pre-fill the export range from the alignment
    if (r.confidence < 0.25) {
      alert(`Aligned at ${fmt(r.offset)} but the match is weak (${r.confidence}). Check it before rendering.`);
    }
  });
});

/* ----------------------------------------------- two-track audition mixer
   After an alignment exists, the video player and a hidden <audio> with the
   clean recording play as two layered tracks (like a video editor): toggle
   either to compare the camera audio with the aligned recording. */

let refAudio = null;
let recOn = true, camOn = false;

function setupMixer() {
  const wrap = $("track-mixer");
  if (refAudio) { refAudio.pause(); refAudio.removeAttribute("src"); refAudio = null; }
  const s = current && current.sync;
  if (!s || !s.ref_path) {
    wrap.hidden = true;
    $("player").muted = false;
    return;
  }
  wrap.hidden = false;
  refAudio = new Audio(`/api/videos/${current.id}/sync-audio/file`);
  refAudio.preload = "auto";
  // default mirrors the export: recording audible, camera muted
  recOn = true; camOn = false;
  applyMix();
  // show where the aligned recording sits on the video's timeline
  const dur = current.meta.duration || 1;
  const bar = $("rec-span");
  bar.style.left = `${(s.offset / dur) * 100}%`;
  bar.style.width = `${Math.min(100, (s.duration / dur) * 100)}%`;
}

function applyMix() {
  $("player").muted = !camOn;
  if (refAudio) refAudio.muted = !recOn;
  $("trk-rec").classList.toggle("on", recOn);
  $("trk-cam").classList.toggle("on", camOn);
}

$("trk-rec").onclick = () => { recOn = !recOn; applyMix(); };
$("trk-cam").onclick = () => { camOn = !camOn; applyMix(); };

function syncRefAudio() {
  if (!refAudio || !current || !current.sync) return;
  const p = $("player");
  const t = p.currentTime - current.sync.offset; // video time -> recording time
  if (t >= 0 && t <= current.sync.duration) {
    if (Math.abs(refAudio.currentTime - t) > 0.15) refAudio.currentTime = t;
    refAudio.playbackRate = p.playbackRate;
    if (!p.paused && refAudio.paused) refAudio.play().catch(() => {});
    if (p.paused && !refAudio.paused) refAudio.pause();
  } else if (!refAudio.paused) {
    refAudio.pause(); // playhead is outside the aligned span
  }
}

for (const ev of ["play", "pause", "seeked", "timeupdate", "ratechange"]) {
  $("player").addEventListener(ev, syncRefAudio);
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
refreshLibrary();

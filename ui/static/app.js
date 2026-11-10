// CA-universal replay bench: vanilla JS, no build step. CA frames come only from the export, computed in worker.js.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const fmtS = (s) => (s == null ? "–" : s < 120 ? `${(+s).toFixed(1)} s` : `${(s / 60).toFixed(1)} min`);
const b64u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const COL = { on: [236, 232, 223], off: [12, 12, 11], fn: [255, 90, 78], fp: [78, 168, 255] };

async function api(path, opts = {}) {
  const o = { ...opts, headers: { ...(opts.headers || {}) } };
  if (o.method === "POST") o.headers["X-CA-UI"] = "1";
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, o), j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

let toastTimer;
function toast(msg, bad = false) {
  const t = $("toast"); t.textContent = msg; t.className = "toast"; void t.offsetWidth; t.className = "toast on" + (bad ? " bad" : "");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = "toast"), 4500);
}

// ---------- header pixel field (header only; stops while the video plays, when hidden, and under reduced motion) ----------
const field = (() => {
  const cv = $("field"), ctx = cv.getContext("2d"), G = 28, SZ = 4; let cells = new Map(), timer = null, cols = 0;
  const fit = () => { cv.width = cv.clientWidth; cv.height = cv.clientHeight; cols = Math.ceil(cv.width / G); };
  const tick = () => {
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (const [k, a] of cells) {
      const na = a * 0.88; if (na < 0.03) { cells.delete(k); continue; }
      cells.set(k, na); ctx.fillStyle = `rgba(255,180,58,${na})`; ctx.fillRect((k % cols) * G + G / 2 - SZ / 2, Math.floor(k / cols) * G + G / 2 - SZ / 2, SZ, SZ);
    }
    if (!cells.size) stop();
  };
  const stop = () => { clearInterval(timer); timer = null; ctx.clearRect(0, 0, cv.width, cv.height); };
  const poke = (e) => {
    if (reduced || player.playing || document.hidden) return;
    const r = cv.getBoundingClientRect(), cx = Math.floor((e.clientX - r.left) / G), cy = Math.floor((e.clientY - r.top) / G);
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const d = Math.hypot(dx, dy); if (d > 2.2 || Math.random() < 0.45) continue; // sparse
      const k = (cy + dy) * cols + cx + dx; if (cx + dx >= 0 && cx + dx < cols && cy + dy >= 0) cells.set(k, Math.max(cells.get(k) || 0, 1 - d / 3));
    }
    if (!timer) timer = setInterval(tick, 48);
  };
  fit(); addEventListener("resize", fit); cv.parentElement.addEventListener("pointermove", poke);
  document.addEventListener("visibilitychange", () => document.hidden && stop());
  return { stop: () => { cells.clear(); stop(); } };
})();

// ---------- runs shelf ----------
let runs = [], cfg = null;
async function loadRuns() {
  runs = await api("/api/runs");
  for (const kind of ["example", "run", "imported"]) {
    const ul = $("list-" + kind), items = runs.filter((r) => r.kind === kind);
    ul.innerHTML = items.length ? items.map((r) => `<li><button data-run="${esc(r.name)}" aria-pressed="${r.name === player.name}">
      <span class="t">${esc(r.label)}</span>
      <span class="m"><span class="st ${esc(r.status)}">${esc(r.status)}${r.stage ? " · " + esc(r.stage) : ""}</span>
      ${r.n_params ? fmtInt(r.n_params) + " params" : ""}${r.export_bytes ? " · " + fmtInt(r.export_bytes) + " B" : ""}
      ${r.frames ? ` · ${r.frames}f ${r.width ?? "?"}×${r.height ?? "?"}` : ""}${r.preset ? " · " + esc(r.preset) : ""}${r.elapsed_s ? " · " + fmtS(r.elapsed_s) : ""}
      ${r.has_original ? "" : " · CA only"}</span></button></li>`).join("") : `<li class="cm" style="padding:8px 6px">none yet</li>`;
  }
}
document.addEventListener("click", (e) => { const b = e.target.closest("[data-run]"); if (b) openRun(b.dataset.run); });

$("import-file").addEventListener("change", async (e) => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  try {
    const r = await api("/api/import-export", { method: "POST", body: f, headers: { "X-Filename": f.name.replace(/[^\x20-\x7e]/g, "_") } });
    toast(`imported ${r.name} (${fmtInt(r.export.n_params)} params, ${r.T} frames, CA-only playback)`); await loadRuns(); openRun(r.name);
  } catch (err) { toast("import rejected: " + err.message, true); }
});

// ---------- player ----------
const player = { name: null, T: 0, H: 0, W: 0, vis: null, cont: null, orig: null, t: 0, playing: false, fps: 30, src: null, srcT: null, sha: null };
const cache = new Map(); // run name + export size -> {vis, cont, facts, sha, ms}
const worker = new Worker("/static/worker.js", { type: "module" });
let reqId = 0; const pending = new Map();
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id); if (!p) return;
  if (data.progress != null) return p.progress(data.progress);
  pending.delete(data.id); data.error ? p.reject(new Error(data.error)) : p.resolve(data);
};
const computeCA = (url, progress) => new Promise((resolve, reject) => { const id = ++reqId; pending.set(id, { resolve, reject, progress }); worker.postMessage({ id, url }); });

const ctxs = { orig: $("cv-orig").getContext("2d"), ca: $("cv-ca").getContext("2d"), err: $("cv-err").getContext("2d") };
function paint(ctx, fn) {
  const { W, H } = player, img = ctx.createImageData(W, H), d = img.data;
  for (let i = 0; i < W * H; i++) { const c = fn(i); d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = 255; }
  ctx.putImageData(img, 0, 0);
}
function blank(ctx, msg) { const { canvas: c } = ctx; ctx.fillStyle = "#0c0c0b"; ctx.fillRect(0, 0, c.width, c.height); c.title = msg || ""; }

function render() {
  const { t, W, H, T, vis, cont, orig } = player; if (!vis) return;
  const N = W * H, o = t * N, cv = $("t-cont").checked;
  paint(ctxs.ca, cv ? (i) => { const g = Math.round(Math.min(1, Math.max(0, cont[o + i])) * 255); return [g, g, g]; } : (i) => (vis[o + i] ? COL.on : COL.off));
  if (orig) {
    paint(ctxs.orig, (i) => (orig[o + i] ? COL.on : COL.off));
    if ($("t-err").checked) paint(ctxs.err, (i) => { const a = orig[o + i], b = vis[o + i]; return a && b ? COL.on : !a && !b ? COL.off : a ? COL.fn : COL.fp; });
  } else { blank(ctxs.orig, "no original"); blank(ctxs.err, "no original"); }
  $("scrub").value = t;
  let wrong = 0; if (orig) for (let i = 0; i < N; i++) wrong += orig[o + i] !== vis[o + i];
  const src = player.src ? ` · source frame #${player.src[t]}` : "", time = player.srcT ? ` · t=${player.srcT[t].toFixed(3)} s` : ` · t=${(t / player.fps).toFixed(3)} s`;
  $("counter").textContent = `frame ${String(t).padStart(String(T - 1).length, "0")}/${T - 1}${src}${time}${orig ? ` · ${wrong} px differ (${(100 * wrong / N).toFixed(2)}%)` : " · no original"}`;
}
function setZoom() {
  const z = +$("zoom").value;
  for (const c of [$("cv-orig"), $("cv-ca"), $("cv-err")]) { c.width = player.W || 1; c.height = player.H || 1; c.style.width = `${(player.W || 1) * z}px`; c.style.height = `${(player.H || 1) * z}px`; }
  render();
}
$("zoom").onchange = setZoom; $("t-cont").onchange = () => { $("ca-cap").innerHTML = $("t-cont").checked ? 'CA visible value <span class="cm">gray = clip(v,0,1), raw state thresholded at 0.5, not a probability</span>' : 'CA replay <span class="cm">(export only)</span>'; render(); };
$("t-err").onchange = () => { $("fig-err").hidden = !$("t-err").checked; render(); };

let raf = null, last = 0, acc = 0;
function loop(ts) {
  if (!player.playing) return;
  acc += ((ts - last) / 1000) * player.fps * +$("speed").value; last = ts;
  while (acc >= 1) { acc -= 1; if (player.t >= player.T - 1) { setPlaying(false); return; } player.t++; render(); }
  raf = requestAnimationFrame(loop);
}
function setPlaying(on) {
  if (!player.vis) return;
  player.playing = on; $("b-play").textContent = on ? "[ pause ]" : "[ play ]";
  if (on) { field.stop(); if (player.t >= player.T - 1) player.t = 0; last = performance.now(); acc = 0; raf = requestAnimationFrame(loop); render(); }
  else cancelAnimationFrame(raf);
}
const step = (d) => { setPlaying(false); player.t = Math.max(0, Math.min(player.T - 1, player.t + d)); render(); };
$("b-play").onclick = () => setPlaying(!player.playing); $("b-prev").onclick = () => step(-1); $("b-next").onclick = () => step(1);
$("scrub").oninput = (e) => { setPlaying(false); player.t = +e.target.value; render(); };
document.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) && e.target.type !== "checkbox" || tag === "BUTTON" && e.key === " ") return;
  if (!player.vis) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!player.playing); }
  else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
});

let openSeq = 0;
async function openRun(name) {
  const seq = ++openSeq; setPlaying(false);
  const det = await api(`/api/runs/${encodeURIComponent(name)}`).catch((e) => (toast(e.message, true), null)); if (!det || seq !== openSeq) return;
  player.name = name; document.querySelectorAll("[data-run]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.run === name));
  $("player-title").textContent = "// " + det.summary.label; renderModel(det); if (det.summary.own) watchJob(name);
  for (const b of ["b-play", "b-prev", "b-next", "scrub"]) $(b).disabled = true;
  player.vis = null;
  if (!det.summary.has_export) { $("compute-line").textContent = `no export yet (status: ${det.summary.status}); the player needs the exported model file`; renderFacts(det, null); return; }
  const man = det.manifest || {}, pp = man.data?.preprocessing, key = `${name}:${man.export?.sizes?.file ?? det.status?.sizes?.file}`;
  let ca = cache.get(key);
  if (!ca) {
    $("compute-line").textContent = "computing CA frames from the export in a worker…";
    const t0 = performance.now();
    try { ca = await computeCA(`/api/runs/${encodeURIComponent(name)}/export`, (p) => seq === openSeq && ($("compute-line").textContent = `computing CA frames from the export in a worker… ${Math.round(p * 100)}%`)); }
    catch (e) { if (seq === openSeq) $("compute-line").textContent = "export could not be replayed: " + e.message; return; }
    ca.wall = performance.now() - t0; cache.set(key, ca);
  }
  let orig = null;
  if (det.summary.has_original) { try { const o = await api(`/api/runs/${encodeURIComponent(name)}/original`); orig = { ...o, u8: b64u8(o.bin.b64) }; } catch (e) { orig = null; } }
  if (seq !== openSeq) return;
  const f = ca.facts;
  Object.assign(player, { T: f.T, H: f.H, W: f.W, vis: ca.vis, cont: ca.cont, sha: ca.sha, t: 0, fps: orig?.fps || pp?.fps || f.meta?.fps || 30,
    orig: orig && orig.bin.shape[0] >= f.T && orig.bin.shape[1] === f.H && orig.bin.shape[2] === f.W ? orig.u8 : null,
    src: orig?.source_frame_index || null, srcT: orig?.source_time_s || null });
  window.__bench = { name, sha256: ca.sha, T: f.T, H: f.H, W: f.W, ms: ca.ms };
  $("scrub").max = f.T - 1; for (const b of ["b-play", "b-prev", "b-next", "scrub"]) $(b).disabled = false;
  $("compute-line").textContent = `CA frames computed in this browser from the export: ${f.T} frames × ${f.S} updates, ${ca.ms.toFixed(0)} ms (cached for this session) · sha256 of binary frames ${ca.sha.slice(0, 16)}…` + (player.orig ? "" : " · no original available: CA-only playback");
  renderFacts(det, ca); setZoom();
}

function renderFacts(det, ca) {
  const man = det.manifest || {}, ex = man.export || {}, sz = ex.sizes || det.status?.sizes || {}, f = ca?.facts || {}, pr = det.progress, imp = man.imported || {};
  const train = pr ? `${fmtS(pr.train_s)} (${fmtInt(pr.step)}/${fmtInt(pr.steps)} steps)` : imp.t ? `${fmtS(imp.t)} to best checkpoint at step ${fmtInt(imp.it)} (recorded by the source run${imp.start ? `; continued from ${imp.start}, earlier training not included` : ""})` : "–";
  const rows = [["parameters", fmtInt(f.n_params ?? ex.n_params)], ["C channels", f.C ? `${f.C} (1 visible + ${f.C - 1} hidden)` : "–"], ["Fh features", f.Fh ?? "–"],
    ["S updates/frame", f.S ?? "–"], ["weight dtype", f.weight_dtype ?? ex.weight_dtype ?? "–"],
    ["export bytes", sz.file ? `${fmtInt(sz.file)} B file` + (sz.weights ? ` (raw: weights ${fmtInt(sz.weights)} + first frame ${fmtInt(sz.first_frame)} + header ${fmtInt(sz.header)} = ${fmtInt(sz.raw_total)}, deflated)` : "") : f.bytes ? `${fmtInt(f.bytes)} B file` : "–"],
    ["frames", f.T ? `${f.T} × ${f.H}×${f.W}` : "–"], ["training time", train], ["status", det.summary.status + (det.status?.error ? ` — ${det.status.error}` : "")],
    ["replay check", det.status?.replay ? `numpy vs Node: ${det.status.replay.binary_mismatch_px} px mismatch` : "–"]];
  $("facts").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
}

// ---------- model panel ----------
function renderModel(det) {
  const man = det.manifest || {}, s = man.settings || {}, imp = man.imported || {}, C = s.C ?? imp.C ?? man.C, Fh = s.Fh ?? imp.Fh ?? man.Fh, S = imp.S ?? man.S ?? 4;
  const np = man.export?.n_params ?? (C && Fh ? Fh * C * 9 + Fh + C * Fh + C : null);
  $("arch").innerHTML = [["C channels", C ? `${C}: 1 visible + ${C - 1} hidden (continuous state)` : "–"], ["Fh features", Fh ?? "–"], ["S updates", `${S} per video frame`],
    ["parameters", np ? `${fmtInt(np)} = Fh·C·9 + Fh + C·Fh + C (learned weights, not rules)` : "–"], ["preset", man.preset ? `${man.preset}: ${s.note || ""}` : "–"]]
    .map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  $("sch-c").textContent = C ?? "C"; $("sch-c2").textContent = C ?? "C"; $("sch-fh").textContent = Fh ?? "Fh"; $("sch-s").textContent = S;
  renderCurve(det.hist || []);
}
function renderCurve(hist) {
  const svg = $("curve"), W = 560, H = 240, L = 48, R = 12, Tp = 12, B = 30;
  if (!hist.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}">no training history (example or imported export)</text>`; $("ckpts").innerHTML = '<p class="cm" style="padding:6px">no training history for this model</p>'; return; }
  const names = [...new Set(hist.map((h) => h.name))], xmax = Math.max(1, ...hist.map((h) => h.step)), ymax = Math.max(1e-4, ...hist.map((h) => h.bin_err)) * 1.05;
  const x = (v) => L + (v / xmax) * (W - L - R), y = (v) => Tp + (1 - v / ymax) * (H - Tp - B);
  const dash = ["", "4 3", "1 3", "6 2 1 2"];
  let out = `<line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${Tp}" x2="${L}" y2="${H - B}"/>`;
  for (const v of [0, ymax / 2, ymax]) out += `<text x="${L - 6}" y="${y(v) + 3}" style="text-anchor:end">${v === 0 ? "0" : v < 0.01 ? v.toExponential(1) : v.toFixed(3)}</text>`;
  for (const v of [0, xmax / 2, xmax]) out += `<text x="${x(v)}" y="${H - B + 14}">${Math.round(v)}</text>`;
  out += `<text x="${(L + W) / 2}" y="${H - 4}">step</text><text x="${L + 6}" y="${Tp + 8}" style="text-anchor:start">bin_err</text>`;
  const bestRows = new Set();
  names.forEach((n, k) => {
    const hs = hist.filter((h) => h.name === n); let best = Infinity, bi = null;
    hs.forEach((h) => { if (h.bin_err < best) { best = h.bin_err; bi = h; bestRows.add(h); } });
    out += `<polyline fill="none" stroke="var(--paper)" stroke-width="1" stroke-dasharray="${dash[k % 4]}" points="${hs.map((h) => `${x(h.step).toFixed(1)},${y(h.bin_err).toFixed(1)}`).join(" ")}"/>`;
    if (bi) out += `<rect x="${x(bi.step) - 4}" y="${y(bi.bin_err) - 4}" width="8" height="8" fill="var(--acc)" transform="rotate(45 ${x(bi.step)} ${y(bi.bin_err)})"><title>${esc(n)} best ${bi.bin_err.toFixed(4)} at step ${bi.step}</title></rect>`;
    out += `<text x="${W - R}" y="${Tp + 10 + 12 * k}" style="text-anchor:end;fill:var(--paper)">${esc(n)} best ${best.toFixed(4)}</text>`;
  });
  svg.innerHTML = out;
  $("ckpts").innerHTML = `<table><thead><tr><th>step</th><th>model</th><th>bin_err</th><th></th></tr></thead><tbody>${hist.map((h) =>
    `<tr class="${bestRows.has(h) ? "best" : ""}"><td>${h.step}</td><td>${esc(h.name)}</td><td>${h.bin_err.toFixed(4)}</td><td>${bestRows.has(h) ? "★" : ""}</td></tr>`).join("")}</tbody></table>`;
}

// ---------- new run: video, preview, training ----------
async function loadVideos(select) {
  const vs = await api("/api/videos"), sel = $("video");
  sel.innerHTML = vs.map((v) => `<option value="${esc(v.id)}">${esc(v.builtin ? v.name : "upload: " + v.name)}</option>`).join("");
  if (select) sel.value = select; probeSel();
}
async function probeSel() {
  $("probe").textContent = "probing…";
  try { const p = await api(`/api/video/${encodeURIComponent($("video").value)}/probe`); $("probe").textContent = `${p.width}×${p.height} · ${p.fps} fps · ${fmtInt(p.frames)} frames · ${p.duration_s?.toFixed(2)} s · ${p.codec}/${p.pix_fmt} · ${fmtInt(p.bytes)} B`; }
  catch (e) { $("probe").textContent = "probe failed: " + e.message; }
}
$("video").onchange = probeSel;
$("upload").addEventListener("change", async (e) => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  if (f.size > (cfg?.max_upload ?? 200 << 20)) return toast("file larger than 200 MB", true);
  $("probe").textContent = `uploading ${f.name}…`;
  try { const r = await api("/api/upload", { method: "POST", body: f, headers: { "X-Filename": f.name.replace(/[^\x20-\x7e]/g, "_") } }); toast(`uploaded ${r.name}`); await loadVideos(r.id); }
  catch (err) { $("probe").textContent = "upload rejected: " + err.message; toast("upload rejected: " + err.message, true); }
});
const params = () => ({ video: $("video").value, start: $("p-start").value, frames: $("p-frames").value, width: $("p-width").value,
  threshold: $("p-threshold").value, fps: $("p-fps").value || null, invert: $("p-invert").checked });

const pv = { T: 0, H: 0, W: 0, gray: null, bin: null, man: null };
function pvRender() {
  if (!pv.gray) return;
  const t = +$("pv-scrub").value, N = pv.W * pv.H, z = Math.max(1, Math.floor(300 / pv.W));
  for (const [id, fn] of [["pv-gray", (i) => pv.gray[t * N + i]], ["pv-bin", (i) => (pv.bin[t * N + i] ? 236 : 12)]]) {
    const c = $(id); c.width = pv.W; c.height = pv.H; c.style.width = `${pv.W * z}px`; c.style.height = `${pv.H * z}px`;
    const ctx = c.getContext("2d"), img = ctx.createImageData(pv.W, pv.H);
    for (let i = 0; i < N; i++) { const g = fn(i); img.data.set([g, g, g, 255], 4 * i); } ctx.putImageData(img, 0, 0);
  }
  const pp = pv.man.preprocessing; $("pv-counter").textContent = `frame ${t}/${pv.T - 1} · source frame #${pp.source_frame_index[t]} · t=${pp.source_time_s[t].toFixed(3)} s`;
}
$("pv-scrub").oninput = pvRender;
$("b-preview").onclick = async () => {
  $("pv-info").textContent = "decoding interval…"; $("b-preview").disabled = true;
  try {
    const r = await api("/api/preview", { method: "POST", json: params() });
    [pv.T, pv.H, pv.W] = r.bin.shape; pv.gray = b64u8(r.gray.b64); pv.bin = b64u8(r.bin.b64); pv.man = r.manifest;
    const pp = r.manifest.preprocessing, on = pv.bin.reduce((a, b) => a + b, 0) / pv.bin.length;
    $("pv-scrub").max = pv.T - 1; $("pv-scrub").value = 0; $("pv-scrub").disabled = false;
    $("pv-info").textContent = `${pv.T} frames × ${pv.H}×${pv.W} (height from aspect) · fps ${pp.fps} · ${pp.binarize} with threshold ${pp.threshold} · ${(100 * on).toFixed(1)}% pixels on · cache key ${r.manifest.key}`;
    pvRender();
  } catch (e) { $("pv-info").textContent = "preview rejected: " + e.message; } finally { $("b-preview").disabled = false; }
};
$("preset").onchange = () => { const p = cfg.presets[$("preset").value]; $("preset-note").textContent = `${p.note}. Defaults: C=${p.C}, Fh=${p.Fh}, steps=${fmtInt(p.steps)}, lr=${p.lr}. Your preprocessing values are used as entered; nothing is replaced silently.`; };

$("b-train").onclick = async () => {
  $("b-train").disabled = true;
  try {
    const r = await api("/api/train", { method: "POST", json: { ...params(), preset: $("preset").value, steps: $("t-steps").value || null, device: $("device").value } });
    toast(`job started: ${r.name}`); await loadRuns(); watchJob(r.name);
  } catch (e) { toast("training not started: " + e.message, true); } finally { $("b-train").disabled = false; }
};

let watch = { name: null, timer: null, last: null };
function watchJob(name) { clearTimeout(watch.timer); watch = { name, timer: null, last: null }; $("job").hidden = false; pollJob(); }
async function pollJob() {
  const name = watch.name; let d;
  try { d = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (e) { $("job-err").textContent = e.message; return; }
  if (name !== watch.name) return;
  const s = d.summary.status, pr = d.progress || {}, best = (pr.best || []).map((b, i) => `${pr.names?.[i] ?? i}: ${b.err?.toFixed(4)} @ ${b.step}`).join(", ");
  $("job-name").textContent = `┌─ job ${name}`;
  const frac = s === "finished" ? 1 : pr.steps ? pr.step / pr.steps : 0;
  $("job-bar").firstElementChild.style.width = `${(100 * frac).toFixed(1)}%`; $("job-bar").setAttribute("aria-valuenow", Math.round(100 * frac));
  $("job-facts").innerHTML = [["status", s], ["stage", d.status?.stage ?? "–"], ["step", pr.steps ? `${fmtInt(pr.step)}/${fmtInt(pr.steps)}` : "–"],
    ["train time", fmtS(pr.train_s)], ["best bin_err", best || "–"], ["device", d.job?.device ?? "–"]].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  $("job-err").textContent = d.status?.error ? `failed: ${d.status.error}` : s === "died" ? "process ended without writing a final status (see log). Resume continues from the last saved state." : "";
  $("job-log").textContent = d.log_tail.join("\n"); $("job-log").scrollTop = 1e9;
  const live = ["running", "starting", "queued"].includes(s);
  $("b-cancel").disabled = !live || !d.summary.own; $("b-resume").disabled = live || s === "finished" || !d.summary.own; $("b-open").disabled = !d.summary.has_export;
  if (watch.last && watch.last !== s) {
    toast(`job ${name}: ${s}`, ["failed", "died"].includes(s)); loadRuns();
    if (s === "finished" && player.name === name) openRun(name);
  }
  watch.last = s;
  if (live) watch.timer = setTimeout(pollJob, 1500);
}
$("b-cancel").onclick = async () => { try { await api(`/api/runs/${encodeURIComponent(watch.name)}/cancel`, { method: "POST" }); } catch (e) { toast(e.message, true); } pollJob(); };
$("b-resume").onclick = async () => { try { await api(`/api/runs/${encodeURIComponent(watch.name)}/resume`, { method: "POST" }); toast(`job resumed: ${watch.name}`); watch.last = "resumed"; } catch (e) { toast(e.message, true); } pollJob(); };
$("b-open").onclick = () => { openRun(watch.name); $("player").scrollIntoView({ behavior: reduced ? "auto" : "smooth" }); };

// ---------- boot ----------
(async () => {
  cfg = await api("/api/config");
  $("preset").innerHTML = Object.keys(cfg.presets).map((k) => `<option>${esc(k)}</option>`).join(""); $("preset").onchange();
  $("p-width").max = cfg.limits.max_width; $("p-width").min = cfg.limits.min_width; $("p-frames").max = cfg.limits.max_frames;
  await Promise.all([loadRuns(), loadVideos()]);
  const first = runs.find((r) => r.kind === "example");
  const want = new URLSearchParams(location.search).get("run") || first?.name;
  if (want) openRun(want);
})().catch((e) => toast("startup failed: " + e.message, true));

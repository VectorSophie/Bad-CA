// CA-universal replay bench: vanilla JS, no build step. CA frames come only from the export, computed in worker.js
// (NCA) or here with player/rule_core.mjs (explicit rules). Substep states and the inspector use the same core functions.
import { parseNcab, parseNcaJson, ruleAt, substep, traceCell, ncabHeader } from "/player/nca_core.mjs";
import { parseRules, run as ruleRun, step as ruleStep, programAt, explainCell } from "/player/rule_core.mjs";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const fmtS = (s) => (s == null ? "–" : s < 120 ? `${(+s).toFixed(1)} s` : `${(s / 60).toFixed(1)} min`);
const b64u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const COL = { on: [236, 232, 223], off: [12, 12, 11], fn: [255, 90, 78], fp: [78, 168, 255], sel: [255, 180, 58] };

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
  for (const kind of ["example", "run", "imported", "c12"]) {
    const ul = $("list-" + kind), items = runs.filter((r) => r.kind === kind);
    ul.innerHTML = items.length ? items.map((r) => `<li><button data-run="${esc(r.name)}" aria-pressed="${r.name === player.name}">
      <span class="t">${esc(r.label)}</span>
      <span class="m"><span class="st ${esc(r.status)}">${esc(r.status)}${r.stage ? " · " + esc(r.stage) : ""}</span>
      ${r.format ? esc(r.format) + " · " : ""}${r.rules != null ? `${r.rules} rules / ${r.programs} programs · ` : ""}${r.phases ? `${r.phases} phases (${esc(r.mode)}) · ` : ""}${r.n_params ? fmtInt(r.n_params) + " params" : ""}${r.export_bytes ? " · " + fmtInt(r.export_bytes) + " B" : ""}
      ${r.frames ? ` · ${r.frames}f ${r.width ?? "?"}×${r.height ?? "?"}` : ""}${r.preset ? " · " + esc(r.preset) : ""}${r.elapsed_s ? " · " + fmtS(r.elapsed_s) : ""}
      ${r.has_original ? "" : " · CA only"}</span></button></li>`).join("") : `<li class="cm" style="padding:8px 6px">none yet</li>`;
  }
  const bs = $("b-sel"), keep = bs.value;
  bs.innerHTML = '<option value="">none</option>' + runs.filter((r) => r.has_export).map((r) => `<option value="${esc(r.name)}">${esc(r.kind)}: ${esc(r.label)}</option>`).join("");
  bs.value = keep;
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
const player = { name: null, T: 0, H: 0, W: 0, vis: null, cont: null, orig: null, t: 0, sub: 0, playing: false, fps: 30, fpsSrc: "assumed", src: null, srcT: null, sha: null, det: null };
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
function paint(ctx, fn, W = player.W, H = player.H) {
  const img = ctx.createImageData(W, H), d = img.data;
  for (let i = 0; i < W * H; i++) { const c = fn(i); d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = 255; }
  ctx.putImageData(img, 0, 0);
}
function blank(ctx, msg) { const { canvas: c } = ctx; ctx.fillStyle = "#0c0c0b"; ctx.fillRect(0, 0, c.width, c.height); c.title = msg || ""; }

function render() {
  const { t, W, H, T, vis, cont, orig } = player; if (!vis) return;
  const N = W * H, o = t * N, sub = player.sub;
  let caVis = (i) => vis[o + i], caCont = cont ? (i) => cont[o + i] : null;
  if (sub > 0) { // between frames: the state after t*S + sub substeps, from cached frame states (never from the original)
    const st = stateAt(eng, t * eng.S + sub);
    if (eng.kind === "rule") { caVis = (i) => st.v[i]; caCont = null; } else { caVis = (i) => (st[i] > eng.m.threshold ? 1 : 0); caCont = (i) => st[i]; }
  }
  const cv = $("t-cont").checked && caCont, sel = insp.i;
  paint(ctxs.ca, (i) => (i === sel ? COL.sel : cv ? ((g) => [g, g, g])(Math.round(Math.min(1, Math.max(0, caCont(i))) * 255)) : caVis(i) ? COL.on : COL.off));
  renderB();
  if (orig) {
    paint(ctxs.orig, (i) => (orig[o + i] ? COL.on : COL.off));
    if ($("t-err").checked) paint(ctxs.err, (i) => { const a = orig[o + i], b = vis[o + i]; return a && b ? COL.on : !a && !b ? COL.off : a ? COL.fn : COL.fp; });
  } else { blank(ctxs.orig, "no original"); blank(ctxs.err, "no original"); }
  $("scrub").value = t;
  let wrong = 0; if (orig) for (let i = 0; i < N; i++) wrong += orig[o + i] !== vis[o + i];
  const src = player.src ? ` · source frame #${player.src[t]}` : "", time = player.srcT ? ` · t=${player.srcT[t].toFixed(3)} s` : ` · t=${(t / player.fps).toFixed(3)} s`;
  const S = eng?.S ?? 1, subTxt = ` · CA substep ${t * S + sub} = ${t}·S + ${sub} (S=${S})` + (sub ? ` — CA panel is ${sub}/${S} of the way to frame ${t + 1}; original/error show frame ${t}` : "");
  $("counter").textContent = `frame ${String(t).padStart(String(T - 1).length, "0")}/${T - 1}${src}${time}${orig ? ` · ${wrong} px differ (${(100 * wrong / N).toFixed(2)}%)` : " · no original"}${subTxt}`;
  $("b-sub").disabled = $("b-frame").disabled = t >= T - 1;
  renderTimelines(); if (!player.playing) renderInspector(); // the inspector re-runs a substep; not during playback
}
function setZoom() {
  const z = +$("zoom").value;
  const sz = (c, W, H) => { c.width = W || 1; c.height = H || 1; c.style.width = `${(W || 1) * z}px`; c.style.height = `${(H || 1) * z}px`; };
  for (const c of [$("cv-orig"), $("cv-ca"), $("cv-err")]) sz(c, player.W, player.H);
  for (const c of [$("cv-b"), $("cv-berr")]) sz(c, B.e?.m.W, B.e?.m.H);
  render();
}
$("zoom").onchange = setZoom; $("t-cont").onchange = () => { $("ca-cap").innerHTML = $("t-cont").checked ? 'CA visible value <span class="cm">gray = clip(v,0,1), raw state thresholded at 0.5, not a probability</span>' : 'CA replay <span class="cm">(export only)</span>'; render(); };
$("t-err").onchange = () => { $("fig-err").hidden = !$("t-err").checked; render(); };

let raf = null, last = 0, acc = 0;
function loop(ts) {
  if (!player.playing) return;
  acc += ((ts - last) / 1000) * player.fps * +$("speed").value; last = ts;
  while (acc >= 1) { acc -= 1; if (player.t >= player.T - 1) { setPlaying(false); return; } player.t++; player.sub = 0; render(); }
  raf = requestAnimationFrame(loop);
}
function setPlaying(on) {
  if (!player.vis) return;
  player.playing = on; $("b-play").textContent = on ? "[ pause ]" : "[ play ]";
  if (on) { field.stop(); if (player.t >= player.T - 1) player.t = 0; last = performance.now(); acc = 0; raf = requestAnimationFrame(loop); render(); }
  else { cancelAnimationFrame(raf); renderInspector(); }
}
const step = (d) => { setPlaying(false); player.sub = 0; player.t = Math.max(0, Math.min(player.T - 1, player.t + d)); render(); };
const seek = (t) => { setPlaying(false); player.sub = 0; player.t = Math.max(0, Math.min(player.T - 1, t)); render(); };
$("b-sub").onclick = () => { setPlaying(false); if (player.t >= player.T - 1) return; if (++player.sub >= eng.S) { player.sub = 0; player.t++; } render(); };
$("b-frame").onclick = () => step(1);
$("b-play").onclick = () => setPlaying(!player.playing); $("b-prev").onclick = () => step(-1); $("b-next").onclick = () => step(1);
$("scrub").oninput = (e) => seek(+e.target.value);
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
  for (const b of ["b-play", "b-prev", "b-next", "scrub", "b-sub", "b-frame"]) $(b).disabled = true;
  player.vis = null; eng = null; insp.i = null; player.sub = 0; player.det = det;
  if (!det.summary.has_export) { $("compute-line").textContent = `no export yet (status: ${det.summary.status}); the player needs the exported model file`; renderFacts(det, null); return; }
  const man = det.manifest || {}, pp = man.data?.preprocessing, key = `${name}:${man.export?.sizes?.file ?? det.status?.sizes?.file}`;
  let ca;
  $("compute-line").textContent = "computing CA frames from the export…";
  try { ca = await loadEngine(name, key, (p) => seq === openSeq && ($("compute-line").textContent = `computing CA frames from the export in a worker… ${Math.round(p * 100)}%`)); }
  catch (e) { if (seq === openSeq) $("compute-line").textContent = "export could not be replayed: " + e.message; return; }
  let orig = null;
  if (det.summary.has_original) { try { const o = await api(`/api/runs/${encodeURIComponent(name)}/original`); orig = { ...o, u8: b64u8(o.bin.b64) }; } catch (e) { orig = null; } }
  if (seq !== openSeq) return;
  const f = ca.facts; eng = ca;
  if (orig?.first_frame_mismatch_px) toast(`original resolved from export meta differs from the export's first frame in ${orig.first_frame_mismatch_px} px`, true);
  player.fpsSrc = orig?.fps ? "source video" : pp?.fps ? "run manifest" : f.meta?.fps ? "export meta" : "assumed (no fps recorded)";
  Object.assign(player, { T: f.T, H: f.H, W: f.W, vis: ca.vis, cont: ca.cont, sha: ca.sha, t: 0, fps: orig?.fps || pp?.fps || f.meta?.fps || 30,
    orig: orig && orig.bin.shape[0] >= f.T && orig.bin.shape[1] === f.H && orig.bin.shape[2] === f.W ? orig.u8 : null,
    src: orig?.source_frame_index || null, srcT: orig?.source_time_s || null });
  window.__bench = { name, sha256: ca.sha, T: f.T, H: f.H, W: f.W, ms: ca.ms };
  $("scrub").max = f.T - 1; for (const b of ["b-play", "b-prev", "b-next", "scrub", "b-sub", "b-frame"]) $(b).disabled = false;
  computeLine();
  renderFacts(det, ca); renderModel(det); renderRules(); setZoom();
}

function renderFacts(det, ca) {
  if (ca?.kind === "rule") {
    const f = ca.facts;
    $("facts").innerHTML = [["format", "dlrule-v1 explicit rules (executable decision lists)"], ["state", `v + ${f.k} hidden bits`], ["S substeps/frame", f.S], ["programs / rules", `${f.programs} / ${f.rules}`],
      ["export bytes", `${fmtInt(f.bytes)} B file`], ["frames", `${f.T} × ${f.H}×${f.W}`], ["source", det.manifest?.c12?.path ?? "–"], ["meta", JSON.stringify(f.meta || {})]]
      .map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
    return;
  }
  const man = det.manifest || {}, ex = man.export || {}, sz = ex.sizes || det.status?.sizes || {}, f = ca?.facts || {}, pr = det.progress, imp = man.imported || {};
  const train = pr ? `${fmtS(pr.train_s)} (${fmtInt(pr.step)}/${fmtInt(pr.steps)} steps)` : imp.t ? `${fmtS(imp.t)} to best checkpoint at step ${fmtInt(imp.it)} (recorded by the source run${imp.start ? `; continued from ${imp.start}, earlier training not included` : ""})` : "–";
  const rows = [["parameters", ca?.m?.version === 2 ? `${fmtInt(ex.n_params)} stored values (${ca.m.K} rule${ca.m.K > 1 ? "s" : ""}, mode ${ca.m.mode}; see model.rules)` : fmtInt(f.n_params ?? ex.n_params)], ["C channels", f.C ? `${f.C} (1 visible + ${f.C - 1} hidden)` : "–"], ["Fh features", f.Fh ?? "–"],
    ["S updates/frame", f.S ?? "–"], ["weight dtype", f.weight_dtype ?? ex.weight_dtype ?? "–"],
    ["export bytes", sz.file ? `${fmtInt(sz.file)} B file` + (sz.weights ? ` (raw: weights ${fmtInt(sz.weights)} + first frame ${fmtInt(sz.first_frame)} + header ${fmtInt(sz.header)} = ${fmtInt(sz.raw_total)}, deflated)` : "") : f.bytes ? `${fmtInt(f.bytes)} B file` : "–"],
    ["frames", f.T ? `${f.T} × ${f.H}×${f.W}` : "–"], ["training time", train], ["status", det.summary.status + (det.status?.error ? ` — ${det.status.error}` : "")],
    ["replay check", det.status?.replay ? `numpy vs Node: ${det.status.replay.binary_mismatch_px} px mismatch` : "–"]];
  $("facts").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
}

// ---------- model panel ----------
function renderModel(det) {
  const man = det.manifest || {}, s = man.settings || {}, imp = man.imported || {}, c12 = man.c12?.header || {};
  $("schematic").style.display = c12.format === "dlrule-v1" ? "none" : "";
  if (c12.format === "dlrule-v1") {
    $("arch").innerHTML = `<dt>model</dt><dd>explicit rules (dlrule-v1), not a Neural CA: ${c12.k} hidden bits, S=${c12.S}; the rules themselves are listed in model.rules below</dd>`;
    return renderCurve([]);
  }
  const C = s.C ?? imp.C ?? man.C ?? c12.C, Fh = s.Fh ?? imp.Fh ?? man.Fh ?? c12.Fh, S = imp.S ?? man.S ?? c12.S ?? 4;
  const np = man.export?.n_params ?? (C && Fh ? Fh * C * 9 + Fh + C * Fh + C : null);
  $("arch").innerHTML = [["C channels", C ? `${C}: 1 visible + ${C - 1} hidden (continuous state)` : "–"], ["Fh features", Fh ?? "–"], ["S updates", `${S} per video frame`],
    ["parameters", c12.format === "ncab-v2" ? `${fmtInt(np)} stored values: K=${c12.K} rule(s), mode ${c12.mode}, each rule Fh·C·9 + Fh + C·Fh + C = ${fmtInt(Fh * C * 9 + Fh + C * Fh + C)} effective (learned weights, not rules)` : np ? `${fmtInt(np)} = Fh·C·9 + Fh + C·Fh + C (learned weights, not rules)` : "–"], ["preset", man.preset ? `${man.preset}: ${s.note || ""}` : "–"]]
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

// ---------- engines: NCA (worker frames + frame states) or explicit rules (rule_core, here) ----------
let eng = null, verCounter = 1;
const insp = { i: null };
const B = { name: null, e: null };
const inflate = async (b) => new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
const isNcab = (u8) => u8[0] === 0x4e && u8[1] === 0x43 && u8[2] === 0x41 && u8[3] === 0x42;
const hex = async (u8) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", u8))].map((x) => x.toString(16).padStart(2, "0")).join("");
const fmtN = (x) => (x == null ? "·" : x === 0 ? "0" : Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3 ? x.toExponential(3) : x.toPrecision(5));

async function loadEngine(name, key, progress) {
  const url = `/api/runs/${encodeURIComponent(name)}/export`, r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const u8 = new Uint8Array(await r.arrayBuffer());
  if (!isNcab(u8)) { const o = JSON.parse(new TextDecoder().decode(u8)); if (o.format === "ca-universal/dlrule-v1" || (o.format === "ca-universal/rule-v1" && Array.isArray(o.programs))) return ruleEngine(o, u8.length); }
  let ca = cache.get(key);
  if (!ca) { ca = await computeCA(url, progress); cache.set(key, ca); }
  const m = isNcab(u8) ? await parseNcab(u8, inflate) : parseNcaJson(JSON.parse(new TextDecoder().decode(u8)));
  return { ...ca, kind: "nca", m, hdr: isNcab(u8) ? await ncabHeader(u8, inflate) : null, bytes: u8.length, S: m.S, T: m.T, N: m.H * m.W, memo: null, ver: 1, verNote: "export" };
}

async function ruleEngine(o, bytes) {
  const m = parseRules(o), N = m.H * m.W, t0 = performance.now();
  const e = { kind: "rule", m, base: m, json: o, bytes, S: m.S, T: m.T, N, ver: 1, verNote: "export", hybridFrom: null };
  e.progAt = (g) => m.programs[programAt(m, Math.floor(g / m.S) + 1)];
  simulateRule(e, m.programs, 0, null); e.ms = performance.now() - t0;
  const ref = ruleRun(m).frames; e.runMatch = ref.every((f, t) => f.every((x, i) => x === e.vis[t * N + i])); // the substep cache equals run()
  e.sha = await hex(e.vis);
  e.facts = { T: m.T, H: m.H, W: m.W, S: m.S, k: m.k, bytes, meta: m.meta, programs: m.programs.length, rules: m.programs.reduce((a, p) => a + p.rules.length, 0) };
  return e;
}
// seq[g] = full state {v, h} after g substeps. With a prefix, states 0..g0 are kept and later substeps use `programs`.
function simulateRule(e, programs, g0, prefix) {
  const m = e.m, S = m.S, G = (m.T - 1) * S, seq = prefix ? prefix.slice(0, g0 + 1) : [{ v: m.first.slice(), h: m.h0.slice() }];
  for (let g = seq.length - 1; g < G; g++) { const [v, h] = ruleStep(m, programs[programAt(m, Math.floor(g / S) + 1)], seq[g].v, seq[g].h); seq.push({ v, h }); }
  e.seq = seq; e.vis = new Uint8Array(m.T * e.N); e.cont = null;
  for (let t = 0; t < m.T; t++) e.vis.set(seq[t * S].v, t * e.N);
}
// Full state after g substeps (frame t = after t*S substeps). NCA: from the nearest cached frame state, stepping with substep().
function stateAt(e, g) {
  if (e.kind === "rule") return e.seq[g];
  const S = e.S, t = Math.floor(g / S), t0 = t - (t % e.every);
  let from = t0 * S, s = e.states[t0];
  if (e.memo && e.memo.g <= g && e.memo.g >= from) ({ g: from, s } = e.memo);
  for (let x = from; x < g; x++) s = substep(e.m, ruleAt(e.m, Math.floor(x / S) + 1), s);
  e.memo = { g, s }; return s;
}
function computeLine() {
  const e = eng; if (!e) return;
  const what = e.kind === "rule" ? `explicit rules run in this browser with rule_core.mjs (${e.hybridFrom != null ? "hybrid: no single-version run() reference" : e.runMatch ? "identical to run()" : "MISMATCH vs run()"})` : "CA frames computed in this browser from the export";
  $("compute-line").innerHTML = `${esc(what)}: ${e.T} frames × ${e.S} substeps, ${e.ms.toFixed(0)} ms · sha256 of binary frames ${e.sha.slice(0, 16)}… · <span class="verbadge">version ${e.ver}: ${esc(e.verNote)}</span>` + (player.orig ? "" : " · no original available: CA-only playback");
}

// ---------- B (second export, synchronized by frame index) ----------
$("b-sel").onchange = async () => {
  const name = $("b-sel").value; B.name = name; B.e = null; $("fig-b").hidden = $("fig-berr").hidden = true; $("b-line").textContent = "";
  if (name) {
    const r = runs.find((x) => x.name === name); $("b-line").textContent = "computing B from its export…";
    try { const e = await loadEngine(name, `${name}:${r?.export_bytes}`, (p) => ($("b-line").textContent = `computing B… ${Math.round(p * 100)}%`)); if (B.name !== name) return; B.e = e; }
    catch (err) { $("b-line").textContent = "B could not be replayed: " + err.message; return; }
    $("b-cap").innerHTML = `B <span class="cm">${esc(r?.label ?? name)} (export only)</span>`; $("fig-b").hidden = false;
    $("b-line").textContent = `B: ${B.e.T} frames ${B.e.m.W}×${B.e.m.H}, sha256 ${B.e.sha.slice(0, 16)}…` + (B.e.T !== player.T ? ` · T differs from A (${player.T}); B holds its last frame beyond ${B.e.T - 1}` : "");
  }
  setZoom();
};
function renderB() {
  const e = B.e; if (!e) return;
  const t = Math.min(player.t, e.T - 1), N = e.N, o = t * N, { orig } = player, same = e.m.W === player.W && e.m.H === player.H;
  paint(ctxs.b, (i) => (e.vis[o + i] ? COL.on : COL.off), e.m.W, e.m.H);
  const showErr = $("t-err").checked && orig && same && player.t < e.T; $("fig-berr").hidden = !showErr;
  if (showErr) { const oo = player.t * N; paint(ctxs.berr, (i) => { const a = orig[oo + i], b = e.vis[o + i]; return a && b ? COL.on : !a && !b ? COL.off : a ? COL.fn : COL.fp; }, e.m.W, e.m.H); }
}
ctxs.b = $("cv-b").getContext("2d"); ctxs.berr = $("cv-berr").getContext("2d");

// ---------- phase timelines ----------
function scheduleOf(e) {
  if (e.kind === "rule") return e.m.schedule.map((p) => ({ start: p.start, end: p.end, rule: p.program, reason: p.reason ?? null }));
  if (e.m.schedule) return e.m.schedule.map((p) => ({ start: p.start, end: p.end, rule: p.rule, reason: p.reason ?? null }));
  return [{ start: 1, end: e.T, rule: 0, reason: "single rule (ncab-v1)" }];
}
function timelineHTML(tag, e, label) {
  const sch = scheduleOf(e), T = e.T, fps = player.fps, t = Math.min(player.t, T - 1), word = e.kind === "rule" ? "program" : "rule";
  const Wd = Math.max(300, ($("timelines").clientWidth / ($("timelines").classList.contains("two") ? 2 : 1)) - 16 || 800), pad = 4, x = (f) => pad + (f / T) * (Wd - 2 * pad);
  const cur = sch.findIndex((p) => t >= p.start && t < p.end), sec = (f) => (f / fps).toFixed(3);
  let svg = `<rect class="seg" data-seek="0" x="${x(0)}" y="6" width="${x(1) - x(0)}" height="22"><title>frame 0: given first frame</title></rect>`;
  sch.forEach((p, k) => {
    const w = x(p.end) - x(p.start);
    svg += `<rect class="seg${k === cur ? " cur" : ""}" data-seek="${p.start}" x="${x(p.start)}" y="6" width="${w}" height="22"><title>phase ${k}: frames ${p.start}–${p.end - 1} (${sec(p.start)}–${sec(p.end - 1)} s), ${word} ${p.rule}${p.reason ? ", reason: " + esc(p.reason) : ""}</title></rect>`;
    if (w > 34) svg += `<text x="${x(p.start) + w / 2}" y="21" class="${k === cur ? "acc" : ""}">${word[0]}${p.rule}</text>`;
    svg += `<line class="bd" data-seek="${p.start}" x1="${x(p.start)}" x2="${x(p.start)}" y1="2" y2="32"><title>boundary at frame ${p.start} (${sec(p.start)} s)</title></line>`;
  });
  const xn = x(t + 0.5 + (tag === "A" ? player.sub / e.S : 0));
  svg += `<line class="now" x1="${xn}" x2="${xn}" y1="0" y2="34"/><text x="${pad}" y="44" style="text-anchor:start" class="cm">0</text><text x="${Wd - pad}" y="44" style="text-anchor:end" class="cm">${T} frames</text>`;
  const p = sch[cur], curTxt = t === 0 ? "frame 0: the given first frame (no rule applied)" : p ? `phase ${cur}/${sch.length - 1} · frames ${p.start}–${p.end - 1} · ${sec(p.start)}–${sec(p.end - 1)} s · ${word} ${p.rule}${p.reason ? " · reason: " + p.reason : ""}` : "frame outside the schedule (program/rule 0 by default)";
  const rows = sch.map((p, k) => `<tr data-seek="${p.start}" class="${k === cur ? "cur" : ""}"><td>${k}</td><td>${p.start}–${p.end - 1}</td><td>${sec(p.start)}–${sec(p.end - 1)} s</td><td>${word} ${p.rule}</td><td>${esc(p.reason ?? "–")}</td></tr>`).join("");
  return `<div class="tl"><h3 class="minor">${tag} · ${esc(label)} <span class="cm">// ${sch.length} phase${sch.length > 1 ? "s" : ""}; 1 video frame = ${e.S} CA substep${e.S > 1 ? "s" : ""}; seconds = frame / ${fps} fps (${esc(player.fpsSrc)}); click to seek</span></h3>
    <svg viewBox="0 0 ${Wd} 46" width="${Wd}" height="46" role="img" aria-label="${tag} phase timeline">${svg}</svg>
    <p class="cur-line">${esc(curTxt)}</p><div class="ptab"><table><thead><tr><th>#</th><th>frames</th><th>seconds</th><th>${word}</th><th>reason</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}
function renderTimelines() {
  if (!eng) { $("timelines").innerHTML = ""; return; }
  $("timelines").classList.toggle("two", !!B.e);
  $("timelines").innerHTML = timelineHTML("A", eng, player.det?.summary.label ?? player.name) + (B.e ? timelineHTML("B", B.e, runs.find((r) => r.name === B.name)?.label ?? B.name) : "");
}
$("timelines").addEventListener("click", (ev) => { const el = ev.target.closest("[data-seek]"); if (el) seek(+el.dataset.seek); });
addEventListener("resize", () => eng && renderTimelines());

// ---------- rule view ----------
const ruleLine = (r, i, k) => `R${i}: if ${r.if.map(([f, op, c]) => `${f} ${op} ${c}`).join(" and ") || "always"} -> v=${r.then.v}` + (k ? `, h=${r.then.h.toString(2).padStart(k, "0")}` : "");
function progDiff(a, b) { // b vs a: rules keyed by their condition list
  const key = (r) => JSON.stringify(r.if), A = new Map(a.rules.map((r) => [key(r), r])), Bm = new Map(b.rules.map((r) => [key(r), r]));
  const added = b.rules.filter((r) => !A.has(key(r))), removed = a.rules.filter((r) => !Bm.has(key(r))), changed = b.rules.filter((r) => A.has(key(r)) && JSON.stringify(A.get(key(r)).then) !== JSON.stringify(r.then));
  const order = !added.length && !removed.length && a.rules.map(key).join() !== b.rules.map(key).join();
  return { added, removed, changed, order, same: !added.length && !removed.length && !changed.length && !order };
}
const fmtPrograms = (ps) => "[\n" + ps.map((p) => ' {"rules": [\n' + p.rules.map((r) => "  " + JSON.stringify({ if: r.if, then: r.then })).join(",\n") + "\n ]}").join(",\n") + "\n]";

function renderRules() {
  const e = eng; if (!e) return;
  $("rules-kind").textContent = e.kind === "rule" ? "// dlrule-v1: explicit, executable decision lists" : `// ${e.m.version === 2 ? "ncab-v2 phased" : "ncab-v1"} Neural CA: learned weights, not rules`;
  return e.kind === "rule" ? renderRuleV1(e) : renderNcaPhases(e);
}
function renderRuleV1(e) {
  const m = e.m, sch = scheduleOf(e), k = m.k, cnt = e.counts;
  const used = (p) => sch.map((s, i) => (s.rule === p ? i : null)).filter((x) => x != null);
  let h = `<dl class="facts"><dt>state</dt><dd>v + ${k} hidden bit${k === 1 ? "" : "s"} per cell; ${m.S} substep${m.S > 1 ? "s" : ""} per frame; zero boundary; synchronous</dd>
    <dt>semantics</dt><dd>${esc(m.semantics || "first firing rule wins; default keep")}</dd><dt>features</dt><dd>${esc((m.features || []).join(", "))}</dd>
    <dt>size</dt><dd>${m.programs.length} programs, ${m.programs.reduce((a, p) => a + p.rules.length, 0)} rules, ${m.programs.reduce((a, p) => a + p.rules.reduce((b, r) => b + r.if.length, 0), 0)} conditions · export file ${fmtInt(e.bytes)} B (JSON with hex first frame)</dd></dl>`;
  h += `<h3 class="minor">fire counts <span class="cm">// cells × substeps in which each rule fired, from running the engine over frames</span></h3>
    <label>from frame <input type="number" id="fc-from" min="1" max="${m.T - 1}" value="${cnt?.from ?? 1}"></label><label>to frame <input type="number" id="fc-to" min="1" max="${m.T - 1}" value="${cnt?.to ?? m.T - 1}"></label>
    <button class="btn" id="b-count">[ count firings ]</button> <span class="cm">${cnt ? `counted frames ${cnt.from}–${cnt.to} (${fmtInt(cnt.cells)} cell-substeps, version ${cnt.ver})${cnt.note}` : "not counted yet"}</span>`;
  h += `<div class="two-col"><div>`;
  m.programs.forEach((p, pi) => {
    const lines = p.rules.map((r, i) => ruleLine(r, i, k) + (cnt ? `   ×${fmtInt(cnt.c[pi][i + 1])}` : "")).concat("default: keep state" + (cnt ? `   ×${fmtInt(cnt.c[pi][0])}` : ""));
    h += `<div class="prog"><h3 class="minor">program ${pi} <span class="cm">// phases ${used(pi).join(", ") || "none"} · ${p.rules.length} rules in priority order · ${fmtInt(JSON.stringify(p).length)} B JSON</span></h3><pre>${esc(lines.join("\n"))}</pre></div>`;
  });
  h += `</div><div><h3 class="minor">what changes at each phase boundary <span class="cm">// vs the previous phase's program; rules matched by their conditions</span></h3><div class="scrollbox"><table class="rtab"><thead><tr><th>phase</th><th>frames</th><th>program</th><th>vs previous</th></tr></thead><tbody>`;
  sch.forEach((s, i) => {
    let d = "first phase";
    if (i) {
      const a = sch[i - 1].rule, b = s.rule, df = progDiff(m.programs[a], m.programs[b]);
      d = a === b ? "same program" : df.same ? `program ${a} → ${b}: identical rules` : `program ${a} → ${b}: <span class="diff-add">+${df.added.length} added</span> <span class="diff-del">−${df.removed.length} removed</span> <span class="diff-chg">~${df.changed.length} changed</span>${df.order ? " · order changed" : ""}`
        + [...df.added.map((r) => `<br><span class="diff-add">+ ${esc(ruleLine(r, m.programs[b].rules.indexOf(r), k))}</span>`), ...df.removed.map((r) => `<br><span class="diff-del">− ${esc(ruleLine(r, m.programs[a].rules.indexOf(r), k))}</span>`), ...df.changed.map((r) => `<br><span class="diff-chg">~ ${esc(ruleLine(r, m.programs[b].rules.indexOf(r), k))}</span>`)].join("");
    }
    h += `<tr><td>${i}</td><td>${s.start}–${s.end - 1}</td><td>${s.rule}</td><td style="white-space:normal">${d}</td></tr>`;
  });
  h += `</tbody></table></div>
    <h3 class="minor">edit programs <span class="cm">// JSON list of programs; the schedule is kept</span> <span class="verbadge">version ${e.ver}: ${esc(e.verNote)}</span></h3>
    <textarea id="rule-edit" spellcheck="false" aria-label="programs JSON">${esc(fmtPrograms(m.programs))}</textarea>
    <p class="error" id="edit-err" role="alert"></p>
    <button class="btn" id="b-replay">[ replay from start ]</button> <span class="cm">default: recomputes every frame from frame 0 with the edited programs</span><br>
    <button class="btn" id="b-continue">[ experimental: continue from current state ]</button> <span class="cm">keeps substeps 0..current from the previous version, runs the edited programs after it (a hybrid, labelled as such)</span><br>
    <button class="btn" id="b-reset">[ reset to export ]</button></div></div>`;
  $("rules-body").innerHTML = h;
  $("b-count").onclick = () => countFires(e); $("b-replay").onclick = () => applyEdit("start"); $("b-continue").onclick = () => applyEdit("continue"); $("b-reset").onclick = () => applyEdit("reset");
}
function countFires(e) {
  const m = e.m, S = m.S, from = Math.max(1, Math.min(+$("fc-from").value || 1, m.T - 1)), to = Math.max(from, Math.min(+$("fc-to").value || m.T - 1, m.T - 1));
  const g0 = Math.max((from - 1) * S, e.hybridFrom ?? 0), c = m.programs.map((p) => new Array(p.rules.length + 1).fill(0)), fired = new Int32Array(e.N);
  for (let g = g0; g < to * S; g++) { // re-run each substep with the engine's step() and count the firing rule index per cell
    const pi = programAt(m, Math.floor(g / S) + 1), st = e.seq[g]; ruleStep(m, m.programs[pi], st.v, st.h, fired);
    for (let i = 0; i < e.N; i++) c[pi][fired[i] + 1]++;
  }
  e.counts = { from, to, c, cells: (to * S - g0) * e.N, ver: e.ver, note: e.hybridFrom != null && e.hybridFrom > (from - 1) * S ? ` · hybrid version: counted from substep ${e.hybridFrom} on (earlier substeps ran the previous programs)` : "" };
  renderRules();
}
const FEATS = (k) => ["v", ...Array.from({ length: k }, (_, b) => "h" + b), "n8", "r2", "N", "S", "E", "W", ...Array.from({ length: k }, (_, b) => "m" + b)];
function validatePrograms(ps, m) {
  if (!Array.isArray(ps) || !ps.length) throw new Error("expected a non-empty JSON list of programs");
  const need = Math.max(...m.schedule.map((s) => s.program)) + 1, names = new Set(FEATS(m.k));
  if (ps.length < need) throw new Error(`the schedule uses programs 0..${need - 1}; only ${ps.length} given`);
  return ps.map((p, pi) => {
    if (!p || !Array.isArray(p.rules)) throw new Error(`program ${pi}: missing "rules" list`);
    return { rules: p.rules.map((r, ri) => {
      const where = `program ${pi} rule ${ri}`;
      if (!Array.isArray(r.if) || !r.then) throw new Error(`${where}: needs "if" (list) and "then"`);
      for (const c of r.if) if (!Array.isArray(c) || c.length !== 3 || !names.has(c[0]) || !["<=", ">=", "=="].includes(c[1]) || !Number.isInteger(c[2])) throw new Error(`${where}: bad condition ${JSON.stringify(c)} (feature one of ${[...names].join(",")}; op <=, >=, ==; integer constant)`);
      if (![0, 1].includes(r.then.v) || !Number.isInteger(r.then.h ?? 0) || (r.then.h ?? 0) < 0 || (r.then.h ?? 0) >= 2 ** m.k) throw new Error(`${where}: then.v must be 0|1 and then.h 0..${2 ** m.k - 1}`);
      return { if: r.if, then: { v: r.then.v, h: r.then.h ?? 0 } };
    }) };
  });
}
async function applyEdit(mode) {
  const e = eng; if (!e || e.kind !== "rule") return;
  let ps;
  try { ps = mode === "reset" ? e.base.programs : validatePrograms(JSON.parse($("rule-edit").value), e.m); } catch (err) { $("edit-err").textContent = err.message; return; }
  setPlaying(false); const prevVer = e.ver; e.ver = ++verCounter; e.counts = null; e.memo = null;
  const t0 = performance.now(), m2 = { ...e.m, programs: ps };
  if (mode === "continue") {
    const g0 = player.t * e.S + player.sub, old = e.progAt;
    e.m = m2; simulateRule(e, ps, g0, e.seq); e.hybridFrom = g0;
    e.progAt = (g) => (g < g0 ? old(g) : ps[programAt(m2, Math.floor(g / e.S) + 1)]);
    e.verNote = `EXPERIMENTAL hybrid: substeps 0..${g0} from version ${prevVer}, substeps after ${g0} from the edited programs`;
  } else {
    e.m = m2; simulateRule(e, ps, 0, null); e.hybridFrom = null; e.progAt = (g) => ps[programAt(m2, Math.floor(g / e.S) + 1)];
    e.verNote = mode === "reset" ? "export programs, replayed from frame 0" : "edited programs, replayed from frame 0";
  }
  e.ms = performance.now() - t0; e.runMatch = e.hybridFrom == null && ruleRun(e.m).frames.every((f, t) => f.every((x, i) => x === e.vis[t * e.N + i]));
  e.sha = await hex(e.vis); player.vis = e.vis; player.cont = null;  // every cached frame of the old version is replaced
  computeLine(); renderRules(); render(); toast(`rules version ${e.ver}: ${mode === "continue" ? "experimental hybrid" : "recomputed from frame 0"}`);
}

function renderNcaPhases(e) {
  const m = e.m, { C, Fh } = m, hd = e.hdr?.header || {}, w = m.weight_dtype === "f16" ? 2 : 4, per = Fh * C * 9 + Fh + C * Fh + C, sch = scheduleOf(e);
  const mode = m.version === 2 ? m.mode : "v1 (one rule)", blobs = Object.fromEntries((hd.blobs || []).map(([n, s]) => [n, s]));
  const r = blobs.Ap ? blobs.Ap[2] : null, dpar = r ? Fh * r + r * C * 9 + Fh + C * r + r * Fh + C : 0, initVals = blobs.init_g ? blobs.init_g.reduce((a, b) => a * b, 1) : 0;
  let h = `<p class="note">Neural CA parameters are learned real numbers; they are summarised here and never presented as readable rules.</p>
    <dl class="facts"><dt>architecture</dt><dd>C=${C} (1 visible + ${C - 1} hidden) · Fh=${Fh} · S=${m.S} substeps/frame · K=${m.K ?? 1} rule${(m.K ?? 1) > 1 ? "s" : ""} · mode ${esc(mode)}${r ? ` · delta rank r=${r}` : ""}</dd>
    <dt>initial state</dt><dd>${m.init?.kind === "lowres" ? `learned low-res hidden init, factor ${m.init.factor}: ${fmtInt(initVals)} stored values (${fmtInt(initVals * w)} B)` : "visible = first frame, hidden = 0"}</dd>
    <dt>params per rule</dt><dd>${fmtInt(per)} = Fh·C·9 + Fh + C·Fh + C${r ? ` (shared base ${fmtInt(per)} values = ${fmtInt(per * w)} B; per phase rank-${r} delta ${fmtInt(dpar)} = Fh·r + r·C·9 + Fh + C·r + r·Fh + C)` : ""}</dd>
    <dt>file</dt><dd>${fmtInt(e.bytes)} B deflated · ${e.hdr ? fmtInt(e.hdr.raw_bytes) + " B raw" : "nca-v1 JSON"} · weights ${esc(m.weight_dtype)}</dd></dl>
    <div class="scrollbox"><table class="rtab"><thead><tr><th>phase</th><th>frames</th><th>rule</th><th>reason</th><th class="num">effective params</th>${r ? '<th class="num">delta params</th>' : ""}<th class="num">changed vs prev (|Δ|>1e-6)</th><th class="num">L2 of change</th><th class="num">stored bytes</th></tr></thead><tbody>`;
  const seen = new Map();
  sch.forEach((s, i) => {
    const R = m.rules ? m.rules[s.rule] : m, P = i ? (m.rules ? m.rules[sch[i - 1].rule] : m) : null;
    let ch = 0, l2 = 0;
    if (P) for (const n of ["Wp", "bp", "Wu", "bu"]) for (let j = 0; j < R[n].length; j++) { const d = R[n][j] - P[n][j]; if (Math.abs(d) > 1e-6) ch++; l2 += d * d; }
    const bytes = r ? `${fmtInt(dpar * w)} (delta) + shared base` : m.mode === "fixed" || !m.rules ? (i ? "shared (one rule stored)" : fmtInt(per * w)) : seen.has(s.rule) ? `same rule as phase ${seen.get(s.rule)}, stored once` : fmtInt(per * w);
    if (!seen.has(s.rule)) seen.set(s.rule, i);
    h += `<tr><td>${i}</td><td>${s.start}–${s.end - 1}</td><td>${s.rule}</td><td>${esc(s.reason ?? "–")}</td><td class="num">${fmtInt(per)}</td>${r ? `<td class="num">${fmtInt(dpar)}</td>` : ""}<td class="num">${P ? fmtInt(ch) + " / " + fmtInt(per) : "–"}</td><td class="num">${P ? l2 ? Math.sqrt(l2).toExponential(3) : "0" : "–"}</td><td class="num">${bytes}</td></tr>`;
  });
  $("rules-body").innerHTML = h + `</tbody></table></div><p class="cm">effective params of a phase = the parameters that phase's substeps use (delta mode: base + low-rank update, combined in float64 and rounded to float32 once, as in ca/phase_nca.py). Stored bytes are raw float bytes before deflate.</p>`;
}

// ---------- cell inspector ----------
$("cv-ca").addEventListener("click", (ev) => {
  if (!eng) return; const r = ev.target.getBoundingClientRect(), x = Math.floor((ev.clientX - r.left) / (r.width / player.W)), y = Math.floor((ev.clientY - r.top) / (r.height / player.H));
  if (x < 0 || y < 0 || x >= player.W || y >= player.H) return; insp.i = y * player.W + x; render();
});
function renderInspector() {
  const e = eng, i = insp.i; if (!e || i == null) { $("insp-where").textContent = "// click a cell of the CA panel"; $("insp-body").innerHTML = '<p class="cm">no cell selected</p>'; return; }
  const S = e.S, g = player.t * S + player.sub, f = Math.floor(g / S) + 1, last = g >= (e.T - 1) * S, x = i % player.W, y = Math.floor(i / player.W);
  const sch = scheduleOf(e), ph = sch.findIndex((p) => f >= p.start && f < p.end), word = e.kind === "rule" ? "program" : "rule";
  const rid = e.kind === "rule" ? programAt(e.m, f) : e.m.sched ? e.m.sched[Math.min(f, e.T - 1)] : 0;
  $("insp-where").textContent = `// cell (x=${x}, y=${y}), index ${i} · state after substep ${g} = frame ${player.t} + ${player.sub} (frame t = after t·S substeps, S=${S})`;
  let h = `<dl class="facts"><dt>cell</dt><dd>x=${x}, y=${y} (row-major index ${i})</dd><dt>state shown</dt><dd>after ${g} substeps (frame ${player.t}, substep ${player.sub} of ${S}) · version ${e.ver}</dd>
    <dt>next substep</dt><dd>${last ? "none: this is the last frame of the export" : `${g} → ${g + 1}, part of producing frame ${f}; ${word} ${rid} in effect (phase ${ph < 0 ? "–" : ph}${e.kind === "rule" && e.hybridFrom != null && g < e.hybridFrom ? ", pre-edit programs of the hybrid" : ""})`}</dd></dl>`;
  if (e.kind === "rule") {
    const st = stateAt(e, g), m = e.m, prog = e.progAt(Math.min(g, (e.T - 1) * S - 1)), ex = explainCell(m, prog, st.v, st.h, i);
    h += `<h3 class="minor">features <span class="cm">// rule_core.features() on the current grid (zero boundary)</span></h3><pre class="codeblock">${esc(Object.entries(ex.features).map(([k, v]) => `${k}=${v}`).join("  "))}</pre>`;
    if (!last) {
      const nx = stateAt(e, g + 1), agree = nx.v[i] === ex.next.v && ex.next.h.every((b, j) => b === nx.h[j * e.N + i]);
      h += `<h3 class="minor">rule evaluation <span class="cm">// rule_core.fire(): first firing rule wins; ✓/✗ per condition; ✗ in amber = first failing condition</span></h3><div class="scrollbox"><pre class="codeblock" style="max-height:none;border:0">`;
      for (const r of ex.rules) {
        if (r.skipped) { h += `<span class="skip">R${r.r}..R${prog.rules.length - 1}: not evaluated (R${ex.fired} fired first)</span>\n`; break; }
        const conds = r.conds.map((c, j) => `<span class="${c.ok ? "ok" : j === r.firstFail ? "fired" : "bad"}">${esc(`${c.n} ${c.op} ${c.c}`)} [${c.n}=${c.val} ${c.ok ? "✓" : "✗"}]</span>`).join(" and ") || "always";
        h += r.ok ? `<span class="fired">R${r.r} FIRES:</span> if ${conds} -> v=${r.then.v}${m.k ? ", h=" + r.then.h.toString(2).padStart(m.k, "0") : ""}\n`
          : `<span class="skip">R${r.r} fails at</span> ${esc(`${r.conds[r.firstFail].n} ${r.conds[r.firstFail].op} ${r.conds[r.firstFail].c}`)} (${r.conds[r.firstFail].n}=${r.conds[r.firstFail].val}): ${conds}\n`;
      }
      if (ex.fired < 0) h += `<span class="fired">no rule fired → default: keep state</span>\n`;
      h += `</pre></div><dl class="facts"><dt>current</dt><dd>v=${ex.cur.v}${m.k ? ", h=" + ex.cur.h.join("") + " (h0..)" : ""}</dd><dt>next</dt><dd>v=${ex.next.v}${m.k ? ", h=" + ex.next.h.join("") : ""} · ${agree ? "equals the stepped state at substep " + (g + 1) : "DIFFERS from the stepped state (bug)"}</dd></dl>`;
    }
  } else {
    const m = e.m, s = stateAt(e, g), C = m.C, tr = traceCell(m, ruleAt(m, f), s, i);
    const nx = last ? null : stateAt(e, g + 1), agree = nx && tr.next.every((v, c) => Object.is(v, nx[c * e.N + i]));
    h += `<p class="cm">traceCell(): the substep's own arithmetic for this cell (float64 sums, float32 per stored value). Learned weights are not shown as rules.</p>
      <div><div><h3 class="minor">channels <span class="cm">// 3×3 neighbourhood NW N NE / W · E / SW S SE; · = outside (zero padding)</span></h3><div class="scrollbox"><table class="rtab"><thead><tr><th>c</th><th class="num">now</th><th>3×3 neighbourhood</th>${last ? "" : '<th class="num">Wu·p</th><th class="num">bu</th><th class="num">next</th>'}</tr></thead><tbody>`;
    for (let c = 0; c < C; c++) h += `<tr><td>${c === 0 ? "0 vis" : c}</td><td class="num">${fmtN(tr.cur[c])}</td><td>${tr.nbr[c].map(fmtN).join(" ")}</td>${last ? "" : `<td class="num">${fmtN(tr.upd[c])}</td><td class="num">${fmtN(tr.bias[c])}</td><td class="num">${fmtN(tr.next[c])}</td>`}</tr>`;
    h += `</tbody></table></div>${last ? "" : `<p>visible next = ${fmtN(tr.next[0])} → pixel ${tr.next[0] > m.threshold ? "on" : "off"} (threshold ${m.threshold}) · ${agree ? "next state equals the stepped state at substep " + (g + 1) + " (all channels, bit-exact)" : "DIFFERS from the stepped state (bug)"}</p>`}</div>`;
    h += `<div><h3 class="minor">perception <span class="cm">// pre = bp + Σ Wp·nbr (float64), post = float32(relu(pre))</span></h3><div class="scrollbox"><table class="rtab"><thead><tr><th>f</th><th class="num">pre-activation</th><th class="num">post-ReLU</th></tr></thead><tbody>`;
    for (let k = 0; k < m.Fh; k++) h += `<tr><td>${k}</td><td class="num">${fmtN(tr.pre[k])}</td><td class="num${tr.post[k] ? "" : " skip"}">${fmtN(tr.post[k])}</td></tr>`;
    h += `</tbody></table></div><p class="cm">${tr.post.filter((v) => v > 0).length}/${m.Fh} features active</p></div></div>`;
  }
  $("insp-body").innerHTML = h;
}

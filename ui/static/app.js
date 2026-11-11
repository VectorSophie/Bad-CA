// Bad CA!! local UI: vanilla JS, no build step. Three views: demo (one NCA model, the featured one by default),
// rule lab (explicit dlrule-v1 rules), research = "how it was built" (decision flow, new runs, archive of every export).
// CA frames come only from the export: NCA in worker.js (streamed, frame-index based), explicit rules here with
// player/rule_core.mjs. Substep states, the rule panel and the inspectors use the same core functions.
import { parseNcab, parseNcaJson, ruleAt, substep, traceCell, ncabHeader } from "/player/nca_core.mjs";
import { parseRules, run as ruleRun, step as ruleStep, programAt, explainCell } from "/player/rule_core.mjs";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const fmtS = (s) => (s == null ? "–" : s < 120 ? `${(+s).toFixed(1)} s` : `${(s / 60).toFixed(1)} min`);
const fmtN = (x) => (x == null ? "·" : x === 0 ? "0" : Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3 ? x.toExponential(3) : x.toPrecision(5));
const b64u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const COL = { on: [236, 232, 223], off: [12, 12, 11], fn: [255, 90, 78], fp: [78, 168, 255], sel: [255, 180, 58] };
const inflate = async (b) => new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
const isNcab = (u8) => u8[0] === 0x4e && u8[1] === 0x43 && u8[2] === 0x41 && u8[3] === 0x42;
const hex = async (u8) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", u8))].map((x) => x.toString(16).padStart(2, "0")).join("");
const sup = (k) => `<sup>(${k})</sup>`;

async function api(path, opts = {}) {
  const o = { ...opts, headers: { ...(opts.headers || {}) } };
  if (o.method === "POST") o.headers["X-CA-UI"] = "1";
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, o), j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
async function bytesOf(url) { const r = await fetch(url); if (!r.ok) throw new Error(`HTTP ${r.status}`); return new Uint8Array(await r.arrayBuffer()); }

let toastTimer;
function toast(msg, bad = false) {
  const t = $("toast"); t.textContent = msg; t.className = "toast"; void t.offsetWidth; t.className = "toast on" + (bad ? " bad" : "");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = "toast"), 4500);
}

// ---------- header pixel field (header only; stops while a video plays, when hidden, and under reduced motion) ----------
const anyPlaying = () => DEMO.playing || LAB.playing;
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
    if (reduced || anyPlaying() || document.hidden) return;
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

// ---------- views ----------
const VIEWS = ["demo", "lab", "research"];
let view = "demo";
function showView(v) {
  const jump = v === "new"; if (jump) v = "research"; // "Open your own video" lands on run.new
  if (!VIEWS.includes(v)) v = "demo";
  view = v;
  for (const k of VIEWS) { $("v-" + k).hidden = k !== v; $("tab-" + k).setAttribute("aria-selected", k === v); }
  if (v !== "demo") DEMO.setPlaying(false);
  if (v !== "lab") LAB.setPlaying(false);
  if (v === "lab" && !LAB.e && !labLoading) openLab($("lab-sel").value || LAB_DEFAULT);
  (v === "demo" ? DEMO : v === "lab" ? LAB : null)?.fit();
  if (jump) $("new").scrollIntoView();
}
addEventListener("hashchange", () => showView(location.hash.slice(1)));

// ---------- engines ----------
// NCA: frames streamed from worker.js, kept packed (1 bit per pixel); full-state checkpoints every `every` frames;
// any other state is regenerated from the nearest earlier checkpoint with substep() (never from original frames).
function ncaEngine(u8, m, hdr, onUpdate) {
  const N = m.H * m.W, B = Math.ceil(N / 8);
  const e = { kind: "nca", m, hdr, bytes: u8.length, T: m.T, S: m.S, H: m.H, W: m.W, N, B, packed: new Uint8Array(m.T * B), upto: 0, ckpt: new Map(),
    every: 1, sha: null, ms: null, memo: null, error: null, chunks: 0, t0: performance.now() };
  const w = new Worker("/static/worker.js", { type: "module" });
  w.onmessage = ({ data: d }) => {
    if (d.error) e.error = d.error;
    else if (d.info) e.every = d.info.every;
    else if (d.ckpt != null) e.ckpt.set(d.ckpt, d.state);
    else if (d.packed) {
      if (d.from !== e.upto) e.error = `stream gap: chunk starts at frame ${d.from}, expected ${e.upto}`; // never expected
      e.packed.set(d.packed, d.from * B); e.upto = d.upto; e.chunks++;
    } else if (d.done) { e.sha = d.sha; e.ms = d.ms; }
    onUpdate(e);
  };
  const copy = u8.slice(); w.postMessage({ id: 1, bytes: copy.buffer }, [copy.buffer]);
  e.stop = () => w.terminate();
  e.frame = (t) => { const out = new Uint8Array(N), o = t * B; for (let i = 0; i < N; i++) out[i] = (e.packed[o + (i >> 3)] >> (i & 7)) & 1; return out; };
  e.stateAt = (g) => { // full state after g substeps (frame t = after t*S substeps)
    const S = e.S, t = Math.floor(g / S), t0 = t - (t % e.every); let from = t0 * S, s = e.ckpt.get(t0);
    if (!s) return null;
    if (e.memo && e.memo.g <= g && e.memo.g >= from) ({ g: from, s } = e.memo);
    for (let x = from; x < g; x++) s = substep(e.m, ruleAt(e.m, Math.floor(x / S) + 1), s);
    e.memo = { g, s }; return s;
  };
  return e;
}
async function loadNca(u8, onUpdate) {
  const m = isNcab(u8) ? await parseNcab(u8, inflate) : parseNcaJson(JSON.parse(new TextDecoder().decode(u8)));
  return ncaEngine(u8, m, isNcab(u8) ? await ncabHeader(u8, inflate) : null, onUpdate);
}
const isRuleJson = (u8) => { if (isNcab(u8)) return null; try { const o = JSON.parse(new TextDecoder().decode(u8)); return o.format === "ca-universal/dlrule-v1" || (o.format === "ca-universal/rule-v1" && Array.isArray(o.programs)) ? o : null; } catch { return null; } };

async function ruleEngine(o, bytes) {
  const m = parseRules(o), N = m.H * m.W, t0 = performance.now();
  const e = { kind: "rule", m, base: m, json: o, bytes, S: m.S, T: m.T, H: m.H, W: m.W, N, ver: 1, verNote: "export", hybridFrom: null, stop: () => {} };
  e.progAt = (g) => m.programs[programAt(m, Math.floor(g / m.S) + 1)];
  simulateRule(e, m.programs, 0, null); e.ms = performance.now() - t0;
  const ref = ruleRun(m).frames; e.runMatch = ref.every((f, t) => f.every((x, i) => x === e.vis[t * N + i])); // the substep cache equals run()
  e.sha = await hex(e.vis); e.upto = m.T;
  e.frame = (t) => e.vis.subarray(t * N, (t + 1) * N);
  e.stateAt = (g) => e.seq[g];
  return e;
}
// seq[g] = full state {v, h} after g substeps. With a prefix, states 0..g0 are kept and later substeps use `programs`.
function simulateRule(e, programs, g0, prefix) {
  const m = e.m, S = m.S, G = (m.T - 1) * S, seq = prefix ? prefix.slice(0, g0 + 1) : [{ v: m.first.slice(), h: m.h0.slice() }];
  for (let g = seq.length - 1; g < G; g++) { const [v, h] = ruleStep(m, programs[programAt(m, Math.floor(g / S) + 1)], seq[g].v, seq[g].h); seq.push({ v, h }); }
  e.seq = seq; e.vis = new Uint8Array(m.T * e.N);
  for (let t = 0; t < m.T; t++) e.vis.set(seq[t * S].v, t * e.N);
}
function scheduleOf(e) {
  if (e.kind === "rule") return e.m.schedule.map((p) => ({ start: p.start, end: p.end, rule: p.program, reason: p.reason ?? null }));
  if (e.m.schedule) return e.m.schedule.map((p) => ({ start: p.start, end: p.end, rule: p.rule, reason: p.reason ?? null }));
  return [{ start: 1, end: e.T, rule: 0, reason: "single rule (ncab-v1)" }];
}

// ---------- player (one per view; same markup from #player-tpl) ----------
function makePlayer(mount, tlSlot, opts = {}) {
  mount.append($("player-tpl").content.cloneNode(true));
  const q = (r) => mount.querySelector(`[data-r="${r}"]`);
  const P = { e: null, t: 0, sub: 0, playing: false, fps: 30, fpsSrc: "assumed", orig: null, origReq: null, sel: null, want: null, label: "", name: null, B: null, q, opts };
  const ctx = (r) => q(r).getContext("2d");
  const cx = { ca: ctx("cv-ca"), orig: ctx("cv-orig"), err: ctx("cv-err"), b: ctx("cv-b") };
  if (!opts.compareB) q("b-wrap").hidden = true;
  tlSlot.innerHTML = `<div class="tl"><svg role="img" aria-label="timeline: phases, computed range and playhead; click to seek" data-r="tl-svg"></svg>
    <div class="tl-axis"><span data-r="tl-a"></span><span data-r="tl-b"></span></div><p class="cur-line" data-r="tl-cur"></p></div>`;
  const tq = (r) => tlSlot.querySelector(`[data-r="${r}"]`);

  const paint = (c, W, H, fn) => {
    const img = c.createImageData(W, H), d = img.data;
    for (let i = 0; i < W * H; i++) { const k = fn(i); d[4 * i] = k[0]; d[4 * i + 1] = k[1]; d[4 * i + 2] = k[2]; d[4 * i + 3] = 255; }
    c.putImageData(img, 0, 0);
  };
  P.fit = () => {
    const e = P.e; if (!e) return;
    const zv = q("zoom").value, avail = Math.max(60, (q("cv-ca").closest(".stage").clientWidth || 300) - 4);
    const z = zv === "fit" ? Math.max(1, Math.min(16, Math.floor(avail / e.W), Math.floor((innerHeight * 0.6) / e.H))) : +zv;
    const sz = (c, W, H) => { c.width = W; c.height = H; c.style.width = `${W * z}px`; c.style.height = `${H * z}px`; };
    for (const r of ["cv-ca", "cv-orig", "cv-err"]) sz(q(r), e.W, e.H);
    if (P.B?.e) sz(q("cv-b"), P.B.e.W, P.B.e.H);
    P.render(true);
  };
  q("zoom").onchange = P.fit; addEventListener("resize", () => !mount.closest(".view").hidden && P.fit());

  P.render = (force = false) => {
    const e = P.e; if (!e || P.t >= e.upto) return;
    const { t, sub } = P, N = e.N, S = e.S, f = e.frame(t); P.shown = true;
    let vis = (i) => f[i], cont = null;
    if (sub > 0 || (q("t-cont").checked && !P.playing && e.kind === "nca")) {
      const st = e.stateAt(t * S + sub);
      if (st && e.kind === "rule") vis = (i) => st.v[i];
      else if (st) { vis = sub > 0 ? (i) => (st[i] > e.m.threshold ? 1 : 0) : vis; if (q("t-cont").checked && !P.playing) cont = (i) => st[i]; }
    }
    paint(cx.ca, e.W, e.H, (i) => (i === P.sel ? COL.sel : cont ? ((g) => [g, g, g])(Math.round(Math.min(1, Math.max(0, cont(i))) * 255)) : vis(i) ? COL.on : COL.off));
    const toolsOn = !q("tools").hidden, o = P.orig && toolsOn ? P.orig.u8.subarray(t * N, (t + 1) * N) : null;
    q("fig-orig").hidden = !(o && q("t-orig").checked); q("fig-err").hidden = !(o && q("t-err").checked);
    if (o && q("t-orig").checked) paint(cx.orig, e.W, e.H, (i) => (o[i] ? COL.on : COL.off));
    if (o && q("t-err").checked) paint(cx.err, e.W, e.H, (i) => { const a = o[i], b = f[i]; return a && b ? COL.on : !a && !b ? COL.off : a ? COL.fn : COL.fp; });
    renderB();
    let wrong = 0; if (o) for (let i = 0; i < N; i++) wrong += o[i] !== f[i];
    const sec = (t / P.fps).toFixed(2);
    q("counter").textContent = `frame ${String(t).padStart(String(e.T - 1).length, "0")} / ${e.T - 1} · ${sec} s` + (sub ? ` · +${sub}/${S} substeps toward frame ${t + 1}` : "")
      + (o ? ` · ${wrong} px differ from the original (${((100 * wrong) / N).toFixed(2)}%)` : "");
    q("prev").disabled = t === 0 && !sub; q("next").disabled = q("sub").disabled = t >= e.T - 1;
    status(); renderTimeline();
    opts.onFrame?.(P, force);
    if (!P.playing) opts.onPaused?.(P);
  };
  function status() {
    const e = P.e; if (!e) return;
    const done = e.upto >= e.T, waiting = P.want != null && P.want >= e.upto;
    let s = e.kind === "rule"
      ? `explicit rules run in this browser with rule_core.mjs (${e.hybridFrom != null ? "hybrid: no single-version run() reference" : e.runMatch ? "identical to run()" : "MISMATCH vs run()"}): ${e.T} frames × ${e.S} substeps, ${e.ms.toFixed(0)} ms · sha256 ${e.sha.slice(0, 16)}… · version ${e.ver}: ${e.verNote}`
      : done && e.sha ? `all ${fmtInt(e.T)} frames computed from the export in a worker (${(e.ms / 1000).toFixed(1)} s, ${((e.T - 1) / (e.ms / 1000)).toFixed(1)} frames/s) · sha256 of binary frames ${e.sha.slice(0, 16)}…`
      : (e.upto > 1 ? `computing ahead at ${((e.upto - 1) / ((performance.now() - e.t0) / 1000)).toFixed(1)} frames/s (real time is ${P.fps} frames/s; playback waits at the computed edge) · ` : "computing frames from the model file · ")
        + `computed up to frame ${fmtInt(e.upto - 1)} of ${fmtInt(e.T - 1)}`;
    if (waiting) s = `waiting for frame ${P.want} (computed up to ${e.upto - 1}) · ` + s;
    if (e.error) s = "ERROR: " + e.error + " · " + s;
    q("status").textContent = s;
  }
  function renderTimeline() {
    const e = P.e, svg = tq("tl-svg"); if (!e) return;
    const Wd = 1000, T = e.T, x = (fr) => (fr / T) * Wd, sch = scheduleOf(e);
    const g = P.t + P.sub / e.S, ph = phaseAt(e, P.t, P.sub);
    let h = `<rect class="seg0" x="0" y="0" width="${x(1)}" height="26"><title>frame 0: the given first frame</title></rect>`;
    sch.forEach((p, k) => {
      h += `<rect class="seg${k % 2 ? " alt" : ""}${k === ph.k ? " cur" : ""}" x="${x(p.start)}" y="0" width="${Math.max(0.5, x(p.end) - x(p.start))}" height="26"><title>phase ${k + 1}: frames ${p.start}–${p.end - 1} (${(p.start / P.fps).toFixed(2)}–${((p.end - 1) / P.fps).toFixed(2)} s), rule id ${p.rule}</title></rect>`;
      if (k) h += `<line class="bd" x1="${x(p.start)}" x2="${x(p.start)}" y1="0" y2="26"/>`;
    });
    h += `<rect class="buf-bg" x="0" y="30" width="${Wd}" height="6"/><rect class="buf" x="0" y="30" width="${x(e.upto)}" height="6"><title>computed up to frame ${e.upto - 1}</title></rect>`;
    const xn = x(g + 0.5); h += `<line class="now" x1="${xn}" x2="${xn}" y1="0" y2="40"/>`;
    svg.setAttribute("viewBox", `0 0 ${Wd} 40`); svg.setAttribute("preserveAspectRatio", "none"); svg.innerHTML = h;
    tq("tl-a").textContent = `frame 0 · 0 s`; tq("tl-b").textContent = `frame ${T - 1} · ${((T - 1) / P.fps).toFixed(1)} s`;
    const word = e.kind === "rule" ? "program" : "rule id";
    tq("tl-cur").innerHTML = `<span class="leg-seg"></span> ${sch.length} phase${sch.length > 1 ? "s" : ""} (boundaries = thin lines) · <span class="leg-buf"></span> computed: frames 0–${e.upto - 1} of ${T - 1} · <span class="leg-now"></span> playhead · `
      + (ph.k < 0 ? "frame 0 (given first frame)" : `now: phase ${ph.k + 1} of ${sch.length}, ${word} ${sch[ph.k].rule}`) + ` · seconds = frame / ${P.fps} fps (${esc(P.fpsSrc)}) · click or drag to seek`;
  }
  const seekX = (ev) => { const r = tq("tl-svg").getBoundingClientRect(); if (P.e) P.seek(Math.round(((ev.clientX - r.left) / r.width) * P.e.T - 0.5)); };
  tq("tl-svg").addEventListener("pointerdown", (ev) => { seekX(ev); tq("tl-svg").setPointerCapture(ev.pointerId); });
  tq("tl-svg").addEventListener("pointermove", (ev) => ev.buttons & 1 && seekX(ev));

  // playback: wall clock only decides WHEN the next computed frame is shown; frames are never skipped in the simulation
  let raf = null, last = 0, acc = 0;
  function loop(ts) {
    if (!P.playing) return;
    acc += Math.min(0.25, (ts - last) / 1000) * P.fps * +q("speed").value; last = ts; // a backgrounded tab resumes where it was
    let moved = false;
    while (acc >= 1) {
      if (P.t >= P.e.T - 1) { P.setPlaying(false); return; }
      if (P.t + 1 >= P.e.upto) { acc = 0; P.want = P.t + 1; break; } // not computed yet: wait (buffer bar shows progress)
      acc -= 1; P.t++; P.sub = 0; P.want = null; moved = true;
    }
    if (moved) P.render(); else status();
    raf = requestAnimationFrame(loop);
  }
  P.setPlaying = (on) => {
    if (!P.e || P.playing === on) return;
    P.playing = on; q("play").textContent = on ? "[ pause ]" : "[ play ]";
    if (on) { field.stop(); if (P.t >= P.e.T - 1) P.t = 0; P.sub = 0; last = performance.now(); acc = 0; raf = requestAnimationFrame(loop); }
    else { cancelAnimationFrame(raf); P.want = null; P.render(); }
  };
  P.seek = (t, sub = 0) => {
    const e = P.e; if (!e) return; P.setPlaying(false);
    t = Math.max(0, Math.min(e.T - 1, t));
    if (t >= e.upto) { P.want = t; status(); return; } // resumes in onStream when computed
    P.want = null; P.t = t; P.sub = sub; P.render();
  };
  P.step = (d) => { P.setPlaying(false); P.seek(P.sub && d < 0 ? P.t : P.t + d); };
  P.substep = () => { P.setPlaying(false); const e = P.e; if (!e || P.t >= e.T - 1) return; if (P.sub + 1 >= e.S) { if (P.t + 1 < e.upto) P.seek(P.t + 1); } else { P.sub++; P.render(); } };
  P.onStream = () => {
    const e = P.e; if (!e) return;
    if (P.want != null && P.want < e.upto && !P.playing) { const w = P.want; P.want = null; P.seek(w); return; }
    if (!P.playing && (!P.shown || e.sha)) P.render(); else { status(); renderTimeline(); }
    q("play").disabled = e.upto < 2;
  };
  q("play").onclick = () => P.setPlaying(!P.playing); q("prev").onclick = () => P.step(-1); q("next").onclick = () => P.step(1); q("sub").onclick = P.substep;
  q("tools-btn").onclick = () => { const on = q("tools").hidden; q("tools").hidden = !on; q("tools-btn").setAttribute("aria-expanded", on); q("tools-btn").textContent = on ? "[ tools: on ]" : "[ tools ]"; if (on) opts.loadOrig?.(P); P.fit(); };
  for (const r of ["t-orig", "t-err"]) q(r).onchange = () => { opts.loadOrig?.(P); P.render(); };
  q("t-cont").onchange = () => P.render();
  q("cv-ca").addEventListener("click", (ev) => {
    const e = P.e; if (!e) return; const r = ev.target.getBoundingClientRect(), x = Math.floor((ev.clientX - r.left) / (r.width / e.W)), y = Math.floor((ev.clientY - r.top) / (r.height / e.H));
    if (x < 0 || y < 0 || x >= e.W || y >= e.H) return; P.sel = y * e.W + x; P.setPlaying(false); P.render();
  });
  P.load = (e, meta) => {
    P.e?.stop(); if (P.B?.e) { P.B.e.stop(); P.B = null; q("fig-b").hidden = true; q("b-sel").value = ""; q("b-line").textContent = ""; tlB(); }
    P.setPlaying(false); Object.assign(P, { e, t: 0, sub: 0, sel: null, want: null, orig: null, origReq: null, shown: false }, meta); q("warn").textContent = "";
    q("play").disabled = e.upto < 2; P.fit();
  };
  // ---- B: second export synchronised by frame index (demo tools) ----
  function tlB() { q("timeline-b").innerHTML = P.B?.e ? `<p class="cm">B phases: ${scheduleOf(P.B.e).map((p, k) => `${k + 1}: ${p.start}–${p.end - 1} rule ${p.rule}`).join(" · ")}</p>` : ""; }
  function renderB() {
    const be = P.B?.e; if (!be || q("tools").hidden) return;
    const t = Math.min(P.t, be.T - 1);
    q("fig-b").hidden = false;
    if (t < be.upto) { const f = be.frame(t); paint(cx.b, be.W, be.H, (i) => (f[i] ? COL.on : COL.off)); }
  }
  q("b-sel").onchange = async () => {
    const name = q("b-sel").value; P.B?.e?.stop(); P.B = name ? { name, e: null } : null; q("fig-b").hidden = true; q("b-line").textContent = ""; tlB();
    if (!name) return;
    const r = runs.find((x) => x.name === name); q("b-line").textContent = "loading B…";
    try {
      const u8 = await bytesOf(`/api/runs/${encodeURIComponent(name)}/export`), o = isRuleJson(u8);
      const be = o ? await ruleEngine(o, u8.length) : await loadNca(u8, (be) => { q("b-line").textContent = `B: ${esc(r?.label ?? name)} · computed ${be.upto}/${be.T} frames` + (be.sha ? ` · sha256 ${be.sha.slice(0, 16)}…` : "") + (be.T !== P.e.T ? ` · T differs from A (${P.e.T}); B holds its last frame` : ""); renderB(); });
      if (P.B?.name !== name) return be.stop();
      P.B.e = be; q("b-cap").innerHTML = `B <span class="cm">${esc(r?.label ?? name)} (export only)</span>`; tlB(); P.fit();
    } catch (err) { q("b-line").textContent = "B could not be replayed: " + err.message; }
  };
  return P;
}

// ---------- the two players ----------
const DEMO = makePlayer($("demo-player"), $("demo-timeline-slot"), { compareB: true, onFrame: (P, force) => renderRulePanel(P, force), onPaused: () => renderInspector(), loadOrig: loadOrigDemo });
const LAB = makePlayer($("lab-player"), $("lab-timeline-slot"), { onPaused: () => renderLabInspector(), loadOrig: loadOrigLab });
window.__ui = { DEMO, LAB };
document.addEventListener("keydown", (ev) => {
  const tag = ev.target.tagName, P = view === "demo" ? DEMO : view === "lab" ? LAB : null;
  if (!P?.e || (["INPUT", "SELECT", "TEXTAREA"].includes(tag) && ev.target.type !== "checkbox") || (tag === "BUTTON" && ev.key === " ")) return;
  if (ev.key === " ") { ev.preventDefault(); P.setPlaying(!P.playing); }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); P.step(1); }
  else if (ev.key === "ArrowLeft") { ev.preventDefault(); P.step(-1); }
});

async function fetchOrig(P, url) {
  if (P.origReq) return; const e = P.e; P.origReq = url; P.q("warn").textContent = "";
  try {
    const o = await api(url); if (P.e !== e) return;
    if (o.bin.shape[0] < e.T || o.bin.shape[1] !== e.H || o.bin.shape[2] !== e.W) throw new Error(`original is ${o.bin.shape.join("×")}, export is ${e.T}×${e.H}×${e.W}`);
    P.orig = { ...o, u8: b64u8(o.bin.b64) }; if (o.fps) { P.fps = o.fps; P.fpsSrc = "source video"; }
    if (o.first_frame_mismatch_px) toast(`original resolved from export meta differs from the export's first frame in ${o.first_frame_mismatch_px} px`, true);
    P.render(true);
  } catch (err) { if (P.e === e) P.q("warn").textContent = "no original for comparison: " + err.message; }
}
function loadOrigDemo(P) { if (!P.q("t-orig").checked && !P.q("t-err").checked) return; fetchOrig(P, demoSrc.kind === "featured" ? "/api/featured/original" : `/api/runs/${encodeURIComponent(demoSrc.name)}/original`); }
function loadOrigLab(P) { if ((P.q("t-orig").checked || P.q("t-err").checked) && LAB.name) fetchOrig(P, `/api/runs/${encodeURIComponent(LAB.name)}/original`); }

// ---------- demo: featured model (default) or a research export ----------
let demoSrc = { kind: "featured" }, demoSeq = 0;
async function openFeatured() {
  const seq = ++demoSeq; demoSrc = { kind: "featured" };
  $("demo-banner").hidden = true; $("demo-warn").textContent = "";
  let f;
  try { f = await api("/api/featured"); } catch (err) { $("demo-title").textContent = "// no demo model: " + err.message; return; }
  const ft = f.manifest.featured;
  $("demo-title").textContent = `// ${ft.display_name || "demo model"}`; $("demo-dl").href = "/api/featured/export";
  const u8 = await bytesOf("/api/featured/export"); if (seq !== demoSeq) return;
  const sha = await hex(u8);  // verify what is actually played, in the browser
  if (sha !== ft.sha256) $("demo-warn").textContent = `WARNING: the model file's sha256 is ${sha}, but the manifest says ${ft.sha256 || "(none)"}. This is not the file the manifest describes (see Model details).`;
  const facts = [["source file", f.path], ["sha256", `${sha} · ${sha === ft.sha256 ? "matches the manifest" : `MISMATCH: manifest says ${ft.sha256 || "(none)"}`}`],
    ["model", ft.model], ["status", ft.status], ["why this model", ft.reason], ["trained from", ft.source], ["metrics", ft.metrics && Object.entries(ft.metrics).map(([k, v]) => `${k}: ${v}`).join(" · ")]];
  await startDemo(u8, seq, { fps: f.fps, fpsSrc: "featured manifest", name: null, label: ft.display_name, shaOk: sha === ft.sha256 }, facts);
}
async function openResearchNca(name, det) {
  const seq = ++demoSeq; demoSrc = { kind: "run", name };
  const label = det.summary.label;
  $("demo-title").textContent = `// research export: ${label}`; $("demo-warn").textContent = ""; $("demo-dl").href = `/api/runs/${encodeURIComponent(name)}/export`;
  $("demo-banner").hidden = false; $("demo-banner").innerHTML = `viewing a research export, not the featured model · <button class="btn" id="b-featured">[ back to the featured model ]</button>`;
  $("b-featured").onclick = () => openFeatured();
  const u8 = await bytesOf(`/api/runs/${encodeURIComponent(name)}/export`); if (seq !== demoSeq) return;
  const man = det.manifest || {}, pp = man.data?.preprocessing;
  await startDemo(u8, seq, { fps: pp?.fps || 30, fpsSrc: pp?.fps ? "run manifest" : "assumed 30 (no fps recorded)", name, label },
    [["run", name], ["label", label], ["sha256", await hex(u8)], ["source", man.c12?.path ?? (det.summary.kind === "run" || det.summary.kind === "example" ? `runs/pipe/${name}` : "imported file")], ["preset", man.preset]]);
}
// provenance shown only inside the collapsed "Model details" (errors that affect playback stay visible above it)
function renderDemoFacts(e, u8, facts) {
  const bi = blobInfo(e), n = (ks) => Object.entries(bi).filter(([k]) => ks(k)).reduce((t, [, s]) => t + s.reduce((x, y) => x * y, 1), 0), W = (k) => ["Wp", "bp", "Wu", "bu"].includes(k);
  const K = scheduleOf(e).length, rest = n((k) => !W(k));
  const rows = [["parameters", `${fmtInt(n(W) || e.m.Fh * e.m.C * 9 + e.m.Fh + e.m.C * e.m.Fh + e.m.C)} learned weights${rest ? ` + ${fmtInt(rest)} other stored values (initial hidden state or per-phase deltas)` : ""} · ${K} phase${K > 1 ? "s" : ""}, ${e.m.C} channels, ${e.m.Fh} features, ${e.S} updates per frame`],
    ["precision", e.m.weight_dtype === "f16" ? "float16 stored, computed in float32" : "float32"], ["file size", `${fmtInt(u8.length)} B`], ["frames", `${fmtInt(e.T)} at ${e.W}×${e.H}`], ...facts];
  $("demo-facts").innerHTML = rows.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}
async function startDemo(u8, seq, meta, facts = []) {
  try {
    const e = await loadNca(u8, (e) => { if (DEMO.e === e) { DEMO.onStream(); if (e.sha) window.__bench = { sha256: e.sha, T: e.T, H: e.H, W: e.W, ms: e.ms, chunks: e.chunks, error: e.error }; } });
    if (seq !== demoSeq) return e.stop();
    e.phaseStats = null; mx = null; DEMO.load(e, meta); renderRulePanel(DEMO, true); renderInspector(); renderDemoFacts(e, u8, facts);
    if (meta.fps && e.m.meta?.fps && e.m.meta.fps !== meta.fps) toast(`fps: manifest ${meta.fps}, export meta ${e.m.meta.fps}`, true);
  } catch (err) { $("demo-warn").textContent = "export could not be replayed: " + err.message; }
}

// ---------- demo right panel: the rule in effect ----------
function phaseAt(e, t, sub) {
  const sch = scheduleOf(e), f = sub > 0 || t === 0 ? t + 1 : t; // the frame whose rule produced / is producing what is shown
  const k = sch.findIndex((p) => f >= p.start && f < p.end);
  return { k, f, boundary: k > 0 && sub === 0 && t === sch[k].start, first: t === 0 && sub === 0 };
}
function blobInfo(e) { // stored blob names/shapes (ncab-v2 header.blobs, ncab-v1 header.params)
  const hd = e.hdr?.header || {}; return Object.fromEntries((hd.blobs || hd.params || []).map(([n, s]) => [n, s]));
}
function ruleParams(e, rid) { return e.m.rules ? e.m.rules[rid] : e.m; }
function phaseStats(e) {
  if (e.phaseStats) return e.phaseStats;
  const sch = scheduleOf(e);
  e.phaseStats = sch.map((s, k) => {
    if (!k) return null;
    const R = ruleParams(e, s.rule), Pv = ruleParams(e, sch[k - 1].rule), per = {};
    let ch = 0, l2 = 0, mx = 0;
    for (const n of ["Wp", "bp", "Wu", "bu"]) { let c = 0; for (let j = 0; j < R[n].length; j++) { const d = R[n][j] - Pv[n][j], a = Math.abs(d); if (a > 1e-6) c++; l2 += d * d; if (a > mx) mx = a; } per[n] = c; ch += c; }
    return { changed: ch, l2: Math.sqrt(l2), max: mx, per, prevRule: sch[k - 1].rule };
  });
  return e.phaseStats;
}
let mx = null, panelKey = "";
function renderRulePanel(P, force) {
  const e = P.e, el = $("rulepanel"); if (!e) return;
  const ph = phaseAt(e, P.t, P.sub), key = `${e.T}:${ph.k}:${ph.boundary}:${JSON.stringify(mx)}`;
  if (!force && key === panelKey) return; panelKey = key;
  const m = e.m, { C, Fh, S } = m, sch = scheduleOf(e), K = sch.length, p = sch[Math.max(0, ph.k)], k = p.rule, bi = blobInfo(e), v2 = m.version === 2;
  const users = sch.map((s, i) => (s.rule === k ? i + 1 : null)).filter(Boolean), stats = phaseStats(e)[Math.max(0, ph.k)], per = Fh * C * 9 + Fh + C * Fh + C;
  const sec = (fr) => (fr / P.fps).toFixed(2), delta = m.mode === "delta";
  const blobTxt = (n) => (v2 ? (delta ? `${n} (base) + per-phase delta` : `${n}[${k}] of ${n} ${bi[n]?.join("×")}`) : `stored param ${n} ${bi[n]?.join("×") ?? ""}`);
  const PM = (n, idx) => `<button class="pm" data-mx="${n}" title="show the values of ${n}${sup(k)}">${n}${sup(k)}${idx}</button>`;
  let h = `<h2 class="tag">┌─ rule.in_effect</h2>
    <p class="ph-head"><b>phase ${ph.k < 0 ? 1 : ph.k + 1} of ${K}</b> · frames ${p.start}–${p.end - 1} · ${sec(p.start)}–${sec(p.end - 1)} s · <b>rule id ${k}</b>${users.length > 1 ? ` <span class="cm">(rule id ${k} is used by phases ${users.join(", ")})</span>` : K > 1 ? ` <span class="cm">(used only by this phase)</span>` : ""}</p>
    <p class="cm">1 video frame = ${S} CA substeps · frame t (t ≥ 1) is produced from frame t−1 by ${S} substeps of its phase's rule · ${ph.first ? "showing frame 0 = the given first frame; the next update uses this rule" : P.sub ? `showing ${P.sub}/${S} of the way from frame ${P.t} to ${P.t + 1}` : `showing frame ${P.t}, produced by this rule`}</p>`;
  if (ph.boundary) h += `<p class="switch">rule switch at frame ${P.t}: frame ${P.t - 1} → ${P.t} is the first update with rule id ${k} (frames before used rule id ${stats.prevRule}). Hard switch: no interpolated weights; the state is carried over unchanged.</p>`;
  h += `<p class="legend"><span class="sx">blue = fixed structure</span> (the same for every phase) · <span class="pm-l">red badge = phase parameters</span> (learned numbers of rule id ${k}; click one to see its values)</p>
  <div class="eq">
    <p class="sx">for every cell (y, x), all cells at once, ${S} substeps per video frame; s<sub>c</sub> = 0 outside the ${m.W}×${m.H} grid (zero padding)</p>
    <p><span class="sx">p<sub>f</sub> = ReLU( Σ<sub>c=0..${C - 1}</sub> Σ<sub>dy,dx∈{−1,0,1}</sub></span> ${PM("Wp", "[f,c,dy,dx]")} <span class="sx">· s<sub>c</sub>(y+dy, x+dx) +</span> ${PM("bp", "[f]")} <span class="sx">),&nbsp; f = 0..${Fh - 1}</span></p>
    <p><span class="sx">s<sub>c</sub> ← s<sub>c</sub> + Σ<sub>f=0..${Fh - 1}</sub></span> ${PM("Wu", "[c,f]")} <span class="sx">· p<sub>f</sub> +</span> ${PM("bu", "[c]")}<span class="sx">,&nbsp; c = 0..${C - 1}</span> <span class="cm">(residual update of all ${C} channels; channel 0 = visible)</span></p>
    <p><span class="sx">shown pixel = 1 if s<sub>0</sub> &gt; ${m.threshold} else 0</span> <span class="cm">(s<sub>0</sub> is a regressed state value, not a probability)</span></p>
    ${delta ? `<p class="cm">delta mode: Wp${sup(k)} = Wp + reshape(Ap[${k}]·Bp[${k}]), bp${sup(k)} = bp + dbp[${k}], Wu${sup(k)} = Wu + Au[${k}]·Bu[${k}], bu${sup(k)} = bu + dbu[${k}]; combined in float64, rounded to float32 once</p>` : ""}
    <p class="cm">arithmetic as in player/nca_core.mjs substep() and ca/nca.py step_np: sums in float64 here (float32 in numpy), each stored value rounded to float32</p>
  </div>
  <div class="hscroll"><table class="rtab ptbl"><thead><tr><th>parameter</th><th>shape</th><th class="num">values</th><th>stored as</th>${stats ? '<th class="num">changed</th>' : ""}</tr></thead><tbody>`;
  for (const [n, shp] of [["Wp", `${Fh}×${C}×3×3`], ["bp", `${Fh}`], ["Wu", `${C}×${Fh}`], ["bu", `${C}`]]) {
    const cnt = ruleParams(e, k)[n].length;
    h += `<tr class="${ph.boundary && stats?.per[n] ? "chg" : ""}"><td>${PM(n, "")}</td><td>${shp}</td><td class="num">${fmtInt(cnt)}</td><td class="wrap">${esc(blobTxt(n))}</td>${stats ? `<td class="num">${fmtInt(stats.per[n])}${ph.boundary && stats.per[n] ? " ◆" : ""}</td>` : ""}</tr>`;
  }
  h += `</tbody></table></div><p class="cm">${fmtInt(per)} parameters in this phase's rule = ${Fh}·${C}·9 + ${Fh} + ${C}·${Fh} + ${C}.</p>`;
  h += `<div class="chg-box${ph.boundary ? " on" : ""}"><b>vs previous phase:</b> ` + (!stats ? `this is the first phase; frame 1 is produced from the given first frame${m.hidden0 ? " and the learned low-res initial hidden state" : " with hidden = 0"}.`
    : stats.prevRule === k ? `same rule id ${k} as the previous phase: no parameter changes.`
    : `phase ${ph.k} used rule id ${stats.prevRule}. ${fmtInt(stats.changed)} of ${fmtInt(per)} values changed (|Δ| &gt; 1e-6) · L2 of the change ${stats.l2.toPrecision(4)} · max |Δ| ${stats.max.toPrecision(4)}`
      + (m.mode === "indep" ? ` · <b>all parameters replaced (independent rule)</b>` : delta ? " · shared base; only the low-rank and bias deltas differ" : "")) + `</div>`;
  if (mx) h += matrixHTML(e, k, sch, ph);
  h += `<details class="allph"><summary>all ${K} phases <span class="cm">(parameter count, change vs previous phase, stored bytes)</span></summary>${phasesTable(e, P)}</details>`;
  el.innerHTML = h;
  el.querySelectorAll("[data-mx]").forEach((b) => (b.onclick = () => { mx = mx?.n === b.dataset.mx ? null : { n: b.dataset.mx, page: 0, mode: "val", pick: null }; renderRulePanel(P, true); }));
  if (mx) wireMatrix(e, k, sch, ph, P);
  el.querySelectorAll("tr[data-seek]").forEach((r) => (r.onclick = () => P.seek(+r.dataset.seek)));
}
function phasesTable(e, P) {
  const m = e.m, { C, Fh } = m, w = m.weight_dtype === "f16" ? 2 : 4, per = Fh * C * 9 + Fh + C * Fh + C, sch = scheduleOf(e), st = phaseStats(e), seen = new Map(), cur = phaseAt(e, P.t, P.sub).k;
  let h = `<div class="scrollbox"><table class="rtab"><thead><tr><th>phase</th><th>frames</th><th>rule id</th><th class="num">params</th><th class="num">changed (|Δ|&gt;1e-6)</th><th class="num">L2 Δ</th><th class="num">max |Δ|</th><th class="num">stored bytes</th><th>reason</th></tr></thead><tbody>`;
  sch.forEach((s, i) => {
    const x = st[i], bytes = m.mode === "delta" ? "delta + shared base" : m.mode === "fixed" || !m.rules ? (i ? "shared (one rule)" : fmtInt(per * w)) : seen.has(s.rule) ? `same as phase ${seen.get(s.rule) + 1}` : fmtInt(per * w);
    if (!seen.has(s.rule)) seen.set(s.rule, i);
    h += `<tr data-seek="${s.start}" class="${i === cur ? "cur" : ""}"><td>${i + 1}</td><td>${s.start}–${s.end - 1}</td><td>${s.rule}</td><td class="num">${fmtInt(per)}</td><td class="num">${x ? fmtInt(x.changed) : "–"}</td><td class="num">${x ? x.l2.toPrecision(3) : "–"}</td><td class="num">${x ? x.max.toPrecision(3) : "–"}</td><td class="num">${bytes}</td><td>${esc(s.reason ?? "–")}</td></tr>`;
  });
  return h + `</tbody></table></div><p class="cm">click a row to seek to its first frame. Stored bytes are raw float bytes before deflate.</p>`;
}

// ---------- matrix viewer: heatmap + paged numbers ----------
const MX = {
  Wp: (C, Fh) => ({ dims: [Fh, C, 3, 3], pageDim: "f", pages: Fh, img: [Fh * 4 - 1, C * 4 - 1] }),
  Wu: (C, Fh) => ({ dims: [C, Fh], pageDim: "c", pages: C, img: [Fh, C] }),
  bp: (C, Fh) => ({ dims: [Fh], pages: 1, img: [Fh, 1] }),
  bu: (C) => ({ dims: [C], pages: 1, img: [C, 1] }),
};
function mxValues(e, rid, n, sch, ph) {
  const v = ruleParams(e, rid)[n];
  if (mx.mode !== "diff" || ph.k < 1) return v;
  const pv = ruleParams(e, sch[ph.k - 1].rule)[n]; return Float64Array.from(v, (x, j) => x - pv[j]);
}
function matrixHTML(e, k, sch, ph) {
  const { C, Fh } = e.m, d = MX[mx.n](C, Fh), v = mxValues(e, k, mx.n, sch, ph);
  let amax = 0; for (const x of v) amax = Math.max(amax, Math.abs(x));
  const [iw, ih] = d.img, zoom = Math.max(2, Math.floor(480 / iw));
  let tbl = "";
  if (mx.n === "Wp") {
    const f = mx.page; tbl = `<table class="rtab mtab"><thead><tr><th>c \\ (dy,dx)</th>${[-1, 0, 1].flatMap((dy) => [-1, 0, 1].map((dx) => `<th class="num">${dy},${dx}</th>`)).join("")}</tr></thead><tbody>`;
    for (let c = 0; c < C; c++) tbl += `<tr><td>${c}${c === 0 ? " vis" : ""}</td>${Array.from({ length: 9 }, (_, j) => `<td class="num">${fmtN(v[(f * C + c) * 9 + j])}</td>`).join("")}</tr>`;
    tbl += "</tbody></table>";
  } else if (mx.n === "Wu") {
    const c = mx.page; tbl = `<table class="rtab mtab"><tbody>`;
    for (let r = 0; r < Fh / 8; r++) tbl += `<tr><td class="cm">f ${r * 8}–${r * 8 + 7}</td>${Array.from({ length: 8 }, (_, j) => `<td class="num">${fmtN(v[c * Fh + r * 8 + j])}</td>`).join("")}</tr>`;
    tbl += "</tbody></table>";
  } else {
    tbl = `<table class="rtab mtab"><tbody>`;
    for (let r = 0; r < Math.ceil(v.length / 8); r++) tbl += `<tr><td class="cm">${r * 8}–${Math.min(v.length, r * 8 + 8) - 1}</td>${Array.from({ length: Math.min(8, v.length - r * 8) }, (_, j) => `<td class="num">${fmtN(v[r * 8 + j])}</td>`).join("")}</tr>`;
    tbl += "</tbody></table>";
  }
  const pg = d.pages > 1 ? `<span class="pager"><button class="btn" data-pg="-1" aria-label="previous page">[ ◂ ]</button> ${d.pageDim} = ${mx.page} of 0..${d.pages - 1} <button class="btn" data-pg="1" aria-label="next page">[ ▸ ]</button></span>` : "";
  const what = mx.n === "Wp" ? `x = feature f (blocks of 3 columns: dx −1,0,1), y = channel c (blocks of 3 rows: dy −1,0,1)` : mx.n === "Wu" ? "x = feature f, y = channel c" : "x = index";
  return `<div class="mx"><h3 class="minor">${mx.n}${sup(k)} <span class="cm">// ${d.dims.join("×")} = ${fmtInt(v.length)} values${mx.mode === "diff" ? ", Δ vs previous phase" : ""}</span></h3>
    <label>show <select id="mx-mode"><option value="val">values</option><option value="diff"${mx.mode === "diff" ? " selected" : ""} ${ph.k < 1 ? "disabled" : ""}>change vs previous phase</option></select></label>
    <canvas id="mx-cv" width="${iw}" height="${ih}" style="width:${iw * zoom}px;height:${ih * zoom}px" aria-label="heatmap of ${mx.n}"></canvas>
    <p class="cm">heatmap: ${what}; amber = positive, light = negative, black = 0; brightness = sqrt(|value| / max |value|), max |value| = ${fmtN(amax)}. Click the heatmap to page to that row and read one value.</p>
    <p id="mx-pick" class="counter">${mx.pick ?? ""}</p>${pg}<div class="scrollbox">${tbl}</div></div>`;
}
function wireMatrix(e, k, sch, ph, P) {
  const { C, Fh } = e.m, d = MX[mx.n](C, Fh), v = mxValues(e, k, mx.n, sch, ph), cv = $("mx-cv"), c2 = cv.getContext("2d"), [iw, ih] = d.img;
  let amax = 1e-30; for (const x of v) amax = Math.max(amax, Math.abs(x));
  const img = c2.createImageData(iw, ih), put = (x, y, val) => { const a = Math.sqrt(Math.min(1, Math.abs(val) / amax)), o = 4 * (y * iw + x), c = val >= 0 ? [255, 180, 58] : [236, 232, 223]; img.data[o] = 12 + a * (c[0] - 12); img.data[o + 1] = 12 + a * (c[1] - 12); img.data[o + 2] = 11 + a * (c[2] - 11); img.data[o + 3] = 255; };
  for (let i = 0; i < iw * ih; i++) { img.data[4 * i] = 52; img.data[4 * i + 1] = 51; img.data[4 * i + 2] = 47; img.data[4 * i + 3] = 255; } // separators in the rule colour
  if (mx.n === "Wp") { for (let f = 0; f < Fh; f++) for (let c = 0; c < C; c++) for (let j = 0; j < 9; j++) put(f * 4 + (j % 3), c * 4 + ((j / 3) | 0), v[(f * C + c) * 9 + j]); }
  else if (mx.n === "Wu") { for (let c = 0; c < C; c++) for (let f = 0; f < Fh; f++) put(f, c, v[c * Fh + f]); }
  else for (let i = 0; i < v.length; i++) put(i, 0, v[i]);
  c2.putImageData(img, 0, 0);
  cv.onclick = (ev) => {
    const r = cv.getBoundingClientRect(), x = Math.floor(((ev.clientX - r.left) / r.width) * iw), y = Math.floor(((ev.clientY - r.top) / r.height) * ih);
    const name = `${mx.n}(${k})`;
    if (mx.n === "Wp") { const f = (x / 4) | 0, c = (y / 4) | 0, dx = x % 4, dy = y % 4; if (dx > 2 || dy > 2) return; mx.page = f; mx.pick = `${name}[f=${f}, c=${c}, dy=${dy - 1}, dx=${dx - 1}] = ${fmtN(v[(f * C + c) * 9 + dy * 3 + dx])} (stored index ${dy},${dx})`; }
    else if (mx.n === "Wu") { mx.page = y; mx.pick = `${name}[c=${y}, f=${x}] = ${fmtN(v[y * Fh + x])}`; }
    else mx.pick = `${name}[${x}] = ${fmtN(v[x])}`;
    if (mx.mode === "diff") mx.pick += " (change vs previous phase)";
    renderRulePanel(P, true);
  };
  $("mx-mode").onchange = (ev) => { mx.mode = ev.target.value; mx.pick = null; renderRulePanel(P, true); };
  document.querySelectorAll("[data-pg]").forEach((b) => (b.onclick = () => { mx.page = (mx.page + +b.dataset.pg + d.pages) % d.pages; renderRulePanel(P, true); }));
}

// ---------- demo cell inspector (NCA): traceCell = the substep's own arithmetic for one cell ----------
function renderInspector() {
  const P = DEMO, e = P.e, i = P.sel;
  if (!e || i == null) { $("insp-where").textContent = "// click a cell of the CA video"; $("insp-body").innerHTML = '<p class="cm">no cell selected</p>'; return; }
  const S = e.S, g = P.t * S + P.sub, f = Math.floor(g / S) + 1, last = g >= (e.T - 1) * S, x = i % e.W, y = Math.floor(i / e.W), m = e.m, C = m.C;
  const sch = scheduleOf(e), ph = sch.findIndex((p) => f >= p.start && f < p.end), rid = m.sched ? m.sched[Math.min(f, e.T - 1)] : 0;
  $("insp-where").textContent = `// cell (x=${x}, y=${y}) · frame ${P.t}${P.sub ? ` + ${P.sub} substeps` : ""}`;
  const s = e.stateAt(g);
  if (!s) { $("insp-body").innerHTML = `<p class="cm">state not available yet (the stored state for frame ${P.t - (P.t % e.every)} is still being computed)</p>`; return; }
  const tr = traceCell(m, ruleAt(m, f), s, i), nx = last ? null : e.stateAt(g + 1), agree = nx && tr.next.every((v, c) => Object.is(v, nx[c * e.N + i]));
  let h = `<dl class="facts"><dt>cell</dt><dd>x=${x}, y=${y} (row-major index ${i}) of the ${e.W}×${e.H} grid</dd>
    <dt>state shown</dt><dd>${P.sub ? `after substep ${g} = frame ${P.t}·${S} + ${P.sub}` : `the state after frame ${P.t} (after ${g} = ${P.t}·${S} substeps)`}; this state is the input of the next update · regenerated with substep() from the full state kept at frame ${P.t - (P.t % e.every)}</dd>
    <dt>next update</dt><dd>${last ? "none: this is the last frame of the export" : `substep ${g} → ${g + 1}, part of producing frame ${f}; rule id ${rid} in effect (phase ${ph + 1} of ${sch.length})`}</dd></dl>`;
  h += `<div class="two-col"><div><h3 class="minor">all ${C} channels <span class="cm">// 3×3 neighbourhood NW N NE / W · E / SW S SE; · = outside the grid (zero padding)</span></h3><div class="scrollbox"><table class="rtab"><thead><tr><th>c</th><th class="num">s now</th><th>3×3 neighbourhood</th>${last ? "" : '<th class="num">Δs = Wu·p + bu</th><th class="num">next s</th>'}</tr></thead><tbody>`;
  for (let c = 0; c < C; c++) h += `<tr><td>${c === 0 ? "0 vis" : c}</td><td class="num">${fmtN(tr.cur[c])}</td><td>${tr.nbr[c].map(fmtN).join(" ")}</td>${last ? "" : `<td class="num">${fmtN(tr.upd[c] + tr.bias[c])}</td><td class="num">${fmtN(tr.next[c])}</td>`}</tr>`;
  h += `</tbody></table></div>${last ? "" : `<p>next s<sub>0</sub> = ${fmtN(tr.next[0])} → pixel ${tr.next[0] > m.threshold ? "on" : "off"} (on if s<sub>0</sub> &gt; ${m.threshold}) · <b class="${agree ? "ok" : "bad"}">${agree ? `equals the real substep's state at substep ${g + 1} (all ${C} channels, float32 bit-exact)` : "DIFFERS from the real substep (bug)"}</b></p>
    <p class="cm">next s<sub>c</sub> = float32(s<sub>c</sub> + float32(Σ<sub>f</sub> Wu[c,f]·p<sub>f</sub>) + bu[c]), the same expression as substep().</p>`}</div>`;
  h += `<div><h3 class="minor">p (perception) <span class="cm">// pre = bp[f] + Σ Wp·neighbourhood (float64), post = float32(ReLU(pre))</span></h3><div class="scrollbox"><table class="rtab"><thead><tr><th>f</th><th class="num">pre-ReLU</th><th class="num">post-ReLU p<sub>f</sub></th></tr></thead><tbody>`;
  for (let k = 0; k < m.Fh; k++) h += `<tr><td>${k}</td><td class="num">${fmtN(tr.pre[k])}</td><td class="num${tr.post[k] ? "" : " skip"}">${fmtN(tr.post[k])}</td></tr>`;
  h += `</tbody></table></div><p class="cm">${tr.post.filter((v) => v > 0).length}/${m.Fh} features active (p<sub>f</sub> &gt; 0)</p></div></div>`;
  $("insp-body").innerHTML = h;
}

// ---------- rule lab (dlrule-v1) ----------
const LAB_DEFAULT = "c12~samples~life_then_grow.rule.json";
let labLoading = false;
// readable names for rule files ("4500_56_direct_k0_K14_L32_b0" -> "frames 4500–4555 · direct · 0 hidden bits · L32 b0")
function labLabel(r) {
  const base = r.name.split("~").pop().replace(/\.rule\.json$/, ""), m = base.match(/^(\d+)_(\d+)_([a-z]+)_k(\d+)_K\d+_(L\d+)(?:_(b[\d.]+))?$/);
  if (r.name.startsWith("c12~samples~")) return `sample: ${base.replace(/_/g, " ")}`;
  return m ? `frames ${m[1]}–${+m[1] + +m[2] - 1} · ${m[3]} · ${m[4]} hidden bit${m[4] === "1" ? "" : "s"} · setting ${m[5]}${m[6] ? " " + m[6] : ""}` : base.replace(/_/g, " ");
}
function fillLabSelect() {
  const rs = runs.filter((r) => r.format === "dlrule-v1"), cur = $("lab-sel").value || LAB_DEFAULT;
  const samples = rs.filter((r) => r.name.startsWith("c12~samples~")), ba = rs.filter((r) => !r.name.startsWith("c12~samples~"));
  $("lab-sel").innerHTML = `<optgroup label="synthetic samples (exact)">${samples.map((r) => `<option value="${esc(r.name)}">${esc(labLabel(r))}</option>`).join("")}</optgroup>`
    + `<optgroup label="Bad Apple · explicit rules (experimental)">${ba.map((r) => `<option value="${esc(r.name)}">${esc(labLabel(r))} · ${r.rules} rules / ${r.programs} programs</option>`).join("")}</optgroup>`;
  $("lab-sel").value = cur;
}
$("lab-sel").onchange = () => openLab($("lab-sel").value);
async function openLab(name) {
  labLoading = true;
  try {
    const u8 = await bytesOf(`/api/runs/${encodeURIComponent(name)}/export`), o = isRuleJson(u8);
    if (!o) throw new Error("not a dlrule-v1 file");
    const e = await ruleEngine(o, u8.length), r = runs.find((x) => x.name === name), lbl = r ? labLabel(r) : name;
    LAB.load(e, { name, label: lbl, fps: o.meta?.fps || 30, fpsSrc: o.meta?.fps ? "export meta" : "assumed 30 (no fps recorded)" });
    $("lab-sel").value = name; LAB.q("ca-cap").innerHTML = `explicit rules <span class="cm">(${esc(lbl)}; computed in this browser from the file)</span>`;
    renderRules(); renderLabInspector();
  } catch (err) { toast("rule file could not be loaded: " + err.message, true); } finally { labLoading = false; }
}
const ruleLine = (r, i, k) => `R${i}: if ${r.if.map(([f, op, c]) => `${f} ${op} ${c}`).join(" and ") || "always"} -> v=${r.then.v}` + (k ? `, h=${r.then.h.toString(2).padStart(k, "0")}` : "");
function progDiff(a, b) { // b vs a: rules keyed by their condition list
  const key = (r) => JSON.stringify(r.if), A = new Map(a.rules.map((r) => [key(r), r])), Bm = new Map(b.rules.map((r) => [key(r), r]));
  const added = b.rules.filter((r) => !A.has(key(r))), removed = a.rules.filter((r) => !Bm.has(key(r))), changed = b.rules.filter((r) => A.has(key(r)) && JSON.stringify(A.get(key(r)).then) !== JSON.stringify(r.then));
  const order = !added.length && !removed.length && a.rules.map(key).join() !== b.rules.map(key).join();
  return { added, removed, changed, order, same: !added.length && !removed.length && !changed.length && !order };
}
const fmtPrograms = (ps) => "[\n" + ps.map((p) => ' {"rules": [\n' + p.rules.map((r) => "  " + JSON.stringify({ if: r.if, then: r.then })).join(",\n") + "\n ]}").join(",\n") + "\n]";
function renderRules() {
  const e = LAB.e; if (!e) return;
  const m = e.m, sch = scheduleOf(e), k = m.k, cnt = e.counts;
  $("rules-kind").textContent = `// explicit if-then rules (first firing rule wins) · ${LAB.label}`;
  const used = (p) => sch.map((s, i) => (s.rule === p ? i : null)).filter((x) => x != null);
  let h = `<dl class="facts"><dt>state</dt><dd>v + ${k} hidden bit${k === 1 ? "" : "s"} per cell; ${m.S} substep${m.S > 1 ? "s" : ""} per frame; zero boundary; synchronous</dd>
    <dt>semantics</dt><dd>${esc(m.semantics || "first firing rule wins; default keep")}</dd><dt>features</dt><dd>${esc((m.features || []).join(", "))}</dd>
    <dt>size</dt><dd>${m.programs.length} programs, ${m.programs.reduce((a, p) => a + p.rules.length, 0)} rules, ${m.programs.reduce((a, p) => a + p.rules.reduce((b, r) => b + r.if.length, 0), 0)} conditions · file ${fmtInt(e.bytes)} B (JSON with hex first frame)</dd></dl>`;
  h += `<h3 class="minor">fire counts <span class="cm">// cells × substeps in which each rule fired, from running the engine over frames</span></h3>
    <label>from frame <input type="number" id="fc-from" min="1" max="${m.T - 1}" value="${cnt?.from ?? 1}"></label><label>to frame <input type="number" id="fc-to" min="1" max="${m.T - 1}" value="${cnt?.to ?? m.T - 1}"></label>
    <button class="btn" id="b-count">[ count firings ]</button> <span class="cm">${cnt ? `counted frames ${cnt.from}–${cnt.to} (${fmtInt(cnt.cells)} cell-substeps, version ${cnt.ver})${cnt.note}` : "not counted yet"}</span>`;
  h += `<div class="two-col"><div>`;
  m.programs.forEach((p, pi) => {
    const lines = p.rules.map((r, i) => ruleLine(r, i, k) + (cnt ? `   ×${fmtInt(cnt.c[pi][i + 1])}` : "")).concat("default: keep state" + (cnt ? `   ×${fmtInt(cnt.c[pi][0])}` : ""));
    h += `<div class="prog"><h3 class="minor">program ${pi} <span class="cm">// phases ${used(pi).map((x) => x + 1).join(", ") || "none"} · ${p.rules.length} rules in priority order · ${fmtInt(JSON.stringify(p).length)} B JSON</span></h3><pre>${esc(lines.join("\n"))}</pre></div>`;
  });
  h += `</div><div><h3 class="minor">what changes at each phase boundary <span class="cm">// vs the previous phase's program; rules matched by their conditions</span></h3><div class="scrollbox"><table class="rtab"><thead><tr><th>phase</th><th>frames</th><th>program</th><th>vs previous</th></tr></thead><tbody>`;
  sch.forEach((s, i) => {
    let d = "first phase";
    if (i) {
      const a = sch[i - 1].rule, b = s.rule, df = progDiff(m.programs[a], m.programs[b]);
      d = a === b ? "same program" : df.same ? `program ${a} → ${b}: identical rules` : `program ${a} → ${b}: <span class="diff-add">+${df.added.length} added</span> <span class="diff-del">−${df.removed.length} removed</span> <span class="diff-chg">~${df.changed.length} changed</span>${df.order ? " · order changed" : ""}`
        + [...df.added.map((r) => `<br><span class="diff-add">+ ${esc(ruleLine(r, m.programs[b].rules.indexOf(r), k))}</span>`), ...df.removed.map((r) => `<br><span class="diff-del">− ${esc(ruleLine(r, m.programs[a].rules.indexOf(r), k))}</span>`), ...df.changed.map((r) => `<br><span class="diff-chg">~ ${esc(ruleLine(r, m.programs[b].rules.indexOf(r), k))}</span>`)].join("");
    }
    h += `<tr data-seek="${s.start}"><td>${i + 1}</td><td>${s.start}–${s.end - 1}</td><td>${s.rule}</td><td style="white-space:normal">${d}</td></tr>`;
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
  $("rules-body").querySelectorAll("tr[data-seek]").forEach((r) => (r.onclick = () => LAB.seek(+r.dataset.seek)));
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
let verCounter = 1;
async function applyEdit(mode) {
  const e = LAB.e; if (!e || e.kind !== "rule") return;
  let ps;
  try { ps = mode === "reset" ? e.base.programs : validatePrograms(JSON.parse($("rule-edit").value), e.m); } catch (err) { $("edit-err").textContent = err.message; return; }
  LAB.setPlaying(false); const prevVer = e.ver; e.ver = ++verCounter; e.counts = null;
  const t0 = performance.now(), m2 = { ...e.m, programs: ps };
  if (mode === "continue") {
    const g0 = LAB.t * e.S + LAB.sub, old = e.progAt;
    e.m = m2; simulateRule(e, ps, g0, e.seq); e.hybridFrom = g0;
    e.progAt = (g) => (g < g0 ? old(g) : ps[programAt(m2, Math.floor(g / e.S) + 1)]);
    e.verNote = `EXPERIMENTAL hybrid: substeps 0..${g0} from version ${prevVer}, substeps after ${g0} from the edited programs`;
  } else {
    e.m = m2; simulateRule(e, ps, 0, null); e.hybridFrom = null; e.progAt = (g) => ps[programAt(m2, Math.floor(g / e.S) + 1)];
    e.verNote = mode === "reset" ? "export programs, replayed from frame 0" : "edited programs, replayed from frame 0";
  }
  e.ms = performance.now() - t0; e.runMatch = e.hybridFrom == null && ruleRun(e.m).frames.every((f, t) => f.every((x, i) => x === e.vis[t * e.N + i]));
  e.sha = await hex(e.vis); // every cached frame of the old version is replaced
  renderRules(); LAB.render(true); toast(`rules version ${e.ver}: ${mode === "continue" ? "experimental hybrid" : "recomputed from frame 0"}`);
}
function renderLabInspector() {
  const P = LAB, e = P.e, i = P.sel;
  if (!e || i == null) { $("linsp-where").textContent = "// click a cell of the rule CA"; $("linsp-body").innerHTML = '<p class="cm">no cell selected</p>'; return; }
  const S = e.S, g = P.t * S + P.sub, f = Math.floor(g / S) + 1, last = g >= (e.T - 1) * S, x = i % e.W, y = Math.floor(i / e.W), m = e.m;
  const sch = scheduleOf(e), ph = sch.findIndex((p) => f >= p.start && f < p.end), rid = programAt(m, f);
  $("linsp-where").textContent = `// cell (x=${x}, y=${y}) · state after substep ${g} = frame ${P.t} + ${P.sub}`;
  let h = `<dl class="facts"><dt>cell</dt><dd>x=${x}, y=${y} (row-major index ${i})</dd><dt>state shown</dt><dd>after ${g} substeps (frame ${P.t}, substep ${P.sub} of ${S}) · version ${e.ver}</dd>
    <dt>next substep</dt><dd>${last ? "none: this is the last frame of the file" : `${g} → ${g + 1}, part of producing frame ${f}; program ${rid} in effect (phase ${ph + 1}${e.hybridFrom != null && g < e.hybridFrom ? ", pre-edit programs of the hybrid" : ""})`}</dd></dl>`;
  const st = e.stateAt(g), prog = e.progAt(Math.min(g, (e.T - 1) * S - 1)), ex = explainCell(m, prog, st.v, st.h, i);
  h += `<h3 class="minor">features <span class="cm">// rule_core.features() on the current grid (zero boundary)</span></h3><pre class="codeblock">${esc(Object.entries(ex.features).map(([k, v]) => `${k}=${v}`).join("  "))}</pre>`;
  if (!last) {
    const nx = e.stateAt(g + 1), agree = nx.v[i] === ex.next.v && ex.next.h.every((b, j) => b === nx.h[j * e.N + i]);
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
  $("linsp-body").innerHTML = h;
}

// ---------- research: shelves, model details ----------
let runs = [], cfg = null;
async function loadRuns() {
  runs = await api("/api/runs");
  for (const kind of ["example", "run", "imported", "c12"]) {
    const ul = $("list-" + kind), items = runs.filter((r) => r.kind === kind);
    ul.innerHTML = items.length ? items.map((r) => `<li><button data-run="${esc(r.name)}">
      <span class="t">${esc(r.label)}</span>
      <span class="m"><span class="st ${esc(r.status)}">${esc(r.status)}${r.stage ? " · " + esc(r.stage) : ""}</span>
      ${r.format ? esc(r.format) + " · " : ""}${r.rules != null ? `${r.rules} rules / ${r.programs} programs · ` : ""}${r.phases ? `${r.phases} phases (${esc(r.mode)}) · ` : ""}${r.n_params ? fmtInt(r.n_params) + " params" : ""}${r.export_bytes ? " · " + fmtInt(r.export_bytes) + " B" : ""}
      ${r.frames ? ` · ${r.frames}f ${r.width ?? "?"}×${r.height ?? "?"}` : ""}${r.preset ? " · " + esc(r.preset) : ""}${r.elapsed_s ? " · " + fmtS(r.elapsed_s) : ""}
      ${r.has_original ? "" : " · CA only"}</span></button></li>`).join("") : `<li class="cm" style="padding:8px 6px">none yet</li>`;
  }
  const bs = DEMO.q("b-sel"), keep = bs.value;
  bs.innerHTML = '<option value="">none</option>' + runs.filter((r) => r.has_export).map((r) => `<option value="${esc(r.name)}">${esc(r.kind)}: ${esc(r.label)}</option>`).join("");
  bs.value = keep; fillLabSelect();
}
document.addEventListener("click", (ev) => { const b = ev.target.closest("[data-run]"); if (b) openRun(b.dataset.run); });
$("import-file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0]; ev.target.value = ""; if (!f) return;
  try {
    const r = await api("/api/import-export", { method: "POST", body: f, headers: { "X-Filename": f.name.replace(/[^\x20-\x7e]/g, "_") } });
    toast(`imported ${r.name} (${fmtInt(r.export.n_params)} params, ${r.T} frames, CA-only playback)`); await loadRuns(); openRun(r.name);
  } catch (err) { toast("import rejected: " + err.message, true); }
});
async function openRun(name) {
  const det = await api(`/api/runs/${encodeURIComponent(name)}`).catch((err) => (toast(err.message, true), null)); if (!det) return;
  document.querySelectorAll("[data-run]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.run === name));
  $("model-name").textContent = "// " + det.summary.label; renderModel(det); renderFacts(det); if (det.summary.own) watchJob(name);
  if (!det.summary.has_export) { toast(`no export yet (status: ${det.summary.status})`); return; }
  if (det.summary.format === "dlrule-v1") { location.hash = "#lab"; return openLab(name); }
  location.hash = "#demo"; return openResearchNca(name, det);
}
function renderFacts(det) {
  const man = det.manifest || {}, ex = man.export || {}, sz = ex.sizes || det.status?.sizes || {}, pr = det.progress, imp = man.imported || {}, hd = man.c12?.header || {};
  const train = pr ? `${fmtS(pr.train_s)} (${fmtInt(pr.step)}/${fmtInt(pr.steps)} steps)` : imp.t ? `${fmtS(imp.t)} to best checkpoint at step ${fmtInt(imp.it)}` : "–";
  const rows = [["format", hd.format ?? "ncab / nca-v1"], ["parameters", fmtInt(ex.n_params ?? sz.n_params)], ["export bytes", sz.file ? `${fmtInt(sz.file)} B file` + (sz.weights ? ` (raw: weights ${fmtInt(sz.weights)} + first frame ${fmtInt(sz.first_frame)} + header ${fmtInt(sz.header)} = ${fmtInt(sz.raw_total)}, deflated)` : "") : "–"],
    ["frames", det.summary.frames ? `${det.summary.frames} × ${det.summary.width}×${det.summary.height}` : "–"], ["training time", train], ["status", det.summary.status + (det.status?.error ? ` — ${det.status.error}` : "")],
    ["replay check", det.status?.replay ? `numpy vs Node: ${det.status.replay.binary_mismatch_px} px mismatch` : "–"], ["source", man.c12?.path ?? "–"],
    ["download", det.summary.has_export ? `<a href="/api/runs/${encodeURIComponent(det.summary.name)}/export">[ export file ]</a>` : "–"]];
  $("facts").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${k === "download" ? v : esc(v)}</dd>`).join("");
}
function renderModel(det) {
  const man = det.manifest || {}, s = man.settings || {}, imp = man.imported || {}, c12 = man.c12?.header || {};
  $("schematic").style.display = c12.format === "dlrule-v1" ? "none" : "";
  if (c12.format === "dlrule-v1") { $("arch").innerHTML = `<dt>model</dt><dd>explicit rules (dlrule-v1), not a Neural CA: ${c12.k} hidden bits, S=${c12.S}; open in the rule lab</dd>`; return renderCurve([]); }
  const C = s.C ?? imp.C ?? man.C ?? c12.C, Fh = s.Fh ?? imp.Fh ?? man.Fh ?? c12.Fh, S = imp.S ?? man.S ?? c12.S ?? 4;
  const np = man.export?.n_params ?? (C && Fh ? Fh * C * 9 + Fh + C * Fh + C : null);
  $("arch").innerHTML = [["C channels", C ? `${C}: 1 visible + ${C - 1} hidden (continuous state)` : "–"], ["Fh features", Fh ?? "–"], ["S updates", `${S} per video frame`],
    ["parameters", c12.format === "ncab-v2" ? `${fmtInt(np)} stored values: K=${c12.K} rule(s), mode ${c12.mode}, each rule Fh·C·9 + Fh + C·Fh + C = ${fmtInt(Fh * C * 9 + Fh + C * Fh + C)} (learned weights, not rules)` : np ? `${fmtInt(np)} = Fh·C·9 + Fh + C·Fh + C (learned weights, not rules)` : "–"], ["preset", man.preset ? `${man.preset}: ${s.note || ""}` : "–"]]
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
  catch (err) { $("probe").textContent = "probe failed: " + err.message; }
}
$("video").onchange = probeSel;
$("upload").addEventListener("change", async (ev) => {
  const f = ev.target.files[0]; ev.target.value = ""; if (!f) return;
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
  } catch (err) { $("pv-info").textContent = "preview rejected: " + err.message; } finally { $("b-preview").disabled = false; }
};
$("preset").onchange = () => { const p = cfg.presets[$("preset").value]; $("preset-note").textContent = `${p.note}. Defaults: C=${p.C}, Fh=${p.Fh}, steps=${fmtInt(p.steps)}, lr=${p.lr}. Your preprocessing values are used as entered; nothing is replaced silently.`; };
$("b-train").onclick = async () => {
  $("b-train").disabled = true;
  try {
    const r = await api("/api/train", { method: "POST", json: { ...params(), preset: $("preset").value, steps: $("t-steps").value || null, device: $("device").value } });
    toast(`job started: ${r.name}`); await loadRuns(); watchJob(r.name);
  } catch (err) { toast("training not started: " + err.message, true); } finally { $("b-train").disabled = false; }
};
let watch = { name: null, timer: null, last: null };
function watchJob(name) { clearTimeout(watch.timer); watch = { name, timer: null, last: null }; $("job").hidden = false; pollJob(); }
async function pollJob() {
  const name = watch.name; let d;
  try { d = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (err) { $("job-err").textContent = err.message; return; }
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
  $("b-dl").hidden = !d.summary.has_export; $("b-dl").href = `/api/runs/${encodeURIComponent(name)}/export`;
  if (watch.last && watch.last !== s) { toast(`job ${name}: ${s}`, ["failed", "died"].includes(s)); loadRuns(); }
  watch.last = s;
  if (live) watch.timer = setTimeout(pollJob, 1500);
}
$("b-cancel").onclick = async () => { try { await api(`/api/runs/${encodeURIComponent(watch.name)}/cancel`, { method: "POST" }); } catch (err) { toast(err.message, true); } pollJob(); };
$("b-resume").onclick = async () => { try { await api(`/api/runs/${encodeURIComponent(watch.name)}/resume`, { method: "POST" }); toast(`job resumed: ${watch.name}`); watch.last = "resumed"; } catch (err) { toast(err.message, true); } pollJob(); };
$("b-open").onclick = () => openRun(watch.name);

// ---------- boot ----------
(async () => {
  showView(location.hash.slice(1) || "demo");
  cfg = await api("/api/config");
  $("preset").innerHTML = Object.keys(cfg.presets).map((k) => `<option>${esc(k)}</option>`).join(""); $("preset").onchange();
  $("p-width").max = cfg.limits.max_width; $("p-width").min = cfg.limits.min_width; $("p-frames").max = cfg.limits.max_frames;
  await Promise.all([loadRuns(), loadVideos()]);
  const want = new URLSearchParams(location.search).get("run");
  if (want) openRun(want); else openFeatured();
  if (view === "lab" && !LAB.e && !labLoading) openLab($("lab-sel").value || LAB_DEFAULT);
})().catch((err) => toast("startup failed: " + err.message, true));

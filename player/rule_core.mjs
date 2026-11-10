// Explicit decision-list rule engine ("ca-universal/dlrule-v1"; files written before the rename say "rule-v1" + have "programs"), independent of Python. Same semantics as ca/rules.py:
// features from the CURRENT grid (zero boundary), first firing rule wins, default keep, synchronous, S substeps/frame,
// external schedule chooses the program per frame. step() can also return, per cell, the firing rule index and features
// (the UI cell inspector uses this exact path, so the explanation is the computation itself).
export function parseRules(o) {
  if (!(o.format === "ca-universal/dlrule-v1" || (o.format === "ca-universal/rule-v1" && Array.isArray(o.programs)))) throw new Error("not dlrule-v1");
  const { H, W, k } = o, hex = (s) => Uint8Array.from(s.match(/../g) || [], (b) => parseInt(b, 16)), bits = (u8, n) => { const a = new Uint8Array(n); for (let i = 0; i < n; i++) a[i] = (u8[i >> 3] >> (i & 7)) & 1; return a; };
  return { ...o, first: bits(hex(o.first), H * W), h0: o.h0 ? bits(hex(o.h0), k * H * W) : new Uint8Array(k * H * W) };
}
export function features(m, v, h, i) {
  const { H, W, k } = m, y = Math.floor(i / W), x = i % W, at = (a, yy, xx) => (yy < 0 || yy >= H || xx < 0 || xx >= W ? 0 : a[yy * W + xx]);
  const f = { v: v[i] };
  for (let b = 0; b < k; b++) f["h" + b] = h[b * H * W + i];
  let n8 = 0, r2 = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const c = Math.max(Math.abs(dy), Math.abs(dx)), val = at(v, y + dy, x + dx); if (c === 1) n8 += val; else if (c === 2) r2 += val; }
  f.n8 = n8; f.r2 = r2;
  f.N = at(v, y - 1, x - 1) + at(v, y - 1, x) + at(v, y - 1, x + 1); f.S = at(v, y + 1, x - 1) + at(v, y + 1, x) + at(v, y + 1, x + 1);
  f.W = at(v, y - 1, x - 1) + at(v, y, x - 1) + at(v, y + 1, x - 1); f.E = at(v, y - 1, x + 1) + at(v, y, x + 1) + at(v, y + 1, x + 1);
  for (let b = 0; b < k; b++) { const hb = h.subarray(b * H * W, (b + 1) * H * W); let s = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dy || dx) s += at(hb, y + dy, x + dx); f["m" + b] = s; }
  return f;
}
const OPS = { "<=": (a, b) => a <= b, ">=": (a, b) => a >= b, "==": (a, b) => a === b };
export function fire(prog, f) { for (let r = 0; r < prog.rules.length; r++) if (prog.rules[r].if.every(([n, op, c]) => OPS[op](f[n], c))) return r; return -1; }
export function step(m, prog, v, h, fired = null) {
  const N = m.H * m.W, v2 = v.slice(), h2 = h.slice();
  for (let i = 0; i < N; i++) {
    const r = fire(prog, features(m, v, h, i)); if (fired) fired[i] = r; if (r < 0) continue;
    const t = prog.rules[r].then; v2[i] = t.v; for (let b = 0; b < m.k; b++) h2[b * N + i] = (t.h >> b) & 1;
  }
  return [v2, h2];
}
export function programAt(m, t) { for (const p of m.schedule) if (t >= p.start && t < p.end) return p.program; return 0; }
// Run from the first frame; returns visible frames (and, if keep, the full state after every substep for the inspector).
export function run(m, T = m.T, keep = false) {
  let v = m.first.slice(), h = m.h0.slice(); const frames = [v.slice()], states = keep ? [[{ v: v.slice(), h: h.slice() }]] : null;
  for (let t = 1; t < T; t++) { const prog = m.programs[programAt(m, t)], sub = []; for (let s = 0; s < m.S; s++) { [v, h] = step(m, prog, v, h); if (keep) sub.push({ v: v.slice(), h: h.slice() }); } frames.push(v.slice()); if (keep) states.push(sub); }
  return { frames, states };
}
// Cell inspector (appended): the explanation is produced by the engine's own features() and fire(). For each rule up to
// the firing one: every condition evaluated, and for the rules that failed, the first failing condition.
export function explainCell(m, prog, v, h, i) {
  const f = features(m, v, h, i), fired = fire(prog, f), N = m.H * m.W, rules = [];
  for (let r = 0; r < prog.rules.length; r++) {
    const conds = prog.rules[r].if.map(([n, op, c]) => ({ n, op, c, val: f[n], ok: OPS[op](f[n], c) }));
    rules.push({ r, conds, ok: r === fired, firstFail: conds.findIndex((x) => !x.ok), then: prog.rules[r].then, skipped: fired >= 0 && r > fired });
  }
  const cur = { v: v[i], h: Array.from({ length: m.k }, (_, b) => h[b * N + i]) };
  const t = fired >= 0 ? prog.rules[fired].then : null;
  return { y: Math.floor(i / m.W), x: i % m.W, features: f, fired, rules, cur, next: t ? { v: t.v, h: Array.from({ length: m.k }, (_, b) => (t.h >> b) & 1) } : cur };
}

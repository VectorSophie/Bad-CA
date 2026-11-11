// Neural CA replay core shared by the Node verifier (nca_replay.mjs) and the local UI (browser). No Python shared.
// Reads an export ONLY (nca-v1 JSON or ncab-v1 binary): weights, architecture, first frame. Never sees target frames.
// Step (identical to the C9/C10 verifier): p = relu(conv3x3(s, Wp) + bp), zero padding, kernel (f, c, dy, dx);
// s += Wu @ p + bu. Sums accumulated in float64, stored as float32 (Math.fround) per value.
// ncab-v1 layout: "NCAB" | u8 version=1 | u8 flags (bit0: rest is zlib/deflate) | rest =
//   u32le header_len | header JSON (utf8) | params in header.params order, header.weight_dtype ("f32" | "f16", LE) |
//   first frame packed bits (little bit order, H*W bits).
const b64bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

function check(m) {
  const { C, Fh } = m;
  if (m.Wp.length !== Fh * C * 9 || m.Wu.length !== C * Fh || m.bp.length !== Fh || m.bu.length !== C) throw new Error("bad parameter shapes");
  if (m.first.length < Math.ceil((m.H * m.W) / 8)) throw new Error("first frame too short");
  return m;
}

export function parseNcaJson(ex) {
  if (ex.format !== "ca-universal/nca-v1") throw new Error("not nca-v1");
  const f32 = (p) => { const b = b64bytes(p.b64); return new Float32Array(b.buffer, b.byteOffset, b.length / 4).slice(); };
  return check({ C: ex.C, Fh: ex.Fh, S: ex.S, T: ex.n_frames, H: ex.height, W: ex.width, threshold: ex.threshold,
    Wp: f32(ex.params.Wp), bp: f32(ex.params.bp), Wu: f32(ex.params.Wu), bu: f32(ex.params.bu), first: b64bytes(ex.first_frame_b64),
    weight_dtype: "f32", meta: ex.meta || {} });
}

// ncab-v2 (phased, external schedule): header.blobs lists every float blob in order (Wp,bp,Wu,bu stacked per phase for
// fixed/indep; base + Ap,Bp,dbp,Au,Bu,dbu for delta; optional init_g = learned low-res initial hidden). Effective per-phase
// params are combined in float64 and rounded to float32 once (same as ca/phase_nca.py). schedule: [{start,end,rule}].
function parseV2(hd, body, off) {
  const w = hd.weight_dtype === "f16" ? 2 : 4, R = {};
  for (const [name, shape] of hd.blobs) {
    const n = shape.reduce((a, b) => a * b, 1), raw = body.slice(off, off + n * w); off += n * w;
    R[name] = { a: w === 4 ? new Float32Array(raw.buffer) : f16ToF32(new Uint16Array(raw.buffer)), shape };
  }
  const first = body.slice(off, off + Math.ceil((hd.H * hd.W) / 8)), { C, Fh, K } = hd, rules = [];
  for (let k = 0; k < K; k++) {
    const pick = (nm, n) => R[nm].a.subarray(hd.mode === "delta" ? 0 : k * n, (hd.mode === "delta" ? 0 : k * n) + n);
    const Wp = Float64Array.from(pick("Wp", Fh * C * 9)), bp = Float64Array.from(pick("bp", Fh)), Wu = Float64Array.from(pick("Wu", C * Fh)), bu = Float64Array.from(pick("bu", C));
    if (hd.mode === "delta") {
      const r = R.Ap.shape[2], Ap = R.Ap.a, Bp = R.Bp.a, Au = R.Au.a, Bu = R.Bu.a;
      for (let f = 0; f < Fh; f++) for (let c = 0; c < C * 9; c++) { let acc = 0; for (let q = 0; q < r; q++) acc += Ap[(k * Fh + f) * r + q] * Bp[(k * r + q) * C * 9 + c]; Wp[f * C * 9 + c] += acc; }
      for (let c = 0; c < C; c++) for (let f = 0; f < Fh; f++) { let acc = 0; for (let q = 0; q < r; q++) acc += Au[(k * C + c) * r + q] * Bu[(k * r + q) * Fh + f]; Wu[c * Fh + f] += acc; }
      for (let f = 0; f < Fh; f++) bp[f] += R.dbp.a[k * Fh + f];
      for (let c = 0; c < C; c++) bu[c] += R.dbu.a[k * C + c];
    }
    rules.push({ Wp: Float32Array.from(Wp), bp: Float32Array.from(bp), Wu: Float32Array.from(Wu), bu: Float32Array.from(bu) });
  }
  const sched = new Int32Array(hd.T);
  for (const p of hd.schedule) for (let t = p.start; t < p.end; t++) sched[t] = p.rule;
  let hidden0 = null;
  if (hd.init.kind === "lowres") {
    const g = R.init_g, [Cg, h0, w0] = g.shape, f = hd.init.factor; hidden0 = new Float32Array(Cg * hd.H * hd.W);
    for (let c = 0; c < Cg; c++) for (let y = 0; y < hd.H; y++) for (let x = 0; x < hd.W; x++) hidden0[(c * hd.H + y) * hd.W + x] = g.a[(c * h0 + Math.floor(y / f)) * w0 + Math.floor(x / f)];
  }
  return check({ C, Fh, S: hd.S, T: hd.T, H: hd.H, W: hd.W, threshold: hd.threshold, ...rules[0], rules, sched, hidden0, first, weight_dtype: hd.weight_dtype,
    meta: hd.meta || {}, mode: hd.mode, K, schedule: hd.schedule, init: hd.init, version: 2 });
}

export async function parseNcab(u8, inflate) {
  if (String.fromCharCode(...u8.subarray(0, 4)) !== "NCAB" || (u8[4] !== 1 && u8[4] !== 2)) throw new Error("not ncab-v1/v2");
  let body = u8.subarray(6);
  if (u8[5] & 1) body = await inflate(body);
  const dv = new DataView(body.buffer, body.byteOffset, body.byteLength), hl = dv.getUint32(0, true);
  const hd = JSON.parse(new TextDecoder().decode(body.subarray(4, 4 + hl)));
  if (u8[4] === 2) return parseV2(hd, body, 4 + hl);
  let off = 4 + hl; const P = {}, w = hd.weight_dtype === "f16" ? 2 : 4;
  for (const [name, shape] of hd.params) {
    const n = shape.reduce((a, b) => a * b, 1), raw = body.slice(off, off + n * w); off += n * w;
    P[name] = w === 4 ? new Float32Array(raw.buffer) : f16ToF32(new Uint16Array(raw.buffer));
  }
  const first = body.slice(off, off + Math.ceil((hd.H * hd.W) / 8));
  return check({ C: hd.C, Fh: hd.Fh, S: hd.S, T: hd.T, H: hd.H, W: hd.W, threshold: hd.threshold, ...P, first, weight_dtype: hd.weight_dtype, meta: hd.meta || {} });
}

// Autonomous rollout from the first frame, hidden = 0. Returns {vis: Uint8Array[T], cont: Float32Array[T]} (visible channel).
// v1: one rule. v2: rule m.rules[m.sched[t]] produces frame t; hidden starts at m.hidden0 (learned) or 0.
export function rollout(m, T = m.T, onFrame = null) {
  const { C, S, H, W, threshold } = m, N = H * W;
  let s = new Float32Array(C * N);
  for (let i = 0; i < N; i++) s[i] = (m.first[i >> 3] >> (i & 7)) & 1;
  if (m.hidden0) s.set(m.hidden0, N);
  const vis = [], cont = [];
  for (let t = 0; t < T; t++) {
    const r = m.rules ? m.rules[m.sched[t]] : m;
    if (t > 0) for (let k = 0; k < S; k++) s = substep(m, r, s);
    const b = new Uint8Array(N); for (let i = 0; i < N; i++) b[i] = t === 0 ? s[i] : (s[i] > threshold ? 1 : 0);
    vis.push(b); cont.push(Float32Array.from(s.subarray(0, N)));
    if (onFrame) onFrame(t, b, cont[t]);
  }
  return { vis, cont };
}

// ---------- UI inspector helpers (appended; rollout() above now calls the optimised substep() below) ----------
// Initial full state (C*N, channel-major): visible = first frame, hidden = m.hidden0 or 0. Same as rollout's start.
export function initState(m) {
  const N = m.H * m.W, s = new Float32Array(m.C * N);
  for (let i = 0; i < N; i++) s[i] = (m.first[i >> 3] >> (i & 7)) & 1;
  if (m.hidden0) s.set(m.hidden0, N);
  return s;
}
// Parameters producing frame t (v1: the one rule; v2: rules[sched[t]]).
export const ruleAt = (m, t) => (m.rules ? m.rules[m.sched[Math.min(t, m.T - 1)]] : m);
// Reference substep (the original naive loops; kept for ui/test_inspector.mjs). float64 sums, Math.fround per stored value.
export function substepRef(m, r, s) {
  const { C, Fh, H, W } = m, N = H * W, { Wp, bp, Wu, bu } = r, p = new Float32Array(Fh * N), n = new Float32Array(C * N);
  for (let f = 0; f < Fh; f++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let acc = bp[f];
    for (let c = 0; c < C; c++) for (let dy = 0; dy < 3; dy++) {
      const yy = y + dy - 1; if (yy < 0 || yy >= H) continue;
      for (let dx = 0; dx < 3; dx++) { const xx = x + dx - 1; if (xx < 0 || xx >= W) continue; acc += Wp[((f * C + c) * 3 + dy) * 3 + dx] * s[c * N + yy * W + xx]; }
    }
    p[f * N + y * W + x] = Math.fround(acc > 0 ? acc : 0);
  }
  for (let c = 0; c < C; c++) for (let i = 0; i < N; i++) {
    let acc = 0; for (let f = 0; f < Fh; f++) acc += Wu[c * Fh + f] * p[f * N + i];
    n[c * N + i] = Math.fround(s[c * N + i] + Math.fround(acc) + bu[c]);
  }
  return n;
}
// One substep (used by rollout, rolloutStates, the UI worker). Same arithmetic as substepRef, bit for bit: every output
// value gets the same float64 terms in the same order (bp, then c, dy, dx; then f for Wu·p). Only the loop order changed:
// weight outer, pixels inner over a zero-padded float64 copy of the state (padding adds +0 terms, which leave the sum
// unchanged), so the inner loop is contiguous. Returns a new state.
const scratch = new Map();
export function substep(m, r, s) {
  const { C, Fh, H, W } = m, N = H * W, W2 = W + 2, N2 = (H + 2) * W2, L = H * W2, { Wp, bp, Wu, bu } = r, key = `${C},${Fh},${H},${W}`;
  let z = scratch.get(key);
  if (!z) scratch.set(key, (z = { pad: new Float64Array(C * N2 + 2), acc: new Float64Array(4 * L), p: new Float64Array(Fh * N), u: new Float64Array(4 * N) }));
  const { pad, acc, p, u } = z, n = new Float32Array(C * N);
  for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) { const o = c * N2 + (y + 1) * W2 + 1, q = c * N + y * W; for (let x = 0; x < W; x++) pad[o + x] = s[q + x]; }
  // 4 features at a time when Fh % 4 === 0 (else one by one): each padded value is loaded once for 4 sums
  const step4 = Fh % 4 === 0 ? 4 : 1;
  for (let f = 0; f < Fh; f += step4) {
    const nf = step4;
    for (let q = 0; q < nf; q++) acc.fill(bp[f + q], q * L, (q + 1) * L);
    for (let c = 0; c < C; c++) for (let k = 0; k < 9; k++) {
      const b0 = c * N2 + ((k / 3) | 0) * W2 + (k % 3);
      if (nf === 4) {
        const w0 = Wp[(f * C + c) * 9 + k], w1 = Wp[((f + 1) * C + c) * 9 + k], w2 = Wp[((f + 2) * C + c) * 9 + k], w3 = Wp[((f + 3) * C + c) * 9 + k];
        for (let j = 0; j < L; j++) { const v = pad[b0 + j]; acc[j] += w0 * v; acc[L + j] += w1 * v; acc[2 * L + j] += w2 * v; acc[3 * L + j] += w3 * v; }
      } else { const w = Wp[(f * C + c) * 9 + k]; for (let j = 0; j < L; j++) acc[j] += w * pad[b0 + j]; }
    }
    for (let q = 0; q < nf; q++) { const po = (f + q) * N, ao = q * L; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const a = acc[ao + y * W2 + x]; p[po + y * W + x] = Math.fround(a > 0 ? a : 0); } }
  }
  // Wu·p: 4 channels at a time when C % 4 === 0 (same per-value order over f)
  const c4 = C % 4 === 0 ? 4 : 1;
  for (let c = 0; c < C; c += c4) {
    u.fill(0, 0, c4 * N);
    for (let f = 0; f < Fh; f++) {
      const po = f * N;
      if (c4 === 4) {
        const w0 = Wu[c * Fh + f], w1 = Wu[(c + 1) * Fh + f], w2 = Wu[(c + 2) * Fh + f], w3 = Wu[(c + 3) * Fh + f];
        for (let i = 0; i < N; i++) { const v = p[po + i]; u[i] += w0 * v; u[N + i] += w1 * v; u[2 * N + i] += w2 * v; u[3 * N + i] += w3 * v; }
      } else { const w = Wu[c * Fh + f]; for (let i = 0; i < N; i++) u[i] += w * p[po + i]; }
    }
    for (let q = 0; q < c4; q++) { const b = bu[c + q], o = (c + q) * N, uo = q * N; for (let i = 0; i < N; i++) n[o + i] = Math.fround(s[o + i] + Math.fround(u[uo + i]) + b); }
  }
  return n;
}

// rollout() that also keeps the full state at every `every`-th frame (states[t] or null). vis/cont as in rollout.
export function rolloutStates(m, T = m.T, onFrame = null, every = 1) {
  const N = m.H * m.W; let s = initState(m); const vis = [], cont = [], states = [];
  for (let t = 0; t < T; t++) {
    if (t > 0) { const r = ruleAt(m, t); for (let k = 0; k < m.S; k++) s = substep(m, r, s); }
    const b = new Uint8Array(N); for (let i = 0; i < N; i++) b[i] = t === 0 ? s[i] : (s[i] > m.threshold ? 1 : 0);
    vis.push(b); cont.push(Float32Array.from(s.subarray(0, N))); states.push(t % every === 0 ? s : null);
    if (onFrame) onFrame(t, b, cont[t]);
  }
  return { vis, cont, states };
}
// Everything the substep computes for cell i: 3x3 neighbourhood of every channel (null outside = zero padding),
// perception pre-activation (float64 sum) and post-ReLU (float32), update Wu·p (float32), bias, next state per channel.
export function traceCell(m, r, s, i) {
  const { C, Fh, H, W } = m, N = H * W, y = Math.floor(i / W), x = i % W, { Wp, bp, Wu, bu } = r;
  const nbr = [], pre = new Float64Array(Fh), post = new Float32Array(Fh), upd = new Float32Array(C), next = new Float32Array(C);
  for (let c = 0; c < C; c++) { const g = []; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const yy = y + dy, xx = x + dx; g.push(yy < 0 || yy >= H || xx < 0 || xx >= W ? null : s[c * N + yy * W + xx]); } nbr.push(g); }
  for (let f = 0; f < Fh; f++) {
    let acc = bp[f];
    for (let c = 0; c < C; c++) for (let dy = 0; dy < 3; dy++) {
      const yy = y + dy - 1; if (yy < 0 || yy >= H) continue;
      for (let dx = 0; dx < 3; dx++) { const xx = x + dx - 1; if (xx < 0 || xx >= W) continue; acc += Wp[((f * C + c) * 3 + dy) * 3 + dx] * s[c * N + yy * W + xx]; }
    }
    pre[f] = acc; post[f] = Math.fround(acc > 0 ? acc : 0);
  }
  for (let c = 0; c < C; c++) {
    let acc = 0; for (let f = 0; f < Fh; f++) acc += Wu[c * Fh + f] * post[f];
    upd[c] = Math.fround(acc); next[c] = Math.fround(s[c * N + i] + upd[c] + bu[c]);
  }
  return { y, x, cur: Float32Array.from({ length: C }, (_, c) => s[c * N + i]), nbr, pre, post, upd, bias: Float32Array.from(bu), next };
}
// Header JSON of an ncab (v1 or v2) without decoding weights; for the per-phase storage summary.
export async function ncabHeader(u8, inflate) {
  let body = u8.subarray(6); if (u8[5] & 1) body = await inflate(body);
  const hl = new DataView(body.buffer, body.byteOffset, body.byteLength).getUint32(0, true);
  return { version: u8[4], header: JSON.parse(new TextDecoder().decode(body.subarray(4, 4 + hl))), file_bytes: u8.length, raw_bytes: body.length + 6 };
}

// Independent decoder + player for Campaign 05 container files of CA methods (ca_circuit = 1,
// ca_lut_exc = 2). Reads ONLY the file bytes (no JSON, no Python, no source clip), rebuilds the rule
// and the first frame, runs the CA autonomously and prints the SHA-256 over all T observed frames
// (1 byte per cell, row-major, frame-major) -- the same digest as ca.container.sha.
// Bit layout: see ca/container.py and ca/costs.py (MSB-first bits, Elias-gamma integers).
// circuit: gate j = [tt (4 bits), a, b (ceil(log2(n + j)) bits each, min 1)], pool = [taps, gates];
// tt bit q = output for (a, b) = (q >> 1, q & 1). Gates are evaluated one by one per cell (no LUT).
// usage: node decode_file.mjs file.bin [frames_out.bin]
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";

const data = readFileSync(process.argv[2]);
let pos = 0;
const bit = () => { if (pos >= data.length * 8) throw new Error("read past end"); const v = (data[pos >> 3] >> (7 - (pos & 7))) & 1; pos++; return v; };
const bits = (n) => { let v = 0; for (let i = 0; i < n; i++) v = v * 2 + bit(); return v; };
const gamma = () => { let z = 0; while (bit() === 0) z++; return 2 ** z + bits(z); };
const clog2 = (x) => Math.max(1, x <= 1 ? 0 : (x - 1).toString(2).length);

const method = gamma();
if (method !== 1 && method !== 2) throw new Error("not a CA method: " + method);
const W = gamma(), H = gamma(), T = gamma(), N = W * H;
const f0 = new Uint8Array(N);
if (bit()) { let i = 0, v = 0; while (i < N) { const run = gamma() - 1; for (let r = 0; r < run; r++) f0[i++] = v; v ^= 1; } if (i !== N) throw new Error("bad first frame"); }
else for (let i = 0; i < N; i++) f0[i] = bit();

const family = gamma(), k = gamma() - 1, S = gamma(), bcode = bits(2), hinit = bits(1);
if (!(family >= 1 && family <= 3) || bcode > 2 || S < 1 || k < 0) throw new Error("bad rule header");
const boundary = bcode === 2 ? "wrap" : bcode;
const C = k + 1, taps = [];
for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) taps.push([0, dy, dx]);
for (let c = 1; c <= k; c++) {
  if (family === 3) { for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) taps.push([c, dy, dx]); }
  else taps.push([c, 0, 0]);
}
const n = taps.length;

let gates = null, outputs = null, luts = null;
if (method === 1) {
  const G = gamma() - 1; gates = [];
  for (let j = 0; j < G; j++) { const nb = clog2(n + j); const tt = bits(4), a = bits(nb), b = bits(nb); if (a >= n + j || b >= n + j) throw new Error("bad wire"); gates.push([tt, a, b]); }
  const nb = clog2(n + G); outputs = []; for (let c = 0; c < C; c++) { const o = bits(nb); if (o >= n + G) throw new Error("bad output"); outputs.push(o); }
} else {
  if (gamma() !== 1) throw new Error("unknown default");
  const E = 2 ** n; luts = [];
  const own = (c) => (c === 0 ? 4 : taps.findIndex(([cc, dy, dx]) => cc === c && dy === 0 && dx === 0));
  for (let c = 0; c < C; c++) { const l = new Uint8Array(E), o = own(c); for (let e = 0; e < E; e++) l[e] = (e >> o) & 1; luts.push(l); }
  const m = gamma() - 1; let prev = -1;
  for (let i = 0; i < m; i++) { prev += gamma(); luts[Math.floor(prev / E)][prev % E] ^= 1; }
}

const cell = (ch, y, x) => {
  if (boundary === "wrap") return ch[((y + H) % H) * W + ((x + W) % W)];
  if (y < 0 || y >= H || x < 0 || x >= W) return boundary;
  return ch[y * W + x];
};
const pool = new Uint8Array(n + (gates ? gates.length : 0));
const step = (src) => {
  const dst = Array.from({ length: C }, () => new Uint8Array(N));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (gates) {
      for (let t = 0; t < n; t++) pool[t] = cell(src[taps[t][0]], y + taps[t][1], x + taps[t][2]);
      for (let j = 0; j < gates.length; j++) { const [tt, a, b] = gates[j]; pool[n + j] = (tt >> (pool[a] * 2 + pool[b])) & 1; }
      for (let c = 0; c < C; c++) dst[c][y * W + x] = pool[outputs[c]];
    } else {
      let idx = 0;
      for (let t = 0; t < n; t++) idx += cell(src[taps[t][0]], y + taps[t][1], x + taps[t][2]) * 2 ** t;
      for (let c = 0; c < C; c++) dst[c][y * W + x] = luts[c][idx];
    }
  }
  return dst;
};

const t0 = process.hrtime.bigint();
let s = Array.from({ length: C }, () => new Uint8Array(N).fill(hinit)); s[0] = f0;
const h = createHash("sha256"), frames = [];
for (let f = 0; f < T; f++) {
  if (f > 0) for (let g = 0; g < S; g++) s = step(s);
  h.update(s[0]); frames.push(Buffer.from(s[0]));
}
const ms = Number(process.hrtime.bigint() - t0) / 1e6;
if (process.argv[3]) writeFileSync(process.argv[3], Buffer.concat(frames));
console.log(JSON.stringify({ sha256: h.digest("hex"), method, W, H, T, family, k, S, boundary, hidden_init: hinit,
  gates: gates ? gates.length : null, updates: (T - 1) * S * N, decode_ms: Math.round(ms * 100) / 100, file_bytes: data.length }));

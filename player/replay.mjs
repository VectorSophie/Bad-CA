// Independent replay of a ca-universal/rule-v1 (LUTs) or circuit-v1 (2-input gates) export.
// Shares no code with the Python search. circuit-v1 is evaluated gate by gate per cell (no LUT is built):
// gates[j] = [tt, a, b] over the pool [tap inputs 0..n-1, gate 0, gate 1, ...]; tt bit q = output for
// (a, b) = (q >> 1, q & 1); outputs[c] = pool index of channel c's next value.
// usage: node replay.mjs export.json [frames_out.bin]
// prints JSON {sha256 (over observed-channel frames, 1 byte/cell), white_per_frame}
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";

const ex = JSON.parse(readFileSync(process.argv[2], "utf8"));
const CIRCUIT = ex.format === "ca-universal/circuit-v1";
if (!(CIRCUIT || ex.format === "ca-universal/rule-v1")) throw new Error("unknown format " + ex.format);
const { width: W, height: H, channels: C, taps, steps_per_frame: S, n_frames: N, observed_channel: OBS } = ex;
const n = taps.length;
if (!(W > 0 && H > 0 && C >= 1 && S >= 1 && N >= 1 && n >= 1 && n <= 24 && OBS >= 0 && OBS < C)) throw new Error("bad header");
if (!(ex.boundary === "wrap" || ex.boundary === 0 || ex.boundary === 1)) throw new Error("bad boundary");
if (!(ex.hidden_init === 0 || ex.hidden_init === 1)) throw new Error("bad hidden_init");
for (const [c, dy, dx] of taps) if (!(c >= 0 && c < C && Math.abs(dy) <= 1 && Math.abs(dx) <= 1)) throw new Error("bad tap");
if (CIRCUIT) {
  ex.gates.forEach(([tt, a, b], j) => { if (!(tt >= 0 && tt < 16 && a >= 0 && b >= 0 && a < n + j && b < n + j)) throw new Error("bad gate " + j); });
  if (ex.outputs.length !== C || !ex.outputs.every((o) => o >= 0 && o < n + ex.gates.length)) throw new Error("bad outputs");
} else if (ex.luts_b64.length !== C) throw new Error("need one LUT per channel");

const bits = (b64, len) => {          // little-endian bit order within each byte
  const buf = Buffer.from(b64, "base64");
  if (buf.length !== Math.ceil(len / 8)) throw new Error("length mismatch");
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = (buf[i >> 3] >> (i & 7)) & 1;
  return out;
};
const luts = CIRCUIT ? null : ex.luts_b64.map((b) => bits(b, 2 ** n));
const pool = new Uint8Array(n + (CIRCUIT ? ex.gates.length : 0));
let g = Array.from({ length: C }, () => new Uint8Array(W * H).fill(ex.hidden_init));
g[OBS] = bits(ex.seed_visible_b64, W * H);

const cell = (ch, y, x) => {
  if (ex.boundary === "wrap") return ch[((y + H) % H) * W + ((x + W) % W)];
  if (y < 0 || y >= H || x < 0 || x >= W) return ex.boundary;
  return ch[y * W + x];
};
const step = (src) => {
  const dst = Array.from({ length: C }, () => new Uint8Array(W * H));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (CIRCUIT) {
      taps.forEach(([c, dy, dx], k) => { pool[k] = cell(src[c], y + dy, x + dx); });
      ex.gates.forEach(([tt, a, b], j) => { pool[n + j] = (tt >> (pool[a] * 2 + pool[b])) & 1; });
      for (let c = 0; c < C; c++) dst[c][y * W + x] = pool[ex.outputs[c]];
      continue;
    }
    let idx = 0;
    taps.forEach(([c, dy, dx], k) => { idx += cell(src[c], y + dy, x + dx) * 2 ** k; });
    for (let c = 0; c < C; c++) dst[c][y * W + x] = luts[c][idx];
  }
  return dst;
};

const h = createHash("sha256"), frames = [], white = [];
for (let f = 0; f < N; f++) {
  if (f > 0) for (let s = 0; s < S; s++) g = step(g);
  h.update(g[OBS]); frames.push(Buffer.from(g[OBS])); white.push(g[OBS].reduce((a, b) => a + b, 0));
}
if (process.argv[3]) writeFileSync(process.argv[3], Buffer.concat(frames));
console.log(JSON.stringify({ sha256: h.digest("hex"), white_per_frame: white }));

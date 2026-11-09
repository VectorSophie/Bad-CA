// Independent replay of a ca-universal/nca-v1 export (Neural CA, continuous float32 state). Shares no code with Python.
// Reads ONLY the export JSON (weights, config, first frame). State: C x H x W Float32Array, channel 0 visible
// (= first frame), hidden 0. Step: p = relu(conv3x3(s, Wp) + bp) with zero padding, kernel (f, c, dy, dx), dy/dx 0..2
// for offsets -1..1; s += Wu @ p + bu. Sums are accumulated in float64 and stored as float32 (Math.fround) per value.
// Output: binary visible (> threshold) at every observed frame (every S steps), SHA-256 over T*H*W bytes, and the
// continuous visible values (Float32 little-endian) if a second path is given.
// usage: node nca_replay.mjs export.json [visible_f32.bin]
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";

const ex = JSON.parse(readFileSync(process.argv[2], "utf8"));
if (ex.format !== "ca-universal/nca-v1") throw new Error("not nca-v1");
const { C, Fh, S, n_frames: T, height: H, width: W, threshold } = ex;
const f32 = (p) => { const b = Buffer.from(p.b64, "base64"); const a = new Float32Array(b.buffer, b.byteOffset, b.length / 4); return Float32Array.from(a); };
const Wp = f32(ex.params.Wp), bp = f32(ex.params.bp), Wu = f32(ex.params.Wu), bu = f32(ex.params.bu);
if (Wp.length !== Fh * C * 9 || Wu.length !== C * Fh || bp.length !== Fh || bu.length !== C) throw new Error("bad shapes");
const N = H * W;
const ff = Buffer.from(ex.first_frame_b64, "base64");
let s = new Float32Array(C * N);
for (let i = 0; i < N; i++) s[i] = (ff[i >> 3] >> (i & 7)) & 1;
const p = new Float32Array(Fh * N);
const step = () => {
  for (let f = 0; f < Fh; f++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let acc = bp[f];
    for (let c = 0; c < C; c++) for (let dy = 0; dy < 3; dy++) {
      const yy = y + dy - 1; if (yy < 0 || yy >= H) continue;
      for (let dx = 0; dx < 3; dx++) {
        const xx = x + dx - 1; if (xx < 0 || xx >= W) continue;
        acc += Wp[((f * C + c) * 3 + dy) * 3 + dx] * s[c * N + yy * W + xx];
      }
    }
    p[f * N + y * W + x] = Math.fround(acc > 0 ? acc : 0);
  }
  const n = new Float32Array(C * N);
  for (let c = 0; c < C; c++) for (let i = 0; i < N; i++) {
    let acc = 0; for (let f = 0; f < Fh; f++) acc += Wu[c * Fh + f] * p[f * N + i];
    n[c * N + i] = Math.fround(s[c * N + i] + Math.fround(acc) + bu[c]);
  }
  s = n;
};
const h = createHash("sha256"), vis = [];
for (let t = 0; t < T; t++) {
  if (t > 0) for (let k = 0; k < S; k++) step();
  const b = new Uint8Array(N); for (let i = 0; i < N; i++) b[i] = t === 0 ? s[i] : (s[i] > threshold ? 1 : 0);
  h.update(b); vis.push(Float32Array.from(s.subarray(0, N)));
}
if (process.argv[3]) writeFileSync(process.argv[3], Buffer.concat(vis.map((v) => Buffer.from(v.buffer))));
console.log(JSON.stringify({ sha256: h.digest("hex"), T, H, W, C, S }));

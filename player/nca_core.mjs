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

export async function parseNcab(u8, inflate) {
  if (String.fromCharCode(...u8.subarray(0, 4)) !== "NCAB" || u8[4] !== 1) throw new Error("not ncab-v1");
  let body = u8.subarray(6);
  if (u8[5] & 1) body = await inflate(body);
  const dv = new DataView(body.buffer, body.byteOffset, body.byteLength), hl = dv.getUint32(0, true);
  const hd = JSON.parse(new TextDecoder().decode(body.subarray(4, 4 + hl)));
  let off = 4 + hl; const P = {}, w = hd.weight_dtype === "f16" ? 2 : 4;
  for (const [name, shape] of hd.params) {
    const n = shape.reduce((a, b) => a * b, 1), raw = body.slice(off, off + n * w); off += n * w;
    P[name] = w === 4 ? new Float32Array(raw.buffer) : f16ToF32(new Uint16Array(raw.buffer));
  }
  const first = body.slice(off, off + Math.ceil((hd.H * hd.W) / 8));
  return check({ C: hd.C, Fh: hd.Fh, S: hd.S, T: hd.T, H: hd.H, W: hd.W, threshold: hd.threshold, ...P, first, weight_dtype: hd.weight_dtype, meta: hd.meta || {} });
}

// Autonomous rollout from the first frame, hidden = 0. Returns {vis: Uint8Array[T], cont: Float32Array[T]} (visible channel).
export function rollout(m, T = m.T, onFrame = null) {
  const { C, Fh, S, H, W, threshold, Wp, bp, Wu, bu } = m, N = H * W;
  let s = new Float32Array(C * N);
  for (let i = 0; i < N; i++) s[i] = (m.first[i >> 3] >> (i & 7)) & 1;
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
  const vis = [], cont = [];
  for (let t = 0; t < T; t++) {
    if (t > 0) for (let k = 0; k < S; k++) step();
    const b = new Uint8Array(N); for (let i = 0; i < N; i++) b[i] = t === 0 ? s[i] : (s[i] > threshold ? 1 : 0);
    vis.push(b); cont.push(Float32Array.from(s.subarray(0, N)));
    if (onFrame) onFrame(t, b, cont[t]);
  }
  return { vis, cont };
}

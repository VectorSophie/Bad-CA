// Computes CA frames from an export ONLY (weights + first frame) with the shared player core. Never receives original frames.
// Streaming and frame-index based: frame t is computed from frame t-1 by S substeps (the same substep() rollout() uses,
// so the frames equal `node player/nca_replay.mjs`), strictly in order, whatever the page is doing. Nothing here reads a
// clock to decide what to compute, so a hidden tab or slow painting can only delay frames, never skip them.
// Messages out: {id, info} once; {id, from, packed, upto} chunks of packed binary frames (bit i of frame t = byte
// t*B + (i>>3), bit i&7; B = ceil(H*W/8)); {id, ckpt: t, state} full state (C*H*W float32) every `every` frames;
// {id, done, sha, ms} at the end (sha256 over the T*H*W 0/1 bytes, like nca_replay.mjs).
import { parseNcab, parseNcaJson, initState, ruleAt, substep } from "/player/nca_core.mjs";

const inflate = async (b) => new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
const CKPT_BUDGET = 48 << 20; // bytes of full-state checkpoints kept by the page
let current = 0;
// yield to the event loop between chunks (MessageChannel: not clamped like timers) so a newer request can cancel this one
const tick = () => new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = r; c.port2.postMessage(0); });

self.onmessage = async ({ data: { id, bytes } }) => {
  current = id;
  try {
    const u8 = new Uint8Array(bytes), isNcab = u8[0] === 0x4e && u8[1] === 0x43 && u8[2] === 0x41 && u8[3] === 0x42; // "NCAB"
    const m = isNcab ? await parseNcab(u8, inflate) : parseNcaJson(JSON.parse(new TextDecoder().decode(u8)));
    const { T, S, C, H, W } = m, N = H * W, B = Math.ceil(N / 8), every = Math.max(4, Math.ceil((T * C * N * 4) / CKPT_BUDGET));
    self.postMessage({ id, info: { T, S, C, H, W, N, B, every } });
    const all = new Uint8Array(T * N), t0 = performance.now();
    let s = initState(m), from = 0, lastPost = t0;
    for (let t = 0; t < T; t++) {
      if (t > 0) { const r = ruleAt(m, t); for (let k = 0; k < S; k++) s = substep(m, r, s); }
      for (let i = 0; i < N; i++) all[t * N + i] = t === 0 ? s[i] : s[i] > m.threshold ? 1 : 0;
      if (t % every === 0) self.postMessage({ id, ckpt: t, state: s.slice() });
      const now = performance.now();
      if (t === T - 1 || now - lastPost > 100) {
        const n = t + 1 - from, packed = new Uint8Array(n * B);
        for (let q = 0; q < n; q++) for (let i = 0; i < N; i++) if (all[(from + q) * N + i]) packed[q * B + (i >> 3)] |= 1 << (i & 7);
        self.postMessage({ id, from, packed, upto: t + 1 }, [packed.buffer]);
        from = t + 1; lastPost = now; await tick();
        if (current !== id) return; // superseded by a newer model
      }
    }
    const ms = performance.now() - t0;
    const sha = [...new Uint8Array(await crypto.subtle.digest("SHA-256", all))].map((x) => x.toString(16).padStart(2, "0")).join("");
    self.postMessage({ id, done: true, sha, ms });
  } catch (e) {
    self.postMessage({ id, error: String((e && e.message) || e) });
  }
};

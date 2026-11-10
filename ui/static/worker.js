// Computes CA frames from an export ONLY (weights + first frame), using the shared player core. Never receives original frames.
// rolloutStates = rollout() plus the full state at frame boundaries (ui/test_inspector.mjs checks it is bit-identical to rollout()).
import { parseNcab, parseNcaJson, rolloutStates } from "/player/nca_core.mjs";

const inflate = async (b) => new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
const STATE_BUDGET = 128 << 20; // bytes of full frame states kept; above that only every k-th frame is kept

self.onmessage = async ({ data: { id, url } }) => {
  try {
    const u8 = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const isNcab = u8[0] === 0x4e && u8[1] === 0x43 && u8[2] === 0x41 && u8[3] === 0x42; // "NCAB"
    const m = isNcab ? await parseNcab(u8, inflate) : parseNcaJson(JSON.parse(new TextDecoder().decode(u8)));
    const N = m.H * m.W, every = Math.max(1, Math.ceil((m.C * N * 4 * m.T) / STATE_BUDGET));
    const t0 = performance.now();
    const { vis, cont, states } = rolloutStates(m, m.T, (t) => self.postMessage({ id, progress: (t + 1) / m.T }), every);
    const ms = performance.now() - t0;
    const all = new Uint8Array(m.T * N), cv = new Float32Array(m.T * N);
    vis.forEach((b, t) => all.set(b, t * N)); cont.forEach((c, t) => cv.set(c, t * N));
    const sha = [...new Uint8Array(await crypto.subtle.digest("SHA-256", all))].map((x) => x.toString(16).padStart(2, "0")).join("");
    const facts = { C: m.C, Fh: m.Fh, S: m.S, T: m.T, H: m.H, W: m.W, weight_dtype: m.weight_dtype, meta: m.meta, bytes: u8.length,
      n_params: m.Wp.length + m.bp.length + m.Wu.length + m.bu.length };
    const kept = states.filter(Boolean);
    self.postMessage({ id, done: true, vis: all, cont: cv, sha, ms, facts, states, every }, [all.buffer, cv.buffer, ...kept.map((s) => s.buffer)]);
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message || e) });
  }
};

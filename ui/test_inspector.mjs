// Checks the UI cell inspectors against the engines.  node ui/test_inspector.mjs [extra.ncab | extra.rule.json ...]
// NCA: (1) rolloutStates (used by the UI worker) gives bit-identical visible values to rollout() for every frame;
//      (2) traceCell's next state equals substep()'s next state for every channel (float32 bit-exact) at sampled cells.
//      (3) the optimised substep() equals the naive substepRef() (all channels, float32 bit-exact) along a rollout,
//          for the base exports and the featured model (runs/featured/manifest.json), across phase boundaries.
// rule-v1: explainCell's firing rule and next state equal step()'s fired index and next state at sampled cells.
import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { parseNcab, rollout, rolloutStates, ruleAt, substep, substepRef, traceCell, initState } from "../player/nca_core.mjs";
import { parseRules, run, step, programAt, explainCell } from "../player/rule_core.mjs";

const ROOT = new URL("..", import.meta.url).pathname;
const files = process.argv.length > 2 ? process.argv.slice(2) : [];
const base = [ROOT + "runs/c12/samples/c10_main_as_v2_2phase.ncab", ROOT + "runs/pipe/example_c10_main/model.ncab", ROOT + "runs/c12/samples/life_then_grow.rule.json"];
let fail = 0;
const ok = (cond, msg) => { console.log(`${cond ? "PASS" : "FAIL"} ${msg}`); if (!cond) fail++; };
const bits = (a) => new Uint32Array(Float32Array.from(a).buffer);
const same = (a, b) => a.length === b.length && bits(a).every((x, i) => x === bits(b)[i]);

for (const path of [...base, ...files]) {
  const name = path.replace(ROOT, "");
  if (path.endsWith(".json")) {
    const m = parseRules(JSON.parse(readFileSync(path, "utf8"))), N = m.H * m.W, { states } = run(m, m.T, true);
    let checked = 0, bad = 0;
    for (let t = 1; t < m.T; t++) for (let k = 0; k < m.S; k++) {
      const st = k === 0 ? (t === 1 ? states[0][0] : states[t - 1][m.S - 1]) : states[t][k - 1], prog = m.programs[programAt(m, t)], fired = new Int32Array(N);
      const [v2, h2] = step(m, prog, st.v, st.h, fired);
      for (let i = 0; i < N; i += 7) {
        const e = explainCell(m, prog, st.v, st.h, i); checked++;
        if (e.fired !== fired[i] || e.next.v !== v2[i] || e.next.h.some((b, j) => b !== h2[j * N + i])) bad++;
        if (e.fired >= 0 && (e.rules.slice(0, e.fired).some((r) => r.firstFail < 0) || e.rules[e.fired].firstFail !== -1)) bad++;
      }
    }
    ok(bad === 0 && checked > 0, `${name}: explainCell == step() at ${checked} cell-substeps (fired rule, next v/h, earlier rules each fail)`);
    continue;
  }
  const m = await parseNcab(new Uint8Array(readFileSync(path)), async (b) => new Uint8Array(inflateSync(b))), N = m.H * m.W;
  const ref = rollout(m), mine = rolloutStates(m);
  ok(ref.cont.every((c, t) => same(c, mine.cont[t])) && ref.vis.every((b, t) => b.every((x, i) => x === mine.vis[t][i])), `${name}: rolloutStates visible == rollout() for all ${m.T} frames (bit-exact)`);
  let checked = 0, bad = 0;
  for (const t of [0, 1, Math.floor(m.T / 2), m.T - 2]) {
    let s = mine.states[t]; const r = ruleAt(m, t + 1);
    for (let k = 0; k < m.S; k++) {
      const n = substep(m, r, s);
      for (const i of [0, m.W - 1, Math.floor(N / 2) + 3, N - 1, 5 * m.W + 7]) {
        const tr = traceCell(m, r, s, i); checked++;
        if (!same(tr.next, Array.from({ length: m.C }, (_, c) => n[c * N + i]))) bad++;
      }
      s = n;
    }
    if (t + 1 < m.T && !same(s, mine.states[t + 1])) bad++;   // S substeps from frame t reach the stored frame t+1 state
  }
  ok(bad === 0, `${name}: traceCell next == substep next, all ${m.C} channels, ${checked} cell-substeps (float32 bit-exact)`);
  ok(fastEqualsRef(m, Math.min(m.T - 1, 24)), `${name}: substep() == substepRef() for every substep of frames 1..${Math.min(m.T - 1, 24)} (float32 bit-exact)`);
}
function fastEqualsRef(m, upto, start = 1, s0 = initState(m)) {
  let s = s0;
  for (let t = start; t <= upto; t++) for (let k = 0; k < m.S; k++) { const r = ruleAt(m, t), a = substep(m, r, s), b = substepRef(m, r, s); if (!same(a, b)) return false; s = a; }
  return true;
}
// featured model: walk to its first phase boundaries with the fast step, checking it against the reference on the way
const fman = JSON.parse(readFileSync(ROOT + "runs/featured/manifest.json", "utf8")).featured;
{
  const m = await parseNcab(new Uint8Array(readFileSync(ROOT + fman.export)), async (b) => new Uint8Array(inflateSync(b)));
  const b2 = m.schedule[Math.min(2, m.schedule.length - 1)].start;
  ok(fastEqualsRef(m, Math.min(m.T - 1, b2 + 1)), `${fman.export}: substep() == substepRef() for every substep of frames 1..${Math.min(m.T - 1, b2 + 1)} (crosses boundaries at ${m.schedule.slice(1, 3).map((p) => p.start).join(", ")})`);
}
console.log(fail ? `FAIL (${fail})` : "ALL PASS");
process.exit(fail ? 1 : 0);

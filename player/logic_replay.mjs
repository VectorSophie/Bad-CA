// node logic_replay.mjs file.logic.json [steps] -> {sha256 over (steps+1)*H*W*C state bytes, row-major (H, W, C)}
import { readFileSync } from "node:fs"; import { createHash } from "node:crypto"; import { parseLogic, run } from "./logic_core.mjs";
const m = parseLogic(JSON.parse(readFileSync(process.argv[2], "utf8"))), T = process.argv[3] ? +process.argv[3] : m.steps, { frames } = run(m, T), h = createHash("sha256");
for (const f of frames) h.update(f);
console.log(JSON.stringify({ sha256: h.digest("hex"), steps: T, H: m.H, W: m.W, C: m.channels, gates: m.net.gates.length }));

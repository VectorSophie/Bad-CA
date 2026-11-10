// node rule_replay.mjs file.rule.json -> {sha256 over T*H*W visible bytes}
import { readFileSync } from "node:fs"; import { createHash } from "node:crypto"; import { parseRules, run } from "./rule_core.mjs";
const m = parseRules(JSON.parse(readFileSync(process.argv[2], "utf8"))), { frames } = run(m), h = createHash("sha256"); for (const f of frames) h.update(f);
console.log(JSON.stringify({ sha256: h.digest("hex"), T: m.T, H: m.H, W: m.W, k: m.k, S: m.S }));

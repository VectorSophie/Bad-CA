// Independent replay of a Neural CA export (ca-universal/nca-v1 JSON or ncab-v1 binary). Shares no code with Python.
// Reads ONLY the export (weights, config, first frame); the step math lives in nca_core.mjs (float64 sums, fround).
// Output: SHA-256 over the T*H*W binary visible bytes, and the continuous visible values (Float32 LE) if a second
// path is given.  usage: node nca_replay.mjs export.(json|ncab) [visible_f32.bin]
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { inflateSync } from "node:zlib";
import { parseNcaJson, parseNcab, rollout } from "./nca_core.mjs";

const buf = readFileSync(process.argv[2]);
const m = buf.subarray(0, 4).toString() === "NCAB" ? await parseNcab(new Uint8Array(buf), async (b) => new Uint8Array(inflateSync(b))) : parseNcaJson(JSON.parse(buf.toString("utf8")));
const { vis, cont } = rollout(m);
const h = createHash("sha256"); for (const b of vis) h.update(b);
if (process.argv[3]) writeFileSync(process.argv[3], Buffer.concat(cont.map((v) => Buffer.from(v.buffer))));
console.log(JSON.stringify({ sha256: h.digest("hex"), T: m.T, H: m.H, W: m.W, C: m.C, S: m.S, weight_dtype: m.weight_dtype }));

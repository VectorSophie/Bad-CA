// Full replay of an export INSIDE a headless browser with the UI's engine (player/nca_core.mjs served by ui/server.py),
// SHA-256 over the T*H*W binary frames, compared by the caller with `node player/nca_replay.mjs EXPORT`.
// usage: node experiments/browser_replay_check.mjs BASE_URL EXPORT PLAYWRIGHT_CORE_DIR HEADLESS_SHELL
import { createRequire } from "node:module"; import { readFileSync } from "node:fs";
const [base, file, pwDir, shell] = process.argv.slice(2);
const { chromium } = createRequire(pwDir + "/")("playwright-core");
const b = await chromium.launch({ executablePath: shell }); const p = await b.newPage(); await p.goto(base + "/");
const b64 = readFileSync(file).toString("base64"); const t0 = Date.now();
const r = await p.evaluate(async (b64) => {
  const c = await import("/player/nca_core.mjs"); const u8 = Uint8Array.from(atob(b64), (x) => x.charCodeAt(0));
  const inflate = async (x) => new Uint8Array(await new Response(new Blob([x]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
  const m = await c.parseNcab(u8, inflate); const N = m.H * m.W, all = new Uint8Array(m.T * N); let k = 0;
  c.rollout(m, m.T, (t, vis) => { all.set(vis, t * N); k++; });
  const h = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", all))).map((x) => x.toString(16).padStart(2, "0")).join("");
  return { sha256: h, frames: k, T: m.T, phases: m.schedule.length };
}, b64);
console.log(JSON.stringify({ ...r, seconds: (Date.now() - t0) / 1000 })); await b.close();

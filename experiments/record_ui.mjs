// Campaign 12: short real screen recording of the local UI (headless Chromium via playwright-core, read-only use of an
// installed copy). Opens a phased NCA export and an explicit-rule export, seeks to a phase boundary, clicks cells,
// steps CA substeps across the boundary and captures the page after each action -> PNG frames (assembled by Python).
// usage: node experiments/record_ui.mjs BASE_URL OUT_DIR PLAYWRIGHT_CORE_DIR HEADLESS_SHELL
import { createRequire } from "node:module"; import { mkdirSync } from "node:fs";
const [base, out, pwDir, shell] = process.argv.slice(2); mkdirSync(out, { recursive: true });
const { chromium } = createRequire(pwDir + "/")("playwright-core");
const b = await chromium.launch({ executablePath: shell }); const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
let n = 0; const shot = async (note) => { await p.waitForTimeout(400); await p.screenshot({ path: `${out}/f${String(n++).padStart(3, "0")}.png` }); console.log(n - 1, note); };
const ready = async () => { await p.waitForFunction(() => /sha256 of binary frames/.test(document.getElementById("compute-line")?.textContent || ""), null, { timeout: 180000 }); };
const scrubTo = async (t) => { await p.$eval("#scrub", (el, t) => { el.value = t; el.dispatchEvent(new Event("input", { bubbles: true })); }, t); };
const clickCell = async (fx, fy) => { const bb = await (await p.$("#cv-ca")).boundingBox(); await p.mouse.click(bb.x + bb.width * fx, bb.y + bb.height * fy); };
await p.goto(`${base}/?run=c12~pilot~uniform4~export_s0.ncab`); await ready();
await p.$eval("#player", (el) => el.scrollIntoView()); await shot("phased NCA loaded (uniform4, L240)");
await scrubTo(118); await shot("seek to frame 118 (phase 1 ends at 119)");
await clickCell(0.55, 0.45); await p.$eval("#insp", (el) => el.scrollIntoView()); await shot("cell inspector: NCA cell, phase 1");
for (let i = 0; i < 6; i++) { await p.click("#b-sub"); await shot(`+1 substep (${i + 1})`); }
await p.$eval("#timelines", (el) => el.scrollIntoView()); await shot("timeline after crossing into phase 2");
await p.goto(`${base}/?run=c12~rules~4500_56_direct_k0_K14_L32_b0.rule.json`); await ready();
await p.$eval("#player", (el) => el.scrollIntoView()); await shot("explicit rules loaded (14 programs)");
await scrubTo(20); await clickCell(0.45, 0.35); await p.$eval("#insp", (el) => el.scrollIntoView()); await shot("rule inspector: firing rule and failed earlier rules");
for (let i = 0; i < 4; i++) { await p.click("#b-frame"); await shot(`+1 frame (${i + 1})`); }
await p.$eval("#rules", (el) => el.scrollIntoView()); await shot("rule view: programs and fire counts");
await b.close();

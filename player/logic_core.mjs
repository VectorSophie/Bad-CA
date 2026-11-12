// Logic-gate CA engine ("badca/logic-ca-v1"), independent of the Python/JAX code: a fixed Boolean netlist applied to every
// cell's 3x3 neighbourhood, synchronous update, zero or periodic boundary. Node i < 9C is input (patch position p, channel c)
// with i = p*C + c (row-major, centre p = 4); gate [op, a, b] = bit ((a << 1) | b) of op_truth_tables[op].
export function parseLogic(o) {
  if (o.format !== "badca/logic-ca-v1") throw new Error("not badca/logic-ca-v1");
  const C = o.channels, n = (o.H || 0) * (o.W || 0) * C, init = new Uint8Array(n);
  if (o.init) { const u8 = Uint8Array.from(o.init.match(/../g), (b) => parseInt(b, 16)); for (let i = 0; i < n; i++) init[i] = (u8[i >> 3] >> (7 - (i & 7))) & 1; }
  return { ...o, init, tt: o.op_truth_tables };
}
// one cell: returns every node value (inputs then gates) so an inspector can show the actual computation
export function evalCell(m, s, y, x) {
  const { H, W, channels: C, net, tt } = m, per = m.boundary === "periodic", vals = new Uint8Array(net.n_in + net.gates.length);
  for (let p = 0; p < 9; p++) {
    let yy = y + Math.floor(p / 3) - 1, xx = x + (p % 3) - 1;
    if (per) { yy = (yy + H) % H; xx = (xx + W) % W; }
    for (let c = 0; c < C; c++) vals[p * C + c] = yy < 0 || yy >= H || xx < 0 || xx >= W ? 0 : s[(yy * W + xx) * C + c];
  }
  for (let g = 0; g < net.gates.length; g++) { const [op, a, b] = net.gates[g]; vals[net.n_in + g] = (tt[op] >> ((vals[a] << 1) | vals[b])) & 1; }
  return vals;
}
export function step(m, s) {
  const { H, W, channels: C, net } = m, out = new Uint8Array(s.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const v = evalCell(m, s, y, x); for (let c = 0; c < C; c++) out[(y * W + x) * C + c] = v[net.outputs[c]]; }
  return out;
}
export function run(m, T = m.steps) { const frames = [m.init]; for (let t = 0; t < T; t++) frames.push(step(m, frames[frames.length - 1])); return { frames }; }

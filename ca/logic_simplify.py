"""Path D, stage D2: simplification of badca/logic-ca-v1 netlists.

Exact (every rewrite preserves the local function, hence -- same start, same boundary, synchronous -- every rollout):
  simplify(net): pass-through/wire removal, constant propagation, gates with one relevant input, equal inputs, NOT
  absorption into the consuming gate, canonical operand order + common-subexpression merging, dead-gate removal.
  abc(net): established multi-level synthesis (yosys + ABC `abc -g`, the wrapper in ca.reduce) on the whole netlist.
  equivalent(net1, net2): SAT miter (pysat), exact for any input width.
Approximate (changes the local function; behaviour must be measured): tie(net, node, bit) ties a gate to a constant
("unit tying", cf. Lee et al., ICML 2026), followed by simplify().
"""
import json, os, subprocess, sys, tempfile, zlib
import numpy as np
from ca.dlca import OP_TT, _gate

TT_OP = {t: i for i, t in enumerate(OP_TT)}
NOT_TT, CONST = 0b0011, {0: 0, 1: 15}          # NOT a as a one-input gate: tt over (a<<1)|b = 1,1,0,0 -> 0b0011


def _f(tt, a, b): return (tt >> ((a << 1) | b)) & 1


def _swap(tt): return sum(_f(tt, (q & 1), (q >> 1)) << q for q in range(4))           # g(a, b) = f(b, a)
def _neg_a(tt): return sum(_f(tt, 1 - (q >> 1), q & 1) << q for q in range(4))         # g(a, b) = f(~a, b)
def _neg_b(tt): return sum(_f(tt, q >> 1, 1 - (q & 1)) << q for q in range(4))


def simplify(net):
    """Exact rewrite to a canonical, smaller netlist. Nodes are hash-consed terms:
    ('in', i) | ('c', bit) | ('not', x) | ('g', tt, x, y) with x < y and tt depending on both inputs."""
    n_in = net["n_in"]; terms, ids = [], {}
    def mk(t):
        if t not in ids: ids[t] = len(terms); terms.append(t)
        return ids[t]
    def neg(x):
        t = terms[x]
        if t[0] == "c": return mk(("c", 1 - t[1]))
        if t[0] == "not": return t[1]
        return mk(("not", x))
    def gate(tt, x, y):
        tx, ty = terms[x], terms[y]
        if tx[0] == "c":                                   # fold constant a
            f0, f1 = _f(tt, tx[1], 0), _f(tt, tx[1], 1); return unary(f0, f1, y)
        if ty[0] == "c": f0, f1 = _f(tt, 0, ty[1]), _f(tt, 1, ty[1]); return unary(f0, f1, x)
        if x == y: return unary(_f(tt, 0, 0), _f(tt, 1, 1), x)
        if tx[0] == "not": return gate(_neg_a(tt), tx[1], y)
        if ty[0] == "not": return gate(_neg_b(tt), x, ty[1])
        dep_a = any(_f(tt, 0, b) != _f(tt, 1, b) for b in (0, 1)); dep_b = any(_f(tt, a, 0) != _f(tt, a, 1) for a in (0, 1))
        if not dep_a and not dep_b: return mk(("c", _f(tt, 0, 0)))
        if not dep_b: return unary(_f(tt, 0, 0), _f(tt, 1, 0), x)
        if not dep_a: return unary(_f(tt, 0, 0), _f(tt, 0, 1), y)
        if x > y: tt, x, y = _swap(tt), y, x
        return mk(("g", tt, x, y))
    def unary(f0, f1, x):
        if f0 == f1: return mk(("c", f0))
        return x if (f0, f1) == (0, 1) else neg(x)
    rep = [mk(("in", i)) for i in range(n_in)]
    for op, a, b in net["gates"]: rep.append(gate(OP_TT[op], rep[a], rep[b]))
    outs = [rep[o] for o in net["outputs"]]
    # emit reachable terms in topological (creation) order
    need = set(); stack = list(outs)
    while stack:
        t = stack.pop()
        if t in need: continue
        need.add(t); k = terms[t]
        if k[0] == "not": stack.append(k[1])
        elif k[0] == "g": stack += [k[2], k[3]]
    idx = {}; gates = []
    for t in sorted(need):
        k = terms[t]
        if k[0] == "in": idx[t] = k[1]; continue
        if k[0] == "c": gates.append([CONST[k[1]], 0, 0])
        elif k[0] == "not": gates.append([12, idx[k[1]], idx[k[1]]])
        else: gates.append([TT_OP[k[1]], idx[k[2]], idx[k[3]]])
        idx[t] = n_in + len(gates) - 1
    return dict(n_in=n_in, gates=gates, outputs=[idx[o] for o in outs])


def stats(net):
    """gates (every remaining gate, NOT and constants included), NOT gates, depth (longest input->output path in gates),
    serialized bytes (compact JSON, raw and zlib)."""
    n = net["n_in"]; d = [0] * n
    for op, a, b in net["gates"]: d.append(1 + max(d[a], d[b]))
    blob = json.dumps(net, separators=(",", ":")).encode()
    return dict(gates=len(net["gates"]), not_gates=sum(op in (10, 12) for op, _, _ in net["gates"]), depth=max([d[o] for o in net["outputs"]] + [0]),
                json_bytes=len(blob), zlib_bytes=len(zlib.compress(blob, 9)))


def tie(net, node, bit):
    """Unit tying: gate node -> constant bit (approximate), then exact simplification."""
    g = [list(x) for x in net["gates"]]; g[node - net["n_in"]] = [CONST[bit], 0, 0]
    return simplify(dict(net, gates=g))


def node_values(net, P):
    """Every node's value on patches P (..., 9, C): list of arrays (inputs then gates)."""
    C = P.shape[-1]; vals = [P[..., i // C, i % C] for i in range(net["n_in"])]
    for op, a, b in net["gates"]: vals.append(_gate(op, vals[a], vals[b]))
    return vals


def _cnf(net, base, inputs):
    """Tseitin clauses of a netlist; inputs = list of input vars; returns (clauses, node vars, next free var)."""
    var = list(inputs); cl = []; nv = base
    for op, a, b in net["gates"]:
        nv += 1; z = nv; tt = OP_TT[op]; x, y = var[a], var[b]
        for q in range(4):
            va, vb = q >> 1, q & 1; o = _f(tt, va, vb)
            cl.append([(-x if va else x), (-y if vb else y), (z if o else -z)])
        var.append(z)
    return cl, var, nv


def equivalent(n1, n2, timeout=60):
    """Exact check: True if the two netlists compute the same outputs for every input (SAT miter UNSAT), False with a
    counterexample otherwise, None on timeout."""
    from pysat.solvers import Glucose4
    import threading
    assert n1["n_in"] == n2["n_in"] and len(n1["outputs"]) == len(n2["outputs"])
    ins = list(range(1, n1["n_in"] + 1)); c1, v1, nv = _cnf(n1, n1["n_in"], ins); c2, v2, nv = _cnf(n2, nv, ins); cl = c1 + c2; diffs = []
    for o1, o2 in zip(n1["outputs"], n2["outputs"]):
        nv += 1; d = nv; x, y = v1[o1], v2[o2]; diffs.append(d)
        cl += [[-d, x, y], [-d, -x, -y], [d, -x, y], [d, x, -y]]
    cl.append(diffs)
    with Glucose4(bootstrap_with=cl) as s:
        tm = threading.Timer(timeout, s.interrupt); tm.start(); r = s.solve_limited(expect_interrupt=True); tm.cancel()
        if r is None: return None, None
        if not r: return True, None
        m = set(x for x in s.get_model() if x > 0); return False, [int(i in m) for i in ins]


def verilog(net):
    n = net["n_in"]; L = [f"module rule({', '.join([f'input x{i}' for i in range(n)] + [f'output y{c}' for c in range(len(net['outputs']))])});"]
    name = lambda i: f"x{i}" if i < n else f"g{i - n}"
    for j, (op, a, b) in enumerate(net["gates"]):
        tt = OP_TT[op]; terms = [f"({'' if (q >> 1) else '~'}{name(a)} & {'' if (q & 1) else '~'}{name(b)})" for q in range(4) if _f(tt, q >> 1, q & 1)]
        L.append(f"  wire g{j}; assign g{j} = {' | '.join(terms) if terms else '1' + chr(39) + 'b0'};")
    for c, o in enumerate(net["outputs"]): L.append(f"  assign y{c} = {name(o)};")
    return "\n".join(L + ["endmodule"]) + "\n"


def abc(net, flow="abc_x4"):
    """Established multi-level synthesis of the whole netlist (yosys synth + ABC mapping to 2-input gates), via the
    wrapper and BLIF reader in ca.reduce. Returns a netlist (exactly equivalent: check with equivalent())."""
    from ca.reduce import YOSYS, FLOWS, _blif_gates
    n = net["n_in"]; C = len(net["outputs"])
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "r.v"), "w").write(verilog(net))
        cmd = f"read_verilog r.v; synth -flatten -top rule; {FLOWS[flow]}; opt_clean -purge; write_blif r.blif"
        p = subprocess.run([sys.executable, YOSYS, "-q", "-p", cmd], cwd=d, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)   # yosys (wasm) hangs, not fails, under prlimit --as=16G: use --as=64G
        if p.returncode: raise RuntimeError((p.stdout + p.stderr)[-2000:])
        gates, outputs = _blif_gates(os.path.join(d, "r.blif"), n, C)
    out = dict(n_in=n, gates=[[TT_OP[tt], a, b] for tt, a, b in gates], outputs=list(outputs))
    return out


def truth_table(net):
    n = net["n_in"]; assert n <= 20
    idx = np.arange(2 ** n, dtype=np.int64); C = len(net["outputs"])
    P = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8).reshape(-1, 9, n // 9)
    vals = node_values(net, P); return np.stack([vals[o] for o in net["outputs"]], -1)


def describe_totalistic(net):
    """For a 1-channel circuit: if the local function depends only on (centre, number of live neighbours), return the
    exact birth/survival statement; otherwise None. Checked on all 512 neighbourhoods, so the statement is exact."""
    if net["n_in"] != 9: return None
    T = truth_table(net)[:, 0]; idx = np.arange(512); bits = (idx[:, None] >> np.arange(9)) & 1
    v = bits[:, 4]; nb = bits.sum(1) - v; table = {}
    for i in range(512):
        k = (int(v[i]), int(nb[i]))
        if table.setdefault(k, int(T[i])) != int(T[i]): return None
    born = [n for n in range(9) if table.get((0, n)) == 1]; surv = [n for n in range(9) if table.get((1, n)) == 1]
    return dict(rule=f"B{''.join(map(str, born))}/S{''.join(map(str, surv))}",
                text=f"a dead cell becomes alive if it has {' or '.join(map(str, born)) or 'no number of'} live neighbours; "
                     f"a live cell stays alive if it has {' or '.join(map(str, surv)) or 'no number of'} live neighbours; otherwise the cell is dead")

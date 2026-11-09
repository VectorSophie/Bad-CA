"""Shrink LUT rules into small 2-input-gate circuits with existing tools (no custom optimiser):
espresso (pyeda) multi-output two-level minimisation with don't-cares, then yosys + ABC multi-level
synthesis mapped to 2-input gates (`abc -g`). The result is in ca.costs.rule_circuit form:
gates [(tt, a, b)] over the pool [inputs 0..n-1, gate 0, ...], outputs = pool index per channel.
Constants and inverters are gates too, so the size counts them with their wiring."""
import os, subprocess, sys, tempfile
import numpy as np
from pyeda.inter import exprvars, truthtable
from pyeda.boolalg.minimization import espresso_tts
from pyeda.boolalg import expr as E

YOSYS = os.path.join(os.path.dirname(sys.executable), "yowasp-yosys")
GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT"
FLOWS = {  # yosys `abc` runs ABC's default strash/synthesis script before mapping
    "abc_default": f"abc -g {GATES}",
    # shortcut: custom ABC scripts (-script) exit silently in the yowasp (wasm) build, so the heavier
    # flow re-runs the default mapping on its own output; use a native ABC for dc2/resyn2rs/compress2rs.
    "abc_x4": "; ".join([f"abc -g {GATES}; opt -full"] * 4),
}


def espresso(luts, care, n):
    """Multi-output espresso with every non-care entry a don't-care. pyeda's truth table uses
    inputs[0] as the least significant bit, the same as the LUT index (tap k = bit k)."""
    X = exprvars("x", n); dc = np.ones(2 ** n, bool); dc[np.asarray(care)] = False
    tts = [truthtable(X, "".join("-" if d else str(int(v)) for v, d in zip(l, dc))) for l in luts]
    return espresso_tts(*tts)


def _v(e):
    if e is E.Zero: return "1'b0"
    if e is E.One: return "1'b1"
    if isinstance(e, E.Variable): return f"x{e.indices[0]}"
    if isinstance(e, E.Complement): return f"~x{(~e).indices[0]}"
    if isinstance(e, E.AndOp): return "(" + " & ".join(_v(a) for a in e.xs) + ")"
    if isinstance(e, E.OrOp): return "(" + " | ".join(_v(a) for a in e.xs) + ")"
    raise TypeError(type(e))


def verilog(exprs, n):
    io = ", ".join([f"input x{i}" for i in range(n)] + [f"output y{c}" for c in range(len(exprs))])
    body = "\n".join(f"  assign y{c} = {_v(e)};" for c, e in enumerate(exprs))
    return f"module rule({io});\n{body}\nendmodule\n"


def _blif_gates(path, n, C):
    names, outs_order = {}, [f"y{c}" for c in range(C)]
    lines = [l.strip() for l in open(path)]
    i = 0
    while i < len(lines):
        if lines[i].startswith(".names"):
            sig = lines[i].split()[1:]; rows = []; i += 1
            while i < len(lines) and lines[i] and not lines[i].startswith("."):
                rows.append(lines[i].split()); i += 1
            names[sig[-1]] = (sig[:-1], rows)
        else:
            i += 1
    gates, memo = [], {}
    def node(sig):
        if sig.startswith("x") and sig[1:].isdigit() and sig not in names: return int(sig[1:])
        if sig in memo: return memo[sig]
        ins, rows = names[sig]
        def f(vals):
            hit = any(all(p == "-" or int(p) == v for p, v in zip(r[0], vals)) for r in rows if len(r) == 2)
            if not ins: hit = any(r == ["1"] for r in rows)
            on = all(r[-1] == "1" for r in rows) if rows else True
            return int(hit if on else not hit)
        if not ins:
            gates.append((15 if f(()) else 0, 0, 0)); r = n + len(gates) - 1
        elif len(ins) == 1:
            a = node(ins[0]); fa = (f((0,)), f((1,)))
            if fa == (0, 1): r = a  # buffer: a wire costs nothing
            else:
                tt = sum(fa[q >> 1] << q for q in range(4)); gates.append((tt, a, a)); r = n + len(gates) - 1
        elif len(ins) == 2:
            a, b = node(ins[0]), node(ins[1])
            tt = sum(f((q >> 1, q & 1)) << q for q in range(4)); gates.append((tt, a, b)); r = n + len(gates) - 1
        else:
            raise ValueError(f"{sig}: {len(ins)}-input node, mapping did not reach 2-input gates")
        memo[sig] = r
        return r
    outputs = [node(o) for o in outs_order]
    return gates, outputs


def synth(exprs, n, flow):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "r.v"), "w").write(verilog(exprs, n))
        cmd = f"read_verilog r.v; synth -flatten -top rule; {FLOWS[flow]}; opt_clean -purge; write_blif r.blif"
        p = subprocess.run([YOSYS, "-q", "-p", cmd], cwd=d, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
        if p.returncode: raise RuntimeError((p.stdout + p.stderr)[-2000:])
        return _blif_gates(os.path.join(d, "r.blif"), n, len(exprs))

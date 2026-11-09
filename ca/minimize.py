"""Two-level (SOP) minimisation of LUT outputs with unvisited entries as don't-cares.
shortcut: sympy Quine-McCluskey, fine to ~11 inputs; switch to espresso/ABC for wider rules."""
import numpy as np
from sympy import symbols, lambdify
from sympy.logic import SOPform


def sop(lut, care, n):
    xs = symbols(f"x0:{n}")
    care = np.asarray(care); dc = np.setdiff1d(np.arange(2 ** n), care)
    bits = lambda i: [int(b) for b in format(int(i), f"0{n}b")]  # MSB first -> reversed var list
    expr = SOPform(list(reversed(xs)), [bits(i) for i in care if lut[i]], [bits(i) for i in dc])
    idx = np.arange(2 ** n)
    f = lambdify(xs, expr, "numpy")
    new = (np.asarray(f(*[(idx >> k) & 1 for k in range(n)])) * np.ones(2 ** n)).astype(np.uint8)
    assert (new[care] == lut[care]).all()
    terms = expr.args if expr.func.__name__ == "Or" else (expr,)
    lits = sum(len(t.args) if t.func.__name__ == "And" else 1 for t in terms) if expr not in (True, False) else 0
    return dict(expr=str(expr), terms=len(terms) if expr not in (True, False) else 0, literals=lits), new

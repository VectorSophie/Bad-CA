"""Exp18 (Campaign 06): was the Campaign 05 truth-table sign error a pure variable renaming?

The pilot encoder (snapshot 893056a8f9c7) and the fixed encoder are built on the same instances.
Claim checked: mapping every truth-table variable tt[j][q] -> tt[j][3-q] (and nothing else) turns the
old clause multiset EXACTLY into the new one. If so, both CNFs are equisatisfiable (same SAT/UNSAT
verdicts); only the decoding of models (the witness reading) was wrong. Writes runs/c6/bugcheck.json.
"""
import collections, importlib.util, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.circuit_sat import CircuitSAT as New
from ca.rule import family_taps
from c5_targets import STRUCTS, target

spec = importlib.util.spec_from_file_location("old_cs", os.path.join(ROOT, "runs", "frozen", "893056a8f9c7", "ca", "circuit_sat.py"))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

out = []
for t, st, G, lex in [("pcA", "k1_S1", 2, False), ("pcB", "k1_S2", 3, False), ("pcB", "k1_S2", 3, True), ("disc12", "k1_S2", 3, False),
                      ("disc12", "k1_S2", 4, True), ("bar12", "k1_S4", 2, False), ("ba32_3000_5", "k1_S2", 3, False), ("ba32_3000_5", "k1_S2", 4, True)]:
    k, seen, S = STRUCTS[st]; F = target(t); taps = family_taps(k, seen)
    o = old.CircuitSAT(F, taps, k + 1, G, S, lex=lex); nw = New(F, taps, k + 1, G, S, lex=lex)
    ren = {}
    for j in range(G):
        for q in range(4):
            assert o.tt[j][q] == nw.tt[j][q]
            ren[o.tt[j][q]] = nw.tt[j][3 - q]
    m = lambda l: (1 if l > 0 else -1) * ren.get(abs(l), abs(l))
    A = collections.Counter(tuple(sorted(m(l) for l in c)) for c in o.cl)
    B = collections.Counter(tuple(sorted(c)) for c in nw.cl)
    same_unrenamed = collections.Counter(tuple(sorted(c)) for c in o.cl) == B
    rec = dict(target=t, struct=st, G=G, lex=lex, n_vars_old=o.nv, n_vars_new=nw.nv, n_clauses=len(nw.cl),
               identical_after_tt_renaming=A == B, identical_without_renaming=same_unrenamed)
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(dict(claim="old CNF == new CNF under tt[j][q] -> tt[j][3-q]", results=out, all_hold=all(r["identical_after_tt_renaming"] for r in out)),
          open(os.path.join(ROOT, "runs", "c6", "bugcheck.json"), "w"), indent=1)

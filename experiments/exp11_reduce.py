"""Exp11 (Campaign 04, part 3): shrink exact LUT rules into small 2-input-gate circuits. CPU only.

For every exported rule that reproduces its target exactly, two equivalence levels:
  full   the circuit equals the LUT on ALL 2^n inputs, every channel (truth-table equivalence);
  reach  only the entries the clip's own rollout consults must match; every other entry is a
         don't-care (equivalent on this clip, not as a function).
Candidates per level: keep-exception LUT bitstream (reach: non-consulted entries set to keep), and
the smallest circuit from ca.reduce (espresso -> yosys/ABC 2-input gates, each flow), either of the rule
itself or as next_c = own_c XOR delta_c(x) with delta = rule XOR keep synthesised the same way.
Every candidate is checked three ways:
  1. its implemented LUT equals the rule on the level's care set (all entries for `full`);
  2. it is serialised into its decode-checked bitstream (ca.costs), DECODED, and rolled out in Python;
  3. the decoded rule is exported (rule-v1, or circuit-v1 evaluated gate by gate) and the independent
     Node player replays it from the seed frame, hidden init and boundary; its SHA-256 must equal the
     target clip's.
Circuit bits = header + gate count + per gate (4-bit op + two wire references of ceil(log2 pool) bits)
+ output references. Constants and inverters are gates. Wires (buffers) cost nothing.

usage: exp11_reduce.py OUTDIR RULE_JSON [RULE_JSON ...]
"""
import base64, hashlib, json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.rule import family_taps, rollout
from ca.export import export_rule
from ca import costs, reduce as R
from exp03_sat_ablation import target

OUT = os.path.join(ROOT, "runs", sys.argv[1]); os.makedirs(OUT, exist_ok=True)
SNAP = os.path.basename(os.path.realpath(ROOT))
sha = lambda F: hashlib.sha256(np.ascontiguousarray(F, np.uint8).tobytes()).hexdigest()


def node(path):
    return json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path],
                                     capture_output=True, text=True, check=True).stdout)["sha256"]


def load(path):
    ex = json.load(open(path))
    n = len(ex["taps"]); C = ex["channels"]
    luts = np.stack([np.unpackbits(np.frombuffer(base64.b64decode(b), np.uint8), bitorder="little")[:2 ** n] for b in ex["luts_b64"]])
    tname = ex["meta"].get("source")
    if tname is None:  # exp07/exp12 exports: the target is in the run's result.json
        r = json.load(open(os.path.join(os.path.dirname(path), "result.json")))
        # exp07/exp08 (D1) results predate the target field; every one of them ran on ba32_3000_5 (CHECKPOINT_03)
        tname = r.get("target") or ("ba32_3000_5" if "/exp07/" in os.path.abspath(path) else None)
        assert tname, f"no target for {path}"
    k = C - 1
    fam = "memoryless" if k == 0 else ("F_priv" if n == 9 + k else "F_shared")
    taps = [tuple(t) for t in ex["taps"]]
    assert taps == family_taps(k, "moore" if fam == "F_shared" else "self"), "taps are not a known family"
    assert ex["boundary"] == 0 and ex["hidden_init"] == 0
    return ex, taps, luts, tname, fam, k


def check(label, F, S, taps, fam, k, kind, obj, luts, care, stem):
    """Serialise -> decode -> Python rollout -> Node replay. Returns the size record."""
    n = len(taps)
    if kind == "lut":
        w = costs.rule_lut_exceptions(obj, taps, fam, k, S); d = costs.rule_lut_exceptions_decode(w.b); L = d["luts"]
        path = stem + ".rule.json"
        export_rule(path, d["taps"], L, F[0], len(F), d["S"], d["boundary"], d["hidden_init"], meta=dict(label=label), expected=F)
        rec = dict(bits=len(w), exceptions=int((obj != costs.default_luts(taps, k + 1)).sum()))
    else:
        g, o = obj
        w = costs.rule_circuit(g, o, n, fam, k, S); d = costs.rule_circuit_decode(w.b)
        L = costs.eval_circuit(d["gates"], d["outputs"], n)
        path = stem + ".circuit.json"
        ex = export_rule(path, d["taps"], L, F[0], len(F), d["S"], d["boundary"], d["hidden_init"], meta=dict(label=label), expected=F)
        ex.pop("luts_b64"); ex.update(format="ca-universal/circuit-v1", gates=[list(map(int, x)) for x in d["gates"]],
                                      outputs=list(map(int, d["outputs"])))
        json.dump(ex, open(path, "w"))
        n_const = sum(1 for tt, _, _ in g if tt in (0, 15))
        rec = dict(bits=len(w), gates=len(g), constant_gates=n_const, inverter_gates=sum(1 for tt, a, b in g if tt in (3, 5) or (a == b and tt in (3, 12))))
    assert (L[:, care] == luts[:, care]).all(), f"{label}: differs from the rule on its care set"
    vis, _, _ = rollout(F[0], d["taps"], L, len(F), d["S"], d["boundary"], d["hidden_init"])
    rec.update(full_tt_equal=bool((L == luts).all()), decoded_python_rollout_exact=bool((vis == F).all()),
               node_replay_matches_target=node(path) == sha(F), export=os.path.relpath(path, ROOT))
    return rec


results = []
for path in sys.argv[2:]:
    t0 = time.perf_counter()
    ex, taps, luts, tname, fam, k = load(path)
    F = target(tname); S = ex["steps_per_frame"]; n = len(taps)
    vis, _, seen = rollout(F[0], taps, luts, len(F), S)
    label = os.path.relpath(path, os.path.join(ROOT, "runs")).replace("/", "__").replace(".rule.json", "").replace(".json", "")
    res = dict(rule=os.path.relpath(path, ROOT), target=tname, family=fam, k=k, S=S, n_inputs=n, code_snapshot=SNAP,
               exact=bool((vis == F).all()), consulted_entries=len(seen), recording_bits=costs.recording_costs(F))
    if not res["exact"]:
        res["skipped"] = "not exact on its target"; results.append(res); print(json.dumps(res), flush=True); continue
    for level, care in (("full", np.arange(2 ** n)), ("reach", np.array(seen))):
        stem = os.path.join(OUT, f"{label}.{level}")
        base = luts if level == "full" else np.where(np.isin(np.arange(2 ** n), care), luts, costs.default_luts(taps, k + 1))
        lv = dict(care_entries=int(len(care)))
        lv["lut_keep_exceptions"] = check(label, F, S, taps, fam, k, "lut", base, luts, care, stem + ".lut")
        t1 = time.perf_counter(); cover = R.espresso(luts, care, n); lv["espresso_s"] = round(time.perf_counter() - t1, 2)
        lv["sop_literals"] = sum(str(e).count("x[") for e in cover)  # two-level reference size (variable occurrences)
        flows = {}
        for fl in R.FLOWS:
            t1 = time.perf_counter(); gc = R.synth(cover, n, fl)
            flows[fl] = check(label, F, S, taps, fam, k, "circuit", gc, luts, care, f"{stem}.{fl}")
            flows[fl]["synth_s"] = round(time.perf_counter() - t1, 2)
        # structured candidate: next_c = own_c XOR delta_c(x), delta = rule XOR keep, synthesised by the same tools
        own = [taps.index((c, 0, 0)) for c in range(k + 1)]
        dcover = R.espresso(luts ^ costs.default_luts(taps, k + 1), care, n)
        for fl in R.FLOWS:
            t1 = time.perf_counter(); g, o = R.synth(dcover, n, fl); g = list(g)
            outs = []
            for c in range(k + 1):
                g.append((6, o[c], own[c])); outs.append(n + len(g) - 1)  # tt 6 = XOR
            flows["keepxor_" + fl] = check(label, F, S, taps, fam, k, "circuit", (g, outs), luts, care, f"{stem}.keepxor_{fl}")
            flows["keepxor_" + fl]["synth_s"] = round(time.perf_counter() - t1, 2)
        lv["circuits"] = flows
        best = min(flows, key=lambda f: flows[f]["bits"]); lv["best_circuit"] = dict(flow=best, **flows[best])
        if level == "full":
            assert all(v["full_tt_equal"] for v in flows.values()) and lv["lut_keep_exceptions"]["full_tt_equal"]
        res[level] = lv
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    results.append(res)
    print(json.dumps({kk: (v if kk not in ("full", "reach") else dict(lut=v["lut_keep_exceptions"]["bits"], circuit=v["best_circuit"]["bits"],
                                                                      gates=v["best_circuit"]["gates"])) for kk, v in res.items()}), flush=True)
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)

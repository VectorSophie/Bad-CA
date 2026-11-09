"""Campaign 08 pre-registered judgement (docs/CAMPAIGN_08.md): segment followed iff F1tol >= 0.25, tolerant precision
>= 0.20, change-mass ratio in [0.25, 3], err < hold. success = all three segments; partial = early followed, late not."""
SEGS = ("early", "mid", "late")


def followed(s):
    return s["f1_tol"] >= 0.25 and s["prec_tol"] >= 0.20 and 0.25 <= s["change_mass_ratio"] <= 3 and s["err"] < s["hold"]


def level(rep):
    f = {k: followed(rep[k]) for k in SEGS}
    if all(f.values()): return "success", f
    if f["early"] and not f["late"]: return "partial", f
    return "fail", f

"""Path D4: shape / motion / degeneration diagnostics and side-by-side renders for binary rollouts.
All metrics compare a predicted sequence B (n, H, W) with the original F (n, H, W) frame by frame (same indices).
Foreground = the pixels that differ from the frame's background colour, where the background colour is the majority
colour of the ORIGINAL frame's border (Bad Apple silhouettes are black-on-white or white-on-black); the same colour
defines foreground in the prediction, so a prediction that inverts or fills is penalised."""
import numpy as np
from scipy import ndimage

TOL = 1   # boundary tolerance in pixels (declared)


def bg_colour(f):
    b = np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]]); return int(b.mean() >= 0.5)


def boundary(m):
    return m & ~ndimage.binary_erosion(m, border_value=0)


def frame_shape(p, t):
    """Shape metrics of one predicted frame p against target t."""
    bg = bg_colour(t); P, T = p != bg, t != bg
    inter, union = (P & T).sum(), (P | T).sum(); iou = float(inter / union) if union else 1.0
    bp, bt = boundary(P), boundary(T); st = ndimage.generate_binary_structure(2, 2)
    dp, dt = ndimage.binary_dilation(bp, st, TOL), ndimage.binary_dilation(bt, st, TOL)
    prec = float((bp & dt).sum() / bp.sum()) if bp.sum() else (1.0 if not bt.sum() else 0.0)
    rec = float((bt & dp).sum() / bt.sum()) if bt.sum() else (1.0 if not bp.sum() else 0.0)
    bf1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    thin = T & ~ndimage.binary_opening(T, st)                         # target structures thinner than 3 px
    thin_rec = float((thin & P).sum() / thin.sum()) if thin.sum() else None
    nP, nT = ndimage.label(P, st)[1], ndimage.label(T, st)[1]
    cen = lambda m: np.array(ndimage.center_of_mass(m)) if m.any() else np.array([np.nan, np.nan])
    return dict(iou=iou, boundary_f1=bf1, thin_recall=thin_rec, cc_pred=int(nP), cc_true=int(nT), area_ratio=float(P.sum() / max(1, T.sum())),
                centroid_drift=float(np.linalg.norm(cen(P) - cen(T))) if P.any() and T.any() else None,
                edge_ratio=float(bp.sum() / max(1, bt.sum())))


def change_metrics(B, F, lag=1):
    """Change F1 at lag: predicted changes B[t] != B[t-lag] vs true changes F[t] != F[t-lag] (same pixel, same sign)."""
    if len(F) <= lag: return dict(prec=None, rec=None, f1=None, mass=None, on_off_pred=None, on_off_true=None)
    pc = B[lag:].astype(int) - B[:-lag]; tc = F[lag:].astype(int) - F[:-lag]
    hit = ((pc != 0) & (pc == tc)).sum(); npred, ntrue = (pc != 0).sum(), (tc != 0).sum()
    prec = hit / npred if npred else (1.0 if not ntrue else 0.0); rec = hit / ntrue if ntrue else (1.0 if not npred else 0.0)
    return dict(prec=float(prec), rec=float(rec), f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, mass=float(npred / ntrue) if ntrue else None,
                on_off_pred=[int((pc > 0).sum()), int((pc < 0).sum())], on_off_true=[int((tc > 0).sum()), int((tc < 0).sum())])


def entropy2x2(f):
    """Spatial entropy of 2x2 block patterns (bits): geometric blocks/stripes and constant frames have low values."""
    c = (f[:-1, :-1] + 2 * f[:-1, 1:] + 4 * f[1:, :-1] + 8 * f[1:, 1:]).ravel(); p = np.bincount(c, minlength=16) / len(c); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sequence_metrics(B, F, first_free=0):
    """B, F: (n, H, W) including the start frame at index 0. first_free = index of the first predicted (not given) frame."""
    S = [frame_shape(B[i], F[i]) for i in range(first_free, len(F))]
    agg = lambda k: float(np.nanmean([np.nan if s[k] is None else s[k] for s in S])) if S else None
    err = (B[first_free:] != F[first_free:]).mean((1, 2))
    l1, l4 = change_metrics(B, F, 1), change_metrics(B, F, 4)
    # degeneration proxies, prediction vs original over the predicted frames
    ent_p = np.mean([entropy2x2(b) for b in B[first_free:]]); ent_t = np.mean([entropy2x2(f) for f in F[first_free:]])
    const = [bool((b == b.flat[0]).all()) for b in B[first_free:]]
    period = None                                                        # smallest p <= 4 with the last 8 predicted frames repeating
    tail = B[-8:]
    for p in (1, 2, 3, 4):
        if len(tail) > p and all((tail[i] == tail[i - p]).all() for i in range(p, len(tail))): period = p; break
    true_period = None
    tt = F[-8:]
    for p in (1, 2, 3, 4):
        if len(tt) > p and all((tt[i] == tt[i - p]).all() for i in range(p, len(tt))): true_period = p; break
    collapse = next((first_free + i for i, s in enumerate(S) if s["iou"] < 0.5), None)
    sig = next((i for i in range(1, len(F)) if (B[:i + 1] != F[:i + 1]).mean() > 0.1), None)
    return dict(err=float(err.mean()), hold=float((F[first_free:] != F[max(0, first_free - 1)]).mean()), iou=agg("iou"), boundary_f1=agg("boundary_f1"),
                thin_recall=agg("thin_recall"), area_ratio=agg("area_ratio"), centroid_drift=agg("centroid_drift"), edge_ratio=agg("edge_ratio"),
                cc_pred_mean=agg("cc_pred"), cc_true_mean=agg("cc_true"), lag1=l1, lag4=l4, entropy_pred=float(ent_p), entropy_true=float(ent_t),
                constant_frames=int(sum(const)), terminal_period_pred=period, terminal_period_true=true_period,
                first_iou_below_0_5=collapse, first_cumulative_err_above_0_1=sig)


def contact_sheet(path, F, B, idx, labels, every=1, scale=4, title=""):
    """Rows: original, CA, difference (red = CA on but should be off, blue = CA off but should be on). One column per
    frame (every k-th). labels[i] in {'prefix','train','val','test'} is printed above each column."""
    from PIL import Image, ImageDraw
    cols = list(range(0, len(F), every)); H, W = F.shape[1:]; pad = 2; top = 26; lab_h = 12
    img = Image.new("RGB", (len(cols) * (W + pad) * scale // scale * 1 + 4, 0))
    cw, ch = W * scale + pad, H * scale + pad; out = Image.new("RGB", (len(cols) * cw + 60, top + 3 * ch + lab_h + 4), (246, 245, 242))
    d = ImageDraw.Draw(out); d.text((4, 2), title, fill=(20, 20, 20))
    col = {"prefix": (120, 120, 120), "train": (21, 128, 61), "val": (180, 120, 0), "test": (185, 28, 28)}
    for r, name in enumerate(("original", "CA", "diff")): d.text((2, top + lab_h + r * ch + ch // 2 - 6), name, fill=(60, 60, 60))
    for j, i in enumerate(cols):
        x0 = 56 + j * cw; d.text((x0, top - 2), f"{idx[i]}", fill=(60, 60, 60)); d.rectangle([x0, top + 9, x0 + W * scale - 1, top + 11], fill=col.get(labels[i], (0, 0, 0)))
        f, b = F[i].astype(bool), B[i].astype(bool)
        g = lambda a: np.repeat(np.repeat(a, scale, 0), scale, 1)
        rows = [np.stack([g(f) * 255] * 3, -1), np.stack([g(b) * 255] * 3, -1)]
        dif = np.full((H, W, 3), 235, np.uint8); dif[b & ~f] = (220, 40, 40); dif[~b & f] = (40, 90, 220); rows.append(np.repeat(np.repeat(dif, scale, 0), scale, 1))
        for r, a in enumerate(rows): out.paste(Image.fromarray(a.astype(np.uint8)), (x0, top + lab_h + r * ch))
    out.save(path)

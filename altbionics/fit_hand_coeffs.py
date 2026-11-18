"""
Offline per-joint least-squares fit for the Surge finger IK, from the JSONL pairs
collected by HandCalibrator during in-situ teleop.

Each JSONL row carries "sensors" (5 x [x,y,z], meters) and "state_deg"
({code: measured degrees}). We fit, per joint, a polynomial in that joint's
sensor position (X,T -> 0, I -> 1, M -> 2, R -> 3, P -> 4):

    --degree 1 :  deg = c0 + cx*x + cy*y + cz*z                 (4 coeffs)
    --degree 2 :  deg = c0 + cx*x + cy*y + cz*z
                      + cxx*x^2 + cyy*y^2 + czz*z^2
                      + cxy*xy + cxz*xz + cyz*yz                (10 coeffs)

Term order is ik.POLY_TERMS, shared with ik.poly_row so the fit and the live
evaluation can never disagree. Output coefficients.json carries manus_mode=0 (GJA
targets are true degrees) and the degree; ik.load_coefficients() reads it, and a
joint's degree is implied by its coeff-list length (4 or 10), so linear and
quadratic joints can coexist in one file.

Overfit guard: degree 2 is 10 coeffs, so on a few dozen samples in-sample R^2
flatters. Every joint reports leave-one-out CV R^2 (exact, via the hat matrix)
alongside in-sample R^2 -- trust the CV number. A big in-sample/CV gap means the
quadratic is memorizing poses, not learning the map; collect more varied data or
drop that joint back to degree 1 (fit it separately with --only ... --degree 1
--append).

Conditioning: with x,y,z in meters the degree-2 monomials span very different
scales, so the fit standardizes the design columns, solves, then folds the
standardization back into RAW-coordinate coefficients -- ik.py still just plugs in
raw meters, no stored means/scales.

Curation: fit different joints from different files with --only / --exclude, and
merge into an existing coefficients.json with --append (overwrites only the joints
fitted this run, keeps the rest -- so you can mix degrees per joint).

Examples
--------
    python -m altbionics.fit_hand_coeffs ~/alt-bionics-sdk/examples/hand_calibrate/pairs_*.jsonl \
        --degree 2 -o ~/alt-bionics-sdk/examples/hand_calibrate/coefficients.json
    python -m altbionics.fit_hand_coeffs run.jsonl --degree 2 --only I,M,R,P -o coefficients.json
    python -m altbionics.fit_hand_coeffs run.jsonl --degree 1 --only T --append coefficients.json -o coefficients.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from altbionics import ik   # single source of truth for polynomial term order

SENSOR_IDX = {"X": 0, "T": 0, "I": 1, "M": 2, "R": 3, "P": 4}
CODES = ("X", "T", "I", "M", "R", "P")

# Variable name + feeding sensor for each joint, in ik.py's block order. Used
# only to render the paste-ready ik.py block (format_ik_py_block). -anr
IK_PY_META = {
    "X": ("THUMB_X", 0),
    "T": ("THUMB_T", 0),
    "I": ("INDEX",   1),
    "M": ("MIDDLE",  2),
    "R": ("RING",    3),
    "P": ("PINKY",   4),
}


def _iter_json_objects(text):
    """Yield successive top-level JSON objects from `text`, tolerant of ANY
    whitespace (including none) between them. Uses raw_decode to consume one
    object at a time from wherever the last one ended.

    This reads every layout the calibrator has ever written:
      * legacy compact : one object per line (older files);
      * multi-line, blank-separated : object spread over lines, blank line between;
      * multi-line, no blank line   : object spread over lines, next object's '{'
        immediately after the previous object's '}' (what _format_row actually
        produces today) — this is the case the earlier line-based reader could
        not parse. -anr
    """
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        # skip any whitespace (spaces, newlines, blank lines) before the object
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        yield obj
        i = end


def load_rows(paths):
    """Load capture records from one or more pairs_*.jsonl files.

    Robust to every layout the calibrator writes (see _iter_json_objects): one
    object per line, multi-line records separated by a blank line, and multi-line
    records butted together with no blank line between them. Parses by JSON object
    boundaries, not by lines, so whitespace between records is irrelevant.

    Returns (rows, matched_paths). A pattern that matches NO file, and a file that
    parses to zero records, each print a warning -- so an empty/misnamed path can't
    silently look like "nothing to fit". -anr
    """
    rows = []
    matched = []
    for pat in paths:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if not hits:
            print(f"  [warn] no file matches: {pat}")
            continue
        matched += hits
        for p in hits:
            with open(p) as f:
                text = f.read()
            n0 = len(rows)
            try:
                rows.extend(_iter_json_objects(text))
            except ValueError as e:   # malformed JSON somewhere in the file
                print(f"  [warn] {p}: parse error, skipped the rest ({e})")
            got = len(rows) - n0
            if got == 0:
                print(f"  [warn] {p}: 0 records parsed ({len(text)} bytes)")
            else:
                print(f"  [ok]   {p}: {got} records")
    return rows, matched


# Ridge penalty grid searched per joint when --ridge is on. 0.0 is OLS, so the
# auto-selected fit can never be worse out-of-sample than plain least squares.
# Penalties are in STANDARDIZED feature units (each monomial scaled to unit
# variance first), so the same grid is sensible for every joint and degree. -anr
RIDGE_GRID = (0.0, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def _ridge_loo(As, b, alpha, tot):
    """Ridge solve in standardized space + exact leave-one-out CV, penalizing
    every column except the intercept (col 0). Returns (a, r2_loo) where `a` is
    the standardized-space coefficient vector and r2_loo is NaN if the LOO hat
    diagonal is degenerate (h_ii -> 1)."""
    import numpy as np

    p = As.shape[1]
    lam = alpha * np.eye(p)
    lam[0, 0] = 0.0                                   # never penalize the intercept
    try:
        Minv = np.linalg.inv(As.T @ As + lam)
    except np.linalg.LinAlgError:
        return None, float("nan")
    a = Minv @ (As.T @ b)
    pred = As @ a
    resid = b - pred
    h = np.einsum("ij,jk,ik->i", As, Minv, As)        # diag of the ridge hat matrix
    denom = 1.0 - h
    if np.any(denom <= 1e-9) or tot <= 0:
        return a, float("nan")
    press = float(np.sum((resid / denom) ** 2))
    return a, 1.0 - press / tot


def _fit_one(A, b, ridge=False):
    """Fit b ~ A with column standardization folded back to RAW coefficients,
    plus exact leave-one-out CV. A[:,0] is the all-ones intercept; A[:,1:] are the
    raw monomials.

    ridge=False: plain OLS (alpha 0). ridge=True: search RIDGE_GRID per joint and
    keep the penalty with the best LOO-CV R^2 (ties/NaN lose to a finite score).

    Returns (coeffs_raw, r2_in, r2_loo, rmse, alpha). coeffs_raw are in the ORIGINAL
    (unstandardized) feature space, so ik.py evaluates them directly on raw x,y,z.
    """
    import numpy as np

    n, p = A.shape
    # Standardize the non-intercept columns. Column space is unchanged for OLS, so
    # that fit is identical in exact arithmetic; for ridge, standardizing is what
    # makes one penalty grid fair across wildly different monomial scales, and it
    # keeps the (A'A + lam) inverse well-conditioned for the LOO diagonal. -anr
    mu = A[:, 1:].mean(axis=0)
    sd = A[:, 1:].std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    As = A.copy()
    As[:, 1:] = (A[:, 1:] - mu) / sd

    tot = float(np.sum((b - b.mean()) ** 2))
    alphas = RIDGE_GRID if ridge else (0.0,)
    best = None  # (r2_loo_sortkey, alpha, a, r2_loo)
    for alpha in alphas:
        a, r2_loo = _ridge_loo(As, b, alpha, tot)
        if a is None:
            continue
        key = r2_loo if r2_loo == r2_loo else float("-inf")  # NaN -> worst
        if best is None or key > best[0]:
            best = (key, alpha, a, r2_loo)
    if best is None:                                   # every alpha singular (pathological)
        a, r2_loo, alpha = np.linalg.lstsq(As, b, rcond=None)[0], float("nan"), 0.0
    else:
        _, alpha, a, r2_loo = best

    # Fold standardization back into raw-coordinate coeffs:
    #   b ~ a0 + sum_j a_j (m_j - mu_j)/sd_j
    #     = [a0 - sum_j a_j mu_j/sd_j] + sum_j (a_j/sd_j) m_j
    c = np.empty(p)
    c[1:] = a[1:] / sd
    c[0] = a[0] - np.sum(a[1:] * mu / sd)

    resid = b - A @ c
    r2_in = 1.0 - float(np.sum(resid ** 2)) / tot if tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return [float(v) for v in c], r2_in, r2_loo, rmse, float(alpha)


def fit(rows, codes, degree, ridge=False):
    import numpy as np

    ncoef = ik.N_COEFFS[degree]
    out = {}
    for code in codes:
        si = SENSOR_IDX[code]
        X, y = [], []
        for r in rows:
            s = r.get("sensors")
            st = r.get("state_deg", {})
            if not s or len(s) <= si or code not in st:
                continue
            x_, y_, z_ = s[si]
            X.append(ik.poly_row(x_, y_, z_, degree))
            y.append(st[code])
        # Need more rows than coefficients for a meaningful fit (and for LOO to
        # exist at all). Degree 2 wants a comfortable margin over its 10 coeffs.
        if len(y) <= ncoef:
            print(f"{code}: only {len(y)} samples for {ncoef} coeffs -- skipped")
            continue
        c, r2_in, r2_loo, rmse, alpha = _fit_one(np.array(X), np.array(y), ridge=ridge)
        out[code] = c
        loo = f"{r2_loo:.3f}" if r2_loo == r2_loo else "  n/a"   # NaN check
        atxt = f"  a={alpha:<6g}" if ridge else ""
        print(f"{code}: n={len(y):3d}  deg={degree}  "
              f"R^2={r2_in:.3f}  (CV {loo})  RMSE={rmse:5.2f}{atxt}")
    return out


def format_ik_py_block(manus_mode, coeffs):
    """Render `coeffs` as a paste-ready ik.py block, e.g.

        IK_REGRESSION_MANUS_MODE = 0
        # Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
        THUMB_X = IKCoeffs(73.9794, -749.0167, -1142.0210, -484.8022)  # sensor0 -> X
        ...

    Drops straight into ik.py's baked-in coefficient block. `coeffs` is the
    {code: [c0,...]} dict written to coefficients.json; each list is length 4
    (degree 1) or 10 (degree 2) and IKCoeffs infers the degree from the count, so
    the block needs no degree annotation. After an --append merge this emits the
    full six-joint set (possibly of MIXED degree), not just the joints fit this
    run. A joint absent from `coeffs` is written as a commented, clearly marked
    line so the block never silently drops a joint. The '=' columns and the
    trailing '# sensorN -> CODE' comments are aligned. -anr
    """
    name_w = max(len(name) for name, _ in IK_PY_META.values())   # align '='
    calls = {
        code: "IKCoeffs(" + ", ".join(f"{v:.4f}" for v in c) + ")"
        for code, c in coeffs.items() if code in IK_PY_META
    }
    call_w = max((len(s) for s in calls.values()), default=0)    # align comments

    lines = [
        f"IK_REGRESSION_MANUS_MODE = {int(manus_mode)}",
        "# Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).",
    ]
    for code, (name, si) in IK_PY_META.items():
        if code in calls:
            lines.append(f"{name:<{name_w}} = {calls[code]:<{call_w}}  # sensor{si} -> {code}")
        else:
            # Commented stub: keep it short rather than padding to the fitted-line
            # width, so a partial (--only ...) block stays readable. -anr
            lines.append(f"# {name:<{name_w}} = IKCoeffs(...)   # sensor{si} -> {code}  (NOT FITTED)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="JSONL file(s) or globs")
    ap.add_argument("-o", "--out", default="coefficients.json", help="output JSON path")
    ap.add_argument("--degree", type=int, default=1, choices=(1, 2),
                    help="polynomial degree to fit (1=linear default, 2=quadratic)")
    ap.add_argument("--ridge", action="store_true",
                    help="L2-regularize; auto-pick the penalty per joint by best "
                         "leave-one-out CV (OLS is in the grid, so never worse). "
                         "Strongly recommended for --degree 2 on limited data.")
    ap.add_argument("--only", default=None, help="comma list of codes to fit (e.g. I,M,R,P)")
    ap.add_argument("--exclude", default=None, help="comma list of codes to skip")
    ap.add_argument("--append", default=None,
                    help="existing coefficients.json to start from; only fitted joints are overwritten")
    ap.add_argument("--manus-mode", type=int, default=0,
                    help="manus_mode to stamp into the output (default 0: GJA targets are true degrees)")
    args = ap.parse_args()

    codes = list(CODES)
    if args.only:
        codes = [c.strip().upper() for c in args.only.split(",") if c.strip()]
    if args.exclude:
        ex = {c.strip().upper() for c in args.exclude.split(",")}
        codes = [c for c in codes if c not in ex]

    rows, matched = load_rows(args.paths)
    print(f"loaded {len(rows)} rows from {len(matched)} file(s); "
          f"fitting {codes} at degree {args.degree}"
          f"{' with ridge (auto-CV)' if args.ridge else ''}")
    if not rows:
        print("no rows loaded -- check the path(s) above; nothing written")
        return
    fitted = fit(rows, codes, args.degree, ridge=args.ridge)
    if not fitted:
        print("nothing fitted -- no output written")
        return

    base = {"manus_mode": args.manus_mode, "degree": args.degree, "coeffs": {}}
    if args.append and os.path.exists(os.path.expanduser(args.append)):
        with open(os.path.expanduser(args.append)) as f:
            base = json.load(f)
        base.setdefault("coeffs", {})
    base["manus_mode"] = args.manus_mode
    base["coeffs"].update(fitted)
    # Advisory top-level degree: the mix if joints differ, else the single value.
    degs = {len(v) for v in base["coeffs"].values()}
    base["degree"] = args.degree if len(degs) == 1 else "mixed"

    out = os.path.expanduser(args.out)
    with open(out, "w") as f:
        json.dump(base, f, indent=2)
    print(f"\nwrote {out}  (manus_mode={base['manus_mode']}, degree={base['degree']}, "
          f"joints={sorted(base['coeffs'])})")

    # Paste-ready ik.py block (the full merged set, so it includes joints carried
    # over from --append, at whatever degree each was fit). -anr
    print("\n# ---- paste-ready ik.py block (baked-in fallback coefficients) ----")
    print(format_ik_py_block(base["manus_mode"], base["coeffs"]))


if __name__ == "__main__":
    main()

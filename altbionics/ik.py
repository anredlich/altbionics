"""
Finger inverse kinematics for the Alt-Bionics Surge hand driven by a
Manus MetaGloves Pro glove.

The Manus glove reports, per finger, a 3-D fingertip position (meters) relative
to the transmitter box on the back of the hand. Those raw EMF sensor positions
are clean and reproducible (invariant to whole-hand movement). We map each
finger's fingertip position to a Surge joint angle (0-127, true degrees in
mode-0 control) with a per-joint polynomial regression:

    degree 1:  angle = c0 + cx*x + cy*y + cz*z
    degree 2:  angle = c0 + cx*x + cy*y + cz*z
                       + cxx*x^2 + cyy*y^2 + czz*z^2
                       + cxy*x*y + cxz*x*z + cyz*y*z      (then clamp to [0,127])

A joint's model order is carried implicitly by how many coefficients it has: 4 =>
degree 1, 10 => degree 2 (see POLY_ROW / N_COEFFS). The two are mixable per joint
inside one coefficients.json, so a linear finger and a quadratic thumb coexist.
fit_hand_coeffs.py chooses the degree and reports leave-one-out CV R^2 so a
degree-2 fit is only kept where it actually generalizes, not just where it drives
in-sample R^2 up.

Sensor index -> Surge joint:
    sensor 0 (thumb)  -> X (thumb rotation)  and  T (thumb flex)   [two joints, one sensor]
    sensor 1 (index)  -> I
    sensor 2 (middle) -> M
    sensor 3 (ring)   -> R
    sensor 4 (pinky)  -> P

This module is intentionally dependency-light (numpy only, and only inside the
offline refit helper) and holds no hardware or SDK references, so it is reused
unchanged by teleoperation and by policy inference, and is trivially unit-testable.
"""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# Canonical polynomial term order. eval() below inlines these same terms for
# speed (it runs in the teleop loop); poly_row() returns them as a list so the
# OFFLINE fitter (fit_hand_coeffs.py) builds its design matrix in exactly this
# order and the fit can never disagree with the evaluation. Keep the two in sync.
#   degree 1 -> 4 terms:  1, x, y, z
#   degree 2 -> 10 terms: + x^2, y^2, z^2, xy, xz, yz
# ---------------------------------------------------------------------------
POLY_TERMS = ("1", "x", "y", "z", "x^2", "y^2", "z^2", "xy", "xz", "yz")
N_COEFFS = {1: 4, 2: 10}            # degree -> number of coefficients
_DEGREE_FOR = {4: 1, 10: 2}         # number of coefficients -> degree


def poly_row(x: float, y: float, z: float, degree: int) -> "list[float]":
    """Feature row in POLY_TERMS order, truncated to `degree`. The offline fit
    imports this so its design matrix matches eval()'s term order exactly."""
    r = [1.0, x, y, z]
    if degree >= 2:
        r += [x * x, y * y, z * z, x * y, x * z, y * z]
    return r


# ---------------------------------------------------------------------------
# Fitted coefficients.
#
# angle = <polynomial in POLY_TERMS order>, clamped to [0,127]. IKCoeffs takes
# either 4 args (degree 1) or 10 args (degree 2); the count IS the degree, so the
# paste-ready block fit_hand_coeffs.py prints drops straight in here with no other
# change. The baked-in values below are the degree-1 mode-0 fallback; a re-fit in
# coefficients.json (any degree) overrides them on import (see load_coefficients).
# ---------------------------------------------------------------------------
class IKCoeffs:
    """Per-joint polynomial coefficients + clamped evaluation. Degree is inferred
    from the number of coefficients (4 -> linear, 10 -> quadratic)."""

    __slots__ = ("c", "degree")

    def __init__(self, *coeffs: float):
        if len(coeffs) not in _DEGREE_FOR:
            raise ValueError(
                f"IKCoeffs needs {sorted(_DEGREE_FOR)} coefficients "
                f"(got {len(coeffs)}); 4 => degree 1, 10 => degree 2."
            )
        self.c = tuple(float(v) for v in coeffs)
        self.degree = _DEGREE_FOR[len(self.c)]

    # Back-compat accessors for the first four (linear) coefficients. -anr
    @property
    def c0(self) -> float: return self.c[0]
    @property
    def cx(self) -> float: return self.c[1]
    @property
    def cy(self) -> float: return self.c[2]
    @property
    def cz(self) -> float: return self.c[3]

    def eval(self, x: float, y: float, z: float) -> float:
        """Polynomial model, clamped to the Surge command range [0, 127]. The
        terms are inlined (not via poly_row) to keep the teleop hot loop cheap,
        but MUST stay in POLY_TERMS order."""
        c = self.c
        a = c[0] + c[1] * x + c[2] * y + c[3] * z
        if len(c) == 10:
            a += (c[4] * x * x + c[5] * y * y + c[6] * z * z
                  + c[7] * x * y + c[8] * x * z + c[9] * y * z)
        if a < 0.0:
            return 0.0
        if a > 127.0:
            return 127.0
        return a


# ---------------------------------------------------------------------------
# Manus mode the coefficients below were fit in.
#
# The regression data was collected with the Surge hand in Manus mode 1
# (AT+SMC=1). The hand BOOTS into mode 1, and AT+SJA angle commands are
# interpreted differently in mode 1 vs mode 0, so the coefficients are only
# valid when the hand is driven in this same mode. Any code that drives the
# hand from these coefficients (teleop, inference) MUST put the hand in this
# mode first. This mirrors the C++ SDKClient's IK_REGRESSION_MANUS_MODE
# constant and its setManusMode(IK_REGRESSION_MANUS_MODE) call on startup / [R].
#
# If you ever re-fit from mode-0 raw data, set this to 0 together with the new
# coefficients, so the mode always travels with the numbers it belongs to.
# ---------------------------------------------------------------------------
IK_REGRESSION_MANUS_MODE = 1
# Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
THUMB_X = IKCoeffs(86.1629, -752.1872, -698.0955, 118.7896)   # sensor0 -> X
THUMB_T = IKCoeffs(36.9310, 160.2511, -300.9056, 36.0033)     # sensor0 -> T
INDEX   = IKCoeffs(30.1531, -635.1319, -488.8804, -644.5459)  # sensor1 -> I
MIDDLE  = IKCoeffs(3.7775, -357.4082, 1042.8632, -618.2953)   # sensor2 -> M
RING    = IKCoeffs(24.1787, -869.9276, -1311.3787, -371.9058) # sensor3 -> R
PINKY   = IKCoeffs(58.7754, -1248.2520, -471.1196, 4.7252)    # sensor4 -> P

IK_REGRESSION_MANUS_MODE = 0
# Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
THUMB_X = IKCoeffs(73.9794, -749.0167, -1142.0210, -484.8022)   # sensor0 -> X
THUMB_T = IKCoeffs(-2.7922, 1777.8480, -154.8030, -242.7610)     # sensor0 -> T
INDEX   = IKCoeffs(51.7316, -514.6595, 939.8438, -81.8321)  # sensor1 -> I
MIDDLE  = IKCoeffs(60.0123, -211.6275, 1493.8986, -91.4477)   # sensor2 -> M
RING    = IKCoeffs(91.2018, -139.5266, 1612.6392, 30.7096) # sensor3 -> R
PINKY   = IKCoeffs(72.4048, -530.3939, 634.6522, -232.7074)    # sensor4 -> P

IK_REGRESSION_MANUS_MODE = 0
# Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
THUMB_X = IKCoeffs(177.7175, -1074.1551, -1515.4751, 236.8993)  # sensor0 -> X
THUMB_T = IKCoeffs(-91.2599, 2081.2255, -17.3383, -957.9701)    # sensor0 -> T
INDEX   = IKCoeffs(94.8746, -463.5839, 673.8458, 313.2947)      # sensor1 -> I
MIDDLE  = IKCoeffs(95.1237, -243.0449, 768.2416, 249.1654)      # sensor2 -> M
RING    = IKCoeffs(115.0983, -129.0815, 1048.3159, 375.6816)    # sensor3 -> R
PINKY   = IKCoeffs(129.0874, -459.0139, 867.8822, 330.4992)     # sensor4 -> P

IK_REGRESSION_MANUS_MODE = 0
# Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
THUMB_X = IKCoeffs(14.3443, -105.6296, -898.3670, -890.9109)  # sensor0 -> X
THUMB_T = IKCoeffs(135.3448, 480.1815, -622.2234, 690.9735)   # sensor0 -> T
INDEX   = IKCoeffs(73.5009, -504.2023, 643.6362, 86.6491)     # sensor1 -> I
MIDDLE  = IKCoeffs(70.9112, -317.5807, 767.6790, -9.0410)     # sensor2 -> M
RING    = IKCoeffs(81.7838, -260.4128, 1007.9843, 31.0440)    # sensor3 -> R
PINKY   = IKCoeffs(62.3424, -492.3201, 738.8399, -356.0375)   # sensor4 -> P

# IK_REGRESSION_MANUS_MODE = 0
# # Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
# THUMB_X = IKCoeffs(36.6177, -44.5281, -262.8345, -376.4712, -3358.5466, -2249.5367, 1893.0186, -6293.3345, -2069.0372, 2669.3065)                  # sensor0 -> X
# THUMB_T = IKCoeffs(2849.3154, -45189.0690, -28233.6227, 32326.7322, 192917.2283, 50713.3328, 79856.4601, 153729.8220, -308544.1341, -195373.6179)  # sensor0 -> T
# INDEX   = IKCoeffs(473.5043, -2354.2931, -9919.7049, 2358.4463, -15699.8143, 21329.0399, -6287.1459, 59560.2518, -15432.8325, -53546.7408)         # sensor1 -> I
# MIDDLE  = IKCoeffs(174.2661, -441.4678, -443.4002, 302.6210, -8395.1625, 14532.0998, -5062.3567, 28979.1446, -4048.4724, 9240.4270)                # sensor2 -> M
# RING    = IKCoeffs(184.9523, -98.6771, 181.9810, 646.2879, -10513.5566, -6390.4805, -4136.8558, 19366.2163, -5268.9264, 6664.7533)                 # sensor3 -> R
# PINKY   = IKCoeffs(55.0264, -974.3836, 2261.0294, -2013.9497, 1267.1746, 21293.1423, -17957.2467, 30808.7395, -15331.6723, 12008.8941)             # sensor4 -> P

# # IK_REGRESSION_MANUS_MODE = 0
# # # Thumb uses sensor 0 for BOTH its rotation (X) and its flex (T).
# THUMB_X = IKCoeffs(14.3443, -105.6296, -898.3670, -890.9109)  # sensor0 -> X
# THUMB_T = IKCoeffs(2849.3154, -45189.0690, -28233.6227, 32326.7322, 192917.2283, 50713.3328, 79856.4601, 153729.8220, -308544.1341, -195373.6179)  # sensor0 -> T
# INDEX   = IKCoeffs(473.5043, -2354.2931, -9919.7049, 2358.4463, -15699.8143, 21329.0399, -6287.1459, 59560.2518, -15432.8325, -53546.7408)         # sensor1 -> I
# MIDDLE  = IKCoeffs(174.2661, -441.4678, -443.4002, 302.6210, -8395.1625, 14532.0998, -5062.3567, 28979.1446, -4048.4724, 9240.4270)                # sensor2 -> M
# RING    = IKCoeffs(184.9523, -98.6771, 181.9810, 646.2879, -10513.5566, -6390.4805, -4136.8558, 19366.2163, -5268.9264, 6664.7533)                 # sensor3 -> R
# PINKY   = IKCoeffs(55.0264, -974.3836, 2261.0294, -2013.9497, 1267.1746, 21293.1423, -17957.2467, 30808.7395, -15331.6723, 12008.8941)             # sensor4 -> P

# Surge joint command codes, in the order SurgeHand expects them.
JOINT_CODES = ("X", "T", "I", "M", "R", "P")

# Which glove sensor feeds each joint, and the live coefficient table that
# solve() reads (load_coefficients may replace entries from JSON).
_SENSOR_IDX = {"X": 0, "T": 0, "I": 1, "M": 2, "R": 3, "P": 4}
_COEFFS = {"X": THUMB_X, "T": THUMB_T, "I": INDEX, "M": MIDDLE, "R": RING, "P": PINKY}


def solve(sensors: "list[tuple[float, float, float]]") -> "dict[str, float]":
    """
    Map 5 fingertip positions -> 6 Surge joint angles.

    Parameters
    ----------
    sensors : list of 5 (x, y, z) tuples, in meters, in Manus sensor order
              (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky).

    Returns
    -------
    dict mapping each joint code ("X","T","I","M","R","P") to an angle in
    [0, 127]. Thumb X and T both read sensor 0.
    """
    if len(sensors) < 5:
        raise ValueError(f"need 5 sensor positions, got {len(sensors)}")

    return {code: _COEFFS[code].eval(*sensors[_SENSOR_IDX[code]])
            for code in JOINT_CODES}


def refit_from_csv(csv_path: str) -> "dict[str, IKCoeffs]":
    """
    Re-fit all six joints (DEGREE 1) from an ik_pairs.csv (21 columns: 5 sensors *
    xyz, then 6 angles X,T,I,M,R,P). Returns a dict of joint code -> IKCoeffs.

    Legacy helper for the old Windows CSV path. The current pipeline collects
    JSONL pairs and fits (degree 1 or 2) with fit_hand_coeffs.py instead.
    """
    import numpy as np

    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split(",")]
            if len(vals) >= 21:
                rows.append(vals)
    D = np.array(rows)
    N = len(D)
    if N < 5:
        raise ValueError(f"need several rows to fit, got {N}")

    # (sensor index, angle column) for each joint. Thumb X/T both use sensor 0.
    spec = {
        "X": (0, 15), "T": (0, 16),
        "I": (1, 17), "M": (2, 18), "R": (3, 19), "P": (4, 20),
    }
    out: "dict[str, IKCoeffs]" = {}
    for code, (si, ac) in spec.items():
        x, y, z = D[:, si * 3 + 0], D[:, si * 3 + 1], D[:, si * 3 + 2]
        ang = D[:, ac]
        A = np.column_stack([np.ones(N), x, y, z])
        c, *_ = np.linalg.lstsq(A, ang, rcond=None)
        pred = A @ c
        rmse = float(np.sqrt(np.mean((ang - pred) ** 2)))
        ss, st = np.sum((ang - pred) ** 2), np.sum((ang - np.mean(ang)) ** 2)
        r2 = 1 - ss / st if st > 0 else 0.0
        out[code] = IKCoeffs(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
        print(f"{code}: IKCoeffs({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, {c[3]:.4f})"
              f"   RMSE={rmse:.2f}  R^2={r2:.3f}")
    return out


# ---------------------------------------------------------------------------
# Optional coefficient override from JSON, so a re-fit (fit_hand_coeffs.py)
# takes effect WITHOUT editing this file, and the Manus mode travels WITH the
# numbers. The baked-in coefficients above remain the fallback.
#   schema: {"manus_mode": 0, "degree": 2,
#            "coeffs": {"X":[c0,cx,cy,cz,cxx,cyy,czz,cxy,cxz,cyz], "T":[...4 or 10...], ...}}
# A coeff list of length 4 is degree 1, length 10 is degree 2; joints may mix.
# The optional top-level "degree" is advisory only (per-joint length is truth).
# ---------------------------------------------------------------------------
DEFAULT_COEFFS_PATH = os.environ.get(
    "ALTBIONICS_IK_COEFFS",
    os.path.expanduser("~/lerobot_trossen/hand_calib/coefficients.json"),
)


def load_coefficients(path: "str | None" = None, *, verbose: bool = True) -> bool:
    """Override _COEFFS and IK_REGRESSION_MANUS_MODE from JSON. Returns True if a
    file was loaded, False if none found (defaults kept). Each joint's coeff list
    may be length 4 (degree 1) or 10 (degree 2); a bad length is skipped."""
    global IK_REGRESSION_MANUS_MODE
    p = path or DEFAULT_COEFFS_PATH
    if not os.path.exists(p):
        return False
    with open(p) as f:
        d = json.load(f)
    for code, v in d.get("coeffs", {}).items():
        if code in _COEFFS and len(v) in _DEGREE_FOR:
            _COEFFS[code] = IKCoeffs(*(float(t) for t in v))
        elif code in _COEFFS:
            print(f"[ik] skipped {code}: {len(v)} coeffs (need 4 or 10)")
    if "manus_mode" in d:
        IK_REGRESSION_MANUS_MODE = int(d["manus_mode"])
    if verbose:
        degs = {c: _COEFFS[c].degree for c in JOINT_CODES}
        print(f"[ik] loaded coefficients from {p} "
              f"(manus_mode={IK_REGRESSION_MANUS_MODE}, degrees={degs})")
    return True


# Auto-load a re-fit if present, so teleop / record / inference pick it up with
# no code change. A bad file must never break imports.
try:
    load_coefficients(verbose=True)
except Exception as _e:  # noqa: BLE001
    print(f"[ik] coefficient auto-load skipped: {_e}")

"""
In-situ finger-IK calibration for the Surge hand + Manus glove.

Collects (glove fingertip xyz -> MEASURED hand angle) pairs during live
teleoperation, so the finger IK can be re-fit from data gathered under natural
manipulation. Keyboard-driven: teleoperate normally (the hand runs the current
IK as a 0th approximation), trim each finger until the Surge finger matches your
real finger, then press Enter to record the pose.

The recorded TARGET is the hand's MEASURED angle (AT+GJA), which is TRUE DEGREES
regardless of Manus mode -- it captures where the finger actually is, not where
we asked it to go. So you can collect while running in Manus mode 1 (good 0th
approximation) and still fit mode-0 coefficients, because the targets are already
true degrees. Fit offline with fit_hand_coeffs.py, which writes coefficients.json
that ik.load_coefficients() picks up (carrying manus_mode=0 with the numbers).

Scope note: this object holds ALL the calibration state and logic. The lerobot
surge leader only (a) forwards raw keypresses via on_key, (b) forwards each tick's
follower observation via observe (where the measured hand angle rides as
"hand_<code>.pos"), (c) feeds the glove sensors via set_sensors, and (d) reads
`trim` to add to the IK output. Nothing here imports lerobot or pynput, so it
stays reusable and unit-testable.

Thread-safety: on_key runs on the keyboard-listener thread; set_sensors/observe
run on the main loop thread. Shared snapshots (`_sensors`, `_meas_deg`) are
REPLACED wholesale (never mutated in place), so the capture reads a consistent
snapshot without a lock. `trim` entries are independent float writes (GIL-atomic);
a one-step race is harmless.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

from altbionics import ik

# Finger-trim keys (match the Windows [K] / python app): up, down per joint.
# Added to the IK command for that joint so the physical finger moves while you
# match it to your real finger. The trim is in the IK command's own scale
# (mode-1: 0..127; mode-0: degrees) -- it only has to drive the finger to the
# right place; the RECORDED target is the measured GJA angle (true degrees).
TRIM_KEYS = {
    "h": ("X", +1.0), "n": ("X", -1.0),   # thumb rotation
    "g": ("T", +1.0), "b": ("T", -1.0),   # thumb flex
    "a": ("I", +1.0), "z": ("I", -1.0),   # index
    "s": ("M", +1.0), "x": ("M", -1.0),   # middle
    "d": ("R", +1.0), "c": ("R", -1.0),   # ring
    "f": ("P", +1.0), "v": ("P", -1.0),   # pinky
}

DEFAULT_DIR = os.path.expanduser("~/lerobot_trossen/hand_calib")


class HandCalibrator:
    """Trim + capture state for in-situ finger-IK calibration. See module docstring."""

    def __init__(
        self,
        data_dir: str = DEFAULT_DIR,
        session: Optional[str] = None,
        step_deg: float = 1.0,
        reset_trim_on_capture: bool = True,
    ):
        self.step_deg = float(step_deg)
        # Zero every joint's trim right AFTER each capture, so the next pose is
        # trimmed from scratch and stale nudges don't silently carry over (which
        # was confusing in the lerobot in-situ version). The row is written with
        # the live trims first, then they reset. -anr
        self.reset_trim_on_capture = bool(reset_trim_on_capture)
        self.trim: "dict[str, float]" = {c: 0.0 for c in ik.JOINT_CODES}
        # snapshots replaced wholesale so the key thread reads them atomically.
        self._sensors: Optional[list] = None
        self._meas_deg: "dict[str, float]" = {c: 0.0 for c in ik.JOINT_CODES}
        self._have_meas = False
        self._n = 0
        # Actual Manus mode the hand is being DRIVEN in at capture time, recorded
        # as the row's run_manus_mode. None -> fall back to the live
        # ik.IK_REGRESSION_MANUS_MODE (the lerobot path never sets this, so it
        # keeps recording the constant exactly as before). A driver that switches
        # the hand's mode (e.g. the standalone app's trim-only -> mode 0) sets this
        # so the provenance matches reality. The fit ignores it regardless. -anr
        self.run_manus_mode: "Optional[int]" = None
        os.makedirs(data_dir, exist_ok=True)
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(data_dir, f"pairs_{self.session}.jsonl")

    # -- fed by the leader each tick -----------------------------------------

    def set_sensors(self, sensors) -> None:
        """Latest glove fingertip positions (list of 5 (x,y,z)), or None."""
        self._sensors = list(sensors) if sensors is not None else None

    def observe(self, obs: dict) -> None:
        """Latest follower observation -> pull the MEASURED hand angles. The
        follower writes hand_<code>.pos in radians (from GJA true degrees), so we
        convert back to degrees. A missing key holds the last value for that joint."""
        m = {}
        ok = True
        for c in ik.JOINT_CODES:
            v = obs.get(f"hand_{c}.pos")
            if v is None:
                ok = False
                m[c] = self._meas_deg.get(c, 0.0)
            else:
                m[c] = math.degrees(v)
        self._meas_deg = m
        self._have_meas = self._have_meas or ok

    def set_measured(self, meas_deg: dict) -> None:
        """Latest MEASURED hand angles in DEGREES, fed directly (e.g. a standalone
        app that owns the SurgeHand and reads AT+GJA itself, rather than pulling
        hand_<code>.pos out of a follower observation). Mirrors observe() but takes
        degrees already. A missing joint holds its last value. -anr"""
        m = {}
        ok = True
        for c in ik.JOINT_CODES:
            v = meas_deg.get(c)
            if v is None:
                ok = False
                m[c] = self._meas_deg.get(c, 0.0)
            else:
                m[c] = float(v)
        self._meas_deg = m
        self._have_meas = self._have_meas or ok

    # -- keyboard (raw pynput key, duck-typed so we needn't import pynput) ----

    def on_key(self, key) -> None:
        name = getattr(key, "name", None)   # named keys (enter, esc) expose .name
        ch = getattr(key, "char", None)     # character keys expose .char
        if ch is not None:
            ch = ch.lower()
            if ch in TRIM_KEYS:
                code, sign = TRIM_KEYS[ch]
                self.trim[code] = self.trim[code] + sign * self.step_deg
                return
            if ch == "r":
                self.trim = {c: 0.0 for c in ik.JOINT_CODES}
                print("  [calib] trims reset to 0", flush=True)
                return
        if name == "enter":
            self._capture()

    # -- capture --------------------------------------------------------------

    def _capture(self) -> None:
        s = self._sensors
        if s is None or len(s) < 5:
            print("  [calib] no glove frame yet -- not captured", flush=True)
            return
        if not self._have_meas:
            print("  [calib] no measured hand state yet -- not captured", flush=True)
            return
        try:
            ik_deg = ik.solve(s)
        except Exception:
            ik_deg = {c: 0.0 for c in ik.JOINT_CODES}
        row = {
            "t": time.time(),
            "session": self.session,
            "run_manus_mode": (self.run_manus_mode if self.run_manus_mode is not None
                               else ik.IK_REGRESSION_MANUS_MODE),
            "sensors": [[float(a) for a in xyz] for xyz in s[:5]],
            "state_deg": {c: float(self._meas_deg[c]) for c in ik.JOINT_CODES},
            "ik_deg": {c: float(ik_deg[c]) for c in ik.JOINT_CODES},
            "trim_deg": {c: float(self.trim[c]) for c in ik.JOINT_CODES},
        }
        with open(self.path, "a") as f:
            f.write(_format_row(row) + "\n\n")
        self._n += 1
        st = "  ".join(f"{c}{self._meas_deg[c]:5.1f}" for c in ik.JOINT_CODES)
        print(f"  [calib] #{self._n} captured (state deg): {st}", flush=True)
        # Record is written WITH the live trims (above). Now zero them so the next
        # pose is trimmed from scratch — no stale carry-over. -anr
        if self.reset_trim_on_capture:
            self.trim = {c: 0.0 for c in ik.JOINT_CODES}
            
# ---------------------------------------------------------------------------
# Human-readable JSONL layout.
#
# Each captured record is written as ONE JSON object (still valid JSON; readable
# back with json.loads on the whole block, which fit_hand_coeffs does), but laid
# out so the four payload groups each sit on their own line and the three angle
# dicts align joint-for-joint under each other:
#
#   {"t": ..., "session": ..., "run_manus_mode": 1,
#    "sensors":   [[x,y,z], [x,y,z], [x,y,z], [x,y,z], [x,y,z]],
#    "state_deg": {"X":  .., "T":  .., "I":  .., "M":  .., "R":  .., "P":  ..},
#    "ik_deg":    {"X":  .., "T":  .., "I":  .., "M":  .., "R":  .., "P":  ..},
#    "trim_deg":  {"X":  .., "T":  .., "I":  .., "M":  .., "R":  .., "P":  ..}}
#
# ALL fields are kept (ik_deg and trim_deg are diagnostics/provenance, not
# regression inputs, but recorded so every capture can be audited). The reader
# in fit_hand_coeffs parses by JSON-object boundaries, so it reads records with
# OR without a blank line between them, and the legacy one-object-per-line files
# too. -anr
# ---------------------------------------------------------------------------

def _format_row(row: dict) -> str:
    codes = list(ik.JOINT_CODES)

    def _ang_dict(d):
        # one joint per entry, fixed width, fixed joint order -> columns align
        inner = ", ".join(f'"{c}": {float(d[c]):8.3f}' for c in codes)
        return "{" + inner + "}"

    def _sensors(sv):
        return "[" + ", ".join(
            "[" + ", ".join(f"{float(a):11.7f}" for a in xyz) + "]"
            for xyz in sv
        ) + "]"

    head = (f'{{"t": {row["t"]!r}, "session": {json.dumps(row["session"])}, '
            f'"run_manus_mode": {int(row["run_manus_mode"])},')
    lines = [
        head,
        f' "sensors":   {_sensors(row["sensors"])},',
        f' "state_deg": {_ang_dict(row["state_deg"])},',
        f' "ik_deg":    {_ang_dict(row["ik_deg"])},',
        f' "trim_deg":  {_ang_dict(row["trim_deg"])}}}',
    ]
    return "\n".join(lines)

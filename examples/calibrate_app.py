#!/usr/bin/env python3
"""
Standalone glove -> Surge-hand IK calibrator (pygame GUI).

This is the BENCH version of the in-situ lerobot calibration: NO arm. The glove
drives the hand through the current IK (altbionics.ik) as a 0th approximation,
you trim each finger with the keyboard until the Surge finger matches your real
finger, and press Enter to capture a (glove sensors -> MEASURED hand angle) pair.
The captured pairs are written in the SAME JSONL format as the lerobot path
(altbionics.calibrate.HandCalibrator), so altbionics.fit_hand_coeffs fits them
unchanged.

Why standalone: trimming to match your finger while ALSO flying the arm around
(the lerobot in-situ version) is hard to do well. Here you rest the hand on the
bench, pose the glove, settle, trim, capture — and watch the commanded-vs-measured
bars to KNOW a joint has settled before you capture (a capture taken mid-motion
pairs a sensor reading with a not-yet-arrived angle -> a bad fit row).

Mode: the hand is driven in ik.IK_REGRESSION_MANUS_MODE (the mode the current
coefficients were fit in), asserted on connect, exactly like teleop.py. So the
0th-approximation IK means something while you trim. The RECORDED target is the
measured AT+GJA angle (true degrees regardless of mode), so a mode-1 collect can
fit mode-0 coefficients (fit_hand_coeffs stamps manus_mode=0). When you later
re-fit to mode 0 and flip IK_REGRESSION_MANUS_MODE, this app collects in mode 0
with no change.

Controls
--------
Per-joint trim up / down (step adjustable with [ and ]):
    a / z  index      s / x  middle     d / c  ring
    f / v  pinky      g / b  thumb-flex h / n  thumb-rotation
    r      CLEAR all trims back to 0 (without capturing)
    i      toggle TRIM-ONLY mode: drive the hand from the trim keys alone with
           the IK 0th-approx off (for the mode-0 bootstrap, or to sweep a joint
           whose IK is off). The glove still supplies the sensor reading at
           capture, so pairs are collected exactly as in IK+trim mode. Also a
           --trim-only start flag.
    ENTER / SPACE  capture the current pose (writes one JSONL record)
           -> trims reset to 0 after capture (HandCalibrator.reset_trim_on_capture)
    [ / ]  trim step size      q / ESC  quit

Requires pygame + the Manus SDK (altbionics.glove) + a connected Surge hand.
Run:  python examples/calibrate_app.py
      python examples/calibrate_app.py --width 1600 --height 1000 --font 26
"""

import argparse
import os
import sys
import time

from altbionics import ik
from altbionics.hand import SurgeHand, SurgeConfig, autodetect_port, list_serial_ports
from altbionics.glove import ManusGlove
from altbionics.calibrate import HandCalibrator, TRIM_KEYS

# Joints to iterate over for control/read (order here is irrelevant: the hand,
# IK and capture are all keyed by code, and get_angles_burst returns a dict).
JOINT_ORDER = list(ik.JOINT_CODES)                      # ("X","T","I","M","R","P")
# DISPLAY order only: fingers first, then the thumb (T, X) at the bottom —
# matching hand_test.py's vertical order. Does NOT affect capture/IK. -anr
DISPLAY_ORDER = ["I", "M", "R", "P", "T", "X"]
JOINT_LABEL = {"I": "Index", "M": "Middle", "R": "Ring",
               "P": "Pinky", "T": "Thumb Flex", "X": "Thumb Rot"}
# A rough display span per joint for the bars (degrees). Measured GJA can exceed
# the nominal mechanical max (e.g. thumb at its stop), so pad the top a little.
BAR_MAX = {"I": 95.0, "M": 95.0, "R": 95.0, "P": 95.0, "T": 100.0, "X": 100.0}

# ---- colors (match hand_test.py) -------------------------------------------
BG   = (18, 20, 24)
FG   = (230, 232, 236)
DIM  = (120, 126, 134)
BAR_BG  = (40, 44, 50)
BAR_CMD = (80, 170, 255)     # commanded (IK + trim) fill
MARK_ACT = (255, 190, 60)    # measured GJA marker
ACCENT = (90, 200, 140)
WARN = (240, 180, 90)
SETTLED = (90, 200, 140)
MOVING = (240, 180, 90)


class CalibApp:
    """Glove -> IK(+trim) -> hand, read back GJA, feed the HandCalibrator, draw."""

    # how still the measured angle must be (deg, max-min over the window) to call
    # a joint "settled" and show green — purely a capture-readiness cue.
    SETTLE_WINDOW = 6          # readings
    SETTLE_EPS = 1.0           # deg span within the window => settled

    def __init__(self, hand, glove, calib, width, height, font_px,
                 trim_only=False):
        self.hand = hand
        self.glove = glove
        self.calib = calib
        # trim-only mode: drive the hand from trim_deg ALONE (IK 0th-approx
        # removed), so you pose each joint by hand. The glove still supplies the
        # sensor reading recorded at capture, so the (sensors -> measured angle)
        # pairs are collected exactly as in IK+trim mode. Meant for the mode-0
        # bootstrap (no good IK yet) or sweeping a joint whose IK is off. -anr
        self.trim_only = bool(trim_only)
        self.W, self.H = width, height
        self.font_px = font_px

        self.cmd_deg = {c: 0.0 for c in JOINT_ORDER}      # IK + trim, last sent
        self.meas_deg = {c: 0.0 for c in JOINT_ORDER}     # last good GJA
        self._meas_hist = {c: [] for c in JOINT_ORDER}    # for settle detection
        self.status = "Pose the glove, trim to match, ENTER or SPACE to capture."
        self.n_captured = 0

        # Hold-to-repeat trim: while a trim key is held, re-apply its step on a
        # timer so the finger glides instead of stepping once per press. The key
        # DOWN applies the first step immediately (so a tap still moves 1 step);
        # after HOLD_DELAY it repeats every HOLD_INTERVAL. -anr
        self._held = {}            # trim-key char -> time it was pressed
        self._next_repeat = {}     # trim-key char -> next time to apply a step
        self.HOLD_DELAY = 0.30     # s before auto-repeat kicks in
        self.HOLD_INTERVAL = 0.045 # s between repeats while held (~22/s)
        self._last_sensors = None

        # rolling tick-rate readout (tick-to-tick period averaged over a window).
        self._tick_hist = []
        self._last_tick_t = None
        self.tick_hz = 0.0

    # -- one control + read tick --------------------------------------------
    def tick(self):
        # measure true tick-to-tick period (incl. draw + fps wait of the prev
        # frame), averaged over the last N ticks -> displayed as tick_hz.
        now = time.time()
        if self._last_tick_t is not None:
            dt = now - self._last_tick_t
            if dt > 0:
                self._tick_hist.append(dt)
                if len(self._tick_hist) > 50:
                    self._tick_hist.pop(0)
                self.tick_hz = len(self._tick_hist) / sum(self._tick_hist)
        self._last_tick_t = now

        sensors = self.glove.read_sensors()               # list[5] (x,y,z) or None
        self._last_sensors = sensors
        self.calib.set_sensors(sensors)                   # feed capture input

        if self.trim_only:
            # command = trim ALONE (no IK). Doesn't need a glove frame to drive
            # the hand; a frame is still needed to CAPTURE (handled in _capture).
            hand_deg = {c: self.calib.trim[c] for c in JOINT_ORDER}
            sent = self.hand.set_fingers(hand_deg)
            self.cmd_deg = {c: float(sent.get(c, hand_deg[c])) for c in JOINT_ORDER}
        elif sensors is not None:
            hand_deg = ik.solve(sensors)                  # 0th-approx command
            for c in JOINT_ORDER:                          # add live trim
                hand_deg[c] = hand_deg[c] + self.calib.trim[c]
            sent = self.hand.set_fingers(hand_deg)         # drive the hand
            self.cmd_deg = {c: float(sent.get(c, hand_deg[c])) for c in JOINT_ORDER}

        # read back the TRUE measured angle (burst: one round-trip for all six).
        vals = self.hand.get_angles_burst(JOINT_ORDER, timeout=0.08)
        meas = {}
        for c in JOINT_ORDER:
            v = vals.get(c)
            # accept only physically-valid reads; else hold last (same rule as
            # the follower's get_observation). 25.5/255/278 glitches fall outside.
            if v is not None and -5.0 <= v <= 130.0:
                self.meas_deg[c] = v
            meas[c] = self.meas_deg[c]
            h = self._meas_hist[c]
            h.append(self.meas_deg[c])
            if len(h) > self.SETTLE_WINDOW:
                h.pop(0)
        # feed the calibrator the measured degrees directly (no obs dict here).
        self.calib.set_measured(meas)

    def is_settled(self, c):
        h = self._meas_hist[c]
        if len(h) < self.SETTLE_WINDOW:
            return False
        return (max(h) - min(h)) <= self.SETTLE_EPS

    def all_settled(self):
        return all(self.is_settled(c) for c in JOINT_ORDER)

    # -- keyboard ------------------------------------------------------------
    def capture(self):
        if self._last_sensors is None:
            self.status = "No glove frame yet — cannot capture."
            return
        before = self.calib._n
        self.calib._capture()                             # writes the row, resets trims
        if self.calib._n > before:
            self.n_captured = self.calib._n
            kept = "trims kept" if not self.calib.reset_trim_on_capture else "trims reset"
            self.status = (f"Captured #{self.n_captured} "
                           f"({kept}) -> {os.path.basename(self.calib.path)}")
        else:
            self.status = "Not captured (no measured state yet?)."

    def reset_trims(self):
        self.calib.trim = {c: 0.0 for c in ik.JOINT_CODES}
        self._held.clear(); self._next_repeat.clear()
        self.status = "Trims cleared to 0."

    def prime_measured(self):
        """One GJA burst read to populate meas_deg WITHOUT commanding the hand.
        Used at startup before seeding trims, so --trim-only can start from the
        hand's actual pose. -anr"""
        try:
            vals = self.hand.get_angles_burst(JOINT_ORDER, timeout=0.08)
            for c in JOINT_ORDER:
                v = vals.get(c)
                if v is not None and -5.0 <= v <= 130.0:
                    self.meas_deg[c] = v
        except Exception:
            pass

    def seed_trims_from_measured(self):
        """Seed each trim with the current MEASURED angle (GJA, true degrees), so
        that on entering trim-only (mode 0, where command == trim in true degrees)
        the first command equals where the finger already is and the hand barely
        moves. -anr"""
        self.calib.trim = {c: float(self.meas_deg[c]) for c in ik.JOINT_CODES}
        self._held.clear(); self._next_repeat.clear()
        self.status = "Trim-only: seeded from current pose (hand holds)."

    def _command_now(self):
        """Compute and SEND the current drive-mode command immediately (not waiting
        for the next tick), from the latest trim/sensors. Called right after a Manus
        mode switch: switching mode makes the hand reinterpret its LAST command in
        the new mode (e.g. a mode-1 thumb command of 26, which meant ~56 deg, becomes
        26 deg in mode 0 -> the thumb drops). Sending the correct new-mode target at
        once, before the hand can act on that reinterpreted command, keeps the hand
        from jumping. -anr"""
        if self.trim_only:
            hand_deg = {c: self.calib.trim[c] for c in JOINT_ORDER}
        else:
            s = self._last_sensors
            if s is None:
                return
            hand_deg = ik.solve(s)
            for c in JOINT_ORDER:
                hand_deg[c] = hand_deg[c] + self.calib.trim[c]
        try:
            self.hand.set_fingers(hand_deg)
        except Exception:
            pass

    # -- drive mode: hand Manus mode + capture-reset policy -------------------
    def apply_drive_mode(self):
        """Apply everything that follows from the current DRIVE mode (trim-only vs
        IK+trim):

        1. Hand Manus mode:
           * trim-only -> mode 0 (direct/true-degrees), so keyboard commands are
             linear true degrees, not glove-retargeted (mode-1 is nonlinear, esp.
             the thumb), and the command floors switch to the mode-0 set.
           * IK + trim -> ik.IK_REGRESSION_MANUS_MODE, so IK commands match the mode
             their coefficients were fit in.
           Only switches when the mode actually changes (no needless hand twitch);
           in the mode-0 bootstrap (IK_REGRESSION_MANUS_MODE==0) both drive modes
           are mode 0 and nothing switches.
        2. run_manus_mode recorded in captured rows = the actual drive mode.
        3. Capture-reset policy: in IK+trim the trim is a CORRECTION delta, so it
           resets to 0 after each capture (no stale carry-over). In trim-only the
           trim IS the absolute commanded pose, so it is KEPT after capture (don't
           snap the hand open between poses). -anr"""
        target = 0 if self.trim_only else int(ik.IK_REGRESSION_MANUS_MODE)
        if getattr(self.hand, "_manus_mode", None) != target:
            self.hand.set_manus_mode(target == 1)
            # Brief settle for the hand to process AT+SMC, then IMMEDIATELY command
            # the correct NEW-mode target so it doesn't dwell on its last command
            # reinterpreted in the new mode (which is what jumped the thumb). The
            # trim must already hold the intended pose here: the caller seeds it
            # (entering trim-only) or clears it (leaving) BEFORE calling this. -anr
            time.sleep(0.02)
            self._command_now()
            try:
                self.hand.flush_input()
            except Exception:
                pass
        self.calib.run_manus_mode = target
        # keep trims after capture in trim-only; reset them in IK+trim.
        self.calib.reset_trim_on_capture = (not self.trim_only)

    # -- hold-to-repeat trim --------------------------------------------------
    def _apply_trim(self, ch):
        """Apply one trim step for a trim-key char via the calibrator's own
        on_key, so there is ONE trim implementation (step size, joint map). -anr"""
        if ch not in TRIM_KEYS:
            return
        code, _ = TRIM_KEYS[ch]
        class _K:
            char = None; name = None
        k = _K(); k.char = ch
        self.calib.on_key(k)
        self.status = f"{JOINT_LABEL[code]} trim {self.calib.trim[code]:+.1f}"

    def key_down_trim(self, ch):
        """A trim key was pressed: apply one step now and arm auto-repeat."""
        self._apply_trim(ch)
        now = time.time()
        self._held[ch] = now
        self._next_repeat[ch] = now + self.HOLD_DELAY

    def key_up_trim(self, ch):
        self._held.pop(ch, None)
        self._next_repeat.pop(ch, None)

    def repeat_held(self):
        """Called each frame: for every still-held trim key, apply a step when its
        repeat timer is due. Gives smooth hold-to-glide without OS key-repeat."""
        now = time.time()
        for ch in list(self._held):
            if now >= self._next_repeat.get(ch, now):
                self._apply_trim(ch)
                self._next_repeat[ch] = now + self.HOLD_INTERVAL

    # -- rendering -----------------------------------------------------------
    def draw(self, pg, screen, font, small, tiny):
        screen.fill(BG)
        screen.blit(font.render("Surge Hand — Glove IK Calibration (bench)",
                                True, FG), (24, 16))
        # show the hand's ACTUAL Manus mode (trim-only switches it to 0), not the
        # IK's fitted-mode constant. -anr
        mode = getattr(self.hand, "_manus_mode", ik.IK_REGRESSION_MANUS_MODE)
        drive = "TRIM-ONLY (IK off)" if self.trim_only else "IK + trim"
        sub = (f"Manus mode {mode} ({'glove/retargeted' if mode else 'direct/true-deg'})"
               f"   drive: {drive}"
               f"   trim step {self.calib.step_deg:.0f} deg"
               f"   {self.tick_hz:4.1f} Hz"
               f"   captured {self.n_captured}"
               f"   file {os.path.basename(self.calib.path)}")
        # highlight the subtitle when in trim-only so the mode is unmistakable
        screen.blit(small.render(sub, True, WARN if self.trim_only else DIM),
                    (24, 16 + self.font_px + 6))

        # Three columns, laid out so they CANNOT overlap at any window size:
        #   [ bars ]  [ cmd/act/trim readout ]  [ key legend ]
        # The readout column width is measured from a sample string; the legend
        # starts a fixed gap to its right. If that would run past the window, the
        # bars shrink so everything still fits. -anr
        x0 = 24
        y0 = 16 + self.font_px + 6 + small.get_height() + 20
        bar_h = max(26, int(self.font_px * 1.3))
        gap = max(16, int(bar_h * 0.6))

        readout_sample = "cmd 123.4   act 123.4   trim +12.0"
        readout_w = small.size(readout_sample)[0]
        legend_w = max(tiny.size(s)[0] for s in (
            "i            IK on/off (trim-only)", "ENTER/SPACE  capture pose",
            "r            clear trims (to 0)", "green label = settled"))
        col_gap = 28
        # width available for the bars = window - margins - readout - legend - gaps
        bar_w = self.W - x0 - readout_w - legend_w - 3 * col_gap - x0
        bar_w = max(240, min(bar_w, int(self.W * 0.52)))  # sane bounds

        readout_x = x0 + bar_w + col_gap
        legend_x = readout_x + readout_w + col_gap

        for i, c in enumerate(DISPLAY_ORDER):
            y = y0 + i * (bar_h + gap)
            hi = BAR_MAX[c]
            cmd = self.cmd_deg[c]; act = self.meas_deg[c]
            settled = self.is_settled(c)
            lab_col = SETTLED if settled else MOVING
            screen.blit(small.render(
                f"{JOINT_LABEL[c]:<11}", True, lab_col), (x0, y - small.get_height() - 2))
            pg.draw.rect(screen, BAR_BG, (x0, y, bar_w, bar_h), border_radius=6)
            frac = max(0.0, min(1.0, cmd / hi))
            pg.draw.rect(screen, BAR_CMD, (x0, y, int(bar_w * frac), bar_h),
                         border_radius=6)
            af = max(0.0, min(1.0, act / hi))
            mx = x0 + int(bar_w * af)
            pg.draw.rect(screen, MARK_ACT, (mx - 2, y - 4, 4, bar_h + 8))
            txt = f"cmd {cmd:5.1f}   act {act:5.1f}   trim {self.calib.trim[c]:+5.1f}"
            screen.blit(small.render(txt, True, DIM), (readout_x, y + 4))

        # legend (its own column, to the right of the readout)
        ly = y0 - small.get_height() - 2
        legend = [
            "TRIM KEYS (hold to repeat)", "",
            "a/z  index", "s/x  middle", "d/c  ring",
            "f/v  pinky", "g/b  thumb flex", "h/n  thumb rot", "",
            "ENTER/SPACE  capture pose", "r            clear trims (to 0)",
            "i            IK on/off (trim-only)",
            "[ ]    trim step", "q/ESC  quit", "",
            "green label = settled",
        ]
        bars_bottom = y0 + len(DISPLAY_ORDER) * (bar_h + gap)
        # if somehow still too tight, drop the legend below the bars
        if legend_x + legend_w > self.W:
            legend_x, ly = x0, bars_bottom + 20
        for j, line in enumerate(legend):
            col = FG if line.startswith("TRIM KEYS") else DIM
            screen.blit(tiny.render(line, True, col),
                        (legend_x, ly + j * (tiny.get_height() + 4)))

        # status + capture-ready banner
        ready = self.all_settled() and self._last_sensors is not None
        banner = "SETTLED — ready to capture (ENTER or SPACE)" if ready else "settling…"
        screen.blit(small.render(banner, True, SETTLED if ready else WARN),
                    (24, self.H - 2 * small.get_height() - 16))
        screen.blit(small.render(self.status, True, ACCENT),
                    (24, self.H - small.get_height() - 10))


def _key_to_calib(pg, ev):
    """Translate a pygame key event into the duck-typed object HandCalibrator.on_key
    expects (has .char for letters, .name=='enter' for Enter). Returns None for keys
    the calibrator doesn't use."""
    name = pg.key.name(ev.key)
    class _K:  # minimal duck for HandCalibrator.on_key
        char = None
        name = None
    k = _K()
    if ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
        k.name = "enter"
        return k
    if len(name) == 1 and name.isalpha():
        k.char = name
        return k
    return None


def _redirect_native_logs(logfile_path):
    """Send the Manus Core SDK's native stdout/stderr (its periodic 'Datarate...
    data discarded' heartbeat, printed from the C library below Python's logging)
    to a file, while keeping Python's own print()/traceback on the real terminal.

    Mechanism: dup the real terminal fds, point OS fds 1 & 2 at the log file (so
    the C library's writes land there), then rebind sys.stdout/sys.stderr to the
    saved terminal fds (so our [calib] prints and Python tracebacks still show).
    Returns a restore() callable; benign no-op if anything is unsupported. -anr"""
    try:
        saved_out_fd = os.dup(1)
        saved_err_fd = os.dup(2)
        logf = open(logfile_path, "a", buffering=1)
        os.dup2(logf.fileno(), 1)
        os.dup2(logf.fileno(), 2)
        term_out = os.fdopen(saved_out_fd, "w", buffering=1)
        term_err = os.fdopen(saved_err_fd, "w", buffering=1)
        sys.stdout = term_out
        sys.stderr = term_err

        def restore():
            try:
                # put the OS fds back on the terminal so native writes return too,
                # and keep Python streams on the saved terminal streams (NOT
                # sys.__stdout__, which can be None under some launchers). -anr
                os.dup2(saved_out_fd, 1)
                os.dup2(saved_err_fd, 2)
                sys.stdout = term_out
                sys.stderr = term_err
                logf.close()
            except Exception:
                pass
        return restore
    except Exception as e:
        print(f"(could not redirect native glove logs: {e})")
        return lambda: None


def main():
    ap = argparse.ArgumentParser(description="Standalone glove->Surge IK calibrator")
    ap.add_argument("port", nargs="?", default=None, help="Surge serial port")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--so", default=None, help="path to libManusSDK_Integrated.so")
    ap.add_argument("--width", type=int, default=1700, help="window width px (default 1700)")
    ap.add_argument("--height", type=int, default=1050, help="window height px (default 1050)")
    ap.add_argument("--font", type=int, default=24, help="base font px (default 24)")
    ap.add_argument("--rate", type=float, default=30.0, help="control/draw rate Hz (default 30)")
    ap.add_argument("--step", type=float, default=1.0, help="trim step deg (default 1)")
    ap.add_argument("--trim-only", action="store_true",
                    help="start in trim-only mode: drive the hand from trim keys "
                         "alone with the IK 0th-approx off (for the mode-0 bootstrap). "
                         "Toggle live with the 'i' key.")
    ap.add_argument("--data-dir", default=None,
                    help="where pairs_*.jsonl is written "
                         "(default: examples/hand_calibrate next to this script)")
    ap.add_argument("--show-glove-log", action="store_true",
                    help="show the Manus SDK's native datarate/heartbeat lines on "
                         "the terminal (by default they're redirected to a log file)")
    args = ap.parse_args()

    # Default data dir = examples/hand_calibrate, resolved next to THIS script so
    # it's the same folder no matter the working directory. Overridable with
    # --data-dir. -anr
    default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "hand_calibrate")

    ports = list_serial_ports()
    print("Visible serial ports:", ports or "(none)")
    if args.list:
        return

    # --- hand (required) ----------------------------------------------------
    port = args.port or autodetect_port()
    if port is None:
        print("No Surge serial port found. Is the hand plugged in?")
        return
    # Direct control: do NOT auto-send AT+SMC=0; we assert the IK's mode below.
    hcfg = SurgeConfig(port=port, disable_manus_on_connect=False,
                       send_smc_on_connect=False,
                       log_commands=False, log_incoming=False)
    hand = SurgeHand(config=hcfg)
    hand.connect()
    # Assert the mode the current coefficients were fit in (mirror of teleop.py).
    hand.set_manus_mode(ik.IK_REGRESSION_MANUS_MODE == 1)
    time.sleep(0.3); hand.flush_input()
    print(f"Hand connected on {port}; Manus mode {ik.IK_REGRESSION_MANUS_MODE}.")

    # Resolve the data dir now so the native-log file can live beside the pairs.
    data_dir = args.data_dir or default_data_dir
    os.makedirs(data_dir, exist_ok=True)

    # Redirect the Manus SDK's native heartbeat off the terminal (to a log file)
    # BEFORE starting the glove, so its periodic datarate lines don't clutter the
    # console. Python's own prints stay on the terminal. --show-glove-log keeps it
    # visible. restore_logs() is called in finally. -anr
    restore_logs = (lambda: None)
    if not args.show_glove_log:
        glove_log = os.path.join(data_dir, "manus_core.log")
        restore_logs = _redirect_native_logs(glove_log)
        print(f"Manus SDK heartbeat -> {glove_log} (use --show-glove-log to see it)")

    # --- glove (required) ---------------------------------------------------
    glove = ManusGlove(so_path=args.so) if args.so else ManusGlove()
    glove.start()
    print("Glove started. Waiting for frames (move fingers)...")

    # --- calibrator (same class/format as the lerobot path) -----------------
    calib = HandCalibrator(step_deg=args.step, reset_trim_on_capture=True,
                           data_dir=data_dir)
    print(f"Logging captures to {calib.path}")

    try:
        import pygame as pg
    except ImportError:
        print("pygame is required: pip install pygame")
        glove.stop(); hand.disconnect(); restore_logs()
        return

    try:
        pg.init()
        screen = pg.display.set_mode((args.width, args.height), pg.RESIZABLE)
        pg.display.set_caption("Surge Glove IK Calibration")
        font  = pg.font.SysFont("menlo,consolas,monospace", int(args.font * 1.2))
        small = pg.font.SysFont("menlo,consolas,monospace", args.font)
        tiny  = pg.font.SysFont("menlo,consolas,monospace", max(14, int(args.font * 0.8)))
        clock = pg.time.Clock()

        app = CalibApp(hand, glove, calib, args.width, args.height, args.font,
                       trim_only=args.trim_only)
        # put the hand in the mode the initial drive mode needs (so --trim-only
        # starts in mode 0; otherwise a no-op — connect already set the IK mode),
        # set the capture-reset policy, and record the run mode for captures.
        # For --trim-only, read + seed the pose FIRST (while still in connect's
        # mode) so apply_drive_mode holds it across the switch. -anr
        if app.trim_only:
            app.prime_measured()
            app.seed_trims_from_measured()
        app.apply_drive_mode()

        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.VIDEORESIZE:
                    app.W, app.H = ev.w, ev.h
                    screen = pg.display.set_mode((ev.w, ev.h), pg.RESIZABLE)
                elif ev.type == pg.KEYDOWN:
                    if ev.key in (pg.K_q, pg.K_ESCAPE):
                        running = False
                        continue
                    if ev.key == pg.K_LEFTBRACKET:
                        calib.step_deg = max(0.5, calib.step_deg - 0.5)
                        app.status = f"trim step {calib.step_deg:.1f} deg"
                        continue
                    if ev.key == pg.K_RIGHTBRACKET:
                        calib.step_deg = min(20.0, calib.step_deg + 0.5)
                        app.status = f"trim step {calib.step_deg:.1f} deg"
                        continue
                    if ev.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
                        # ENTER or SPACE captures — SPACE is free in this app (no
                        # start-gate like lerobot) and is easier to reach. -anr
                        app.capture()           # capture() updates status/counter
                        continue
                    name = pg.key.name(ev.key)
                    if name in TRIM_KEYS:
                        # trim key: first step now, then hold-to-repeat (app owns
                        # the repeat, which calls the SAME calib.on_key path). -anr
                        app.key_down_trim(name)
                        continue
                    if name == "r":
                        app.reset_trims()
                        continue
                    if name == "i":
                        # toggle trim-only (IK off). Set the trim to the pose we want
                        # to HOLD *before* switching mode, so apply_drive_mode's
                        # command-now lands the right target the instant the mode
                        # flips (the switch makes the hand reinterpret its last
                        # command, which otherwise jumps the thumb):
                        #   entering -> read the TRUE pose NOW (still in the old mode)
                        #               and seed trims from it;
                        #   leaving  -> clear trims so pure IK resumes.
                        # apply_drive_mode then switches the Manus mode, commands the
                        # held pose at once, and sets the capture-reset policy. -anr
                        app.trim_only = not app.trim_only
                        if app.trim_only:
                            app.prime_measured()            # read BEFORE the switch
                            app.seed_trims_from_measured()
                        else:
                            app.reset_trims()
                        app.apply_drive_mode()
                        m = 0 if app.trim_only else int(ik.IK_REGRESSION_MANUS_MODE)
                        app.status = (f"TRIM-ONLY (IK off, Manus {m}) — seeded from pose"
                                      if app.trim_only else f"IK + trim mode (Manus {m})")
                        continue
                elif ev.type == pg.KEYUP:
                    name = pg.key.name(ev.key)
                    if name in TRIM_KEYS:
                        app.key_up_trim(name)
            app.repeat_held()               # apply held-key trim steps on their timer
            app.tick()
            app.draw(pg, screen, font, small, tiny)
            pg.display.flip()
            clock.tick(int(args.rate) if args.rate > 0 else 30)

        # park the hand open on exit
        try:
            hand.open_all()
        except Exception:
            pass
        pg.quit()
    finally:
        glove.stop()
        hand.disconnect()
        restore_logs()      # put stdout/stderr back on the terminal
    print(f"Done. {calib._n} pairs in {calib.path}")


if __name__ == "__main__":
    main()

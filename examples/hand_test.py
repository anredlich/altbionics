#!/usr/bin/env python3
"""
Surge hand test — pygame GUI for live keyboard control and state observation.

The window opens immediately. On launch it enters a STARTUP-CAPTURE phase: it
listens (non-blocking) for the hand's power-up / calibration messages and shows
them in the window. Power-cycle the hand to see the sequence, or press S to skip
straight to live control (useful if the hand is already running).

After startup: one bar per joint shows the commanded angle (blue fill) and the
hand's actual polled angle (orange marker). A key legend is on the right, and a
panel is reserved for the Sensible Robotics tactile sensors (added later).

Runs from a terminal OR the VS Code debugger (pygame handles key capture).
Requires pygame:  pip install pygame  (already in environment.yml).

Controls
--------
Startup phase:   C continue to control  (power-cycle the hand to capture startup)
Per-joint flex / extend (step adjustable with [ and ]):
    a / z  index      s / x  middle     d / c  ring
    f / v  pinky      g / b  thumb-flex h / n  thumb-rotation
Whole hand:  SPACE open all   ENTER close fingers
Preset poses (0-6): open power chuck fine_pinch key hook birds_up
Other:  p toggle actual-angle polling   r recalibrate   [ / ] step   q/ESC quit
        k toggle Manus mode 0/1 (0=direct/true-deg, 1=glove/retargeted)
"""

import argparse
import os
import sys
import time

from altbionics import (
    SurgeHand, SurgeConfig, FINGER_CODES, ANGLE_LIMITS,
    list_serial_ports, autodetect_port,
)

# Sensor panel lives alongside this script in examples/. Import it so we can
# show tactile heatmaps; if it (or its mido dep) is unavailable, the app still
# runs hand-only.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from sensor_panel import SensorPanel
except Exception:
    SensorPanel = None

KEYMAP = {
    "a": ("I", +1), "z": ("I", -1),
    "s": ("M", +1), "x": ("M", -1),
    "d": ("R", +1), "c": ("R", -1),
    "f": ("P", +1), "v": ("P", -1),
    "g": ("T", +1), "b": ("T", -1),
    "h": ("X", +1), "n": ("X", -1),
}
JOINT_ORDER = ["I", "M", "R", "P", "T", "X"]
JOINT_LABEL = {
    "I": "Index", "M": "Middle", "R": "Ring",
    "P": "Pinky", "T": "Thumb Flex", "X": "Thumb Rot",
}

# ---- window layout ----------------------------------------------------------
W, H = 1200, 760
BG = (18, 20, 24)
FG = (230, 232, 236)
DIM = (120, 126, 134)
BAR_BG = (40, 44, 50)
BAR_CMD = (80, 170, 255)
MARK_ACT = (255, 190, 60)
PANEL = (28, 31, 36)
ACCENT = (90, 200, 140)
WARN = (240, 180, 90)

# Set True to print per-read timing + the ENTER-CONTROL wire-state diagnostic
# used while tuning the GJA read path. Harmless to leave False in normal use.
DIAGNOSTICS = True

def _clamp(code, val):
    lo, hi = ANGLE_LIMITS[code]
    return max(lo, min(hi, val))


class TeleopApp:
    # phases
    STARTUP = "startup"
    CONTROL = "control"

    # polling pacing (seconds): gentle so it never floods the serial line
    POLL_INTERVAL = 0.05           # min time between actual-angle polls
    POLL_QUIET_AFTER_CMD = 0.0   # don't poll within this long after a command

    # startup joint-init nudge (workaround for phantom 25.5 readback)
    NUDGE_DEG = 10.0              # small angle to move each joint to and back
    NUDGE_HOLD = 0.25            # seconds to hold each nudge step

    def __init__(self, hand, step: float = 10.0, poll: bool = True,
                 wait_startup: bool = True, nudge_on_start: bool = True,
                 panel=None, smooth: bool = False, speed: float = 30.0,
                 rate: float = 30.0):
        self.hand = hand                     # may be None (sensors-only mode)
        self.panel = panel                   # SensorPanel or None
        self.step = step
        self.poll = poll
        self.nudge_on_start = nudge_on_start
        # smooth velocity control: hold a key to glide a joint at `speed` deg/sec
        self.smooth = smooth
        self.speed = speed                   # deg/sec (smooth mode)
        self.rate = rate                     # control/render loop rate (Hz)
        self.held = {}                       # code -> direction for keys held down
        self._last_vel = time.time()
        self.cmd = {c: float(ANGLE_LIMITS[c][0]) for c in JOINT_ORDER}
        self.actual = {c: 0.0 for c in JOINT_ORDER}   # show 0 until moved+polled
        self.moved = set()                              # joints commanded at least once
        self._poll_i = 0
        self.status = "Ready."
        self.manus_mode = 0            # current Manus mode (0=direct, 1=glove).
                                       # finish_startup() forces 0; 'k' toggles.
        self.sensor_frame = None
        self.connected_ok = True
        self._last_poll = 0.0
        self._last_cmd = 0.0
        self._last_joint = None
        self.burst = False            # False = round-robin poll_one; True = poll_all_burst
        self._burst_i = 0             # batch counter (diagnostics)

        # startup-capture state (only meaningful when a hand is present)
        self.phase = self.STARTUP if (wait_startup and hand is not None) else self.CONTROL
        self.startup_lines: list[str] = []
        self.startup_info = {"calibration_passed": False, "hand_version": None,
                             "firmware_version": None}
        self._startup_start = time.time()
        self.startup_timeout = 60.0
        if self.phase == self.CONTROL and self.hand is not None:
            self.hand.open_all()
            # settle + flush so the first polls don't latch open-command acks
            time.sleep(0.3)
            try:
                self.hand.flush_input()
            except Exception:
                pass
            self._last_cmd = time.time()
            self._last_poll = time.time()

    # -- startup capture (called each frame while in STARTUP phase) -----------
    def pump_startup(self):
        """Read any available startup lines without blocking the GUI."""
        # read a few lines per frame if present
        for _ in range(20):
            line = self.hand.read_line(timeout=0.0)
            if not line:
                break
            self.startup_lines.append(line)
            low = line.lower()
            if "hand version" in low and ":" in line:
                self.startup_info["hand_version"] = line.split(":", 1)[1].strip()
            elif "firmware version" in low and ":" in line:
                self.startup_info["firmware_version"] = line.split(":", 1)[1].strip()
            if "calibration passed" in low:
                self.startup_info["calibration_passed"] = True
            if "startup sequence complete" in low:
                self.finish_startup("Startup complete.")
                return
        if time.time() - self._startup_start > self.startup_timeout:
            self.finish_startup("Startup wait timed out — entering control.")

    def finish_startup(self, msg="Entering control."):
        self.phase = self.CONTROL
        self.status = msg
        # Disable Manus mode so AT+SJA maps to true degrees (hand boots into
        # Manus mode). The hand moves slightly as the mode switches.
        try:
            self.hand.set_manus_mode(False)
            self.manus_mode = 0
            time.sleep(0.3)
            self.hand.flush_input()
        except Exception:
            pass
        self.hand.open_all()
        # The six open commands leave a burst of OK acks draining on the line.
        # Let it settle and flush so later reads are clean. We do NOT nudge or
        # poll here: joints show 0 until you move them (moving a joint also gives
        # it its first command, which clears the firmware's phantom ~25.5 read).
        time.sleep(0.3)
        try:
            self.hand.flush_input()
        except Exception:
            pass
        self._last_cmd = time.time()
        self._last_poll = time.time()
        if DIAGNOSTICS:
            # show anything still on the wire entering CONTROL (read-path tuning).
            time.sleep(0.05)
            leftover = self.hand.read_available()
            print(f"[ENTER CONTROL] leftover on wire: {leftover!r}", flush=True)

    def nudge_init(self):
        """
        Briefly command each joint to a small angle and back to 0, one at a
        time. On firmware v1.4 some joints (M, P, T) report a phantom ~25.5 deg
        via AT+GJA? until they receive their first movement command; nudging
        initializes their readback so the actual-angle display is correct from
        the start. Motion is small (to NUDGE_DEG then back to 0). This is a
        workaround for a hand-side readback-init quirk (reported to Alt
        Bionics); disable with --no-nudge if you prefer no startup motion.
        """
        self.status = "Initializing joints (nudge)..."
        for code in JOINT_ORDER:
            try:
                self.hand.set_finger(code, self.NUDGE_DEG)
                time.sleep(self.NUDGE_HOLD)
                self.hand.set_finger(code, 0)
                time.sleep(self.NUDGE_HOLD)
                self.cmd[code] = 0.0
            except Exception:
                self.connected_ok = False
                self.status = "Nudge failed — connection problem?"
                return
        try:
            self.hand.flush_input()
        except Exception:
            pass
        self.status = "Joints initialized."

    # -- hand actions (all no-op if no hand connected) ------------------------
    def move(self, code, direction):
        if self.hand is None:
            return
        self.cmd[code] = _clamp(code, self.cmd[code] + direction * self.step)
        # command + immediate true-angle readback (SJA then GJA), like real control
        actual = self.hand.set_finger_read(code, self.cmd[code], timeout=0.1)
        if actual is not None:
            self.actual[code] = actual
        self._last_cmd = time.time()
        self._last_joint = code
        self.moved.add(code)
        self.status = f"{JOINT_LABEL[code]} -> {self.cmd[code]:.0f} deg"

    def open_all(self):
        if self.hand is None:
            return
        self.hand.open_all()
        for c in JOINT_ORDER:
            self.cmd[c] = float(ANGLE_LIMITS[c][0])
        self.moved.update(JOINT_ORDER)   # all joints commanded now
        # clear the batch-command acks so polling reads stay aligned
        time.sleep(0.15)
        try:
            self.hand.flush_input()
        except Exception:
            pass
        self._last_joint = None   # let polling re-cycle all joints
        self._last_cmd = time.time()
        self.status = "Open all"

    def close_fingers(self):
        if self.hand is None:
            return
        self.hand.close_all()
        for c in ("I", "M", "R", "P"):
            self.cmd[c] = float(ANGLE_LIMITS[c][1])
        self.moved.update(("I", "M", "R", "P"))
        # clear the batch-command acks so polling reads stay aligned
        time.sleep(0.15)
        try:
            self.hand.flush_input()
        except Exception:
            pass
        self._last_joint = None   # let polling re-cycle all joints        
        self._last_cmd = time.time()
        self.status = "Close fingers"

    def do_pose(self, n):
        if self.hand is None:
            return
        self.hand.pose(n)
        self.moved.update(JOINT_ORDER)   # poses command all joints
        # clear the batch-command acks so polling reads stay aligned
        time.sleep(0.15)
        try:
            self.hand.flush_input()
        except Exception:
            pass
        self._last_joint = None   # let polling re-cycle all joints        
        self._last_cmd = time.time()
        self.status = f"Pose {n}"

    def recalibrate(self):
        if self.hand is None:
            return
        # re-enter startup capture to show the recalibration sequence
        self.hand._send_raw("AT+RCH=1")
        self.phase = self.STARTUP
        self.startup_lines = []
        self.startup_info = {"calibration_passed": False, "hand_version": None,
                             "firmware_version": None}
        self._startup_start = time.time()
        self.status = "Recalibrating..."

    def toggle_manus_mode(self):
        """
        Flip the hand's Manus control mode (AT+SMC=1/0) live.

        mode 0 = direct: AT+SJA angles are true degrees (what this GUI assumes).
        mode 1 = glove : the hand boots into this; AT+SJA is retargeted.

        Used to probe whether AT+GJA? reports the TRUE physical angle in mode 1
        (the yellow marker) or something mode-dependent: toggle to mode 1, move a
        joint (blue), and watch whether the yellow 'act' readout tracks the real
        finger position. The hand moves slightly as the mode switches.
        """
        if self.hand is None:
            return
        new_mode = 0 if self.manus_mode == 1 else 1
        try:
            self.hand.set_manus_mode(new_mode == 1)
            self.manus_mode = new_mode
            time.sleep(0.2)
            self.hand.flush_input()
            self._last_cmd = time.time()
            self.status = (f"Manus mode {new_mode} "
                           f"({'glove/retargeted' if new_mode else 'direct/true-deg'})")
        except Exception:
            self.connected_ok = False
            self.status = "Mode switch failed — connection problem?"

    # -- sensor tare (delegates to the panel) ---------------------------------
    def tare_sensors(self):
        if self.panel is not None:
            self.panel.tare_all()
            self.status = "Sensors tared."   # completes within a few frames

    def clear_tare_sensors(self):
        if self.panel is not None:
            self.panel.clear_tare_all()
            self.status = "Sensor tare cleared."

    def poll_one(self):
        if self.hand is None or not self.poll or self.phase != self.CONTROL:
            return
        if not self.moved:
            return                      # nothing moved yet; leave displays at 0
        now = time.time()
        if now - self._last_cmd < self.POLL_QUIET_AFTER_CMD:
            return
        if now - self._last_poll < self.POLL_INTERVAL:
            return
        self._last_poll = now
        # Poll among joints that have been moved (so we never read a joint's
        # phantom pre-move value). Prefer the last-moved joint for responsiveness,
        # else cycle through the moved set.
        moved = [c for c in JOINT_ORDER if c in self.moved]
        self._poll_i += 1
        if self._last_joint in self.moved and (self._poll_i % 3 != 0):
            code = self._last_joint
        else:
            code = moved[self._poll_i % len(moved)]
        try:
            if DIAGNOSTICS:
                import time as _t
                _t0 = _t.perf_counter()
                val = self.hand.get_angle_fast(code, timeout=0.05)
                _call_ms = (_t.perf_counter() - _t0) * 1000.0
                _gap_ms = (_t0 - getattr(self, "_last_poll_perf", _t0)) * 1000.0
                self._last_poll_perf = _t.perf_counter()
                if val is not None:
                    self.actual[code] = val
                print(f"code: {code}  val: {val}  call: {_call_ms:5.1f} ms"
                      f"  gap_since_prev: {_gap_ms:6.1f} ms", flush=True)
            else:
                val = self.hand.get_angle_fast(code, timeout=0.05)
                if val is not None:
                    self.actual[code] = val
        except Exception:
            self.connected_ok = False
            self.status = "Serial read problem — is the hand connected/powered?"

    def poll_all_burst(self):
        """Burst-read ALL six joints in one pipelined call (step-2 test path).
        Prints each joint's value + the whole-batch timing, so we can see
        whether this firmware pipelines cleanly and how often a batch aborts."""
        if self.hand is None or not self.poll or self.phase != self.CONTROL:
            return
        if not self.moved:
            return
        now = time.time()
        if now - self._last_poll < self.POLL_INTERVAL:
            return
        self._last_poll = now
        import time as _t
        _t0 = _t.perf_counter()
        vals = self.hand.get_angles_burst(JOINT_ORDER, timeout=0.08)
        _batch_ms = (_t.perf_counter() - _t0) * 1000.0
        _gap_ms = (_t0 - getattr(self, "_last_poll_perf", _t0)) * 1000.0
        self._last_poll_perf = _t.perf_counter()
        for c in JOINT_ORDER:
            v = vals.get(c)
            if v is not None:
                self.actual[c] = v
        if DIAGNOSTICS:
            self._burst_i += 1
            n_none = sum(1 for c in JOINT_ORDER if vals.get(c) is None)
            shown = "  ".join(f"{c}:{(f'{vals[c]:5.1f}' if vals[c] is not None else ' None')}"
                            for c in JOINT_ORDER)
            print(f"BURST#{self._burst_i:<5d} {shown}   batch: {_batch_ms:5.1f} ms"
                f"  nNone: {n_none}  gap: {_gap_ms:6.1f} ms", flush=True)
        
    def poll_all_once(self):
        """Read every joint's actual angle once (used at startup so the display
        is populated immediately, not blank until the first keypress)."""
        if self.hand is None:
            return
        for code in JOINT_ORDER:
            try:
                val = self.hand.get_angle_fast(code, timeout=0.1)
                if val is not None:
                    self.actual[code] = val
            except Exception:
                pass

    # -- smooth velocity control ----------------------------------------------
    def on_key_up(self, code):
        """Stop gliding a joint when its movement key is released."""
        self.held.pop(code, None)

    def update_velocity(self):
        """
        In smooth mode, advance each held joint by speed*dt this frame and send
        the command. Executed once per frame; uses real elapsed dt so the deg/sec
        speed is correct regardless of frame rate.
        """
        now = time.time()
        dt = now - self._last_vel
        self._last_vel = now
        if not self.smooth or self.hand is None or self.phase != self.CONTROL:
            return
        if not self.held:
            return
        delta = self.speed * dt
        for code, direction in list(self.held.items()):
            self.cmd[code] = _clamp(code, self.cmd[code] + direction * delta)
            try:
                self.hand.set_finger(code, self.cmd[code])
            except Exception:
                self.connected_ok = False
                self.status = "Serial write problem — hand connected?"
                return
            self._last_joint = code
            self.moved.add(code)
        self._last_cmd = now
        moving = ", ".join(JOINT_LABEL[c] for c in self.held)
        self.status = f"moving {moving} @ {self.speed:.0f} deg/s"

    # -- rendering ------------------------------------------------------------
    def draw(self, pg, screen, font, small):
        screen.fill(BG)
        screen.blit(font.render("Surge Hand — Live Control", True, FG), (24, 18))

        if self.phase == self.STARTUP:
            self._draw_startup(pg, screen, font, small)
        else:
            self._draw_control(pg, screen, font, small)

        screen.blit(small.render(self.status, True, ACCENT), (24, H - 34))

    def _draw_startup(self, pg, screen, font, small):
        screen.blit(small.render(
            "STARTUP CAPTURE — power-cycle the hand to see calibration, "
            "or press S to skip to control.", True, WARN), (24, 52))
        # scrolling transcript
        x0, y0 = 24, 92
        maxlines = 26
        view = self.startup_lines[-maxlines:]
        for i, line in enumerate(view):
            screen.blit(small.render(line[:120], True, DIM), (x0, y0 + i * 22))
        # summary so far
        info = self.startup_info
        sy = y0 + maxlines * 22 + 10
        screen.blit(small.render(
            f"calibration passed: {info['calibration_passed']}    "
            f"hand: {info['hand_version']}    fw: {info['firmware_version']}",
            True, FG), (x0, sy))

    def _draw_control(self, pg, screen, font, small):
        has_hand = self.hand is not None
        mode = []
        if has_hand:
            if self.smooth:
                mode.append(f"SMOOTH hold-to-move @ {self.speed:.0f} deg/s "
                            f"({self.rate:.0f} Hz)   "
                            f"polling {'ON' if self.poll else 'off'}")
            else:
                mode.append(f"step {self.step:.0f} deg   polling {'ON' if self.poll else 'off'}")
            mode.append(f"Manus mode {self.manus_mode} "
                        f"({'glove' if self.manus_mode else 'direct'})")
        else:
            mode.append("sensors-only (no hand connected)")
        screen.blit(small.render("   ".join(mode), True, DIM), (24, 52))

        x0, y0 = 24, 92
        bar_w, bar_h, gap = 520, 34, 20   # wider bars; sensors now render below

        if has_hand:
            for i, code in enumerate(JOINT_ORDER):
                y = y0 + i * (bar_h + gap)
                lo, hi = ANGLE_LIMITS[code]
                frac = (self.cmd[code] - lo) / (hi - lo) if hi > lo else 0
                screen.blit(small.render(JOINT_LABEL[code], True, FG), (x0, y - 20))
                pg.draw.rect(screen, BAR_BG, (x0, y, bar_w, bar_h), border_radius=6)
                pg.draw.rect(screen, BAR_CMD,
                             (x0, y, int(bar_w * frac), bar_h), border_radius=6)
                act = self.actual[code]
                if act is not None:
                    af = (act - lo) / (hi - lo) if hi > lo else 0
                    mx = x0 + int(bar_w * max(0, min(1, af)))
                    pg.draw.rect(screen, MARK_ACT, (mx - 2, y - 4, 4, bar_h + 8))
                txt = f"{self.cmd[code]:5.0f}"
                if act is not None:
                    txt += f" (act {act:4.0f})"
                screen.blit(small.render(txt, True, DIM), (x0 + bar_w + 12, y + 8))
        else:
            screen.blit(small.render(
                "No hand connected — tactile sensors shown below.",
                True, DIM), (x0, y0))

        # key legend — right-side vertical column (leaves the bottom for sensors)
        legend_x = x0 + bar_w + 320     # well right of the bars + (act NN) readouts
        legend_y = 22                    # KEYS header aligned with title center
        if has_hand:
            legend_lines = [
                "KEYS",
                "",
                "a/z  index",
                "s/x  middle",
                "d/c  ring",
                "f/v  pinky",
                "g/b  thumb flex",
                "h/n  thumb rot",
                "",
                "SPACE  open",
                "ENTER  close",
                "0-6  poses",
                "p  poll   r  recal",
                "k  Manus mode 0/1",
                "[ ]  step size",
                "m  step/smooth mode",
                "j  burst/single poll",
            ]
            for i, line in enumerate(legend_lines):
                col = FG if line == "KEYS" else DIM
                screen.blit(small.render(line, True, col), (legend_x, legend_y + i * 24))
            base = legend_y + len(legend_lines) * 24 + 8
        else:
            base = legend_y
        # sensor + quit keys under the column
        sensor_keys = ("t  tare   y  clear tare   -/=  sensitivity"
                       if self.panel else "")
        screen.blit(small.render(sensor_keys, True, DIM), (legend_x, base))
        screen.blit(small.render("q / ESC  quit", True, DIM), (legend_x, base + 24))

        # -- sensor panel BELOW the bars and clear of the (raised) legend ----
        if self.panel is not None:
            bars_bottom = y0 + 6 * (bar_h + gap) + 24
            sy = max(bars_bottom, base + 56)   # below both the bars and the legend
            sx = x0
            sw = W - 2 * x0
            sh = H - sy - 40
            self.panel.draw(pg, screen, small, sx, sy, sw, sh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Surge hand pygame test/teleop")
    parser.add_argument("port", nargs="?", default=None, help="serial port")
    parser.add_argument("--list", action="store_true", help="list ports, exit")
    parser.add_argument("--step", type=float, default=10.0, help="deg per key")
    parser.add_argument("--no-poll", action="store_true",
                        help="do not poll the hand for actual angles")
    parser.add_argument("--no-wait-startup", action="store_true",
                        help="skip the startup-capture phase, go straight to control")
    parser.add_argument("--no-nudge", action="store_true",
                        help="skip the startup joint-init nudge (no startup motion)")
    parser.add_argument("--no-sensors", action="store_true",
                        help="don't connect the tactile sensors")
    parser.add_argument("--smooth", action="store_true", default=True,
                        help="hold-to-move velocity control (default on)")
    parser.add_argument("--step-mode", dest="smooth", action="store_false",
                        help="use discrete tap-to-step instead of smooth motion")
    parser.add_argument("--speed", type=float, default=30.0,
                        help="deg/sec for smooth mode (default 30)")
    parser.add_argument("--rate", type=float, default=30.0,
                        help="control updates/sec for smooth mode (default 30)")
    parser.add_argument("--burst", action="store_true",
                        help="read all 6 joints per tick via get_angles_burst (step-2 test)")
    args = parser.parse_args()

    ports = list_serial_ports()
    print("Visible serial ports:", ports or "(none)")
    if args.list:
        return

    # --- hand is OPTIONAL: connect if a port is found, else run sensors-only --
    port = args.port or autodetect_port()
    hand = None
    if port is not None:
        try:
            cfg = SurgeConfig(port=port, send_smc_on_connect=False,
                              enable_feedback_on_connect=False,
                              log_commands=False, log_incoming=False)
            hand = SurgeHand(config=cfg)
            hand.connect()
            print(f"Hand connected on {port}.")
        except Exception as exc:
            print(f"Hand connect failed ({exc}); running without the hand.")
            hand = None
    else:
        print("No hand serial port found; running sensors-only.")

    # --- sensors are OPTIONAL too ---
    panel = None
    if not args.no_sensors and SensorPanel is not None:
        panel = SensorPanel()
        if panel.connected_count:
            print(f"Tactile sensors connected: {panel.connected_count}")
        else:
            print("No tactile sensors detected.")

    if hand is None and (panel is None or panel.connected_count == 0):
        print("Nothing to control: no hand and no sensors. Exiting.")
        return

    try:
        import pygame as pg
    except ImportError:
        print("pygame is required: pip install pygame")
        if hand:
            hand.disconnect()
        if panel:
            panel.close()
        return

    try:
        pg.init()
        screen = pg.display.set_mode((W, H))
        pg.display.set_caption("Surge Hand Test")
        font = pg.font.SysFont("menlo,consolas,monospace", 26)
        small = pg.font.SysFont("menlo,consolas,monospace", 18)
        clock = pg.time.Clock()

        app = TeleopApp(hand, step=args.step, poll=not args.no_poll,
                        wait_startup=not args.no_wait_startup and hand is not None,
                        nudge_on_start=not args.no_nudge,
                        panel=panel,
                        smooth=args.smooth, speed=args.speed, rate=args.rate)
        app.burst = args.burst

        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    running = handle_key(app, ev, pg)
                elif ev.type == pg.KEYUP:
                    nm = pg.key.name(ev.key)
                    if nm in KEYMAP:
                        app.on_key_up(KEYMAP[nm][0])
            if app.phase == app.STARTUP:
                app.pump_startup()
            else:
                app.update_velocity()   # advance smooth (held-key) motion
                if app.burst:
                    app.poll_all_burst()
                else:
                    app.poll_one()                
            if panel is not None:
                panel.poll()
            app.draw(pg, screen, font, small)
            pg.display.flip()
            clock.tick(int(app.rate) if app.rate > 0 else 30)

        if hand is not None:
            try:
                hand.open_all()
            except Exception:
                pass
        pg.quit()
    finally:
        if hand is not None:
            hand.disconnect()
        if panel is not None:
            panel.close()
    print("Done.")


def handle_key(app, ev, pg) -> bool:
    """Return False to quit, True to keep running."""
    if ev.key in (pg.K_q, pg.K_ESCAPE):
        return False
    name = pg.key.name(ev.key)

    # During startup capture, only S (skip) and quit are active.
    if app.phase == app.STARTUP:
        if name == "s":
            app.finish_startup("Skipped startup — control ready.")
        return True

    # Sensor tare keys (work with or without a hand). 't' tares both sensors,
    # 'y' clears the tare. ('c' stays Ring-extend for the hand.)
    if name == "t":
        app.tare_sensors(); return True
    if name == "y":
        app.clear_tare_sensors(); return True
    if name in ("-", "minus") and app.panel is not None:
        app.panel.adjust_pmax(0.8)   # more sensitive (blue at lower pressure)
        app.status = f"sensor sensitivity: blue at {app.panel.pmax:.0f}"
        return True
    if name in ("=", "equals") and app.panel is not None:
        app.panel.adjust_pmax(1.25)  # less sensitive
        app.status = f"sensor sensitivity: blue at {app.panel.pmax:.0f}"
        return True

    if ev.key == pg.K_SPACE:
        app.open_all(); return True
    if ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
        app.close_fingers(); return True
    if ev.key == pg.K_p:
        app.poll = not app.poll
        app.status = f"polling {'ON' if app.poll else 'off'}"; return True
    if ev.key == pg.K_r:
        app.recalibrate(); return True
    if ev.key == pg.K_k:
        app.toggle_manus_mode(); return True
    if name == "m":
        app.smooth = not app.smooth
        app.held.clear()   # drop any held joints when switching modes
        app.status = ("SMOOTH hold-to-move mode" if app.smooth
                      else "STEP tap mode")
        return True
    if name == "j":
        app.burst = not app.burst
        app.status = ("BURST: all-6 per tick" if app.burst
                      else "ROUND-ROBIN: one joint per tick")
        return True    
    if ev.key == pg.K_LEFTBRACKET:
        app.step = max(1.0, app.step - 1.0); return True
    if ev.key == pg.K_RIGHTBRACKET:
        app.step = min(45.0, app.step + 1.0); return True
    if name in ("0", "1", "2", "3", "4", "5", "6"):
        app.do_pose(int(name)); return True
    if name in KEYMAP:
        code, d = KEYMAP[name]
        if app.smooth:
            app.held[code] = d        # start gliding; update_velocity does the motion
        else:
            app.move(code, d)         # discrete step (original behavior)
        return True
    return True


if __name__ == "__main__":
    main()

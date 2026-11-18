"""
Tactile sensor panel for hand_test.py.

Self-contained helper that manages up to two Sensible Robotics finger sensors
and renders them as red->blue heatmaps (red = no pressure, deeper blue = more
pressure), matching the vendor's RoboSkin visualizer. Everything is
availability-based: sensors that aren't present are simply skipped, so the app
runs with zero, one, or two sensors.

Kept separate from hand_test.py so the hand app doesn't hard-depend on mido:
the panel imports the sensor driver lazily and degrades gracefully if the
sensor deps (mido/python-rtmidi) aren't installed.

Integration (in hand_test.py):
    from sensor_panel import SensorPanel
    panel = SensorPanel()          # detects & connects available sensors
    ...
    panel.poll()                   # each frame, gentle
    panel.draw(pg, screen, small, x, y, w, h)   # render into a rect
    panel.tare_all() / panel.clear_tare_all()
    panel.close()                  # on exit
"""

from __future__ import annotations

import time
from typing import Optional


# Pressure range for color mapping. Values are post-tare. Lower = more sensitive
# (color ramps up at lighter touches). Adjustable live with '-' and '=' keys.
PRESSURE_MAX = 150.0

# Colors
COL_BLANK = (30, 30, 34)      # inactive taxel (structural blank)
COL_LABEL = (210, 212, 216)
COL_DIM = (120, 126, 134)

# Thermal-style gradient for pressure (colorblind-safe: the pressure colors run
# on the blue<->yellow axis; red is only ever the 'no pressure' background):
#   no touch  -> red
#   light     -> blue         (appears at very low pressure)
#   medium    -> cyan
#   firm      -> yellow
# Stops as (position 0..1, RGB). The red->blue step is kept short so a light
# touch reads as clean blue rather than a muddy purple.
_GRAD = [
    (0.00, (150, 40, 40)),     # ~no pressure: red background
    (0.08, (90, 55, 130)),     # very light: brief purple transition
    (0.18, (40, 80, 220)),     # light: blue
    (0.50, (40, 200, 220)),    # medium: cyan
    (1.00, (245, 230, 70)),    # firm: yellow
]


def _lerp(a, b, t):
    return int(a + (b - a) * t)


def pressure_color(value: float, pmax: float = PRESSURE_MAX):
    """
    Map a taxel pressure to an RGB color along a dark->blue->cyan->yellow
    thermal gradient. Low-end contrast is emphasized so faint touches show as
    clear blue. `pmax` sets the pressure that reaches full yellow.
    """
    t = max(0.0, min(1.0, value / max(1.0, pmax)))
    # find the gradient segment containing t
    for i in range(len(_GRAD) - 1):
        p0, c0 = _GRAD[i]
        p1, c1 = _GRAD[i + 1]
        if t <= p1:
            span = (p1 - p0) or 1.0
            f = (t - p0) / span
            return (_lerp(c0[0], c1[0], f),
                    _lerp(c0[1], c1[1], f),
                    _lerp(c0[2], c1[2], f))
    return _GRAD[-1][1]


class _OneSensor:
    """Wraps a single TactileSensor with its latest frame + connection state."""

    def __init__(self, TactileSensor, index: int):
        self.index = index
        self.name = f"Sensor {index}"
        self.sensor = None
        self.frame = None
        self.error: Optional[str] = None
        self._last_poll = 0.0
        self._TactileSensor = TactileSensor

    def connect(self) -> bool:
        try:
            s = self._TactileSensor(index=self.index)
            if s.port_name is None:
                self.error = "not found"
                return False
            s.connect()
            self.sensor = s
            self.name = s.port_name
            return True
        except Exception as exc:
            self.error = str(exc)[:40]
            return False

    def poll(self, interval: float):
        if self.sensor is None:
            return
        # While taring, don't rate-limit — drain frames every call so the
        # baseline capture completes promptly (a few frames).
        now = time.time()
        if not self.sensor.taring and (now - self._last_poll < interval):
            return
        self._last_poll = now
        try:
            f = self.sensor.read()
            if f is not None:
                self.frame = f
        except Exception as exc:
            self.error = str(exc)[:40]

    def tare(self):
        if self.sensor is not None:
            self.sensor.tare(n_frames=8)

    def clear_tare(self):
        if self.sensor is not None:
            self.sensor.clear_tare()

    def close(self):
        if self.sensor is not None:
            try:
                self.sensor.disconnect()
            except Exception:
                pass


class SensorPanel:
    """Manages both sensors and draws their heatmaps. Availability-based."""

    POLL_INTERVAL = 0.05   # per-sensor min seconds between MIDI drains (gentle)

    def __init__(self):
        self.available = False
        self.sensors: list[_OneSensor] = []
        self.import_error: Optional[str] = None
        self.pmax = PRESSURE_MAX      # color-scale ceiling (lower = more sensitive)
        self._connect_all()

    def adjust_pmax(self, factor: float):
        """Scale the color-sensitivity ceiling (e.g. 0.8 = more sensitive)."""
        self.pmax = max(20.0, min(4000.0, self.pmax * factor))

    def _connect_all(self):
        # Lazy import so the hand app works without mido installed.
        try:
            from altbionics.sensor import TactileSensor
        except Exception as exc:
            self.import_error = (
                "tactile deps not installed (pip install mido python-rtmidi)"
            )
            return
        for idx in (1, 2):
            one = _OneSensor(TactileSensor, idx)
            if one.connect():
                self.sensors.append(one)
        self.available = len(self.sensors) > 0

    @property
    def connected_count(self) -> int:
        return len(self.sensors)

    def poll(self):
        for s in self.sensors:
            s.poll(self.POLL_INTERVAL)

    def tare_all(self):
        for s in self.sensors:
            s.tare()

    def clear_tare_all(self):
        for s in self.sensors:
            s.clear_tare()

    def close(self):
        for s in self.sensors:
            s.close()

    # -- rendering ------------------------------------------------------------

    def draw(self, pg, screen, small, x, y, w, h):
        """Draw the sensor panel into the rect (x,y,w,h)."""
        title = f"Tactile Sensors   (blue\u2192yellow, full@{self.pmax:.0f};  -/= sensitivity)"
        screen.blit(small.render(title, True, COL_LABEL), (x, y))
        y0 = y + 26

        if self.import_error:
            screen.blit(small.render(self.import_error, True, COL_DIM), (x, y0))
            return
        if not self.sensors:
            screen.blit(small.render(
                "no sensors detected (plug in Central PCB, close browser)",
                True, COL_DIM), (x, y0))
            return

        # lay the connected sensors side by side within the rect
        n = len(self.sensors)
        gap = 24
        each_w = (w - gap * (n - 1)) // n
        for i, s in enumerate(self.sensors):
            sx = x + i * (each_w + gap)
            self._draw_one(pg, screen, small, s, sx, y0, each_w, h - 26)

    def _draw_one(self, pg, screen, small, s, x, y, w, h):
        # header line
        tare_s = ""
        if s.sensor is not None:
            if s.sensor.taring:
                tare_s = "  [taring]"
            elif s.sensor.is_tared:
                tare_s = "  [tared]"
        screen.blit(small.render(f"{s.name}{tare_s}", True, COL_DIM), (x, y))

        grid_top = y + 22
        # square-ish cells that fit the available area
        avail_h = h - 22 - 22   # header + footer room
        cell = max(6, min((w) // 10, avail_h // 10))
        gw = cell * 10

        frame = s.frame
        for r in range(10):
            for c in range(10):
                cx = x + c * cell
                cy = grid_top + r * cell
                if frame is None:
                    color = COL_BLANK
                elif not frame.active[r, c]:
                    color = COL_BLANK
                else:
                    color = pressure_color(float(frame.values[r, c]), self.pmax)
                pg.draw.rect(screen, color, (cx, cy, cell - 1, cell - 1))

        # footer: peak / total
        fy = grid_top + gw + 4
        if frame is not None:
            txt = f"peak {frame.max_force():.0f}  total {frame.total_force():.0f}"
        else:
            txt = "(waiting for data)"
        screen.blit(small.render(txt, True, COL_DIM), (x, fy))

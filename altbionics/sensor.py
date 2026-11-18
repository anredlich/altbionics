"""
Sensible Robotics tactile finger sensor driver (Surge hand).

Format confirmed empirically (2171-frame capture) and cross-checked against the
vendor's RoboSkin browser visualizer:

  Transport : USB-MIDI. Each Central PCB enumerates as a MIDI input named
              "Robotic Hand Sensor 1" / "Robotic Hand Sensor 2".
  Framing   : SysEx messages. A full taxel frame has 208 data bytes:
                bytes 0..7  : 8-byte header  (…, byte7 == 10 marks a taxel frame;
                              byte6 is a rolling frame counter)
                bytes 8..   : 100 taxel values, 2 bytes each, big-endian 7-bit
                              MIDI pairs -> value = high*128 + low  (14-bit)
              Short SysEx messages whose header byte7 != 10 (e.g. the
              "…,6,7,…" status/housekeeping messages) are ignored.
  Grid      : the 100 values map to a 10x10 grid. The RAW wire order is
              transposed relative to the physical finger; we transpose on
              decode so row = up/down the finger, matching the GUI.
  Active    : of the 100 grid cells, 80 are real taxels (matches the vendor
              "80 taxels per finger" spec). The other 20 are structural blanks
              (fingertip curvature + a real interior gap at rows 2-3). The
              ACTIVE_MASK below marks the 80 live cells; it was derived from the
              cells that ever reported nonzero across the full capture.

Dependencies (sensors group; not in the base env):
    pip install mido python-rtmidi

Cross-platform: CoreMIDI on macOS, ALSA on Linux.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import mido
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mido is required for the tactile sensor. Install the sensors deps:\n"
        "    pip install mido python-rtmidi"
    ) from exc

import numpy as np


GRID_ROWS = 10
GRID_COLS = 10
N_TAXELS = GRID_ROWS * GRID_COLS          # 100 grid cells
FRAME_DATA_LEN = 208                      # full taxel SysEx: 8 header + 200 value bytes
HEADER_LEN = 8
FRAME_MARKER_INDEX = 7                    # data[7] == 10 marks a full taxel frame
FRAME_MARKER_VALUE = 10
FRAME_COUNTER_INDEX = 6                   # data[6] is a rolling frame counter

# Active-taxel mask in PHYSICAL orientation (row = up/down the finger, matching
# the GUI). 1 = real taxel, 0 = structural blank. 80 active cells total.
# Derived from a 2171-frame all-over-press capture (cells ever nonzero).
ACTIVE_MASK = np.array([
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=bool)

assert ACTIVE_MASK.sum() == 80, "active mask should mark 80 taxels"

DEFAULT_PORT_HINT = "Robotic Hand Sensor"


def list_ports() -> list[str]:
    """Names of all MIDI input ports visible to the OS."""
    return list(mido.get_input_names())


def find_sensor_ports() -> list[str]:
    """MIDI input ports that look like Sensible Robotics sensors."""
    return [p for p in list_ports() if DEFAULT_PORT_HINT.lower() in p.lower()]


# def autodetect_port(index: int = 1) -> Optional[str]:
#     """
#     Best-effort: return the port for sensor `index` (1 or 2). Falls back to the
#     first sensor port, or the only MIDI port if just one exists.
#     """
#     sensors = find_sensor_ports()
#     for p in sensors:
#         if p.strip().endswith(str(index)):
#             return p
#     if sensors:
#         return sensors[0]
#     ports = list_ports()
#     return ports[0] if len(ports) == 1 else None

def autodetect_port(index: int = 1) -> Optional[str]:
    """
    Return the MIDI port for sensor `index` (1 or 2) by matching the distinctive
    "Sensor <index>" substring, which is present in the port name on both macOS
    (CoreMIDI) and Linux (ALSA, where names carry a trailing client:port like
    '...MIDI 1 28:0'). Returns None if that specific sensor isn't present — we do
    NOT fall back to another sensor's port, since that would bind two panels to
    one sensor.
    """
    target = f"sensor {index}"
    for p in find_sensor_ports():
        if target in p.lower():
            return p
    return None

def decode_frame(data) -> Optional[np.ndarray]:
    """
    Decode one SysEx data tuple into a 10x10 float array in PHYSICAL
    orientation, or None if `data` is not a full taxel frame.

    Blanks (inactive taxels) are left as their raw value (typically 0); use
    ACTIVE_MASK to distinguish real taxels. See TactileFrame for a richer view.
    """
    if len(data) != FRAME_DATA_LEN:
        return None
    if data[FRAME_MARKER_INDEX] != FRAME_MARKER_VALUE:
        return None
    payload = data[HEADER_LEN:]
    vals = np.empty(N_TAXELS, dtype=np.float32)
    for i in range(N_TAXELS):
        hi = payload[2 * i]
        lo = payload[2 * i + 1]
        vals[i] = hi * 128 + lo
    raw_grid = vals.reshape(GRID_ROWS, GRID_COLS)
    return raw_grid.T.copy()   # transpose -> physical/GUI orientation


@dataclass
class TactileFrame:
    """
    One decoded tactile snapshot.

    Attributes
    ----------
    values : (10,10) float array
        Pressure per grid cell, physical orientation (row = up/down finger).
        Inactive/blank cells are 0.
    counter : int
        The frame's rolling counter byte (useful to detect dropped frames).
    timestamp : float
        time.time() when decoded.
    """
    values: np.ndarray
    counter: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def active(self) -> np.ndarray:
        """Boolean (10,10) mask of real taxels (True) vs structural blanks."""
        return ACTIVE_MASK

    @property
    def display(self) -> np.ndarray:
        """(10,10) float array with blanks as NaN — nice for heatmaps."""
        out = self.values.copy()
        out[~ACTIVE_MASK] = np.nan
        return out

    @property
    def active_values(self) -> np.ndarray:
        """1-D array of just the 80 active taxel values (row-major)."""
        return self.values[ACTIVE_MASK]

    def total_force(self) -> float:
        """Sum of pressure over active taxels (a simple contact magnitude)."""
        return float(self.values[ACTIVE_MASK].sum())

    def max_force(self) -> float:
        """Peak pressure over active taxels."""
        return float(self.values[ACTIVE_MASK].max())

    def centroid(self) -> tuple[float, float] | None:
        """
        Pressure-weighted centroid (row, col) over active taxels, or None if
        there's no contact. Useful for 'where on the pad is it touching'.
        """
        v = np.where(ACTIVE_MASK, self.values, 0.0)
        s = v.sum()
        if s <= 0:
            return None
        rows, cols = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
        return (float((v * rows).sum() / s), float((v * cols).sum() / s))


class TactileSensor:
    """
    Reads one Sensible Robotics finger sensor over USB-MIDI.

    Example
    -------
    >>> from altbionics import TactileSensor
    >>> s = TactileSensor(index=1)          # "Robotic Hand Sensor 1"
    >>> s.connect()
    >>> frame = s.read()                     # latest TactileFrame, or None
    >>> if frame: print(frame.total_force(), frame.centroid())
    >>> s.disconnect()

    Or as a context manager:
    >>> with TactileSensor(index=2) as s:
    ...     f = s.read()
    """

    def __init__(
        self,
        port: Optional[str] = None,
        *,
        index: int = 1,
        on_frame: Optional[Callable[[TactileFrame], None]] = None,
        clamp_negative: bool = True,
        smoothing: float = 0.0,
    ):
        if port is None:
            port = autodetect_port(index)
        self.port_name = port
        self.on_frame = on_frame
        self._inport = None
        self._latest: Optional[TactileFrame] = None
        self._last_counter: Optional[int] = None
        self.dropped = 0     # count of skipped frames (counter gaps)
        # --- tare (software baseline subtraction) ---------------------------
        # When _baseline is set, each decoded frame has it subtracted so an
        # untouched sensor reads ~0 and you see only the change from contact.
        self._baseline: Optional[np.ndarray] = None
        self._tare_request = 0        # >0: capture baseline from next N frames
        self._tare_accum: Optional[np.ndarray] = None
        self._tare_count = 0
        self.clamp_negative = clamp_negative   # clip tared values at 0 (default on)
        # --- live smoothing (exponential moving average) --------------------
        # Reduces frame-to-frame jitter (helpful for noisy taxels like sensor 1's
        # top row). smoothing in [0,1): 0 = off (no smoothing), higher = smoother
        # but laggier. Applied AFTER tare, to the displayed values.
        #   smoothed = (1-a)*smoothed + a*new,  where a = 1 - smoothing
        self.smoothing = float(smoothing)
        self._smoothed: Optional[np.ndarray] = None

    # -- lifecycle ------------------------------------------------------------

    def connect(self) -> "TactileSensor":
        if self.port_name is None:
            raise ValueError(
                "No sensor MIDI port found. Visible ports: "
                f"{list_ports() or 'none'}. Is the Central PCB plugged in? "
                "(Close the browser visualizer — it holds the port.)"
            )
        self._inport = mido.open_input(self.port_name)
        return self

    def disconnect(self) -> None:
        if self._inport is not None:
            self._inport.close()
        self._inport = None

    @property
    def is_connected(self) -> bool:
        return self._inport is not None and not self._inport.closed

    # -- reading --------------------------------------------------------------

    def poll(self) -> Optional[TactileFrame]:
        """
        Drain pending MIDI, decode any full taxel frames, and return the most
        recent one (or None if no new complete frame). Non-blocking. Call in a
        loop or once per GUI frame.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        newest: Optional[TactileFrame] = None
        for msg in self._inport.iter_pending():
            if msg.type != "sysex":
                continue
            grid = decode_frame(msg.data)
            if grid is None:
                continue  # status/housekeeping message, not a taxel frame
            counter = msg.data[FRAME_COUNTER_INDEX]
            if self._last_counter is not None:
                gap = (counter - self._last_counter) % 128
                if gap > 1:
                    self.dropped += gap - 1
            self._last_counter = counter

            # If a tare capture is in progress, accumulate raw frames toward the
            # baseline average (do this on the RAW grid, before subtraction).
            if self._tare_request > 0:
                if self._tare_accum is None:
                    self._tare_accum = np.zeros_like(grid)
                self._tare_accum += grid
                self._tare_count += 1
                self._tare_request -= 1
                if self._tare_request == 0:
                    self._baseline = self._tare_accum / max(1, self._tare_count)
                    self._tare_accum = None
                    self._tare_count = 0

            # Apply the baseline subtraction (tare) if one is set.
            if self._baseline is not None:
                grid = grid - self._baseline
                if self.clamp_negative:
                    np.clip(grid, 0, None, out=grid)

            # Live smoothing (EMA) to reduce frame-to-frame jitter.
            if self.smoothing > 0.0:
                a = 1.0 - self.smoothing        # blend factor for the new frame
                if self._smoothed is None:
                    self._smoothed = grid.copy()
                else:
                    self._smoothed = (1.0 - a) * self._smoothed + a * grid
                grid = self._smoothed.copy()

            frame = TactileFrame(values=grid, counter=counter)
            newest = frame
            self._latest = frame
            if self.on_frame:
                self.on_frame(frame)
        return newest

    def read(self) -> Optional[TactileFrame]:
        """Poll and return the latest frame seen so far (or None)."""
        self.poll()
        return self._latest

    # -- tare (software baseline) ---------------------------------------------

    def tare(self, n_frames: int = 8) -> None:
        """
        Capture a new baseline by averaging the next `n_frames` frames, then
        subtract it from all subsequent frames (so an untouched sensor reads
        ~0 and you see only the change from contact). Call with the sensor at
        rest / untouched. The baseline is applied as frames arrive, so it takes
        effect once `n_frames` more frames have been read via poll()/read().
        """
        self._tare_request = max(1, n_frames)
        self._tare_accum = None
        self._tare_count = 0

    def clear_tare(self) -> None:
        """Remove the baseline; frames return to raw (un-tared) values."""
        self._baseline = None
        self._tare_request = 0
        self._tare_accum = None
        self._tare_count = 0
        self._smoothed = None   # reset smoothing so it re-seeds cleanly

    def set_smoothing(self, smoothing: float) -> None:
        """
        Set the live EMA smoothing amount, in [0, 1). 0 = off, higher = smoother
        (but more lag). Resets the running average so the new setting takes
        effect cleanly.
        """
        self.smoothing = max(0.0, min(0.99, float(smoothing)))
        self._smoothed = None

    @property
    def is_tared(self) -> bool:
        """True if a baseline is currently applied."""
        return self._baseline is not None

    @property
    def taring(self) -> bool:
        """True while a tare capture is still collecting frames."""
        return self._tare_request > 0

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> "TactileSensor":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

"""
Surge hand driver — pure-Python control for the Alt Bionics Surge hand.

Protocol confirmed against the official Surge Technical Datasheet (v1.3,
Dec 2025) and verified on firmware v1.4 hardware.

    Transport : USB-serial (RS-485 dongle), 115200 baud, 8-N-1
    Terminator: "\n\r"  (datasheet EOL for every AT command)

Key AT commands (see the datasheet "AT Commands" tables):
    AT+SJA=<deg>,<J>   set joint <J> to <deg> degrees        (e.g. AT+SJA=45,I)
    AT+SPP=<pose>      move to a preset grip pose 0..6
    AT+GJA?<J>         get joint <J> current angle
    AT+GCE?<J>         get joint <J> error code
    AT+RCH=1           recalibrate the hand
    AT+SCE=1           clear motor error / release a stuck joint
    AT+SSM=<v>,<J>     stop motor <J> (v=1 stop, v=0 allow move; J='A' = all)
    AT+SEF=<v>         command feedback off(0)/on(1): hand replies OK / ERROR
    AT+SMC=<v>         enable(1)/disable(0) Manus-glove control

Joint (finger) codes — SINGLE LETTERS, from the datasheet "Finger Codes":
    I=Index  M=Middle  R=Ring  P=Pinky  T=Thumb-Flex  X=Thumb-Rotation

Angles are in DEGREES. Ranges from the datasheet mechanical specs:
    fingers (I,M,R,P): 0..90 deg
    thumb flex (T)   : 0..95 deg
    thumb rot  (X)   : 0..85 deg

Runs identically on macOS (/dev/cu.usbserial-*) and Linux (/dev/ttyUSB*,
/dev/ttyACM*). No compilation, no platform binaries.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass

try:
    import serial  # pyserial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyserial is required. Install it with: pip install pyserial "
        "(it is included in environment.yml)."
    ) from exc


EOL = "\n\r"  # datasheet terminator for AT commands

# --- Finger / joint codes (single letters, per datasheet) -------------------
FINGER_CODES = {
    "index": "I",
    "middle": "M",
    "ring": "R",
    "pinky": "P",
    "thumb_flex": "T",       # Thumb flexion
    "thumb_rotation": "X",   # Thumb rotation
}
CODE_TO_NAME = {v: k for k, v in FINGER_CODES.items()}
VALID_CODES = set(FINGER_CODES.values())            # {'I','M','R','P','T','X'}
FINGERS = tuple(FINGER_CODES.keys())                # friendly names
JOINT_ORDER_CANON = ("I", "M", "R", "P", "T", "X")  # stable iteration order
JOINT_ALL = "A"                                     # 'A' = all joints (stop)

# --- Per-joint angle limits in degrees (from datasheet mechanical specs) -----
ANGLE_LIMITS = {
    "I": (0, 90),
    "M": (0, 90),
    "R": (0, 90),
    "P": (0, 90),
    "T": (0, 95),   # thumb flexion
    "X": (0, 85),   # thumb rotation
}
DEFAULT_LIMIT = (0, 90)

# --- Per-mode, per-joint command floors (anti-stop-glitch). See note above. -anr
COMMAND_FLOORS = {
    0: {"I": 2.0, "M": 2.0, "R": 2.0, "P": 2.0, "T": 2.0,  "X": 2.0},
    1: {"I": 2.0, "M": 2.0, "R": 2.0, "P": 2.0, "T": 12.0, "X": 12.0},
}

# --- Preset grip poses (AT+SPP), from datasheet "Pose Codes" -----------------
POSES = {
    "open": 0,
    "power": 1,
    "chuck": 2,
    "fine_pinch": 3,
    "key": 4,
    "hook": 5,
    "birds_up": 6,
}


def resolve_joint(joint: str) -> str:
    """
    Accept a friendly finger name ('index','thumb_flex') or a raw single-letter
    code ('I','X') and return the validated single-letter code. Names are
    case-insensitive; codes are upper-cased.
    """
    j = joint.strip()
    if j in FINGER_CODES:
        return FINGER_CODES[j]
    low = j.lower()
    if low in FINGER_CODES:
        return FINGER_CODES[low]
    up = j.upper()
    if up in VALID_CODES:
        return up
    raise ValueError(
        f"Unknown joint {joint!r}. Use a name in {list(FINGER_CODES)} "
        f"or a code in {sorted(VALID_CODES)}."
    )

def clamp_angle(code: str, angle: float, floor: float = 0.0) -> float:
    """
    Clamp a joint command to the Surge protocol range [0, 127] (float).

    Matches the Alt-Bionics C++ driver (surge_linux.cpp): clips only to 0..127,
    NOT to per-joint degree limits, and WITHOUT rounding to int. 0..127 is
    mode-agnostic — correct in direct mode (mode 0, true degrees) and in Manus
    mode (mode 1, where AT+SJA is retargeted and a joint can exceed its degree
    range). `code` is kept for API compatibility but no longer selects a range.
    """
    if angle < floor:
        return floor
    if angle > 127.0:
        return 127.0
    return float(angle)

@dataclass
class SurgeConfig:
    port: str
    baudrate: int = 115200
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 1.0
    write_timeout: float = 1.0
    # FTDI serial string used to pin THIS hand during autodetect (see autodetect_port).
    preferred_serial: str | None = "FTBC5EKG"
    # AT+SMC=1 enables MANUS-glove control. For DIRECT software control you
    # generally do NOT want this on, so default False.
    send_smc_on_connect: bool = False
    disable_manus_on_connect: bool = True   # send AT+SMC=0 after calibration (direct-control mode)
    # AT+SEF=1 makes the hand reply OK/ERROR to each command.
    enable_feedback_on_connect: bool = False
    log_commands: bool = True
    log_incoming: bool = True
    listen_on_connect: float = 0.0


def list_serial_ports() -> list[str]:
    """Return candidate serial device paths visible to the OS."""
    return [p.device for p in list_ports.comports()]

def autodetect_port(preferred_serial: str | None = "FTBC5EKG") -> str | None:
    """
    Find the Surge hand's serial device.

    The hand is an FTDI FT232 (VID 0x0403) with a UNIQUE serial string (e.g.
    "FTBC5EKG"). 0x0403:6001 is a very common generic ID, so when other FTDI
    devices are attached (notably the Manus glove dongle), identifying the hand
    by its serial string pins THIS hand regardless of enumeration order or port.
    Mirrors the C++ driver's detection order:
        1. the port whose FTDI serial == preferred_serial
        2. the first FTDI device (VID 0x0403)
        3. the first ttyUSB/ttyACM (usbserial/usbmodem on macOS)  [fallback]
    """
    ports = list(list_ports.comports())
    if not ports:
        return None
    # 1. exact serial match — pins THIS hand even with the glove dongle present.
    if preferred_serial:
        for p in ports:
            if p.serial_number == preferred_serial:
                return p.device
    # 2. first FTDI device (VID 0x0403).
    for p in ports:
        if p.vid == 0x0403:
            return p.device
    # 3. first ttyUSB/ttyACM of any kind (previous behaviour).
    if sys.platform == "darwin":
        pref = [p.device for p in ports if "usbserial" in p.device or "usbmodem" in p.device]
    else:
        pref = [p.device for p in ports if "ttyUSB" in p.device or "ttyACM" in p.device]
    if pref:
        return pref[0]
    return ports[0].device

class SurgeHand:
    """
    Controls a single Surge hand over USB-serial (RS-485).

    Example
    -------
    >>> from altbionics import SurgeHand
    >>> with SurgeHand(autodetect=True) as hand:
    ...     hand.wait_until_ready()          # after a power-cycle
    ...     hand.set_finger("index", 45)     # degrees
    ...     hand.pose("power")               # preset grip
    ...     print(hand.get_angle("index"))
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        autodetect: bool = False,
        config: SurgeConfig | None = None,
        **kwargs,
    ):
        if config is None:
            if port is None:
                if autodetect:
                    port = autodetect_port(kwargs.get("preferred_serial", "FTBC5EKG"))
                if port is None:
                    raise ValueError(
                        "No port given and autodetect found none. "
                        f"Visible ports: {list_serial_ports() or 'none'}"
                    )
            config = SurgeConfig(port=port, **kwargs)
        self.config = config
        self._serial: serial.Serial | None = None
        self._rx_buf: bytes = b""
        self._manus_mode = 1   # hand BOOTS into Manus mode 1; updated by set_manus_mode

    # -- connection lifecycle -------------------------------------------------

    def connect(self) -> "SurgeHand":
        c = self.config
        self._serial = serial.Serial(
            port=c.port,
            baudrate=c.baudrate,
            bytesize=c.bytesize,
            parity=c.parity,
            stopbits=c.stopbits,
            timeout=c.timeout,
            write_timeout=c.write_timeout,
        )
        self._rx_buf = b""  # raw receive buffer for line framing
        time.sleep(0.1)
        if c.listen_on_connect > 0:
            self.listen(duration=c.listen_on_connect, idle_timeout=2.0)
        if c.enable_feedback_on_connect:
            self._send_raw("AT+SEF=1")
        if c.send_smc_on_connect:
            self._send_raw("AT+SMC=1")
        return self

    def disconnect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    # -- core: set joint angle (AT+SJA) ---------------------------------------

    def set_finger(self, finger: str, angle: float) -> float:
        """
        Set one joint to `angle`. `finger` may be a friendly name
        ('index','middle','ring','pinky','thumb_flex','thumb_rotation') or a
        raw code ('I','M','R','P','T','X'). The value is clamped to the Surge
        protocol range [0, 127] (see clamp_angle) and sent with one decimal,
        matching the Alt-Bionics C++ driver.

        Returns the value ACTUALLY SENT to the hand (post-clamp), so callers
        (e.g. the dataset recorder) can log the true action. Mirrors the Trossen
        arm's send_action, which also returns the actually-sent target.
        """
        code = resolve_joint(finger)
        floor = COMMAND_FLOORS.get(self._manus_mode, {}).get(code, 0.0)
        sent = clamp_angle(code, angle, floor)        
        self._send_raw(f"AT+SJA={sent:.1f},{code}")
        return sent

    def set_fingers(self, mapping: dict[str, float]) -> dict[str, float]:
        """
        Set several joints from a {finger: angle} dict.

        Returns {code: actually_sent_value} for every joint commanded (post-clamp,
        keyed by the resolved single-letter code). This is the batch the follower
        uses each step: send all joints, get back what the hand received, record it.
        """
        sent: dict[str, float] = {}
        for finger, angle in mapping.items():
            code = resolve_joint(finger)
            sent[code] = self.set_finger(finger, angle)
        return sent
    
    def open_all(self) -> None:
        """Drive every joint to its minimum (open) angle."""
        for code in VALID_CODES:
            self.set_finger(code, ANGLE_LIMITS[code][0])

    def close_all(self) -> None:
        """Drive the four fingers to max flexion (does not force the thumb)."""
        for code in ("I", "M", "R", "P"):
            self.set_finger(code, ANGLE_LIMITS[code][1])

    # -- preset grip poses (AT+SPP) -------------------------------------------

    def pose(self, pose: str | int) -> None:
        """
        Move to a preset grip pose. Accepts a name ('open','power','chuck',
        'fine_pinch','key','hook','birds_up') or a raw code 0..6.
        """
        if isinstance(pose, str):
            key = pose.strip().lower()
            if key not in POSES:
                raise ValueError(
                    f"Unknown pose {pose!r}. Choose from {list(POSES)} or 0..6."
                )
            code = POSES[key]
        else:
            code = int(pose)
            if code not in POSES.values():
                raise ValueError(f"Pose code {code} out of range 0..6.")
        self._send_raw(f"AT+SPP={code}")

    # -- queries (AT+GJA?, AT+GCE?) -------------------------------------------

    def get_angle(self, finger: str, *, wait: float = 0.5) -> float | None:
        """
        Query a joint's current angle in degrees (AT+GJA?<J>). Returns the
        parsed float, or None if no valid reply arrived within `wait`.

        The hand answers a GJA query with a line containing the angle (observed
        format e.g. "+GA=64.30" or a bare number). We flush stale input first
        and skip OK/ERROR acknowledgements, so a late reply from a *previous*
        command can't be misread as this joint's angle.
        """
        code = resolve_joint(finger)
        replies = self.send_and_read(f"AT+GJA?{code}", wait=wait)
        for line in replies:
            up = line.upper()
            if up in ("OK", "ERROR") or up.startswith("OK") and len(line) <= 3:
                continue  # acknowledgement, not the value
            # prefer lines that look like an angle reply (contain '=' or GA/GJA)
            if "=" in line or "GA" in up:
                val = _parse_float(line.split("=")[-1])
                if val is not None:
                    return val
        # fallback: any parseable float that isn't obviously an ack
        for line in replies:
            if line.upper().startswith("OK"):
                continue
            val = _parse_float(line)
            if val is not None:
                return val
        return None

    def get_angle_fast(self, finger: str, *, timeout: float = 0.05) -> float | None:
        """
        Robust single-joint GJA read. Contract: returns a CORRECT angle, or None.
        Never returns a plausible-but-wrong value from a truncated/garbled frame.

        - reset_input_buffer() drops any stale reply so read_until returns THIS
          query's answer, not a leftover (fixes the one-deep pipeline lag).
        - read_until(b"\\n\\r") waits for the FULL two-byte terminator, so a reply
          still arriving isn't cut short at a bare \\r.
        - the endswith / "GJA=" guard rejects a partial frame (returns None) rather
          than parsing a truncated number. A None is safe (caller holds last value);
          a wrong number would poison the dataset.

        ~10 ms per call on this hardware: that is the hand's true round-trip
        (request out, firmware turnaround, reply back), not loop overhead. The
        timeout is a CEILING (only paid on a genuine miss), not a wait.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        code = resolve_joint(finger)
        s = self._serial
        assert s is not None
        s.timeout = timeout
        s.reset_input_buffer()
        s.write((f"AT+GJA?{code}" + EOL).encode("ascii"))
        raw = s.read_until(b"\n\r")
        if not raw.endswith(b"\n\r") or b"GJA=" not in raw:
            return None
        return _parse_float(raw.decode("ascii", errors="replace"))

    def get_angles_burst(self, codes, *, timeout: float = 0.08) -> dict[str, float | None]:
        """
        Pipelined multi-joint GJA read. Flush once, write ALL queries back to
        back, then read the replies IN THE SAME ORDER. Pays the ~10 ms round-trip
        latency ONCE for the whole batch instead of once per joint.

        Relies on: (a) command feedback OFF, so the only bytes on the wire are
        GJA replies, and (b) the hand answering queries in the order received
        (single serial command processor). GJA replies carry no joint code
        (AT+GJA=NN), so attribution is purely positional — which is exactly why
        B2 below refuses a batch that doesn't come back clean.

        Contract (same spirit as get_angle_fast): each joint maps to a CORRECT
        angle or None; a positional read never yields a wrong-but-plausible value.
            - a per-frame guard (full terminator + "GJA=") rejects a truncated
              frame for THAT joint (-> None),
            - but a MISSING frame would shift every later joint in the batch, so
              if any frame is malformed we stop reading and return None for the
              rest: a shifted misattribution is worse than a held-last value.

        Returns {code: angle_or_None} for every requested code, in request order.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        resolved = [resolve_joint(c) for c in codes]
        s = self._serial
        assert s is not None
        s.timeout = timeout
        # one flush for the whole batch, then fire all queries back to back.
        s.reset_input_buffer()
        for code in resolved:
            s.write((f"AT+GJA?{code}" + EOL).encode("ascii"))
        # read the replies in order; one bad frame aborts the rest (B2).
        out: dict[str, float | None] = {}
        aborted = False
        for code in resolved:
            if aborted:
                out[code] = None
                continue
            raw = s.read_until(b"\n\r")
            if not raw.endswith(b"\n\r") or b"GJA=" not in raw:
                # malformed/short frame: positional alignment is now unsafe for
                # every remaining joint, so abort and None-out the rest.
                out[code] = None
                aborted = True
                continue
            out[code] = _parse_float(raw.decode("ascii", errors="replace"))
        return out
            
    def set_finger_read(self, finger: str, angle: float, *,
                        timeout: float = 0.1) -> float | None:
        """
        Command a joint (AT+SJA) and immediately read back its TRUE angle
        (AT+GJA?), returning the measured angle. This is the command+verify
        pattern for control/recording: one SJA, one GJA, minimal latency.

        Note: get_angle/GJA reports the true physical angle regardless of Manus
        mode, so the returned value is the real position the joint reached.
        Returns None if no readback arrived within `timeout`.
        """
        code = resolve_joint(finger)
        floor = COMMAND_FLOORS.get(self._manus_mode, {}).get(code, 0.0)
        sent = clamp_angle(code, angle, floor)
        self._send_raw(f"AT+SJA={sent:.1f},{code}")
        line = self.query_fast(f"AT+GJA?{code}", "GJA=", timeout=timeout)
        if line is not None:
            return _parse_float(line.split("=")[-1])
        return None

    def get_error(self, finger: str, *, wait: float = 0.5) -> str | None:
        """
        Query a joint's error code (AT+GCE?<J>). Returns the reply string
        containing the code (e.g. '0x0001'), or None if nothing valid arrived.
        Flushes stale input first and skips OK/ERROR acks.
        """
        code = resolve_joint(finger)
        replies = self.send_and_read(f"AT+GCE?{code}", wait=wait)
        for line in replies:
            s = line.strip()
            up = s.upper()
            if not s or up in ("OK", "ERROR"):
                continue
            if "0X" in up or "=" in s:      # looks like a code reply
                return s
        # fallback: first non-ack line
        for line in replies:
            if line.strip().upper() not in ("OK", "ERROR") and line.strip():
                return line.strip()
        return None

    def get_all_errors(self, *, wait: float = 0.4) -> dict:
        """
        Query every joint's error register. Returns {code: reply_string}, e.g.
        {'I': '0x0000', 'M': '0x0000', ...}. A value that isn't a clean
        'no error' code (see is_error_ok) indicates a fault on that joint.
        """
        out = {}
        for code in JOINT_ORDER_CANON:
            out[code] = self.get_error(code, wait=wait)
        return out

    @staticmethod
    def is_error_ok(reply: str | None) -> bool:
        """
        Heuristic: is this GCE reply a 'no error' state? Treats None as unknown
        (not ok). Known-good patterns: 0x0000, and the boot report's 0xDD
        ('- No error'). Anything else is treated as a fault to surface.
        """
        if reply is None:
            return False
        up = reply.upper()
        if "NO ERROR" in up:
            return True
        # extract a hex code if present
        import re as _re
        m = _re.search(r"0X([0-9A-F]+)", up)
        if m:
            return m.group(1).lstrip("0") in ("", )  # 0x0000 -> ok
        return False

    def _read_ack(self, wait: float = 0.4) -> bool | None:
        """
        Read an OK/ERROR acknowledgement (AT+SEF must be enabled). Returns True
        for OK, False for ERROR, None if no clear ack arrived.
        """
        lines = self.listen(duration=wait, idle_timeout=wait)
        for line in lines:
            up = line.strip().upper()
            if up.startswith("OK"):
                return True
            if up.startswith("ERROR"):
                return False
        return None

    # -- maintenance (AT+RCH, AT+SCE, AT+SSM) ---------------------------------

    def recalibrate(
        self, *, wait_ready: bool = True, timeout: float = 60.0
    ) -> dict | None:
        """
        Trigger a hand recalibration (AT+RCH=1). If wait_ready, then listen for
        the calibration/startup sequence and return the wait_until_ready dict.
        """
        self.flush_input()
        self._send_raw("AT+RCH=1")
        if wait_ready:
            return self.wait_until_ready(timeout=timeout)
        return None

    def clear_error(self, *, verify: bool = True, wait: float = 0.5) -> bool | None:
        """
        Clear a motor error and release a stuck joint (AT+SCE=1). If verify,
        return the hand's OK/ERROR ack (True/False), or None if no ack.
        """
        self.flush_input()
        self._send_raw("AT+SCE=1")
        if verify:
            return self._read_ack(wait=wait)
        return None

    def set_manus_mode(self, enabled: bool, *, verify: bool = False, wait: float = 0.5):
        """
        Enable (AT+SMC=1) or disable (AT+SMC=0) Manus glove-control mode.

        The hand BOOTS INTO Manus mode. In Manus mode, AT+SJA angle commands are
        routed through glove-retargeting and do NOT map to true degrees (this was
        the cause of the thumb 'nonlinearity'). For direct programmatic/keyboard
        control, disable Manus mode after calibration. Enable it only for glove
        teleop. The hand moves slightly when the mode switches.
        """
        self.flush_input()
        self._send_raw(f"AT+SMC={1 if enabled else 0}")
        self._manus_mode = 1 if enabled else 0
        if verify:
            return self._read_ack(wait=wait)
        return None

    def stop(self, finger: str | None = None) -> None:
        """
        Stop a joint's motor (AT+SSM=1,<J>). With no argument, stops ALL joints
        (code 'A'). Call resume() to allow movement again.
        """
        code = JOINT_ALL if finger is None else resolve_joint(finger)
        self._send_raw(f"AT+SSM=1,{code}")

    def resume(self, finger: str | None = None) -> None:
        """Allow a previously-stopped joint to move again (AT+SSM=0,<J>)."""
        code = JOINT_ALL if finger is None else resolve_joint(finger)
        self._send_raw(f"AT+SSM=0,{code}")

    def set_feedback(self, enabled: bool) -> None:
        """Turn command OK/ERROR feedback on or off (AT+SEF=1/0)."""
        self._send_raw(f"AT+SEF={1 if enabled else 0}")

    def set_manus_control(self, enabled: bool) -> None:
        #"""Enable/disable Manus-glove control of the hand (AT+SMC=1/0)."""
        #self._send_raw(f"AT+SMC={1 if enabled else 0}")
        """Enable/disable Manus-glove control (AT+SMC=1/0). Delegates to
        set_manus_mode so the tracked mode (and command floor) stays in one
        place and can't desync. -anr"""
        self.set_manus_mode(enabled)
        
    # -- low-level I/O --------------------------------------------------------
    #
    # The hand frames responses with CR and/or LF and can send them in
    # fragments; back-to-back commands mean an earlier reply can still be in
    # flight when the next command goes out. To keep responses from being
    # misattributed we:
    #   * accumulate raw bytes in self._rx_buf and split on BOTH \r and \n,
    #   * expose read_line() that returns only COMPLETE buffered lines,
    #   * flush stale input before a query (flush_input),
    #   * tolerate the occasional FTDI "readiness but no data" exception.

    def _send_raw(self, command: str) -> None:
        """Append the datasheet EOL and write an ASCII command."""
        self._write_bytes((command + EOL).encode("ascii"))

    def _write_bytes(self, data: bytes) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        if self.config.log_commands:
            ts = time.strftime("%H:%M:%S")
            ms = int((time.time() % 1) * 1000)
            printable = (
                data.decode("ascii", errors="replace")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
            )
            print(f"[{ts}.{ms:03d}] -> {printable}")
        assert self._serial is not None
        self._serial.write(data)
        self._serial.flush()

    # -- buffered reading -----------------------------------------------------

    def _pump_bytes(self) -> None:
        """Read whatever bytes are available into the rx buffer (non-blocking).
        Tolerates the FTDI/macOS 'readiness but no data' quirk."""
        assert self._serial is not None
        try:
            n = self._serial.in_waiting
        except OSError:
            n = 0
        if not n:
            return
        try:
            chunk = self._serial.read(n)
        except serial.SerialException:
            # transient FTDI hiccup — skip this pump; caller will retry
            return
        if chunk:
            self._rx_buf += chunk

    def _next_buffered_line(self) -> str | None:
        """Pop one complete line from the buffer (split on \r or \n),
        or None if no complete line is buffered yet. Empty fragments are
        skipped. Cleans control/null bytes."""
        while True:
            # find the earliest CR or LF
            idx_r = self._rx_buf.find(b"\r")
            idx_n = self._rx_buf.find(b"\n")
            cands = [i for i in (idx_r, idx_n) if i != -1]
            if not cands:
                return None
            cut = min(cands)
            raw, self._rx_buf = self._rx_buf[:cut], self._rx_buf[cut + 1:]
            text = raw.decode("ascii", errors="replace")
            text = "".join(ch for ch in text if ch == "\t" or 32 <= ord(ch) < 127)
            text = text.strip()
            if text:
                return text
            # empty fragment (e.g. the \n after a \r) — keep looking

    def flush_input(self) -> None:
        """Discard any pending/stale input (buffer + OS) before a query, so a
        late reply from a previous command can't be read as this one's."""
        if not self.is_connected:
            return
        assert self._serial is not None
        try:
            self._serial.reset_input_buffer()
        except Exception:
            pass
        self._rx_buf = b""

    def read_line(self, timeout: float | None = None) -> str | None:
        """
        Return the next complete line from the hand, or None if none arrives
        within `timeout` seconds. timeout=0.0 is a pure non-blocking poll.
        Lines are split on CR or LF and cleaned of control/null bytes.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        deadline = None if timeout is None else (time.time() + timeout)
        while True:
            self._pump_bytes()
            line = self._next_buffered_line()
            if line is not None:
                if self.config.log_incoming:
                    ts = time.strftime("%H:%M:%S")
                    ms = int((time.time() % 1) * 1000)
                    print(f"[{ts}.{ms:03d}] <- {line}")
                return line
            if deadline is not None and time.time() >= deadline:
                return None
            if timeout == 0.0:
                return None
            time.sleep(0.005)

    def read_available(self) -> str:
        """Drain and return everything currently buffered (joined lines)."""
        lines = []
        while True:
            self._pump_bytes()
            line = self._next_buffered_line()
            if line is None:
                break
            lines.append(line)
            if self.config.log_incoming:
                ts = time.strftime("%H:%M:%S")
                ms = int((time.time() % 1) * 1000)
                print(f"[{ts}.{ms:03d}] <- {line}")
        return "\n".join(lines)

    def listen(
        self,
        duration: float = 10.0,
        *,
        stop_on: str | None = None,
        idle_timeout: float | None = None,
    ) -> list[str]:
        """Collect complete lines for up to `duration` seconds; return them."""
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        lines: list[str] = []
        start = time.time()
        last_rx = start
        needle = stop_on.lower() if stop_on else None
        while time.time() - start < duration:
            line = self.read_line(timeout=0.05)
            now = time.time()
            if line:
                lines.append(line)
                last_rx = now
                if needle and needle in line.lower():
                    break
            elif idle_timeout is not None and (now - last_rx) >= idle_timeout:
                break
        return lines

    def send_and_read(
        self, command: str, *, wait: float = 0.5, flush_first: bool = True
    ) -> list[str]:
        """
        Send an AT command (EOL added) and gather reply lines. By default
        flushes stale input first so a previous command's late reply can't be
        mistaken for this one's.
        """
        if flush_first:
            self.flush_input()
        self._send_raw(command)
        return self.listen(duration=wait, idle_timeout=wait)

    def query_fast(self, command: str, match: str, *,
                   timeout: float = 0.1, flush_first: bool = False) -> str | None:
        """
        Send a query and return the FIRST reply line containing `match`, the
        instant it arrives — no waiting out an idle timeout. Much faster than
        send_and_read for request/response polling (e.g. AT+GJA?). Returns the
        matching line, or None if nothing matched within `timeout`.

        flush_first defaults False here: for tight polling we control the
        sequence, so skipping the per-call buffer reset saves time. Set True if
        you suspect stale data in the buffer.
        """
        if flush_first:
            self.flush_input()
        self._send_raw(command)
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.read_line(timeout=0.0)   # non-blocking
            if line and match in line:
                return line
        return None

    # -- startup / calibration handshake --------------------------------------

    CALIB_PASS_MARKER = "calibration passed"
    CALIB_FAIL_MARKERS = ("calibration failed", "test failed")

    def wait_until_ready(
        self, timeout: float = 60.0, *, require_calibration: bool = True
    ) -> dict:
        """
        Listen to the power-up / calibration sequence and return once the hand
        reports ready (or until timeout). Returns a summary dict with keys:
        ready, calibration_passed, hand_version, firmware_version, lines.

        The hand emits this on BOOT (48V power-up) — so connect(), then
        power-cycle the hand, then call this (or call it and power-cycle while
        it listens).
        """
        result = {
            "ready": False,
            "calibration_passed": False,
            "hand_version": None,
            "firmware_version": None,
            "lines": [],
        }
        start = time.time()
        while time.time() - start < timeout:
            line = self.read_line(timeout=0.5)
            if not line:
                continue
            result["lines"].append(line)
            low = line.lower()
            if "hand version" in low and ":" in line:
                result["hand_version"] = line.split(":", 1)[1].strip()
            elif "firmware version" in low and ":" in line:
                result["firmware_version"] = line.split(":", 1)[1].strip()
            if self.CALIB_PASS_MARKER in low:
                result["calibration_passed"] = True
            if "startup sequence complete" in low:
                if not require_calibration or result["calibration_passed"]:
                    result["ready"] = True
                    break
        if self.config.disable_manus_on_connect:
            time.sleep(0.2)
            self.set_manus_mode(False)   # AT+SMC=0 -> direct-control mode
            time.sleep(0.3)              # allow the mode-switch motion to settle
            self.flush_input()

        return result

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> "SurgeHand":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


def _parse_float(text: str) -> float | None:
    """Extract the first float from a string like '49.015' or 'OK 49.0'."""
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

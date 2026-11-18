"""
Glove -> hand teleoperation loop for the Alt-Bionics Surge hand.

This is the Linux/Python equivalent of the Windows SDKClient's [R] "Run Surge"
command: read the Manus glove's raw fingertip positions, map them to the six
Surge joint angles with the fitted regression (altbionics.ik), and stream those
angles to the hand over serial (altbionics.hand.SurgeHand). The hand is driven
in the Manus mode the IK was fit in (ik.IK_REGRESSION_MANUS_MODE), asserted on
connect — the Python mirror of the C++ setManusMode(IK_REGRESSION_MANUS_MODE).

Design for downstream reuse
---------------------------
The control loop is deliberately factored so the three roles the larger project
needs all share the same SurgeHand sink:

    teleoperation (now):   glove -> ik.solve -> SurgeHand          (this file)
    dataset recording:     glove -> ik.solve -> SurgeHand, and the (sensor,angle)
                           pairs are also logged as the action signal  (later)
    policy inference:      policy -> joint angles  -> SurgeHand      (later;
                           the glove source is replaced by the policy output,
                           the SurgeHand sink is identical)

Because the action space is the six Surge joint angles, a policy trained on
teleop data outputs the same six numbers this loop produces, and drives the
same SurgeHand. Keeping the glove read, the IK, and the hand write as separate
collaborators is what makes that swap clean.

No ROS dependency: the glove is read through altbionics.glove.ManusGlove, a
direct ctypes binding to the Manus Core SDK shared library.
"""

from __future__ import annotations

import time
from typing import Optional

from altbionics import ik
from altbionics.hand import SurgeHand, SurgeConfig, autodetect_port
from altbionics.glove import ManusGlove

class GloveHandController:
    """
    Drives a SurgeHand from a ManusGlove using the fitted finger IK.

    Parameters
    ----------
    hand : SurgeHand
        Connected Surge hand. If None, one is autodetected and opened in the
        IK's Manus mode (ik.IK_REGRESSION_MANUS_MODE).
    glove : ManusGlove
        Connected Manus glove source. If None, one is created and started.
    rate_hz : float
        Target control loop rate. The Surge accepts commands comfortably around
        this range; the glove streams ~100+ Hz, so we sample at rate_hz.
    send_on_change_only : bool
        If True, only send a joint command when its rounded angle changed since
        last send (reduces serial traffic / jitter), mirroring the C++ [K]/[R]
        behaviour.
    """

    def __init__(
        self,
        hand: Optional[SurgeHand] = None,
        glove: Optional[ManusGlove] = None,
        rate_hz: float = 50.0,
        send_on_change_only: bool = True,
    ):
        self.hand = hand
        self.glove = glove
        self.rate_hz = float(rate_hz)
        self.send_on_change_only = send_on_change_only
        self._owns_hand = hand is None
        self._owns_glove = glove is None
        self._last_sent: "dict[str, float]" = {}
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> "GloveHandController":
        if self.hand is None:
            # We compute the six joint angles ourselves and stream them as
            # AT+SJA, exactly like the C++ [R] command. The fitted IK
            # coefficients (altbionics.ik) were learned with the hand in
            # Manus mode ik.IK_REGRESSION_MANUS_MODE (= 1), so the hand MUST be
            # in that same mode for the angles to mean what the fit expects.
            #
            # The hand BOOTS into Manus mode 1. SurgeConfig's default
            # disable_manus_on_connect=True would send AT+SMC=0 during
            # wait_until_ready() and break the mapping (this was the poor
            # thumb-control bug). So we disable that auto-SMC=0 here, then
            # explicitly assert the regression mode below — the Python mirror
            # of the C++ setManusMode(IK_REGRESSION_MANUS_MODE).
            port = autodetect_port()
            if port is None:
                raise RuntimeError("No Surge serial port found. Is the hand plugged in?")
            cfg = SurgeConfig(
                port=port,
                disable_manus_on_connect=False,   # do NOT auto-send AT+SMC=0
                send_smc_on_connect=False,        # we assert the mode explicitly below
            )
            self.hand = SurgeHand(config=cfg)
            self.hand.connect()

        # Assert the Manus mode the IK was fit in (power-order independent,
        # same guarantee as the C++ setManusMode): mode travels with the coeffs.
        self.hand.set_manus_mode(ik.IK_REGRESSION_MANUS_MODE == 1)

        if self.glove is None:
            self.glove = ManusGlove()
            self.glove.start()
        return self

    def close(self) -> None:
        if self._owns_glove and self.glove is not None:
            self.glove.stop()
        if self._owns_hand and self.hand is not None:
            self.hand.disconnect()

    def __enter__(self) -> "GloveHandController":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- one control step ----------------------------------------------------

    def step(self) -> "Optional[dict[str, float]]":
        """
        Read one glove frame, solve IK, send to the hand. Returns the dict of
        joint angles sent (or None if no fresh glove data was available).
        """
        sensors = self.glove.read_sensors()   # list[5] of (x,y,z) meters, or None
        if sensors is None:
            return None

        angles = ik.solve(sensors)

        # Build the dict of joints to actually send this step. set_fingers takes
        # a {code: angle} mapping and emits one AT+SJA per entry.
        to_send: "dict[str, float]" = {}
        for code in ik.JOINT_CODES:
            a = round(angles[code], 2)
            if self.send_on_change_only and self._last_sent.get(code) == a:
                continue
            to_send[code] = a
            self._last_sent[code] = a

        if to_send:
            self.hand.set_fingers(to_send)

        return angles

    # -- run loop ------------------------------------------------------------

    def run(self, duration_s: Optional[float] = None, verbose: bool = True) -> None:
        """
        Run the teleop loop until duration_s elapses (or forever if None), or
        until KeyboardInterrupt. This is the [R]-command equivalent.
        """
        period = 1.0 / self.rate_hz
        t_start = time.time()
        self._running = True
        if verbose:
            print(f"Teleop running @ {self.rate_hz:.0f} Hz "
                  f"(glove -> Surge hand, Manus mode "
                  f"{ik.IK_REGRESSION_MANUS_MODE}). Ctrl-C to stop.")
        try:
            while self._running:
                t0 = time.time()
                angles = self.step()
                if verbose and angles is not None:
                    line = "  ".join(f"{c}={angles[c]:5.1f}" for c in ik.JOINT_CODES)
                    print("\r" + line + "   ", end="", flush=True)
                if duration_s is not None and (time.time() - t_start) >= duration_s:
                    break
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            if verbose:
                print("\nStopped.")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


# ---- standalone run: the Python equivalent of the C++ [R] "Run Surge" -------
#   python -m altbionics.teleop               # 50 Hz, runs until Ctrl-C
#   python -m altbionics.teleop --rate 60     # custom rate
#   python -m altbionics.teleop --so /path/to/libManusSDK_Integrated.so
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Manus glove -> Surge hand teleop (C++ [R] equivalent).")
    p.add_argument("--rate", type=float, default=50.0, help="control loop rate (Hz)")
    p.add_argument("--so", type=str, default=None,
                   help="path to libManusSDK_Integrated.so (defaults to glove.DEFAULT_SO)")
    p.add_argument("--all", action="store_true",
                   help="send every joint every step (default: only changed joints)")
    args = p.parse_args()

    glove = ManusGlove(so_path=args.so) if args.so else None
    ctrl = GloveHandController(
        glove=glove,
        rate_hz=args.rate,
        send_on_change_only=not args.all,
    )
    with ctrl:
        ctrl.run()

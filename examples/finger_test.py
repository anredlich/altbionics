#!/usr/bin/env python3
"""
Surge hand finger test — move individual joints, using the datasheet protocol.

The hand must be powered (48V) and the USB dongle connected. If you just
powered it on, it will have calibrated already; this script does not require a
power-cycle (unlike hand_bringup.py).

Usage:
    python examples/finger_test.py                 # gentle index wiggle
    python examples/finger_test.py --finger M      # test the middle finger
    python examples/finger_test.py --sweep         # sweep every joint
    python examples/finger_test.py --pose power    # try a preset grip
    python examples/finger_test.py --list          # list serial ports

Joint codes: I=index M=middle R=ring P=pinky T=thumb_flex X=thumb_rotation
Angles are in DEGREES (fingers 0-90, thumb flex 0-95, thumb rot 0-85).
"""

import argparse
import time

from altbionics import (
    SurgeHand, SurgeConfig, FINGERS, FINGER_CODES, POSES,
    ANGLE_LIMITS, resolve_joint, list_serial_ports, autodetect_port,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Surge finger test")
    parser.add_argument("port", nargs="?", default=None, help="serial port")
    parser.add_argument("--list", action="store_true", help="list ports, exit")
    parser.add_argument("--finger", default="index",
                        help="finger name or code to test (default index)")
    parser.add_argument("--angle", type=float, default=None,
                        help="hold this angle (deg) instead of wiggling")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep every joint in turn")
    parser.add_argument("--pose", default=None,
                        help=f"move to a preset grip: {list(POSES)}")
    args = parser.parse_args()

    ports = list_serial_ports()
    print("Visible serial ports:", ports or "(none)")
    if args.list:
        return

    port = args.port or autodetect_port()
    if port is None:
        print("No port found. Plug in the dongle or pass a port explicitly.")
        return

    # Direct control: don't enable Manus glove; do enable OK/ERROR feedback.
    cfg = SurgeConfig(port=port, send_smc_on_connect=False,
                      enable_feedback_on_connect=True)

    with SurgeHand(config=cfg) as hand:
        print(f"Connected to {port}.\n")

        if args.pose:
            print(f"Moving to preset pose: {args.pose}")
            hand.pose(args.pose)
            time.sleep(1.5)
            print("Returning to open.")
            hand.pose("open")
            return

        if args.sweep:
            for name, code in FINGER_CODES.items():
                lo, hi = ANGLE_LIMITS[code]
                mid = (lo + hi) // 2
                print(f"  {name} ({code}): {lo} -> {mid} -> {lo} deg")
                hand.set_finger(code, mid)
                time.sleep(0.6)
                hand.set_finger(code, lo)
                time.sleep(0.4)
            print("Sweep done.")
            return

        code = resolve_joint(args.finger)
        lo, hi = ANGLE_LIMITS[code]
        if args.angle is not None:
            print(f"Holding {args.finger} ({code}) at {args.angle} deg.")
            hand.set_finger(code, args.angle)
            time.sleep(1.5)
        else:
            mid = (lo + hi) // 2
            print(f"Wiggling {args.finger} ({code}): {lo} -> {mid} -> {lo} deg")
            hand.set_finger(code, mid)
            time.sleep(0.8)
        print(f"Returning {args.finger} to {lo} deg (open).")
        hand.set_finger(code, lo)
        print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Print the Surge hand's full startup / calibration transcript to the terminal.

Unlike the GUI (which shows only the last ~26 lines), this prints EVERY line
the hand emits on power-up, so you can read the whole calibration output —
welcome banner, versions, idle-current test, per-finger min/max, encoder
results, and the pass/fail line.

Usage:
    python examples/startup_log.py                 # autodetect, listen 90s
    python examples/startup_log.py --seconds 120   # listen longer
    python examples/startup_log.py --save cal.txt  # also write to a file
    python examples/startup_log.py /dev/cu.usbserial-XXXX   # explicit port

How to use:
    1. Run the script (hand can already be powered — that's fine).
    2. When it says "listening", UNPLUG the hand's 48V power, wait ~3s, and
       plug it back in. The startup sequence will stream in as it boots.
    3. It stops automatically after it sees "Startup Sequence Complete", or
       after the timeout, whichever comes first.
"""

import argparse
import time

from altbionics import SurgeHand, SurgeConfig, list_serial_ports, autodetect_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Print full Surge startup log")
    parser.add_argument("port", nargs="?", default=None, help="serial port")
    parser.add_argument("--list", action="store_true", help="list ports, exit")
    parser.add_argument("--seconds", type=float, default=90.0,
                        help="max seconds to listen (default 90)")
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="also write the transcript to this file")
    args = parser.parse_args()

    ports = list_serial_ports()
    print("Visible serial ports:", ports or "(none)")
    if args.list:
        return

    port = args.port or autodetect_port()
    if port is None:
        print("No port found. Plug in the dongle or pass a port explicitly.")
        return

    # log_incoming=False here because we print lines ourselves (cleaner output).
    cfg = SurgeConfig(port=port, send_smc_on_connect=False,
                      enable_feedback_on_connect=False,   # stay quiet; pure listen
                      log_commands=False, log_incoming=False)

    lines: list[str] = []
    print(f"\nConnecting to {port} ...")
    with SurgeHand(config=cfg) as hand:
        hand.flush_input()
        print("=" * 70)
        print("LISTENING — now UNPLUG the hand's 48V power, wait ~3s, plug back in.")
        print(f"(Will stop on 'Startup Sequence Complete' or after {args.seconds:.0f}s.)")
        print("=" * 70)

        start = time.time()
        n = 0
        while time.time() - start < args.seconds:
            line = hand.read_line(timeout=0.2)
            if line is None:
                continue
            n += 1
            print(f"{n:3d} | {line}")
            lines.append(line)
            if "startup sequence complete" in line.lower():
                print("=" * 70)
                print("Startup sequence complete — done.")
                break
        else:
            print("=" * 70)
            print(f"Timed out after {args.seconds:.0f}s "
                  f"({n} lines captured).")

    if args.save and lines:
        with open(args.save, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Transcript written to {args.save}")

    # quick summary of the useful bits
    print("\n--- summary ---")
    for line in lines:
        low = line.lower()
        if ("version" in low or "min:" in low or "max:" in low
                or "calibration" in low or "error" in low or "passed" in low
                or "failed" in low):
            print("   ", line)


if __name__ == "__main__":
    main()

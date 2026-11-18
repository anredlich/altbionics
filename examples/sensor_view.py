#!/usr/bin/env python3
"""
Live viewer for a Sensible Robotics tactile finger sensor (terminal).

Connects to a sensor over USB-MIDI, decodes frames, and prints the live 10x10
pressure grid (physical orientation, blanks shown as '.'), plus total/peak
force and the contact centroid. Press the sensor and watch the numbers move.

This is a driver smoke-test: it exercises the full path (MIDI in -> decoded
TactileFrame) independently of the pygame app.

Usage:
    python examples/sensor_view.py                 # sensor 1, live grid
    python examples/sensor_view.py --sensor 2      # sensor 2
    python examples/sensor_view.py --list          # list MIDI ports, exit
    python examples/sensor_view.py --compact        # one summary line/frame
    python examples/sensor_view.py --hz 5           # limit refresh rate

Notes:
  * Close the browser RoboSkin visualizer first — Chrome and Python can't both
    hold the MIDI device.
  * If it says no port found, replug the Central PCB's USB-C (power-cycle) and
    make sure the sensor's JST is seated, then retry.
  * Ctrl-C to quit.
"""

import argparse
import select
import sys
import termios
import time
import tty

# Import the sensor module directly so we don't require pyserial (which the
# package __init__ pulls in via hand.py). This keeps the sensor viewer usable
# on a MIDI-only setup.
try:
    from altbionics.sensor import (
        TactileSensor, ACTIVE_MASK, list_ports, find_sensor_ports,
    )
except ImportError as exc:
    print("Could not import the sensor driver:", exc)
    print("Install MIDI deps if needed:  pip install mido python-rtmidi")
    sys.exit(1)


CLEAR = "\033[2J\033[H"   # ANSI: clear screen + home cursor


def _getch_nonblocking() -> str | None:
    """Return a pressed key without blocking, or None. POSIX (termios)."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def render_grid(values) -> str:
    """Return a printable 10x10 grid, blanks as '.', right-aligned."""
    lines = []
    for r in range(10):
        cells = []
        for c in range(10):
            if not ACTIVE_MASK[r, c]:
                cells.append("    .")
            else:
                cells.append(f"{values[r, c]:5.0f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live tactile sensor viewer")
    parser.add_argument("--sensor", type=int, default=1,
                        help="sensor index 1 or 2 (default 1)")
    parser.add_argument("--port", default=None,
                        help="explicit MIDI port name (overrides --sensor)")
    parser.add_argument("--list", action="store_true",
                        help="list MIDI input ports and exit")
    parser.add_argument("--compact", action="store_true",
                        help="print one summary line per frame (no grid)")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="max screen refreshes per second (default 10)")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="live smoothing 0..0.95 (0=off; try 0.6 for noisy taxels)")
    args = parser.parse_args()

    print("MIDI input ports:", list_ports() or "(none)")
    print("Sensor-like ports:", find_sensor_ports() or "(none)")
    if args.list:
        return

    try:
        sensor = TactileSensor(port=args.port, index=args.sensor)
        sensor.connect()
    except Exception as exc:
        print("\nCould not connect:", exc)
        return

    print(f"\nConnected to: {sensor.port_name}")
    print("Keys:  t=tare (untouched!)  c=clear tare  [ / ]=less/more smoothing  q=quit")
    print("Press the sensor. Ctrl-C also quits.\n")
    time.sleep(0.6)

    # start with the smoothing passed on the command line
    sensor.set_smoothing(args.smooth)

    min_dt = 1.0 / args.hz if args.hz > 0 else 0.0
    last_draw = 0.0

    # Put the terminal in cbreak mode so single keys register without Enter.
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _getch_nonblocking()
            if key:
                k = key.lower()
                if k == "q":
                    break
                elif k == "t":
                    sensor.tare(n_frames=8)
                elif k == "c":
                    sensor.clear_tare()
                elif k == "]":
                    sensor.set_smoothing(min(0.95, sensor.smoothing + 0.1))
                elif k == "[":
                    sensor.set_smoothing(max(0.0, sensor.smoothing - 0.1))

            frame = sensor.read()
            now = time.time()
            if frame is None:
                time.sleep(0.005)
                continue
            if now - last_draw < min_dt:
                continue
            last_draw = now

            if sensor.taring:
                tare_s = "TARING…"
            elif sensor.is_tared:
                tare_s = "tared"
            else:
                tare_s = "raw"

            cen = frame.centroid()
            cen_s = f"({cen[0]:.1f},{cen[1]:.1f})" if cen else "  (no contact)"
            summary = (f"frame#{frame.counter:3d}  [{tare_s}]  "
                       f"smooth={sensor.smoothing:.1f}  "
                       f"total={frame.total_force():8.0f}  "
                       f"peak={frame.max_force():7.0f}  "
                       f"centroid(r,c)={cen_s}  "
                       f"dropped={sensor.dropped}")

            if args.compact:
                print(summary)
            else:
                sys.stdout.write(CLEAR)
                print("Sensible Robotics tactile sensor — live")
                print(f"{sensor.port_name}   [{tare_s}]")
                print()
                print(render_grid(frame.values))
                print()
                print(summary)
                print("\nKeys: t=tare  c=clear  [ ]=smoothing  q=quit")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        sensor.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()

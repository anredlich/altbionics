#!/usr/bin/env python3
"""
Raw MIDI capture for the Sensible Robotics finger sensor.

THIS IS THE FIRST THING TO RUN WHEN THE SENSOR HARDWARE ARRIVES. It dumps
every incoming MIDI message so we can see how taxel data is actually encoded,
which is the one detail needed to finish altbionics/sensor.py.

Usage:
    python examples/capture_sensor_midi.py --list      # list MIDI input ports
    python examples/capture_sensor_midi.py             # autodetect + capture
    python examples/capture_sensor_midi.py "Port Name" # capture a named port
    python examples/capture_sensor_midi.py --save capture.txt   # also log to file

While it runs, press / release the sensor and watch what message types and
values appear. Note the msg.type (control_change / sysex / note_on / ...),
which fields change, and their ranges. Paste a sample back and we finish the
decoder.

First, though: the vendor's own check works on Mac with no Python at all —
plug in the Central PCB, open the finger114 URL in Chrome, allow MIDI, and
confirm the taxels respond. Do that before debugging Python.
"""

import argparse
import sys
import time

try:
    import mido
except ImportError:
    print("mido not installed. Run:  pip install mido python-rtmidi")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw MIDI capture")
    parser.add_argument("port", nargs="?", default=None, help="MIDI input port name")
    parser.add_argument("--list", action="store_true", help="list ports and exit")
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="also append messages to FILE")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="auto-stop after N seconds (0 = run until Ctrl-C)")
    args = parser.parse_args()

    ports = mido.get_input_names()
    print("MIDI input ports:", ports or "(none found)")
    if args.list:
        return
    if not ports:
        print("No MIDI inputs. Plug in the Central PCB (USB-C) and retry.")
        return

    port = args.port
    if port is None:
        if len(ports) == 1:
            port = ports[0]
        else:
            print("Multiple ports found; specify one explicitly.")
            return
    print(f"Opening: {port!r}")
    print("Press/release the sensor. Ctrl-C to stop.\n")

    out = open(args.save, "a") if args.save else None
    count = 0
    t0 = time.time()
    try:
        with mido.open_input(port) as inport:
            for msg in inport:
                ts = time.time() - t0
                line = f"[{ts:8.3f}] {msg}"
                print(line)
                if out:
                    out.write(line + "\n")
                    out.flush()
                count += 1
                if args.seconds and ts >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()
        print(f"\nCaptured {count} messages.")


if __name__ == "__main__":
    main()

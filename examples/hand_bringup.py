#!/usr/bin/env python3
"""
Surge hand bring-up: connect and stream the hand's startup messages.

Run this the first time you plug in the hand's USB communication dongle (with
the hand already powered by its 48V supply). Per the Surge setup sheet, on
power-up the hand emits a welcome message, firmware version, and a calibration
status (pass/fail), then becomes ready to accept AT commands. This script opens
the serial port and prints everything the hand says, so you can watch it boot
and confirm calibration succeeded before sending any finger commands.

Usage:
    python examples/hand_bringup.py                 # autodetect port, listen 20s
    python examples/hand_bringup.py /dev/cu.usbserial-XXXX
    python examples/hand_bringup.py --list          # just list serial ports
    python examples/hand_bringup.py --seconds 30    # listen longer
    python examples/hand_bringup.py --no-smc        # don't send AT+SMC on connect

Notes:
  * This script does NOT move any fingers. It only listens (and, unless
    --no-smc is given, sends the same AT+SMC=1 the driver normally sends on
    connect). It's safe to run repeatedly.
  * The exact text of the welcome/calibration messages isn't documented, so
    this prints raw lines. Once you see what the hand actually sends, we can
    teach the driver to recognize "ready" / "calibration passed" etc.
  * Tip: start this script FIRST, then power-cycle or plug in the hand, so you
    catch the very first power-up lines.
"""

import argparse

from altbionics import SurgeHand, SurgeConfig, list_serial_ports, autodetect_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Surge hand bring-up / listener")
    parser.add_argument("port", nargs="?", default=None,
                        help="serial port (default: autodetect)")
    parser.add_argument("--list", action="store_true",
                        help="list visible serial ports and exit")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to listen for startup messages (default 20)")
    parser.add_argument("--no-smc", action="store_true",
                        help="do NOT send AT+SMC=1 on connect (pure listen)")
    parser.add_argument("--stop-on", default=None,
                        help="stop early when a line contains this text "
                             "(e.g. 'ready' or 'calibration') — leave unset on "
                             "first run until you know the wording")
    args = parser.parse_args()

    ports = list_serial_ports()
    print("Visible serial ports:", ports or "(none found)")
    if args.list:
        return

    port = args.port or autodetect_port()
    if port is None:
        print("No port specified and autodetect found nothing.")
        print("Plug in the hand's USB dongle, or pass a port explicitly.")
        return

    print(f"\nConnecting to {port} ...")
    print("Listening for the hand's startup / calibration / status messages.")
    print("(Power-cycle the hand now if you want to catch the very first lines.)\n")

    cfg = SurgeConfig(
        port=port,
        send_smc_on_connect=not args.no_smc,
        log_incoming=True,     # print every incoming line with a timestamp
        log_commands=True,     # print every outgoing command
        listen_on_connect=0.0, # we'll listen explicitly below instead
    )

    hand = SurgeHand(config=cfg)
    hand.connect()
    try:
        lines = hand.listen(
            duration=args.seconds,
            stop_on=args.stop_on,
            idle_timeout=None,   # listen the full duration on first run
        )
        print("\n--- summary ---")
        if lines:
            print(f"Received {len(lines)} line(s) from the hand:")
            for ln in lines:
                print("   ", ln)
        else:
            print("No data received. Check that:")
            print("  * the hand is powered (48V supply on),")
            print("  * the USB dongle is the RS-485 adapter and is seated,")
            print("  * the port is correct (try --list),")
            print("  * try power-cycling the hand while this script listens.")
    finally:
        hand.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()

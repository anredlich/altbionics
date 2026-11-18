import math, time, threading
from glove import ManusGlove, quat_to_euler_deg


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0

# --- live control via background stdin reader --------------------------------
# ENTER = (re)latch current orientation as the reference; q+ENTER = quit.
# Daemon thread so the readout loop never blocks on input. -anr
_ctl = {"relatch": True, "quit": False}   # start True so the first frame latches
def _key_loop():
    while True:
        try:
            s = input()
        except EOFError:
            return
        if s.strip().lower() == "q":
            _ctl["quit"] = True
            return
        _ctl["relatch"] = True
threading.Thread(target=_key_loop, daemon=True).start()

ref_yaw = None            # reference heading, set on first frame / each ENTER
max_abs_dyaw = 0.0        # worst |drift| since the last latch
_last_cnt, _last_t, fps = 0, time.time(), 0.0

g = ManusGlove().start()
print("Rotate your WHOLE HAND (not fingers).")
print("ENTER = re-zero reference  |  q+ENTER = quit\n")
try:
    while True:
        if _ctl["quit"]:
            break
        r = g.read_hand_rotation()
        if r is None:
            print("\r(waiting for data...)", end="", flush=True)
        else:
            n = math.sqrt(sum(c*c for c in r))
            roll, pitch, yaw = quat_to_euler_deg(r)

            # (re)latch reference on ENTER or first frame
            if _ctl["relatch"] or ref_yaw is None:
                ref_yaw = yaw
                max_abs_dyaw = 0.0
                _ctl["relatch"] = False

            d_yaw = wrap180(yaw - ref_yaw)          # drift from reference pose
            max_abs_dyaw = max(max_abs_dyaw, abs(d_yaw))

            # crude frame-rate: an fps dip during motion == dropped frames
            now = time.time()
            if now - _last_t >= 0.5:
                fps = (g.frame_count - _last_cnt) / (now - _last_t)
                _last_cnt, _last_t = g.frame_count, now

            flags = ""
            if abs(n - 1.0) > 0.02:  flags += " BADQ"     # quaternion degenerate
            if abs(pitch) > 75.0:    flags += " GIMBAL"   # yaw unstable near pitch=+/-90

            print(f"\rframes={g.frame_count} fps={fps:5.1f}  |q|={n:.3f}  "
                  f"roll={roll:+6.1f} pitch={pitch:+6.1f} yaw={yaw:+7.1f}  "
                  f"d_yaw={d_yaw:+6.1f} (max {max_abs_dyaw:4.1f}){flags}   ",
                  end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    g.stop()

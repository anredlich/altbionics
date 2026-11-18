# altbionics

Pure-Python drivers for the **Alt-Bionics Surge** robot hand and the **Manus
MetaGloves Pro**, for teleoperation and imitation-learning data collection
(e.g. with LeRobot).

The **hand** driver speaks the Surge AT-command protocol over RS-485 serial and
needs no proprietary software. The **glove** driver reads fingertip data through
the Manus Core SDK, which you must obtain separately from Manus (see
Requirements).

## Package layout

- `altbionics.hand` — `SurgeHand`: control the 6-DOF Surge hand over RS-485
  serial (AT-command protocol, reimplemented from the public Surge datasheet).
- `altbionics.glove` — `ManusGlove`: read raw fingertip positions from a Manus
  MetaGloves Pro via a ctypes binding to the Manus Core SDK.
- `altbionics.ik` — glove to hand finger retargeting (polynomial regression).
- `altbionics.fit_hand_coeffs` — fit the IK regression coefficients from
  captured glove/hand correspondence data.
- `altbionics.calibrate` — `HandCalibrator`: interactive per-finger trim.
- `altbionics.sensor` — `TactileSensor`: optional tactile-sensor support.
- `altbionics.teleop` — teleoperation helpers.
- `altbionics.rotation_probe` — small diagnostic utility.

`examples/` contains runnable scripts, including `calibrate_app.py` (the
interactive glove/hand calibration app), `hand_test.py`, `hand_bringup.py`, and
tactile-sensor viewers.

## Requirements

### Python
- Python 3.10+
- Core: `numpy`, `pyserial` (installed automatically).
- Examples (GUI / MIDI): `pygame`, `mido`, `python-rtmidi` — install with the
  `examples` extra (below).

### Hardware
- **Alt-Bionics Surge hand** (6-DOF), connected through a USB-to-RS-485 (FTDI)
  adapter, with its separate 48 V power supply.
- **Manus MetaGloves Pro** + the Manus USB dongle (only for the glove half).

### Proprietary / third-party — NOT included in this repo
The **hand** half needs only the Python packages above.

The **glove** half requires the **Manus Core SDK**, which is **not distributed
here** and must be obtained from Manus:

- `libManusSDK_Integrated.so` — the Integrated Linux build of the SDK
  (developed/tested against **v3.1.1**). Point `ManusGlove(so_path=...)` at your
  local copy.
- A **licensed Manus USB dongle** — the Manus SDK only runs with the licensed
  dongle present (the one included with your Manus kit). Without it, the glove
  half will not stream data.
- Get the SDK and license from Manus: https://www.manus-meta.com

> The ctypes struct layouts in `glove.py` mirror Manus's `ManusSDKTypes.h`
> (v3.1.1). If you use a different SDK version, confirm the layouts still match.

### OS support
- **Linux**: hand on `/dev/ttyUSB*` or `/dev/ttyACM*`; glove via
  `libManusSDK_Integrated.so`.
- **macOS**: the hand works (`/dev/cu.usbserial-*`); the Manus Integrated `.so`
  is Linux-only, so the glove half is Linux-only.

## Install

```bash
git clone https://github.com/anredlich/altbionics.git
cd altbionics
pip install -e .              # core: hand + glove + IK
pip install -e ".[examples]"  # also install GUI/MIDI deps for the examples
```

## Quick start

Hand (no proprietary software needed):

```python
from altbionics import SurgeHand

with SurgeHand(autodetect=True) as hand:
    hand.wait_until_ready()        # after a power-cycle
    hand.set_finger("index", 45)   # degrees
    hand.pose("power")             # preset grip
    print(hand.get_angle("index"))
```

Glove (requires the Manus SDK + licensed dongle):

```python
from altbionics.glove import ManusGlove

g = ManusGlove(so_path="~/manus-sdk/.../libManusSDK_Integrated.so").start()
print(g.read_sensors())   # 5 (x,y,z) fingertip positions, or None until first frame
g.stop()
```

## License

MIT — see [LICENSE](LICENSE).

Third-party SDKs (the Manus Core SDK and any other vendor software) are under
their own licenses and are **not** included in this repository.

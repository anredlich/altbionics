"""
altbionics — cross-platform Python control for the Alt Bionics Surge hand,
Sensible Robotics tactile finger sensors, and (later) the Manus glove.

- Hand   : pyserial, AT-command protocol (per Surge datasheet v1.3).
           Cross-platform (macOS + Linux).
- Sensor : USB-MIDI via mido/python-rtmidi. Cross-platform (macOS + Linux).
           Decoder is a skeleton pending real-hardware capture — see
           altbionics/sensor.py and examples/capture_sensor_midi.py.
- Glove  : Manus SDK (libManusSDK_Integrated.so) via ctypes, Linux only.
           Reads raw fingertip positions; feeds the finger IK (altbionics.ik)
           to drive the Surge hand. See altbionics/glove.py, teleop.py.
           
The hand needs only the base environment.yml. The sensor needs the extra
'sensors' dependencies (mido, python-rtmidi); see environment-sensors.yml.
Referencing the sensor symbols does NOT import mido until you actually use
TactileSensor, so this stays importable on a hand-only environment.
"""

from .hand import (
    SurgeHand,
    SurgeConfig,
    FINGERS,
    FINGER_CODES,
    POSES,
    ANGLE_LIMITS,
    resolve_joint,
    list_serial_ports,
    autodetect_port,
)

__all__ = [
    "SurgeHand",
    "SurgeConfig",
    "FINGERS",
    "FINGER_CODES",
    "POSES",
    "ANGLE_LIMITS",
    "resolve_joint",
    "list_serial_ports",
    "autodetect_port",
    "TactileSensor",
    "TactileFrame",
    # glove -> hand teleop (lazy; see __getattr__)
    "ManusGlove",
    "GloveHandController",
    "ik",
]

__version__ = "0.4.0"


def __getattr__(name):
    # Lazy imports so `import altbionics` works on a hand-only environment
    # (no mido, no numpy, no Manus .so). Each symbol is pulled in only when
    # actually referenced.
    #
    # IMPORTANT: submodules (ik, sensor) are imported with importlib.import_module,
    # NOT `from . import ik`. Under an ABSOLUTE import (`from altbionics import ik`)
    # Python is already resolving the package attribute `ik` when it calls this
    # __getattr__; a `from . import ik` here asks for that same attribute again and
    # recurses infinitely. import_module(".ik", __name__) loads the submodule
    # directly and binds it, with no re-entry into __getattr__.
    import importlib
    if name in ("TactileSensor", "TactileFrame"):
        sensor = importlib.import_module(".sensor", __name__)
        return getattr(sensor, name)
    if name == "ManusGlove":                 # Manus glove reader (Linux + libManusSDK.so)
        glove = importlib.import_module(".glove", __name__)
        return glove.ManusGlove
    if name == "GloveHandController":         # glove -> ik -> hand loop
        teleop = importlib.import_module(".teleop", __name__)
        return teleop.GloveHandController
    if name == "ik":                          # finger IK module (needs numpy for refit)
        return importlib.import_module(".ik", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# from .hand import (
#     SurgeHand,
#     SurgeConfig,
#     FINGERS,
#     FINGER_CODES,
#     POSES,
#     ANGLE_LIMITS,
#     resolve_joint,
#     list_serial_ports,
#     autodetect_port,
# )

# __all__ = [
#     "SurgeHand",
#     "SurgeConfig",
#     "FINGERS",
#     "FINGER_CODES",
#     "POSES",
#     "ANGLE_LIMITS",
#     "resolve_joint",
#     "list_serial_ports",
#     "autodetect_port",
#     "TactileSensor",
#     "TactileFrame",
# ]

# __version__ = "0.3.0"


# def __getattr__(name):
#     # Lazy-import the sensor classes so `import altbionics` works even when the
#     # MIDI dependencies (mido/python-rtmidi) aren't installed. They're only
#     # pulled in if you actually reference TactileSensor / TactileFrame.
#     if name in ("TactileSensor", "TactileFrame"):
#         from . import sensor
#         return getattr(sensor, name)
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

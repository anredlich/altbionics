"""
ManusGlove — read raw fingertip (EMF sensor) positions from a Manus MetaGloves
Pro glove on Linux, via a direct ctypes binding to the Manus Core SDK shared
library (libManusSDK_Integrated.so). No ROS dependency.

Why raw device data (not ergonomics/skeleton)?
-----------------------------------------------
The finger IK (altbionics.ik) was fit from the 5 raw fingertip positions the
glove reports per hand (sensor 0=thumb .. 4=pinky), each an (x,y,z) in meters
relative to the transmitter box on the back of the hand. These raw positions
are clean and reproducible; the SDK's ergonomics angles were too noisy for the
thumb. So we bind the raw-device-data stream.

This mirrors, in Python, exactly what the Linux SDKClient.cpp does:
    CoreSdk_InitializeIntegrated()
    (register the raw-device-data callback)
    CoreSdk_InitializeCoordinateSystemWithVUH(z-up, x-from-viewer, right, meters)
    -- then each time the SDK fires the callback on its own thread:
         CoreSdk_GetRawDeviceData(0, &rawDeviceData)
         read rawDeviceData.sensorData[j].position.{x,y,z}  for j in 0..4

All struct layouts below are byte-exact against ManusSDKTypes.h (3.1.1):
    ManusVec3        = 3 floats                      (12 B)
    ManusQuaternion  = 4 floats (w,x,y,z)            (16 B)
    ManusTransform   = position + rotation + scale   (40 B)   <- per-sensor stride
    RawDeviceData    = id + sensorCount + 5xTransform + rotation  (224 B)
    RawDeviceDataInfo= ManusTimestamp(8) + count(4)

read_sensors() returns the latest frame's 5 (x,y,z) tuples (thread-safe), or
None if no frame has arrived yet. read_transforms() additionally exposes the
per-sensor rotation quaternion (available for a future thumb-rotation IK).
"""

from __future__ import annotations

import ctypes as C
import os
import threading
from typing import Optional

DEFAULT_SO = os.path.expanduser(
    "~/manus-sdk/ManusSDK_v3.1.1/SDKClient_Linux/ManusSDK/lib/libManusSDK_Integrated.so"
)

# ---- enum ordinals (from ManusSDKTypes.h; _Invalid = 0, then increment) -----
SIDE_RIGHT = 2             # Side: Invalid=0, Left=1, Right=2, Center=3
AXISVIEW_XFROMVIEWER = 3   # AxisView: Invalid=0, ZFromViewer=1, YFromViewer=2, XFromViewer=3, ...
AXISPOLARITY_POSITIVEZ = 6 # AxisPolarity: Invalid=0, NegZ=1, NegY=2, NegX=3, PosX=4, PosY=5, PosZ=6
SDK_SUCCESS = 0            # SDKReturnCode_Success = 0

MAX_RAW_DEVICE_DATA_SENSOR_SIZE = 5


# ---------------------------------------------------------------------------
# ctypes structs — byte-exact against ManusSDKTypes.h
# ---------------------------------------------------------------------------
class ManusVec3(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]

class ManusQuaternion(C.Structure):
    _fields_ = [("w", C.c_float), ("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]

class ManusTransform(C.Structure):
    _fields_ = [("position", ManusVec3),
                ("rotation", ManusQuaternion),
                ("scale", ManusVec3)]

class RawDeviceData(C.Structure):
    _fields_ = [("id", C.c_uint32),
                ("sensorCount", C.c_uint32),
                ("sensorData", ManusTransform * MAX_RAW_DEVICE_DATA_SENSOR_SIZE),
                ("rotation", ManusQuaternion)]

class ManusTimestamp(C.Structure):
    _fields_ = [("time", C.c_uint64)]

class RawDeviceDataInfo(C.Structure):
    _fields_ = [("publishTime", ManusTimestamp),
                ("rawDeviceDataCount", C.c_uint32)]

class CoordinateSystemVUH(C.Structure):
    _fields_ = [("view", C.c_int),
                ("up", C.c_int),
                ("handedness", C.c_int),
                ("unitScale", C.c_float)]

# ManusHost (passed BY VALUE to CoreSdk_ConnectToHost). For integrated mode we
# connect to an all-zeros host (same as ManusHost_Init + ConnectToHost in the
# C++ client). Sizes from the header: hostName[256], ipAddress[40],
# Version = 3*uint32 + 3*char[16].
MAX_NUM_CHARS_IN_HOST_NAME = 256
MAX_NUM_CHARS_IN_IP_ADDRESS = 40
MAX_NUM_CHARS_IN_VERSION = 16

class Version(C.Structure):
    _fields_ = [("major", C.c_uint32), ("minor", C.c_uint32), ("patch", C.c_uint32),
                ("label", C.c_char * MAX_NUM_CHARS_IN_VERSION),
                ("sha", C.c_char * MAX_NUM_CHARS_IN_VERSION),
                ("tag", C.c_char * MAX_NUM_CHARS_IN_VERSION)]

class ManusHost(C.Structure):
    _fields_ = [("hostName", C.c_char * MAX_NUM_CHARS_IN_HOST_NAME),
                ("ipAddress", C.c_char * MAX_NUM_CHARS_IN_IP_ADDRESS),
                ("manusCoreVersion", Version)]

# Callback type: void (*)(const RawDeviceDataInfo*)
RawDeviceDataStreamCallback_t = C.CFUNCTYPE(None, C.POINTER(RawDeviceDataInfo))

import math
def quat_to_euler_deg(q):
    """(w,x,y,z) -> (roll, pitch, yaw) degrees, ZYX. Clamps pitch at the poles."""
    w, x, y, z = q
    roll  = math.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    sinp  = max(-1.0, min(1.0, 2.0*(w*y - z*x)))
    pitch = math.asin(sinp)
    yaw   = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

class ManusGlove:
    """
    Raw-sensor-position source for one Manus glove, via libManusSDK_Integrated.so.

        g = ManusGlove().start()
        s = g.read_sensors()     # -> [(x,y,z)]*5 meters, or None until first frame
        ...
        g.stop()
    """

    def __init__(self, so_path: str = DEFAULT_SO):
        self.so_path = so_path
        self._lib: Optional[C.CDLL] = None
        self._cb = None                     # keep a ref so the callback isn't GC'd
        self._lock = threading.Lock()
        self._latest_pos: "Optional[list[tuple[float,float,float]]]" = None
        self._latest_quat: "Optional[list[tuple[float,float,float,float]]]" = None
        self._started = False
        self._frame_count = 0
        self._latest_hand_quat: "Optional[tuple[float,float,float,float]]" = None

    # -- public API ----------------------------------------------------------

    def read_sensors(self) -> "Optional[list[tuple[float,float,float]]]":
        """Latest 5 (x,y,z) positions (meters), or None if no frame yet."""
        with self._lock:
            return None if self._latest_pos is None else list(self._latest_pos)

    def read_transforms(self):
        """Latest (positions, quaternions) — 5 each — or (None, None)."""
        with self._lock:
            pos = None if self._latest_pos is None else list(self._latest_pos)
            quat = None if self._latest_quat is None else list(self._latest_quat)
            return pos, quat

    def read_hand_rotation(self):
        """Latest whole-hand orientation quaternion (w,x,y,z) from the device
        IMU (RawDeviceData.rotation), or None. Distinct from the per-sensor
        fingertip quats in read_transforms(). -anr"""
        with self._lock:
            return None if self._latest_hand_quat is None else tuple(self._latest_hand_quat)
        
    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def start(self) -> "ManusGlove":
        self._load()
        self._init_sdk()
        self._register_callback()
        self._init_coordinate_system()
        self._connect()               # integrated: connect to empty host -> starts streams
        self._started = True
        return self

    def stop(self) -> None:
        if self._lib is not None and self._started:
            try:
                self._lib.CoreSdk_ShutDown()
            except Exception:
                pass
        self._started = False

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.stop()

    # -- internals -----------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.so_path):
            raise FileNotFoundError(
                f"Manus SDK library not found: {self.so_path}. "
                "Point so_path at libManusSDK_Integrated.so on this machine."
            )
        self._lib = C.CDLL(self.so_path, mode=C.RTLD_GLOBAL)
        lib = self._lib
        # declare signatures (all CoreSdk_* return SDKReturnCode == c_int)
        lib.CoreSdk_InitializeIntegrated.restype = C.c_int
        lib.CoreSdk_InitializeIntegrated.argtypes = []
        lib.CoreSdk_ShutDown.restype = C.c_int
        lib.CoreSdk_ShutDown.argtypes = []
        lib.CoreSdk_RegisterCallbackForRawDeviceDataStream.restype = C.c_int
        lib.CoreSdk_RegisterCallbackForRawDeviceDataStream.argtypes = [RawDeviceDataStreamCallback_t]
        lib.CoreSdk_GetRawDeviceData.restype = C.c_int
        lib.CoreSdk_GetRawDeviceData.argtypes = [C.c_uint32, C.POINTER(RawDeviceData)]
        # struct passed BY VALUE + a bool
        lib.CoreSdk_InitializeCoordinateSystemWithVUH.restype = C.c_int
        lib.CoreSdk_InitializeCoordinateSystemWithVUH.argtypes = [CoordinateSystemVUH, C.c_bool]
        # connect (integrated: connect to an empty host). ManusHost passed BY VALUE.
        lib.CoreSdk_ConnectToHost.restype = C.c_int
        lib.CoreSdk_ConnectToHost.argtypes = [ManusHost]

    def _check(self, rc: int, what: str):
        if rc != SDK_SUCCESS:
            raise RuntimeError(f"{what} failed (SDKReturnCode={rc})")

    def _init_sdk(self) -> None:
        self._check(self._lib.CoreSdk_InitializeIntegrated(), "CoreSdk_InitializeIntegrated")

    def _register_callback(self) -> None:
        lib = self._lib

        def _on_raw(info_ptr):
            try:
                info = info_ptr.contents
                if info.rawDeviceDataCount < 1:
                    return
                dev = RawDeviceData()
                if lib.CoreSdk_GetRawDeviceData(0, C.byref(dev)) != SDK_SUCCESS:
                    return
                if dev.sensorCount < 5:
                    return
                pos = [(dev.sensorData[j].position.x,
                        dev.sensorData[j].position.y,
                        dev.sensorData[j].position.z) for j in range(5)]
                quat = [(dev.sensorData[j].rotation.w,
                         dev.sensorData[j].rotation.x,
                         dev.sensorData[j].rotation.y,
                         dev.sensorData[j].rotation.z) for j in range(5)]
                with self._lock:
                    self._latest_pos = pos
                    self._latest_quat = quat
                    self._latest_hand_quat = (dev.rotation.w, dev.rotation.x,
                            dev.rotation.y, dev.rotation.z)
                    self._frame_count += 1
            except Exception:
                pass  # never let an exception cross back into the C callback

        self._cb = RawDeviceDataStreamCallback_t(_on_raw)   # MUST keep this ref
        rc = lib.CoreSdk_RegisterCallbackForRawDeviceDataStream(self._cb)
        self._check(rc, "CoreSdk_RegisterCallbackForRawDeviceDataStream")

    def _init_coordinate_system(self) -> None:
        # z-up, x-from-viewer, right-handed, meters — identical to SDKClient.cpp.
        vuh = CoordinateSystemVUH(
            view=AXISVIEW_XFROMVIEWER,
            up=AXISPOLARITY_POSITIVEZ,
            handedness=SIDE_RIGHT,
            unitScale=1.0,
        )
        rc = self._lib.CoreSdk_InitializeCoordinateSystemWithVUH(vuh, True)
        self._check(rc, "CoreSdk_InitializeCoordinateSystemWithVUH")

    def _connect(self) -> None:
        # Integrated mode: Core runs in-process; connect to an all-zeros host
        # (identical to ManusHost_Init + CoreSdk_ConnectToHost in SDKClient.cpp).
        # This is what actually starts the data streams flowing to the callback.
        empty = ManusHost()   # ctypes zero-initializes all fields
        rc = self._lib.CoreSdk_ConnectToHost(empty)
        self._check(rc, "CoreSdk_ConnectToHost")


# ---- standalone test: print the 5 sensor positions, like the C++ [E] menu ---
if __name__ == "__main__":
    import sys, time
    so = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SO
    print(f"Loading {so}")
    g = ManusGlove(so_path=so).start()
    print("Started. Waiting for glove frames (move fingers). Ctrl-C to stop.\n")
    try:
        while True:
            s = g.read_sensors()
            if s is None:
                print("\r(waiting for data...)        ", end="", flush=True)
            else:
                line = "  ".join(f"s{j}:({x:+.4f},{y:+.4f},{z:+.4f})"
                                 for j, (x, y, z) in enumerate(s))
                print(f"\rframes={g.frame_count}  {line}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        g.stop()

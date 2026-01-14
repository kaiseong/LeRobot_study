# src/lerobot/cameras/orbbec/camera_orbbec.py

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any, List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import pyorbbecsdk as ob
except Exception as e:
    ob = None
    logging.info(f"Could not import orbbec: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """
    Minimal Orbbec camera (RGB-only).

    - color stream: OBFormat.RGB only
    - no MJPG/YUY/NV12/convert filter/stride handling
    - raw bytes -> (H,W,3) uint8
    - optional output color_mode (RGB/BGR)
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config=config)
        self.config = config

        # kept for logging only (not used for selection)
        self.serial_or_name = str(config.serial_number_or_name)
        self.serial_number: str | None = self.serial_or_name

        self.fps = int(config.fps) if config.fps else 30
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = float(config.warmup_s)

        self._pipeline: Optional[Any] = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_depth: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.width = config.width
        self.height = config.height

        if self.height and self.width:
            self.capture_width, self.capture_height = int(self.width), int(self.height)
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = int(self.height), int(self.width)
        else:
            # default to 1280x720 if unspecified
            self.capture_width, self.capture_height = 1280, 720

        self._color_format: Any = None
        self._last_logged: bool = False

        # debug counter (optional)
        self._debug_frame_idx: int = 0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_or_name})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        if ob is None:
            raise RuntimeError("pyorbbecsdk is not installed.")

        self._pipeline = ob.Pipeline()
        cfg = ob.Config()

        # color stream: RGB only
        profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)

        req_w, req_h, req_fps = int(self.capture_width), int(self.capture_height), int(self.fps)

        try:
            color_profile = profile_list.get_video_stream_profile(req_w, req_h, ob.OBFormat.RGB, req_fps)
        except Exception:
            logger.warning(f"Exact RGB profile {req_w}x{req_h}@{req_fps} not found. Using default.")
            color_profile = profile_list.get_default_video_stream_profile()

        cfg.enable_stream(color_profile)

        self.capture_width = int(color_profile.get_width())
        self.capture_height = int(color_profile.get_height())
        self.fps = int(color_profile.get_fps())
        try:
            self._color_format = color_profile.get_format()
        except Exception:
            self._color_format = None

        # depth optional (kept minimal)
        if self.use_depth:
            depth_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile = depth_list.get_video_stream_profile(
                    self.capture_width, self.capture_height, ob.OBFormat.Y16, int(self.fps)
                )
            except Exception:
                depth_profile = depth_list.get_default_video_stream_profile()
            cfg.enable_stream(depth_profile)

        # PCDP-ish sync (best effort, ignore if not supported)
        try:
            if hasattr(cfg, "set_frame_aggregate_output_mode") and hasattr(ob, "OBFrameAggregateOutputMode"):
                cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            logger.warning(f"Failed to set FULL_FRAME_REQUIRE: {e}")

        try:
            if hasattr(self._pipeline, "enable_frame_sync"):
                self._pipeline.enable_frame_sync()
        except Exception as e:
            logger.warning(f"Failed to enable frame sync: {e}")

        try:
            self._pipeline.start(cfg)
        except Exception as e:
            self._pipeline = None
            raise ConnectionError(f"Failed to start pipeline: {e}")

        if warmup:
            time.sleep(self.warmup_s)

        logger.info(
            f"{self} connected: color={self.capture_width}x{self.capture_height}@{self.fps}, fmt={self._color_format}"
        )

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"Attempted to disconnect {self}, but it appears already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass

        self._pipeline = None
        logger.info(f"{self} disconnected.")

    def read(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        frames = self._pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError("Timeout waiting for frames.")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("No color frame received.")

        img = self._process_color_rgb_only(color_frame)

        if self.use_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                self.latest_depth = self._process_depth_y16(depth_frame)

        return img

    def _process_color_rgb_only(self, frame: Any) -> np.ndarray:
        """
        Assume: frame data is packed RGB888 (H*W*3 bytes).
        """
        h, w = int(frame.get_height()), int(frame.get_width())

        if not self._last_logged:
            fmt = None
            try:
                fmt = frame.get_format()
            except Exception:
                fmt = self._color_format
            logger.warning(f"Orbbec color fmt={fmt}, size={w}x{h} (RGB-only path)")
            self._last_logged = True

        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty color data")

        # ensure we own the buffer (avoid SDK lifetime issues)
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        elif isinstance(data, memoryview):
            raw = data.tobytes()
        elif isinstance(data, np.ndarray):
            raw = np.asarray(data, dtype=np.uint8).tobytes()
        else:
            raw = bytes(data)

        arr = np.frombuffer(raw, dtype=np.uint8)

        need = h * w * 3
        if arr.size < need:
            raise RuntimeError(f"Color buffer too small: {arr.size} < {need}")
        if arr.size > need:
            arr = arr[:need]

        img_rgb = arr.reshape(h, w, 3)

        # output color mode
        if self.color_mode == ColorMode.BGR:
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            img = img_rgb

        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)

        return np.ascontiguousarray(img)

    def _process_depth_y16(self, frame: Any) -> np.ndarray:
        h, w = int(frame.get_height()), int(frame.get_width())
        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty depth data")

        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        elif isinstance(data, memoryview):
            raw = data.tobytes()
        elif isinstance(data, np.ndarray):
            raw = np.asarray(data, dtype=np.uint16).tobytes()
        else:
            raw = bytes(data)

        arr = np.frombuffer(raw, dtype=np.uint16)
        need = h * w
        if arr.size < need:
            raise RuntimeError(f"Depth buffer too small: {arr.size} < {need}")
        if arr.size > need:
            arr = arr[:need]

        img = arr.reshape(h, w)

        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)

        return np.ascontiguousarray(img)

    # ----------------- threading -----------------
    def _read_loop(self) -> None:
        if self.stop_event is None:
            return
        while not self.stop_event.is_set():
            try:
                frame_data = self.read()
                with self.frame_lock:
                    self.latest_frame = frame_data
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.warning(f"Error in read loop: {e}")
                    time.sleep(0.005)

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        if self.stop_event:
            self.stop_event.set()
        self.stop_event = Event()
        self.new_frame_event.clear()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")
        if not self.thread or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timeout waiting for frame from {self}")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise TimeoutError(f"No frame available yet from {self}")

        return np.ascontiguousarray(frame)

    @staticmethod
    def find_cameras() -> List[dict[str, Any]]:
        if ob is None:
            logger.warning("pyorbbecsdk is not installed.")
            return []

        found: List[dict[str, Any]] = []
        try:
            ctx = ob.Context()
            device_list = ctx.query_devices()
            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                try:
                    info = device.get_device_info()
                    found.append(
                        {
                            "name": info.get_name(),
                            "serial_number": info.get_serial_number(),
                            "type": "Orbbec",
                            "pid": info.get_pid(),
                            "vid": info.get_vid(),
                            "connection_type": info.get_connection_type(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error retrieving info for device index {i}: {e}")
        except Exception as e:
            logger.warning(f"Error querying Orbbec devices: {e}")

        return found

# src/lerobot/cameras/orbbec/camera_orbbec.py

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any, List

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
    """Orbbec Camera implementation for LeRobot."""

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config=config)
        self.config = config

        self.serial_or_name = str(config.serial_number_or_name)
        self.serial_number: str | None = None

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self._pipeline: ob.Pipeline | None = None
        self._device: ob.Device | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None  # color
        self.latest_depth: NDArray[Any] | None = None  # depth (optional cache)
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.width = config.width
        self.height = config.height

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
        else:
            self.capture_width, self.capture_height = None, None

        # 실제 스트림 포맷(디코딩 분기용)
        self._color_format: Any = None
        self._depth_format: Any = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        if ob is None:
            raise RuntimeError("pyorbbecsdk is not installed.")

        ctx = ob.Context()
        dev_list = ctx.query_devices()
        if dev_list.get_count() == 0:
            raise RuntimeError("No Orbbec devices found.")

        device = None
        for i in range(dev_list.get_count()):
            dev = dev_list.get_device_by_index(i)
            try:
                info = dev.get_device_info()
                sn = str(info.get_serial_number())
                name = str(info.get_name())
                if self.serial_or_name == sn or self.serial_or_name == name:
                    device = dev
                    self.serial_number = sn
                    break
            except Exception:
                continue

        if device is None:
            raise RuntimeError(f"Orbbec device '{self.serial_or_name}' not found (match by serial or name).")

        self._device = device
        self._pipeline = ob.Pipeline(device)
        cfg = ob.Config()

        # Color
        profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        target_format = ob.OBFormat.RGB if self.color_mode == ColorMode.RGB else ob.OBFormat.BGR

        req_w = self.capture_width if self.capture_width else 640
        req_h = self.capture_height if self.capture_height else 480
        req_fps = int(self.fps) if self.fps else 30

        try:
            color_profile = profile_list.get_video_stream_profile(req_w, req_h, target_format, req_fps)
        except Exception:
            color_profile = None

        if color_profile is None:
            logger.warning(f"Exact profile ({req_w}x{req_h}@{req_fps}) not found. Using default.")
            color_profile = profile_list.get_default_video_stream_profile()

        cfg.enable_stream(color_profile)

        self.capture_width = int(color_profile.get_width())
        self.capture_height = int(color_profile.get_height())
        self.fps = int(color_profile.get_fps())
        try:
            self._color_format = color_profile.get_format()
        except Exception:
            self._color_format = None

        # Depth (optional)
        if self.use_depth:
            depth_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile = depth_list.get_video_stream_profile(
                    self.capture_width, self.capture_height, ob.OBFormat.Y16, int(self.fps)
                )
            except Exception:
                depth_profile = None
            if depth_profile is None:
                depth_profile = depth_list.get_default_video_stream_profile()
            cfg.enable_stream(depth_profile)
            try:
                self._depth_format = depth_profile.get_format()
            except Exception:
                self._depth_format = None

        try:
            self._pipeline.start(cfg)
        except Exception as e:
            self._pipeline = None
            raise ConnectionError(f"Failed to start pipeline: {e}")

        if warmup:
            time.sleep(self.warmup_s)

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"Attempted to disconnect {self}, but it appears already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self._pipeline is not None:
            self._pipeline.stop()
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

        color_img = self._process_color(color_frame)

        if self.use_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                self.latest_depth = self._process_depth(depth_frame)

        return color_img

    # ----------------- 핵심 수정 1: data 타입 분기 + contiguous 보장 -----------------
    def _to_uint8_1d(self, data: Any) -> np.ndarray:
        """
        data가 bytes류면 frombuffer, ndarray면 asarray로 받아서
        최종적으로 uint8 1D contiguous로 만든다.
        """
        if isinstance(data, np.ndarray):
            arr = np.asarray(data)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            return np.ascontiguousarray(arr.reshape(-1))
        else:
            # bytes / bytearray / memoryview 등
            return np.frombuffer(data, dtype=np.uint8)

    def _to_ndarray_uint8_hwc(self, data: Any, h: int, w: int) -> np.ndarray:
        """
        RGB/BGR raw buffer(또는 ndarray)를 HxWx3 uint8 contiguous로 만든다.
        """
        if isinstance(data, np.ndarray):
            arr = np.asarray(data)
            if arr.ndim == 3:
                # (h,w,3)로 오는 케이스
                arr = arr
            else:
                # 1D로 오는 케이스
                arr = arr.reshape(h, w, 3)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            return np.ascontiguousarray(arr)
        else:
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            return np.ascontiguousarray(arr)

    def _process_color(self, frame: "ob.ColorFrame") -> np.ndarray:
        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty color data")

        h, w = int(frame.get_height()), int(frame.get_width())

        fmt = None
        try:
            fmt = frame.get_format()
        except Exception:
            fmt = self._color_format

        # MJPG 같은 압축 포맷일 가능성 방어
        if fmt is not None and hasattr(ob, "OBFormat") and fmt == ob.OBFormat.MJPG:
            jpg = self._to_uint8_1d(data)
            img_bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode MJPG frame (imdecode returned None)")
            if self.color_mode == ColorMode.RGB:
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img = img_bgr
            if self.rotation is not None:
                img = cv2.rotate(img, self.rotation)
            return np.ascontiguousarray(img)

        # 일반 raw RGB/BGR
        img = self._to_ndarray_uint8_hwc(data, h, w)

        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)

        # 최종 contiguous 강제
        return np.ascontiguousarray(img)

    # ----------------- 핵심 수정 2: depth도 같은 방식으로 contiguous 보장 -----------------
    def _process_depth(self, frame: "ob.DepthFrame") -> np.ndarray:
        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty depth data")

        h, w = int(frame.get_height()), int(frame.get_width())

        if isinstance(data, np.ndarray):
            img = np.asarray(data)
            if img.ndim != 2:
                img = img.reshape(h, w)
            if img.dtype != np.uint16:
                img = img.astype(np.uint16, copy=False)
            img = np.ascontiguousarray(img)
        else:
            img = np.frombuffer(data, dtype=np.uint16).reshape(h, w)
            img = np.ascontiguousarray(img)

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
                    time.sleep(0.005)  # 스팸 방지 + 약간의 backoff

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

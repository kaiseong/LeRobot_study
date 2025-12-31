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
    Orbbec camera.

    정책:
      - serial_number_or_name은 받지만 "디바이스 선택"에는 사용하지 않음
      - 항상 ob.Pipeline() (기본/첫번째 장치)로 구동
      - (가능하면) FULL_FRAME_REQUIRE + frame_sync 활성화
      - frame.get_data() -> bytes로 확실히 복사 후 numpy로 변환
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config=config)
        self.config = config

        # 사용자 입력 값(표기/로그용). 장치 선택에는 사용하지 않음.
        self.serial_or_name = str(config.serial_number_or_name)
        self.serial_number: str | None = self.serial_or_name  # display only

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

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
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
        else:
            self.capture_width, self.capture_height = None, None

        self._color_format: Any = None
        self._depth_format: Any = None

        self._conv_filter: Optional[Any] = None
        self._last_logged_fmt: Any = object()

    def __str__(self) -> str:
        # 시리얼을 "표기"만 한다 (선택에는 사용 X)
        return f"{self.__class__.__name__}({self.serial_or_name})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    # ----------------- helpers -----------------
    def _get_stride_bytes(self, frame: Any) -> Optional[int]:
        if hasattr(frame, "get_stride"):
            try:
                s = int(frame.get_stride())
                return s if s > 0 else None
            except Exception:
                return None
        return None

    def _frame_bytes_copy(self, frame: Any) -> bytes:
        """
        get_data() 결과를 '확실히 복사한 bytes'로 만든다.
        (공식 예제/PCDP 스타일)
        """
        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty frame data")

        if isinstance(data, np.ndarray):
            return np.asarray(data).tobytes()
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, memoryview):
            return data.tobytes()
        return bytes(data)

    def _u8(self, frame: Any) -> np.ndarray:
        return np.frombuffer(self._frame_bytes_copy(frame), dtype=np.uint8)

    def _u16(self, frame: Any) -> np.ndarray:
        return np.frombuffer(self._frame_bytes_copy(frame), dtype=np.uint16)

    def _reshape_hwc3(self, raw_u8: np.ndarray, h: int, w: int, stride_bytes: Optional[int]) -> np.ndarray:
        if stride_bytes is not None and stride_bytes > 0 and stride_bytes != w * 3:
            total = h * stride_bytes
            if raw_u8.size < total:
                raise RuntimeError(
                    f"Color buffer too small: raw={raw_u8.size}, expected>={total} (stride={stride_bytes})"
                )
            mat = raw_u8[:total].reshape(h, stride_bytes)
            mat = mat[:, : w * 3]
            return np.ascontiguousarray(mat.reshape(h, w, 3))

        need = h * w * 3
        if raw_u8.size < need:
            raise RuntimeError(f"Color buffer too small: raw={raw_u8.size}, expected>={need}")
        if raw_u8.size > need:
            raw_u8 = raw_u8[:need]
        return np.ascontiguousarray(raw_u8.reshape(h, w, 3))

    def _reshape_hw_u16(self, raw_u16: np.ndarray, h: int, w: int, stride_bytes: Optional[int]) -> np.ndarray:
        if stride_bytes is not None and stride_bytes > 0:
            stride_u16 = stride_bytes // 2
            if stride_u16 > 0 and stride_u16 != w:
                total = h * stride_u16
                if raw_u16.size < total:
                    raise RuntimeError(
                        f"Depth buffer too small: raw={raw_u16.size}, expected>={total} (stride_bytes={stride_bytes})"
                    )
                mat = raw_u16[:total].reshape(h, stride_u16)
                mat = mat[:, :w]
                return np.ascontiguousarray(mat)

        need = h * w
        if raw_u16.size < need:
            raise RuntimeError(f"Depth buffer too small: raw={raw_u16.size}, expected>={need}")
        if raw_u16.size > need:
            raw_u16 = raw_u16[:need]
        return np.ascontiguousarray(raw_u16.reshape(h, w))

    # ----------------- lifecycle -----------------
    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        if ob is None:
            raise RuntimeError("pyorbbecsdk is not installed.")

        # ✅ 항상 기본(첫번째) 장치 사용: 공식 예제 방식
        self._pipeline = ob.Pipeline()
        cfg = ob.Config()

        # 선택적으로 "현재 연결 장치 수"만 로그 (선택에는 사용 X)
        try:
            ctx = ob.Context()
            dev_list = ctx.query_devices()
            cnt = dev_list.get_count()
            if cnt != 1:
                logger.warning(
                    f"Orbbec devices detected: {cnt}. "
                    f"LeRobot OrbbecCamera is configured to use the default/first device. "
                    f"(serial_or_name '{self.serial_or_name}' is ignored for selection.)"
                )
        except Exception:
            pass

        # Color profile
        profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        target_format = ob.OBFormat.RGB if self.color_mode == ColorMode.RGB else ob.OBFormat.BGR

        req_w = int(self.capture_width) if self.capture_width else 640
        req_h = int(self.capture_height) if self.capture_height else 480
        req_fps = int(self.fps) if self.fps else 30

        color_profile = None
        try:
            color_profile = profile_list.get_video_stream_profile(req_w, req_h, target_format, req_fps)
        except Exception:
            color_profile = None

        # MJPG fallback (많이 안정적)
        if color_profile is None:
            try:
                color_profile = profile_list.get_video_stream_profile(req_w, req_h, ob.OBFormat.MJPG, req_fps)
                logger.warning("Requested RGB/BGR profile not found. Falling back to MJPG profile.")
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

        # Depth optional
        if self.use_depth:
            depth_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = None
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

        # PCDP style: FULL_FRAME_REQUIRE + frame_sync (있으면)
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

        # optional convert filter
        try:
            self._conv_filter = ob.FormatConvertFilter() if hasattr(ob, "FormatConvertFilter") else None
        except Exception:
            self._conv_filter = None

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
        self._conv_filter = None
        logger.info(f"{self} disconnected.")

    # ----------------- read -----------------
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

    def _process_color(self, frame: Any) -> np.ndarray:
        h, w = int(frame.get_height()), int(frame.get_width())

        fmt = None
        try:
            fmt = frame.get_format()
        except Exception:
            fmt = self._color_format

        if fmt is not self._last_logged_fmt:
            # logger.warning(f"Orbbec color fmt={fmt}, size={w}x{h}")
            self._last_logged_fmt = fmt

        stride_bytes = self._get_stride_bytes(frame)

        # MJPG -> decode
        if fmt is not None and hasattr(ob, "OBFormat") and fmt == ob.OBFormat.MJPG:
            jpg = self._u8(frame)
            img_bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode MJPG frame (imdecode returned None)")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if self.color_mode == ColorMode.RGB else img_bgr
            if self.rotation is not None:
                img = cv2.rotate(img, self.rotation)
            return np.ascontiguousarray(img)

        # YUV -> RGB888 via SDK filter (if needed/available)
        if (
            fmt is not None
            and hasattr(ob, "OBFormat")
            and fmt not in (ob.OBFormat.RGB, ob.OBFormat.BGR, ob.OBFormat.MJPG)
            and self._conv_filter is not None
            and hasattr(ob, "OBConvertFormat")
        ):
            convert_format_map = {
                getattr(ob.OBFormat, "YUYV", None): getattr(ob.OBConvertFormat, "YUYV_TO_RGB888", None),
                getattr(ob.OBFormat, "YUY2", None): getattr(ob.OBConvertFormat, "YUYV_TO_RGB888", None),
                getattr(ob.OBFormat, "UYVY", None): getattr(ob.OBConvertFormat, "UYVY_TO_RGB888", None),
                getattr(ob.OBFormat, "NV12", None): getattr(ob.OBConvertFormat, "NV12_TO_RGB888", None),
                getattr(ob.OBFormat, "NV21", None): getattr(ob.OBConvertFormat, "NV21_TO_RGB888", None),
                getattr(ob.OBFormat, "I420", None): getattr(ob.OBConvertFormat, "I420_TO_RGB888", None),
            }
            convert_format = convert_format_map.get(fmt)
            if convert_format is not None:
                try:
                    self._conv_filter.set_format_convert_format(convert_format)
                    rgb_frame = self._conv_filter.process(frame)
                except Exception:
                    rgb_frame = None

                if rgb_frame:
                    hh, ww = int(rgb_frame.get_height()), int(rgb_frame.get_width())
                    stride2 = self._get_stride_bytes(rgb_frame)
                    raw = self._u8(rgb_frame)
                    img_rgb = self._reshape_hwc3(raw, hh, ww, stride2)

                    if self.color_mode == ColorMode.BGR:
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    if self.rotation is not None:
                        img_rgb = cv2.rotate(img_rgb, self.rotation)
                    return np.ascontiguousarray(img_rgb)

        # RGB/BGR raw
        raw = self._u8(frame)
        img = self._reshape_hwc3(raw, h, w, stride_bytes)

        if fmt is not None and hasattr(ob, "OBFormat"):
            if fmt == ob.OBFormat.BGR and self.color_mode == ColorMode.RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif fmt == ob.OBFormat.RGB and self.color_mode == ColorMode.BGR:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)

        return np.ascontiguousarray(img)

    def _process_depth(self, frame: Any) -> np.ndarray:
        h, w = int(frame.get_height()), int(frame.get_width())
        stride_bytes = self._get_stride_bytes(frame)
        raw = self._u16(frame)
        img = self._reshape_hw_u16(raw, h, w, stride_bytes)

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

# src/lerobot/cameras/orbbec/camera_orbbec.py


from __future__ import annotations

import logging
import time

from threading import Event, Lock, Thread
from typing import Any, List, Tuple, Union

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
        self.latest_frame: NDArray[Any] | None = None # COLOR RGB
        self.latest_depth: NDArray[Any] | None = None # DPETH cached
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"


    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
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
        config = ob.Config()

        try:
            # Color Stream
            profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            target_format = ob.OBFormat.RGB if self.color_mode == ColorMode.RGB else ob.OBFormat.BGR
            
            # 설정된 해상도가 없으면 기본값(640x480) 사용
            req_w = self.capture_width if self.capture_width else 640
            req_h = self.capture_height if self.capture_height else 480
            req_fps = int(self.fps) if self.fps else 30

            try:
                # 지원하는 프로필 중 일치하는 것 찾기
                color_profile = profile_list.get_video_stream_profile(req_w, req_h, target_format, req_fps)
            except ob.OBError:
                color_profile = None

            if color_profile is None:
                logger.warning(f"Exact profile ({req_w}x{req_h}@{req_fps}) not found. Using default.")
                color_profile = profile_list.get_default_video_stream_profile()

            config.enable_stream(color_profile)
            
            # 실제 설정된 값으로 업데이트
            self.capture_width = color_profile.get_width()
            self.capture_height = color_profile.get_height()
            self.fps = color_profile.get_fps()

            # Depth Stream (옵션)
            if self.use_depth:
                depth_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                try:
                    # Depth 해상도도 Color와 동일하게 맞춤
                    depth_profile = depth_list.get_video_stream_profile(
                        self.capture_width, self.capture_height, ob.OBFormat.Y16, int(self.fps)
                    )
                except ob.OBError:
                    depth_profile = None

                if depth_profile is None:
                    depth_profile = depth_list.get_default_video_stream_profile()
                
                config.enable_stream(depth_profile)

        except Exception as e:
            raise RuntimeError(f"Failed to configure streams: {e}")

        # 3. 파이프라인 시작
        try:
            # start()는 리턴값이 없습니다. self.pipeline에 대입하면 안 됩니다.
            self._pipeline.start(config)
        except Exception as e:
            self._pipeline = None
            raise ConnectionError(f"Failed to start pipeline: {e}")

        if warmup:
            time.sleep(self.warmup_s)
            
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    
    def read(self) -> np.ndarray:
        """동기식 읽기 (Blocking)"""
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
    
    def _process_color(self, frame: ob.ColorFrame) -> np.ndarray:
        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty color data")
    
        h, w = frame.get_height(), frame.get_width()
    
        # 1) 버퍼 → ndarray (여기서는 view일 수 있음)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    
        # 2) (중요) OpenCV에 넣기 전에 contiguous 강제 (copy 발생 가능)
        img = np.ascontiguousarray(img)
    
        # 3) 회전
        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)
            img = np.ascontiguousarray(img)  # 회전 결과도 안전하게
    
        # 4) 최종 반환도 contiguous 보장
        return np.ascontiguousarray(img)


    def _process_depth(self, frame: ob.DepthFrame) -> np.ndarray:
        data = frame.get_data()
        if data is None:
            raise RuntimeError("Empty depth data")
        
        h, w = frame.get_height(), frame.get_width()
        img = np.frombuffer(data, dtype=np.uint16).reshape(h, w).copy()

        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)
        return img

    # ------------------------------------------------------------------
    # 백그라운드 스레딩 (Asynchronous)
    # ------------------------------------------------------------------
    def _read_loop(self) -> None:
        if self.stop_event is None: return
        while not self.stop_event.is_set():
            try:
                frame_data = self.read()  # always COLOR np.ndarray
                with self.frame_lock:
                    self.latest_frame = frame_data
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.warning(f"Error in read loop: {e}")

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive(): return
        if self.stop_event: self.stop_event.set()
        self.stop_event = Event()
        self.new_frame_event.clear()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event: self.stop_event.set()
        if self.thread: self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> Any:
        if not self.is_connected: raise DeviceNotConnectedError(f"{self} not connected.")
        if not self.thread or not self.thread.is_alive(): self._start_read_thread()
        
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timeout waiting for frame from {self}")
        
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if isinstance(frame, tuple):
            return (np.ascontiguousarray(frame[0]), np.ascontiguousarray(frame[1]))
        return np.ascontiguousarray(frame)
    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def find_cameras() -> List[dict[str, Any]]:
        """
        Detects available Orbbec cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing device info.
        """
        if ob is None:
            logger.warning("pyorbbecsdk is not installed.")
            return []

        found_cameras_info = []
        try:
            ctx = ob.Context()
            device_list = ctx.query_devices()
            
            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                try:
                    device_info = device.get_device_info()
                    name = device_info.get_name()
                    serial = device_info.get_serial_number()
                    pid = device_info.get_pid()
                    vid = device_info.get_vid()
                    connection_type = device_info.get_connection_type()
                    
                    camera_info = {
                        "name": name,
                        "serial_number": serial,
                        "type": "Orbbec",
                        "pid": pid,
                        "vid": vid,
                        "connection_type": connection_type
                    }
                    
                    try:
                        sensor_list = device.get_sensor_list()
                        color_sensor = sensor_list.get_sensor_by_type(ob.OBSensorType.COLOR_SENSOR)
                        if color_sensor:
                            profiles = color_sensor.get_stream_profile_list()
                            default_profile = profiles.get_default_video_stream_profile()
                            if default_profile:
                                camera_info["default_stream_profile"] = {
                                    "stream_type": "Color",
                                    "width": default_profile.get_width(),
                                    "height": default_profile.get_height(),
                                    "fps": default_profile.get_fps(),
                                    "format": default_profile.get_format()
                                }
                    except Exception as e:
                        logger.debug(f"Could not retrieve default profile for {name}: {e}")

                    found_cameras_info.append(camera_info)
                    
                except Exception as e:
                    logger.warning(f"Error retrieving info for device index {i}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error querying Orbbec devices: {e}")
            
        return found_cameras_info
    
    def _find_serial_number_from_name(self, name: str) -> str:
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No Orbbec camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["serial_number"] for dev in found_devices]
            raise ValueError(
                f"Multiple Orbbec cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        return str(found_devices[0]["serial_number"])
    
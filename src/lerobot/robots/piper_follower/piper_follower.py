# src/lerobot/robots/piper_follower/piper_follower.py


from __future__ import annotations
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from piper_sdk import *  

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper_follower import PIPERFollowerConfig

logger = logging.getLogger(__name__)

class PiperFollower(Robot):
    
    config_class = PIPERFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config=config
        self._sdk: Optional[C_PiperInterface_V2] = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_pose: List[float] = [0.0] * 7

    # ------------------------------------------------------------------
    # Properties describing available features
    # ------------------------------------------------------------------
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        motors = {
            f"j{i}": float for i in range(1, 7)
        }
        motors["gripper"] = float
        return {**motors, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        motors = {
            f"j{i}": float for i in range(1, 7)
        }
        motors["gripper"] = float
        return motors

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def is_connected(self) -> bool:
        return self._sdk is not None

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            sdk = C_PiperInterface_V2(can_name = self.config.can_name)
            sdk.ConnectPort()
            sdk.EnableArm(7)
            logger.info(f"{self} connected.")
        except Exception as e:
            sdk=None
            raise e
        

        enable_flag = False
        timeout = 5

        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            enable_flag = sdk.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                sdk.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                sdk.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                sdk.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                sdk.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                sdk.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            sdk.EnableArm(7)
            sdk.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            sdk.GripperCtrl(0,5000,0x01, 0)
            if elapsed_time > timeout:
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
        if(elapsed_time_flag):
            raise DeviceNotConnectedError(f"{self} is failed connect(timeout).")
        
        self._sdk = sdk

        for cam in self.cameras.values():
            cam.connect()
        logger.info(f"{self} connected")

        # Read an initial pose so that relative actions can be computed.
        self._last_pose = self._read_pose()

    def disconnect(self) -> None:
        """Close the connection and optionally disable torque."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            if self.config.disable_torque_on_disconnect:
                # Passing 7 disables all arm joints.
                self._sdk.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                self._sdk.JointCtrl([0, 0, 0, 0, 0, 0])
                time.sleep(5)
                self._sdk.DisableArm(7)
            self._sdk.DisconnectPort()
        finally:
            for cam in self.cameras.values():
                cam.disconnect()
            self._sdk = None
            logger.info(f"{self} disconnected")

    # ------------------------------------------------------------------
    # Calibration and configuration
    # ------------------------------------------------------------------
    def is_calibrated(self) -> bool:
        return self.is_connected()

    def calibrate(self) -> None:
        return

    def configure(self, **kwargs) -> None:
        super().configure(**kwargs)

    # ------------------------------------------------------------------
    # Observation and action
    # ------------------------------------------------------------------
    def _read_pose(self) -> List[float]:
        pos = self._sdk.GetArmJointMsgs()  # type: ignore[attr-defined]
        gripper_msg = self._sdk.GetArmGripperMsgs()
        
        if self.config.use_degrees:
            current_joints = [math.degrees(p) for p in pos]
            gripper = gripper_msg[0]
        else:
            current_joints = list(pos)
            gripper = math.radians(gripper_msg[0])
        

        return current_joints + [gripper]
    
    def get_observation(self) -> Dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        start = time.perf_counter()
        self._last_pose = self._read_pose()

        obs_dict = {f"j{i+1}": p for i, p in enumerate(self._last_pose[:6])}
        obs_dict["gripper"] = self._last_pose[6]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter()-start)*1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict
        

    def send_action(self, action: Dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        target_joints = [action[f"j{i}"] for i in range(1, 7)]
        target_gripper = action["gripper"]

        if self.config.max_relative_target is not None:
            present_pose = self._last_pose 
            
            present_dict = {f"j{i+1}": p for i, p in enumerate(present_pose[:6])}
            present_dict["gripper"] = present_pose[6]
            
            goal_present = {k: (action[k], present_dict[k]) for k in action}
            
            clipped_action = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            
            target_joints = [clipped_action[f"j{i}"] for i in range(1, 7)]
            target_gripper = clipped_action["gripper"]
            action = clipped_action

        if self.config.use_degrees:
            joints_deg = target_joints
            gripper_deg = target_gripper
        else:
            # Radian -> Degree
            joints_deg = [math.degrees(j) for j in target_joints]
            gripper_deg = math.degrees(target_gripper)

        cmd_joints = [int(j * 1000) for j in joints_deg]
        cmd_gripper = int(gripper_deg * 1000)
        
        
        self._sdk.JointCtrl(cmd_joints)
        self._sdk.GripperCtrl(int(cmd_gripper), 1000, 0x01)
        
        return action
        
        

    
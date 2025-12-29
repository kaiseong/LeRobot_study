# src/lerobot/teleoperators/piper_leader/piper_leader.py

from __future__ import annotations

import logging
import time

import math

from typing import Dict, Optional
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from piper_sdk import *  
from ..teleoperator import Teleoperator
from .config_piper_leader import PIPERLeaderConfig

logger = logging.getLogger(__name__)


class PiperLeader(Teleoperator):
    
    config_class = PIPERLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig) -> None:
        super().__init__(config=config)
        self._sdk: Optional[C_PiperInterface_V2] = None
        self.config = config

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "j1": float, # rad
            "j2": float, # rad
            "j3": float, # rad
            "j4": float, # rad
            "j5": float, # rad 
            "j6": float, # rad
            "gripper": float, # degree
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """This teleoperator does not currently provide feedback."""
        return {}

    @property
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
        self._sdk = sdk

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            self._sdk.DisconnectPort()
            logger.info(f"{self} disconnected.")
        finally:
            self._sdk = None

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected()

    def calibrate(self) -> None:
        return

    def configure(self, **kwargs) -> None:
        super().configure(**kwargs)

    def get_action(self) -> Dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        start = time.perf_counter()
        pos = self._sdk.GetArmJointMsgs()  # type: ignore[attr-defined]
        raw_gripper = self._sdk.GetArmGripperMsgs() # degree
        gripper = -2.4 + (101.4/72.75) * (raw_gripper[0]+1.75)

        if self.config.use_degrees:
            pos = [math.degrees(p) for p in pos]
        else:
            gripper = math.radians(gripper)

        dt_ms = (time.perf_counter() - start) *1e3
        logger.debug(f"{self} read action {dt_ms:.1f}ms")
        action = {f"j{i+1}": p for i, p in enumerate(pos)}
        action["gripper"] = gripper
        return action

    def send_feedback(self, feedback: Dict[str, float]) -> None:
        raise NotImplementedError
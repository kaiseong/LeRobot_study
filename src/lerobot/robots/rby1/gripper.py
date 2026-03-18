#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RBY1GripperController:
    def __init__(
        self,
        rby_module: Any,
        *,
        enabled: bool,
        device_ids: Sequence[int],
        baud_rate: int,
        torque_constants: Sequence[float],
        homing_torque: float,
        hold_torque: float,
        homing_sleep_s: float,
        homing_stall_cycles: int,
        direction_reversed: bool,
        home_on_connect: bool,
    ) -> None:
        self._rby = rby_module
        self.enabled = enabled
        self.device_ids = tuple(int(device_id) for device_id in device_ids)
        self.baud_rate = int(baud_rate)
        self.torque_constants = tuple(float(value) for value in torque_constants)
        self.homing_torque = float(homing_torque)
        self.hold_torque = float(hold_torque)
        self.homing_sleep_s = float(homing_sleep_s)
        self.homing_stall_cycles = int(homing_stall_cycles)
        self.direction_reversed = bool(direction_reversed)
        self.home_on_connect = bool(home_on_connect)

        self._bus: Any | None = None
        self._min_q = np.full(len(self.device_ids), np.inf, dtype=float)
        self._max_q = np.full(len(self.device_ids), -np.inf, dtype=float)
        self._target_q: np.ndarray | None = None
        self._operating_mode: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._bus is not None

    def connect(self) -> None:
        if not self.enabled:
            return

        bus = self._rby.DynamixelBus(self._rby.upc.GripperDeviceName)
        if not bus.open_port():
            raise ConnectionError("Failed to open the RBY1 gripper bus.")
        if not bus.set_baud_rate(self.baud_rate):
            raise ConnectionError(f"Failed to configure the RBY1 gripper bus baud rate ({self.baud_rate}).")

        if hasattr(bus, "set_torque_constant"):
            bus.set_torque_constant(list(self.torque_constants))

        # Set USB latency_timer to 1ms (default 16ms causes ~20Hz bottleneck).
        if hasattr(self._rby, "upc") and hasattr(self._rby.upc, "initialize_device"):
            self._rby.upc.initialize_device(self._rby.upc.GripperDeviceName)
            logger.info("RBY1 gripper USB latency_timer optimized to 1ms.")

        missing_devices = [device_id for device_id in self.device_ids if not bus.ping(device_id)]
        if missing_devices:
            raise ConnectionError(f"RBY1 gripper devices are not responding: {missing_devices}")

        self._bus = bus
        self._enable_torque(True)
        if self.home_on_connect:
            self._home()
        else:
            current_q = self._read_encoders()
            self._min_q = current_q - 1.0
            self._max_q = current_q + 1.0
            self._set_operating_mode(self._rby.DynamixelBus.CurrentBasedPositionControlMode)
        if self._target_q is None and np.isfinite(self._min_q).all() and np.isfinite(self._max_q).all():
            self.set_targets([0.0] * len(self.device_ids))

    def disconnect(self) -> None:
        if not self.enabled or self._bus is None:
            return
        self._bus = None
        self._target_q = None
        self._operating_mode = None

    def get_positions(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self._bus is None:
            raise ConnectionError("The RBY1 gripper controller is not connected.")

        encoder_values = self._read_encoders()
        normalized_positions = self._normalize_encoder_values(encoder_values)
        if len(normalized_positions) != 2:
            raise RuntimeError("The RBY1 gripper controller expected exactly two normalized values.")
        return {
            "right_gripper": float(normalized_positions[0]),
            "left_gripper": float(normalized_positions[1]),
        }

    def set_targets(self, normalized_targets: Sequence[float]) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self._bus is None:
            raise ConnectionError("The RBY1 gripper controller is not connected.")

        normalized = np.asarray(list(normalized_targets), dtype=float)
        if normalized.shape != (len(self.device_ids),):
            raise ValueError(
                f"RBY1 gripper targets must contain exactly {len(self.device_ids)} values, got {normalized.shape}."
            )
        normalized = np.clip(normalized, 0.0, 1.0)

        if not np.isfinite(self._min_q).all() or not np.isfinite(self._max_q).all():
            raise RuntimeError("The RBY1 gripper controller is not calibrated.")

        if self.direction_reversed:
            target_q = normalized * (self._max_q - self._min_q) + self._min_q
        else:
            target_q = (1.0 - normalized) * (self._max_q - self._min_q) + self._min_q

        self._target_q = target_q
        self._set_operating_mode(self._rby.DynamixelBus.CurrentBasedPositionControlMode)
        self._bus.group_sync_write_send_torque([(device_id, self.hold_torque) for device_id in self.device_ids])
        self._bus.group_sync_write_send_position(
            [(device_id, float(position)) for device_id, position in zip(self.device_ids, target_q, strict=True)]
        )
        return {
            "right_gripper": float(normalized[0]),
            "left_gripper": float(normalized[1]),
        }

    def _enable_torque(self, enabled: bool) -> None:
        if self._bus is None:
            raise ConnectionError("The RBY1 gripper controller is not connected.")
        value = 1 if enabled else 0
        self._bus.group_sync_write_torque_enable([(device_id, value) for device_id in self.device_ids])

    def _set_operating_mode(self, mode: int) -> None:
        if self._bus is None:
            raise ConnectionError("The RBY1 gripper controller is not connected.")
        if self._operating_mode == mode:
            return
        self._enable_torque(False)
        self._bus.group_sync_write_operating_mode([(device_id, mode) for device_id in self.device_ids])
        self._enable_torque(True)
        self._operating_mode = mode

    def _read_encoders(self) -> np.ndarray:
        if self._bus is None:
            raise ConnectionError("The RBY1 gripper controller is not connected.")
        encoder_readings = self._bus.group_fast_sync_read_encoder(list(self.device_ids))
        if encoder_readings is None:
            raise RuntimeError("Failed to read the RBY1 gripper encoders.")

        encoder_values = np.zeros(len(self.device_ids), dtype=float)
        for index, (_, encoder_value) in enumerate(encoder_readings):
            encoder_values[index] = float(encoder_value)
        return encoder_values

    def _home(self) -> None:
        if self._bus is None:
            raise ConnectionError("The RBY1 gripper controller is not connected.")

        self._set_operating_mode(self._rby.DynamixelBus.CurrentControlMode)
        previous_q = np.zeros(len(self.device_ids), dtype=float)
        direction = 0
        stable_cycles = 0

        while direction < 2:
            target_torque = self.homing_torque if direction == 0 else -self.homing_torque
            self._bus.group_sync_write_send_torque([(device_id, target_torque) for device_id in self.device_ids])
            encoder_values = self._read_encoders()
            self._min_q = np.minimum(self._min_q, encoder_values)
            self._max_q = np.maximum(self._max_q, encoder_values)

            if np.array_equal(previous_q, encoder_values):
                stable_cycles += 1
            else:
                stable_cycles = 0

            previous_q = encoder_values.copy()
            if stable_cycles >= self.homing_stall_cycles:
                direction += 1
                stable_cycles = 0
            time.sleep(self.homing_sleep_s)

        self._set_operating_mode(self._rby.DynamixelBus.CurrentBasedPositionControlMode)
        self._bus.group_sync_write_send_torque([(device_id, self.hold_torque) for device_id in self.device_ids])
        logger.info("RBY1 gripper homing finished.")

    def _normalize_encoder_values(self, encoder_values: np.ndarray) -> np.ndarray:
        if not np.isfinite(self._min_q).all() or not np.isfinite(self._max_q).all():
            logger.debug("RBY1 gripper normalization is not available before calibration.")
            return np.zeros(len(self.device_ids), dtype=float)

        denom = self._max_q - self._min_q
        denom = np.where(np.isclose(denom, 0.0), 1.0, denom)
        normalized = (encoder_values - self._min_q) / denom
        if not self.direction_reversed:
            normalized = 1.0 - normalized
        return np.clip(normalized, 0.0, 1.0)

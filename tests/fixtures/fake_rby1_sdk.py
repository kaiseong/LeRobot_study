#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

_RBY1_M_JOINT_ORDER = (
    "wheel_fr",
    "wheel_fl",
    "wheel_rr",
    "wheel_rl",
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "head_0",
    "head_1",
)


class FakeControlManagerState:
    class State:
        Idle = "idle"
        Enabled = "enabled"
        MinorFault = "minor_fault"
        MajorFault = "major_fault"

    def __init__(self, state: str | None = None):
        self.state = state or self.State.Idle
        self.control_state = "idle"
        self.enabled_joint_idx = []
        self.unlimited_mode_enabled = False


class FakeRobotCommandFeedback:
    class Status:
        Idle = "idle"
        Running = "running"
        Finished = "finished"

    class FinishCode:
        Unknown = "unknown"
        Ok = "ok"

    def __init__(self):
        self.status = self.Status.Running
        self.finish_code = self.FinishCode.Unknown


class FakeCommandHeaderBuilder:
    def __init__(self):
        self.control_hold_time = None

    def set_control_hold_time(self, control_hold_time):
        self.control_hold_time = control_hold_time
        return self


class FakeJointPositionCommandBuilder:
    def __init__(self):
        self.command_header = None
        self.position = None
        self.minimum_time = None

    def set_command_header(self, command_header):
        self.command_header = command_header
        return self

    def set_position(self, position):
        self.position = list(position)
        return self

    def set_minimum_time(self, minimum_time):
        self.minimum_time = minimum_time
        return self


class FakeBodyComponentBasedCommandBuilder:
    def __init__(self):
        self.torso_command = None
        self.right_arm_command = None
        self.left_arm_command = None

    def set_torso_command(self, command):
        self.torso_command = command
        return self

    def set_right_arm_command(self, command):
        self.right_arm_command = command
        return self

    def set_left_arm_command(self, command):
        self.left_arm_command = command
        return self


class FakeComponentBasedCommandBuilder:
    def __init__(self):
        self.body_command = None
        self.head_command = None

    def set_body_command(self, command):
        self.body_command = command
        return self

    def set_head_command(self, command):
        self.head_command = command
        return self


class FakeRobotCommandBuilder:
    def __init__(self):
        self.command = None

    def set_command(self, command):
        self.command = command
        return self


@dataclass
class FakeRobotState:
    position: list[float]


@dataclass
class FakeModel:
    robot_joint_names: list[str]


class FakeCommandStream:
    def __init__(self):
        self.sent_commands = []
        self.cancelled = False

    def send_command(self, builder, timeout_ms=1000):
        self.sent_commands.append((builder, timeout_ms))
        return FakeRobotCommandFeedback()

    def cancel(self):
        self.cancelled = True


class FakeRobot:
    def __init__(self, address: str, model: str):
        self.address = address
        self.requested_model = model
        self._is_connected = False
        self._power_on = False
        self._servo_on = False
        self.control_manager_state = FakeControlManagerState()
        self.disconnect_calls = 0
        self.disable_control_manager_calls = 0
        self.parameters = []
        self.wait_for_control_ready_calls = []
        self.create_command_stream_calls = []
        self.servo_off_calls = []
        self.power_on_calls = []
        self.servo_on_calls = []
        self.reset_fault_calls = 0
        self.position = [0.01 * idx for idx in range(26)]
        self.stream = FakeCommandStream()

    def connect(self, max_retries=5, timeout_ms=1000):
        self._is_connected = True
        self.last_connect = (max_retries, timeout_ms)
        return True

    def disconnect(self):
        self.disconnect_calls += 1
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def is_power_on(self, dev_name):
        self.last_power_query = dev_name
        return self._power_on

    def power_on(self, dev_name):
        self.power_on_calls.append(dev_name)
        self._power_on = True
        return True

    def is_servo_on(self, dev_name):
        self.last_servo_query = dev_name
        return self._servo_on

    def servo_on(self, dev_name):
        self.servo_on_calls.append(dev_name)
        self._servo_on = True
        return True

    def servo_off(self, dev_name):
        self.servo_off_calls.append(dev_name)
        self._servo_on = False
        return True

    def get_control_manager_state(self):
        return self.control_manager_state

    def reset_fault_control_manager(self):
        self.reset_fault_calls += 1
        self.control_manager_state.state = FakeControlManagerState.State.Idle
        return True

    def enable_control_manager(self, unlimited_mode_enabled=False):
        self.control_manager_state.state = FakeControlManagerState.State.Enabled
        self.control_manager_state.unlimited_mode_enabled = unlimited_mode_enabled
        return True

    def disable_control_manager(self):
        self.disable_control_manager_calls += 1
        self.control_manager_state.state = FakeControlManagerState.State.Idle
        return True

    def wait_for_control_ready(self, timeout_ms):
        self.wait_for_control_ready_calls.append(timeout_ms)
        return True

    def create_command_stream(self, priority=1):
        self.create_command_stream_calls.append(priority)
        self.stream = FakeCommandStream()
        return self.stream

    def set_parameter(self, name, value, write_db=True):
        self.parameters.append((name, value, write_db))
        return True

    def get_state(self):
        return FakeRobotState(position=list(self.position))

    def model(self):
        if self.requested_model.lower() != "m":
            raise NotImplementedError(f"Fake model support is limited to model='m', got {self.requested_model!r}")
        return FakeModel(robot_joint_names=list(_RBY1_M_JOINT_ORDER))


class FakeRBY1SDK:
    ControlManagerState = FakeControlManagerState
    RobotCommandFeedback = FakeRobotCommandFeedback
    CommandHeaderBuilder = FakeCommandHeaderBuilder
    JointPositionCommandBuilder = FakeJointPositionCommandBuilder
    BodyComponentBasedCommandBuilder = FakeBodyComponentBasedCommandBuilder
    ComponentBasedCommandBuilder = FakeComponentBasedCommandBuilder
    RobotCommandBuilder = FakeRobotCommandBuilder

    def __init__(self):
        self.created_robots = []

    def create_robot(self, address: str, model: str):
        robot = FakeRobot(address, model)
        self.created_robots.append(robot)
        return robot

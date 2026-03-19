#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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


class FakeRobotCommandHandler:
    def __init__(self, finish_code):
        self._finish_code = finish_code

    def wait(self):
        return self._finish_code

    def get(self):
        return self._finish_code


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


class FakeJointImpedanceControlCommandBuilder:
    def __init__(self):
        self.command_header = None
        self.position = None
        self.minimum_time = None
        self.stiffness = None
        self.torque_limit = None
        self.damping_ratio = None

    def set_command_header(self, command_header):
        self.command_header = command_header
        return self

    def set_position(self, position):
        self.position = list(position)
        return self

    def set_minimum_time(self, minimum_time):
        self.minimum_time = minimum_time
        return self

    def set_stiffness(self, stiffness):
        self.stiffness = list(stiffness)
        return self

    def set_torque_limit(self, torque_limit):
        self.torque_limit = list(torque_limit)
        return self

    def set_damping_ratio(self, damping_ratio):
        self.damping_ratio = list(damping_ratio)
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
    model_name: str = "M"


@dataclass
class FakeDynamicsState:
    joint_names: list[str]


class FakeDynamics:
    def __init__(self, joint_names: list[str]):
        joint_count = len(joint_names)
        self.q_lower = np.full(joint_count, -3.0, dtype=float)
        self.q_upper = np.full(joint_count, 3.0, dtype=float)
        self.qdot_upper = np.full(joint_count, 1.0, dtype=float)
        self.qddot_upper = np.full(joint_count, 2.0, dtype=float)

    def make_state(self, _, joint_names):
        return FakeDynamicsState(joint_names=list(joint_names))

    def get_limit_q_lower(self, _state):
        return self.q_lower.copy()

    def get_limit_q_upper(self, _state):
        return self.q_upper.copy()

    def get_limit_qdot_upper(self, _state):
        return self.qdot_upper.copy()

    def get_limit_qddot_upper(self, _state):
        return self.qddot_upper.copy()


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
        self.enable_control_manager_calls = []
        self.parameters = []
        self.wait_for_control_ready_calls = []
        self.create_command_stream_calls = []
        self.send_command_calls = []
        self.servo_off_calls = []
        self.power_on_calls = []
        self.power_off_calls = []
        self.servo_on_calls = []
        self.tool_flange_voltage_calls = []
        self.tool_flange_voltage = {"right": 0, "left": 0}
        self.reset_fault_calls = 0
        self.position = [0.01 * idx for idx in range(26)]
        self.stream = FakeCommandStream()
        self.dynamics = FakeDynamics(list(_RBY1_M_JOINT_ORDER))

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

    def power_off(self, dev_name):
        self.power_off_calls.append(dev_name)
        self._power_on = False
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

    def set_tool_flange_output_voltage(self, arm, voltage):
        self.tool_flange_voltage_calls.append((arm, voltage))
        self.tool_flange_voltage[str(arm)] = int(voltage)
        FakeDynamixelBus._tool_flange_voltage = dict(self.tool_flange_voltage)
        return True

    def get_control_manager_state(self):
        return self.control_manager_state

    def reset_fault_control_manager(self):
        self.reset_fault_calls += 1
        self.control_manager_state.state = FakeControlManagerState.State.Idle
        return True

    def enable_control_manager(self, unlimited_mode_enabled=False):
        self.enable_control_manager_calls.append(unlimited_mode_enabled)
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
        return FakeModel(robot_joint_names=list(_RBY1_M_JOINT_ORDER), model_name="M")

    def get_dynamics(self):
        return self.dynamics

    def send_command(self, builder, priority=1):
        self.send_command_calls.append((builder, priority))
        self._apply_command(builder)
        return FakeRobotCommandHandler(FakeRobotCommandFeedback.FinishCode.Ok)

    def _apply_command(self, builder):
        command = getattr(builder, "command", None)
        if command is None:
            return
        body_command = getattr(command, "body_command", None)
        head_command = getattr(command, "head_command", None)

        if isinstance(body_command, FakeBodyComponentBasedCommandBuilder):
            if body_command.torso_command is not None and body_command.torso_command.position is not None:
                self.position[4:10] = list(body_command.torso_command.position)
            if body_command.right_arm_command is not None and body_command.right_arm_command.position is not None:
                self.position[10:17] = list(body_command.right_arm_command.position)
            if body_command.left_arm_command is not None and body_command.left_arm_command.position is not None:
                self.position[17:24] = list(body_command.left_arm_command.position)
            if head_command is not None and head_command.position is not None:
                self.position[24:26] = list(head_command.position)
            return

        if isinstance(body_command, (FakeJointPositionCommandBuilder, FakeJointImpedanceControlCommandBuilder)):
            if body_command.position is not None:
                positions = list(body_command.position)
                self.position[4 : 4 + len(positions)] = positions


@dataclass
class FakeButtonState:
    button: int = 0
    trigger: int = 0


@dataclass
class FakeMasterArmState:
    q_joint: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=float))
    qvel_joint: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=float))
    torque_joint: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=float))
    gravity_term: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=float))
    operating_mode: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.int32))
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=float))
    button_right: FakeButtonState = field(default_factory=FakeButtonState)
    button_left: FakeButtonState = field(default_factory=FakeButtonState)
    T_right: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    T_left: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))


class FakeMasterArmControlInput:
    def __init__(self):
        self.target_operating_mode = np.zeros(14, dtype=np.int32)
        self.target_position = np.zeros(14, dtype=float)
        self.target_torque = np.zeros(14, dtype=float)


class FakeMasterArm:
    DOF = 14
    DeviceCount = 16
    TorqueScaling = 0.5
    MaximumTorque = 4.0
    RightToolId = 0x80
    LeftToolId = 0x81

    State = FakeMasterArmState
    ControlInput = FakeMasterArmControlInput

    def __init__(self, dev_name: str):
        self.dev_name = dev_name
        self.model_path = None
        self.control_period = None
        self.initialize_calls = []
        self.active_ids = list(range(self.DeviceCount))
        self.start_control_calls = 0
        self.stop_control_calls = []
        self.enable_torque_calls = 0
        self.disable_torque_calls = 0
        self.control_callback = None
        self.control_running = False
        self.current_state = FakeMasterArmState()
        self.last_control_input = None

    def set_model_path(self, model_path: str):
        self.model_path = model_path

    def set_control_period(self, control_period: float):
        self.control_period = control_period

    def initialize(self, verbose=False):
        self.initialize_calls.append(verbose)
        return list(self.active_ids)

    def start_control(self, control):
        self.start_control_calls += 1
        self.control_callback = control
        self.control_running = True
        self.last_control_input = control(self.current_state)
        return True

    def stop_control(self, torque_disable=False):
        self.stop_control_calls.append(torque_disable)
        self.control_running = False
        return True

    def enable_torque(self):
        self.enable_torque_calls += 1
        return True

    def disable_torque(self):
        self.disable_torque_calls += 1
        return True

    def emit_state(
        self,
        *,
        q_joint=None,
        qvel_joint=None,
        gravity_term=None,
        right_button=None,
        left_button=None,
        right_trigger=None,
        left_trigger=None,
    ):
        if q_joint is not None:
            self.current_state.q_joint = np.asarray(q_joint, dtype=float)
        if qvel_joint is not None:
            self.current_state.qvel_joint = np.asarray(qvel_joint, dtype=float)
        if gravity_term is not None:
            self.current_state.gravity_term = np.asarray(gravity_term, dtype=float)
        if right_button is not None:
            self.current_state.button_right.button = int(right_button)
        if left_button is not None:
            self.current_state.button_left.button = int(left_button)
        if right_trigger is not None:
            self.current_state.button_right.trigger = int(right_trigger)
        if left_trigger is not None:
            self.current_state.button_left.trigger = int(left_trigger)

        if self.control_running and self.control_callback is not None:
            self.last_control_input = self.control_callback(self.current_state)
        return self.last_control_input


class FakeMasterArmFactory:
    DOF = FakeMasterArm.DOF
    DeviceCount = FakeMasterArm.DeviceCount
    TorqueScaling = FakeMasterArm.TorqueScaling
    MaximumTorque = FakeMasterArm.MaximumTorque
    RightToolId = FakeMasterArm.RightToolId
    LeftToolId = FakeMasterArm.LeftToolId
    State = FakeMasterArm.State
    ControlInput = FakeMasterArm.ControlInput

    def __init__(self, upc_module: "FakeUPCModule"):
        self._upc_module = upc_module

    def __call__(self, dev_name: str):
        master_arm = FakeMasterArm(dev_name)
        self._upc_module.created_master_arms.append(master_arm)
        return master_arm


class FakeUPCModule:
    MasterArmDeviceName = "/dev/rby1_master_arm"
    GripperDeviceName = "/dev/rby1_gripper"

    def __init__(self):
        self.initialize_device_calls = []
        self.created_master_arms = []
        self.MasterArm = FakeMasterArmFactory(self)

    def initialize_device(self, dev_name: str):
        self.initialize_device_calls.append(dev_name)
        return True


class FakeDynamixelBus:
    CurrentControlMode = 0
    CurrentBasedPositionControlMode = 5
    _instance_sink = None
    _tool_flange_voltage = {"right": 0, "left": 0}

    def __init__(self, dev_name: str):
        self.dev_name = dev_name
        self.port_open = False
        self.baud_rate = None
        self.torque_constants = None
        self.torque_enable_writes = []
        self.operating_mode_writes = []
        self.torque_writes = []
        self.position_writes = []
        self.encoder_values = [1000.0, 1000.0]
        if self.__class__._instance_sink is not None:
            self.__class__._instance_sink.append(self)

    def open_port(self):
        self.port_open = True
        return True

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
        return True

    def set_torque_constant(self, torque_constants):
        self.torque_constants = list(torque_constants)
        return True

    def ping(self, dev_id):
        if self.dev_name == FakeUPCModule.GripperDeviceName:
            if self._tool_flange_voltage.get("right", 0) != 12:
                return False
            if self._tool_flange_voltage.get("left", 0) != 12:
                return False
        return dev_id in {0, 1}

    def group_sync_write_torque_enable(self, id_and_enable_vector):
        self.torque_enable_writes.append(list(id_and_enable_vector))
        return True

    def group_sync_write_operating_mode(self, id_and_mode_vector):
        self.operating_mode_writes.append(list(id_and_mode_vector))
        return True

    def group_sync_write_send_torque(self, id_and_torque_vector):
        self.torque_writes.append(list(id_and_torque_vector))
        return True

    def group_fast_sync_read_encoder(self, ids):
        return [(dev_id, self.encoder_values[dev_id]) for dev_id in ids]

    def group_sync_write_send_position(self, id_and_position_vector):
        writes = list(id_and_position_vector)
        self.position_writes.append(writes)
        for dev_id, position in writes:
            self.encoder_values[dev_id] = float(position)
        return True


class FakeRBY1SDK:
    ControlManagerState = FakeControlManagerState
    RobotCommandFeedback = FakeRobotCommandFeedback
    CommandHeaderBuilder = FakeCommandHeaderBuilder
    JointPositionCommandBuilder = FakeJointPositionCommandBuilder
    JointImpedanceControlCommandBuilder = FakeJointImpedanceControlCommandBuilder
    BodyComponentBasedCommandBuilder = FakeBodyComponentBasedCommandBuilder
    ComponentBasedCommandBuilder = FakeComponentBasedCommandBuilder
    RobotCommandBuilder = FakeRobotCommandBuilder
    DynamixelBus = FakeDynamixelBus

    def __init__(self):
        self.created_robots = []
        self.upc = FakeUPCModule()
        self.created_buses = []
        FakeDynamixelBus._instance_sink = self.created_buses
        FakeDynamixelBus._tool_flange_voltage = {"right": 0, "left": 0}

    def create_robot(self, address: str, model: str):
        robot = FakeRobot(address, model)
        self.created_robots.append(robot)
        return robot

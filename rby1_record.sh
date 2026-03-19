#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# RBY1 Teleoperation Recording Script
###############################################################################

# ── Robot ──
ROBOT_ADDRESS="192.168.30.1:50051"
ROBOT_MODEL="m"
USE_IMPEDANCE=false
COMMAND_MINIMUM_TIME=0.07
CONTROL_HOLD_TIME=30.0
CUTOFF_FREQUENCY=5.0
COLLISION_CHECK=true

# ── Master Arm ──
MASTER_ARM_URDF="/home/$(whoami)/rby1-dev/external/rby1-sdk/models/master_arm/model.urdf"

# ── Initial & Fixed Poses (degree, auto-converted to radian) ──
FIXED_TORSO_DEG=(0.0 0.0 0.0 10.0 0.0 0.0)
FIXED_HEAD_DEG=(0.0 45.0)
INITIAL_RIGHT_ARM_DEG=(57.0 -14.0 5.0 -120.0 24.0 -37.0 58.0)
INITIAL_LEFT_ARM_DEG=(57.0 14.0 -5.0 -120.0 -24.0 -37.0 -58.0)

# ── Cameras ──
CAMERAS='{
  front:   {type: intelrealsense, serial_number_or_name: "335122270761", width: 640, height: 480, fps: 15},
  right: {type: intelrealsense, serial_number_or_name: "335122272086", width: 480, height: 640, fps: 15, rotation: 90},
  left: {type: intelrealsense, serial_number_or_name: "230422270977", width: 480, height: 640, fps: 15, rotation: 270}
}'

# ── Dataset ──
REPO_ID="kaiseong/rby1-right-arm-task"
SINGLE_TASK="Pick up the object with right arm"
FPS=15
NUM_EPISODES=50
EPISODE_TIME_S=60
RESET_TIME_S=30

# ── Options ──
DISPLAY_DATA=true
STREAMING_ENCODING=true
ENCODER_THREADS=2

###############################################################################

# ── degree → radian 변환 ──
deg2rad() {
  local result=""
  for deg in "$@"; do
    rad=$(python3 -c "import math; print(round(math.radians($deg), 6))")
    result="${result:+$result, }$rad"
  done
  echo "[$result]"
}

FIXED_TORSO=$(deg2rad "${FIXED_TORSO_DEG[@]}")
FIXED_HEAD=$(deg2rad "${FIXED_HEAD_DEG[@]}")
INITIAL_RIGHT_ARM=$(deg2rad "${INITIAL_RIGHT_ARM_DEG[@]}")
INITIAL_LEFT_ARM=$(deg2rad "${INITIAL_LEFT_ARM_DEG[@]}")

lerobot-record \
  --robot.type=rby1 \
  --robot.address="${ROBOT_ADDRESS}" \
  --robot.model="${ROBOT_MODEL}" \
  --robot.use_impedance="${USE_IMPEDANCE}" \
  --robot.command_minimum_time="${COMMAND_MINIMUM_TIME}" \
  --robot.control_hold_time="${CONTROL_HOLD_TIME}" \
  --robot.joint_position_command_cutoff_frequency="${CUTOFF_FREQUENCY}" \
  --robot.collision_check_enabled="${COLLISION_CHECK}" \
  --robot.initial_right_arm_positions="${INITIAL_RIGHT_ARM}" \
  --robot.initial_left_arm_positions="${INITIAL_LEFT_ARM}" \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=master_arm \
  --teleop.robot_address="${ROBOT_ADDRESS}" \
  --teleop.master_arm_model_path="${MASTER_ARM_URDF}" \
  --teleop.fixed_torso_positions="${FIXED_TORSO}" \
  --teleop.fixed_head_positions="${FIXED_HEAD}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.video=true \
  --dataset.streaming_encoding="${STREAMING_ENCODING}" \
  --dataset.encoder_threads="${ENCODER_THREADS}" \
  --display_data="${DISPLAY_DATA}"

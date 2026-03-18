#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# RBY1 Teleoperation Recording Script
###############################################################################

# ── Robot ──
ROBOT_ADDRESS="192.168.30.1:50051"
ROBOT_MODEL="m"
USE_IMPEDANCE=false         

# ── Master Arm ──
MASTER_ARM_URDF="/home/$(whoami)/rby1-dev/external/rby1-sdk/models/master_arm/model.urdf"

# ── Fixed Poses (radian) ──
# torso: [0, 45, -90, 45, 0, 0]°  /  head: [0, 0]°
FIXED_TORSO="[0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0]"
FIXED_HEAD="[0.0, 0.0]"

# ── Cameras ──
CAMERAS='{
  front:   {type: intelrealsense, serial_number_or_name: "", width: 640, height: 480, fps: 15},
  right: {type: intelrealsense, serial_number_or_name: "", width: 480, height: 640, fps: 15, rotation: 90}
}'

# ── Dataset ──
REPO_ID="rainbowrobotics/rby1-right-arm-task"
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

lerobot-record \
  --robot.type=rby1 \
  --robot.address="${ROBOT_ADDRESS}" \
  --robot.model="${ROBOT_MODEL}" \
  --robot.use_impedance="${USE_IMPEDANCE}" \
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

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from threading import Thread
import time
from sensor_msgs.msg import Image
import rclpy.qos
from threading import Thread
from cv_bridge import CvBridge

import rclpy
import logging
from pprint import pformat

from lerobot.common.robots import RobotConfig
import numpy as np

from .robot import Robot
from pydantic import BaseModel
from typing import Any

import rclpy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
class Joints(BaseModel):
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float


def map_value(value, from_min, from_max, to_min, to_max):
    """Maps a value from one range to another."""
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

degrees_to_radians = np.pi / 180.0
        

def make_robot_from_config(config: RobotConfig, teleop=None) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so100_follower_end_effector":
        from .so100_follower import SO100FollowerEndEffector

        return SO100FollowerEndEffector(config)
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower
        class SO101FollowerROS2(SO101Follower):
            def __init__(self, leader, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.leader = leader
                rclpy.init()
                self.node = rclpy.create_node("so101_follower")
                self.publisher = self.node.create_publisher(Float64MultiArray, '/base/position_controller/commands', 10)
                qos_profile = rclpy.qos.QoSProfile(depth=10)
                qos_profile.reliability = rclpy.qos.ReliabilityPolicy.RELIABLE
                self.subscriber = self.node.create_subscription(Image, '/base/gripper_camera_image', self.image_callback, qos_profile)
                self.joint_states_subscriber = self.node.create_subscription(JointState, '/base/joint_states', self.joint_states_callback, rclpy.qos.qos_profile_sensor_data)
                self.joint_states = None
                self.cur_img = None
                self.bridge = CvBridge()
                t = Thread(target=rclpy.spin, args=(self.node,))
                t.start()

            def joint_states_callback(self, msg):
                INVERT_NORM = True
                self.joint_states = {}
                for name, position in zip(msg.name, msg.position):
                    action_name = name.replace("base/", "") + ".pos"
                    self.joint_states[action_name] = position
                
                if INVERT_NORM:
                    self.joint_states['shoulder_pan.pos'] = map_value(self.joint_states['shoulder_pan.pos'] / degrees_to_radians, -110, 100, -100, 100)
                    self.joint_states['shoulder_lift.pos'] = map_value(self.joint_states['shoulder_lift.pos'] / degrees_to_radians, -100, 100, -100, 100)
                    self.joint_states['elbow_flex.pos'] = self.joint_states['elbow_flex.pos'] / 0.0165806
                    self.joint_states['wrist_flex.pos'] = self.joint_states['wrist_flex.pos'] / 0.01658065
                    self.joint_states['wrist_roll.pos'] = self.joint_states['wrist_roll.pos'] / (0.0174533 * 2)
                    self.joint_states['gripper.pos'] = self.joint_states['gripper.pos'] / 0.01919863

            def image_callback(self, msg):
                self.cur_img = self.bridge.imgmsg_to_cv2(msg)
                # remove 4th channel 
                self.cur_img = self.cur_img[:, :, :3]

            def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
                print(action)
                joints = Joints(
                    shoulder_pan=map_value(action['shoulder_pan.pos'], -100, 100, -110, 100) * degrees_to_radians, 
                    #shoulder_pan=action['shoulder_pan.pos'] * 0.009599315,
                    shoulder_lift=map_value(action['shoulder_lift.pos'], -100, 100, -100, 100) * degrees_to_radians, 
                    #shoulder_lift=action['shoulder_lift.pos'] * 0.0279253,
                    elbow_flex=action['elbow_flex.pos'] * 0.0165806,
                    wrist_flex=action['wrist_flex.pos'] * 0.01658065,
                    wrist_roll=action['wrist_roll.pos'] * 0.0174533*2,
                    gripper=action['gripper.pos'] * 0.01919863,
                )
                position_command = Float64MultiArray()
                position_command.data = [joints.shoulder_pan, joints.shoulder_lift, joints.elbow_flex, joints.wrist_flex, joints.wrist_roll, joints.gripper]
                print(f'Publishing: {joints.model_dump_json()}')
                self.publisher.publish(position_command)
                return action

            def get_observation(self) -> dict[str, Any]:
                while self.cur_img is None or self.joint_states is None:
                    rclpy.spin_once(self.node, timeout_sec=0.1)
                obs_dict = self.joint_states.copy()
                for cam_key, cam in self.cameras.items():
                    obs_dict[cam_key] = self.cur_img

                return obs_dict

            def connect(self):
                pass

        return SO101FollowerROS2(teleop, config)

    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot

        return Stretch3Robot(config)
    elif config.type == "viperx":
        from .viperx import ViperX

        return ViperX(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    else:
        raise ValueError(config.type)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions

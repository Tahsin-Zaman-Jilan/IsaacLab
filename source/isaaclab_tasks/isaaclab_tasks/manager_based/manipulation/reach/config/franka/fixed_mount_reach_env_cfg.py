# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import ik_rel_env_cfg


@configclass
class FrankaFixedMountReachEnvCfg(ik_rel_env_cfg.FrankaReachEnvCfg):
    """Franka IK-relative reach task with the arm fixed at an elevated, downward-pointing pose.

    The robot base is mounted in the air (as if bolted to a ceiling beam) and rotated so the
    gripper points down. This only changes the robot's initial pose; everything else (table,
    target marker, IK action, observations, rewards) is inherited unchanged from the parent
    IK-relative reach config.
    """

    def __post_init__(self):
        # post init of parent (sets up Franka + IK-rel action)
        super().__post_init__()

        # Elevated "floating" mount position (placeholder height, to be tuned).
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
        # Scalar-first (w, x, y, z) quaternion: 180 deg about local X so the gripper points down.
        self.scene.robot.init_state.rot = (0.0, 1.0, 0.0, 0.0)

# dual_arm_step2_env_cfg.py
#
# Bisection step 2: arm 1 gets your real Franka+gripper joint init from
# cam_lock_project (panda_joint2=-0.569 etc., fingers open at 0.04). No
# rotation yet (that's a later, separate step). Arm 2 stays the bare
# FRANKA_PANDA_CFG default from step 1 — only ONE variable changes here.
# Reward/command/termination logic: still 100% official, untouched.

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.dual_arm_env_cfg import (
    FrankaDualArmReachEnvCfg,
)


@configclass
class FrankaDualArmStep2EnvCfg(FrankaDualArmReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        current_pos = self.scene.robot.init_state.pos
        current_rot = self.scene.robot.init_state.rot
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=current_pos,
            rot=current_rot,
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        )

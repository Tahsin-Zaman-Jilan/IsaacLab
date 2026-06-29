# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Reach-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reach-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaReachEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Reach-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaReachEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reach-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaReachEnvCfg",
    },
    disable_env_checker=True,
)

##
# Operational Space Control
##

gym.register(
    id="Isaac-Reach-Franka-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-Franka-OSC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)
gym.register(id="Isaac-Reach-Franka-DualArm-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs={"env_cfg_entry_point": f"{__name__}.dual_arm_env_cfg:FrankaDualArmReachEnvCfg", "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"})
gym.register(id="Isaac-Reach-Franka-DualArm-Step2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs={"env_cfg_entry_point": f"{__name__}.dual_arm_step2_env_cfg:FrankaDualArmStep2EnvCfg", "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"})
gym.register(id="Isaac-Reach-Franka-DualArm-RealTable-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs={"env_cfg_entry_point": f"{__name__}.dual_arm_realtable_env_cfg:FrankaDualArmRealTableEnvCfg", "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"})
gym.register(id="Isaac-Reach-Franka-SingleArm-RealTable-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs={"env_cfg_entry_point": f"{__name__}.single_arm_realtable_env_cfg:FrankaSingleArmRealTableEnvCfg", "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"})

##
# Fixed-Mount Reach - IK Relative (elevated, downward-pointing base)
##

gym.register(
    id="FixedMount-Reach-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_mount_reach_env_cfg:FrankaFixedMountReachEnvCfg",
    },
    disable_env_checker=True,
)

##
# Vertical-Lift Fixed-Mount Reach - IK Relative (elevated base + real Z lift joint)
##

gym.register(
    id="VerticalLift-FixedMount-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vertical_lift_reach_env_cfg:FrankaVerticalLiftReachEnvCfg",
    },
    disable_env_checker=True,
)

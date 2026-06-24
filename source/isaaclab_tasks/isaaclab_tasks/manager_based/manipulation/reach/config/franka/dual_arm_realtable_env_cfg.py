# dual_arm_realtable_env_cfg.py
#
# Bisection step: official single-arm Franka reach reward/command/termination,
# UNCHANGED, with two bare arms placed on YOUR real cam-lock cuboid table
# (1.2 x 0.8 x 1.0, from cam_lock_project_env_cfg.py) instead of the official
# Nucleus SeattleLabTable. No gripper joint-pose changes yet, no rotation yet
# -- those are still later, separate steps.

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaDualArmRealTableEnvCfg(FrankaReachEnvCfg):
    """Two bare arms on your real cuboid table. Reward/command/termination
    still 100% official -- only the table asset and arm count change here."""

    def __post_init__(self):
        super().__post_init__()

        # ---- replace official's Nucleus table with your real cuboid ----
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.8, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.88)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )

        # ---- arm 1: move onto the table top (z=1.0), spaced to one side ----
        base_pos = self.scene.robot.init_state.pos
        self.scene.robot.init_state.pos = (base_pos[0], base_pos[1] - 0.3, 1.0)

        # ---- arm 2: bare official arm, other side of the same table ----
        self.scene.robot_2 = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (base_pos[0], base_pos[1] + 0.3, 1.0)

        self.actions.arm_action_2 = mdp.JointPositionActionCfg(
            asset_name="robot_2", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True,
        )

        self.commands.ee_pose_2 = mdp.UniformPoseCommandCfg(
            asset_name="robot_2",
            body_name="panda_hand",
            resampling_time_range=self.commands.ee_pose.resampling_time_range,
            debug_vis=self.commands.ee_pose.debug_vis,
            ranges=self.commands.ee_pose.ranges,
        )

        self.rewards.end_effector_position_tracking_2 = RewTerm(
            func=mdp.position_command_error, weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot_2", body_names=["panda_hand"]), "command_name": "ee_pose_2"},
        )
        self.rewards.end_effector_position_tracking_fine_grained_2 = RewTerm(
            func=mdp.position_command_error_tanh, weight=0.1,
            params={"asset_cfg": SceneEntityCfg("robot_2", body_names=["panda_hand"]), "std": 0.1, "command_name": "ee_pose_2"},
        )
        self.rewards.end_effector_orientation_tracking_2 = RewTerm(
            func=mdp.orientation_command_error, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot_2", body_names=["panda_hand"]), "command_name": "ee_pose_2"},
        )
        self.rewards.joint_vel_2 = RewTerm(
            func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot_2")},
        )

        self.observations.policy.joint_pos_2 = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.policy.joint_vel_2 = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.policy.pose_command_2 = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose_2"},
        )

        self.events.reset_robot_joints_2 = EventTerm(
            func=mdp.reset_joints_by_scale, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot_2"), "position_range": (0.75, 1.25), "velocity_range": (0.0, 0.0)},
        )

        self.curriculum.joint_vel_2 = CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "joint_vel_2", "weight": -0.001, "num_steps": 4500},
        )

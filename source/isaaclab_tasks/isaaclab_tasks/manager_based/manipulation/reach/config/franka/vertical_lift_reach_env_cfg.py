# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.devices import DevicesCfg, Se3GamepadCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .fixed_mount_reach_env_cfg import FrankaFixedMountReachEnvCfg


@configclass
class FrankaVerticalLiftReachEnvCfg(FrankaFixedMountReachEnvCfg):
    """Fixed-mount Franka reach task with a real, commandable vertical "virtual torso" DOF.

    The frozen elevated mount from :class:`FrankaFixedMountReachEnvCfg` is replaced by a real
    prismatic joint along world Z (``lift_joint``), authored into a local copy of the Franka USD
    (``panda_instanceable_vertical_lift.usd``): a non-colliding ``lift_anchor`` body is fixed to
    world at the same pose the mount used to be frozen at, and ``lift_joint`` connects it to
    ``panda_link0`` with +-0.3 m of travel about that pose. At the joint's default position (0.0)
    this is pose-identical to the old frozen mount.

    The lift joint is driven by a separate ``lift_action`` term (relative joint position, alongside
    the inherited arm IK-rel action), so it is an additional, independently controllable DOF and not
    part of the IK solve.
    """

    def __post_init__(self):
        # post init of parent (fixed-mount base pose + inherited IK-rel arm action)
        super().__post_init__()

        # Point the robot spawn at the local USD with the lift_anchor body + lift_joint.
        #
        # NOTE: self.scene.robot is built via FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=...)
        # (see ik_rel_env_cfg.py). dataclasses.replace() only deep-copies fields passed as kwargs;
        # every other field (spawn, actuators, init_state, ...) is the *same shared object* as the
        # FRANKA_PANDA_HIGH_PD_CFG global singleton. Mutating those nested objects in place (e.g.
        # `self.scene.robot.spawn.usd_path = ...`) would silently corrupt that singleton for every
        # other task that imports it (Isaac-Reach-Franka-IK-Rel-v0, FixedMount-*, GripperTeleop-*,
        # ...) if their cfgs are ever instantiated in the same process. So every change below
        # reassigns a freshly-built object/dict instead of mutating one in place.
        lift_usd_path = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Robots", "FrankaEmika", "panda_instanceable_vertical_lift.usd"
        )
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=lift_usd_path)

        # lift_joint starts at 0.0, i.e. exactly at the old frozen mount's pose.
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            joint_pos={**self.scene.robot.init_state.joint_pos, "lift_joint": 0.0}
        )

        # Implicit actuator for the new DOF. Gains mirror panda_hand's finger actuator (the only
        # other prismatic, meters-units joint on this asset) rather than the arm's revolute-joint
        # (radians-units) gains, since the units differ. The whole robot has disable_gravity=True
        # (set by FRANKA_PANDA_HIGH_PD_CFG), so lift_anchor and panda_link0 carry no gravity load
        # through this joint; these gains are untuned placeholders for crisp position tracking,
        # not load-bearing calculations.
        self.scene.robot.actuators = {
            **self.scene.robot.actuators,
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=1.0,
                stiffness=1000.0,
                damping=100.0,
            ),
        }

        # Separate action term for the lift joint, alongside arm_action. Relative (not absolute)
        # position control: matches the IK-rel arm term's "hold key -> keep moving" semantics, unlike
        # plain JointPositionActionCfg which would just re-snap to one fixed target every step.
        self.actions.lift_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=0.5,
        )

        # Enable the keyboard's opt-in lift axis (U/J) so the device's command tensor grows by the
        # one extra dimension lift_action expects. gripper_term stays False (inherited from the base
        # Reach task; this config has no gripper action).
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=False,
                    lift_term=True,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
            },
        )

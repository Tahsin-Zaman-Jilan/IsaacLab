# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices import DevicesCfg, Se3GamepadCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .gripper_teleop_reach_env_cfg import FrankaGripperTeleopReachEnvCfg


@configclass
class FrankaVerticalLiftGripperTeleopReachEnvCfg(FrankaGripperTeleopReachEnvCfg):
    """Fixed-mount Franka reach task combining the real vertical lift joint AND real gripper control.

    Two pieces were developed independently, on sibling branches, each subclassing
    :class:`FrankaFixedMountReachEnvCfg` on its own and never combined into one class:

    * :class:`FrankaGripperTeleopReachEnvCfg`: ``gripper_action`` + a graspable cube.
    * :class:`FrankaVerticalLiftReachEnvCfg`: ``lift_joint`` (Z-axis prismatic) + ``lift_action``.

    This class brings both together explicitly -- every piece either inherited from one parent or
    re-added here by hand, not assumed to "come along for free."

    Inheritance is from :class:`FrankaGripperTeleopReachEnvCfg`, *not* from
    :class:`FrankaVerticalLiftReachEnvCfg`, for a correctness reason, not just style: the base
    ``ActionsCfg`` (in ``reach_env_cfg.py``) pre-declares ``arm_action`` and ``gripper_action`` as
    fields, but never pre-declares ``lift_action`` -- so ``lift_action`` always lands at the end of
    the action term insertion order, wherever it's first assigned. ``Se3Keyboard.advance()`` always
    concatenates its command tensor as [pos+rot (6), gripper (1) if enabled, lift (1) if enabled] --
    gripper before lift, fixed order, not configurable. ``ActionManager.process_action()`` slices the
    incoming flat action tensor by *term insertion order*, not by name. So the action term order here
    must come out as [arm_action, gripper_action, lift_action] to match the device tensor's
    [pos+rot, gripper, lift] layout. Calling ``super().__post_init__()`` on
    :class:`FrankaGripperTeleopReachEnvCfg` first gives exactly [arm_action, gripper_action]; adding
    ``lift_action`` afterwards in this class's own ``__post_init__`` appends it last, producing the
    required order. Inheriting the other way around (lift first, gripper added after) would silently
    cross-wire the two: gripper key-presses would drive the lift joint, and lift key-presses would
    drive the gripper open/close logic.
    """

    def __post_init__(self):
        # post init of parent: fixed-mount base pose + arm IK-rel action + gripper_action + cube +
        # teleop_devices with gripper_term=True. At this point self.actions insertion order is
        # exactly [arm_action, gripper_action].
        super().__post_init__()

        # --- bring in the lift mechanism: USD spawn, joint init state, actuator ---
        # Same values as FrankaVerticalLiftReachEnvCfg. Duplicated rather than inherited from it,
        # since this class inherits from FrankaGripperTeleopReachEnvCfg instead (see class docstring
        # for why that direction is required, not a style choice).
        lift_usd_path = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Robots", "FrankaEmika", "panda_instanceable_vertical_lift.usd"
        )
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=lift_usd_path)

        # lift_joint starts at 0.0, i.e. exactly at the old frozen mount's pose.
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            joint_pos={**self.scene.robot.init_state.joint_pos, "lift_joint": 0.0}
        )

        # Implicit actuator for the lift DOF. See FrankaVerticalLiftReachEnvCfg for the reasoning
        # behind these gains (mirrors panda_hand's finger actuator, the only other prismatic,
        # meters-units joint on this asset; untuned placeholders, not load-bearing calculations).
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

        # --- bring in lift_action, AFTER gripper_action: see class docstring, ordering is required ---
        self.actions.lift_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=0.5,
        )

        # --- rebuild teleop_devices with BOTH gripper_term and lift_term True ---
        # super().__post_init__() already set gripper_term=True (FrankaGripperTeleopReachEnvCfg's own
        # override), but that override has no notion of lift_term, so without rebuilding it here this
        # class's teleop_devices would still be missing the lift axis.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=True,
                    lift_term=True,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=True,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=True,
                    sim_device=self.sim.device,
                ),
            },
        )

        # --- FR3 transparent box prop ---
        # FR3_v2.usd is authored in Y-up / cm units (metersPerUnit=0.01).
        # Isaac Lab's spawn_from_usd does NOT apply the metersPerUnit scale factor; raw USD
        # coordinate values are placed directly into the Z-up, meters scene.  This means:
        #   - Without an explicit scale the box appears 2 m wide × 3 m deep × 0.75 m tall.
        #   - scale=(0.1, 0.1, 0.1) brings it to ~20 cm × 30 cm × 7.5 cm.
        # Adjust scale if the box appears at the wrong size during visual inspection.
        #
        # The +90° rotation around X (quaternion (0.707, 0.707, 0, 0)) converts from Y-up to
        # Z-up so the box stands upright with its opening facing +Z.
        #
        # The Box_v2 child prim inside the USD has an internal translate of (-1.35, 0, 0.195)
        # in USD local coords.  After scale=(0.1) and the +90°-X rotation this becomes a
        # world-frame offset of (-0.135, -0.020, 0) from the asset root, so the box geometry
        # appears ~13.5 cm in -X and 2 cm in -Y from the init_state.pos below.
        #
        # At pos=(0.3, 0.0, 0.0) the box opening (top face after rotation) sits at world
        # Z ≈ 0.075 m, clear of the arm's default resting X position (~-0.15 m) to avoid
        # spawn-time contact.  Tune pos as needed once the gripper orientation is corrected.
        self.scene.fr3_box = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FR3Box",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/exx/Tahsin/tahsin/FR3_v2.usd",
                scale=(0.1, 0.1, 0.1),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.3, 0.0, 0.0),
                rot=(0.7071, 0.7071, 0.0, 0.0),
            ),
        )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg, Se3GamepadCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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

        # --- remove the table ---
        # ``table`` is defined in the grandparent ``ReachSceneCfg`` (reach_env_cfg.py), shared by
        # every Reach task variant, so it can't be deleted there without affecting other configs.
        # Nulling it here instead: ``InteractiveScene._add_entities_from_cfg`` skips any scene field
        # whose value is None, so this fully removes the table (no empty prim, no physics body).
        # Task uses the transparent FR3Box (below) as the real prop instead.
        self.scene.table = None

        # --- bring in the lift mechanism: USD spawn, joint init state, actuator ---
        # Same values as FrankaVerticalLiftReachEnvCfg. Duplicated rather than inherited from it,
        # since this class inherits from FrankaGripperTeleopReachEnvCfg instead (see class docstring
        # for why that direction is required, not a style choice).
        lift_usd_path = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Robots", "FrankaEmika", "panda_instanceable_vertical_lift.usd"
        )
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=lift_usd_path)

        # lift_joint starts at 0.0, i.e. exactly at the old frozen mount's pose.
        #
        # pos.z is also overridden here (not left at FrankaFixedMountReachEnvCfg's pos=(0,0,0.6)):
        # that height put the base *inside* FR3Box, whose real top surface sits at world Z=0.75
        # (measured directly off the box mesh's bounding box -- Box_v2/body spans Y in [0, 0.75] in
        # its own Y-up frame, and the fr3_box asset's rot=(0.7071,0.7071,0,0) maps that Y straight
        # onto world Z with no additional scaling). 1.36 was chosen from the arm's own kinematics:
        # forward-kinematics on the default ready pose (panda_joint1-7 = 0, -0.569, 0, -2.810, 0,
        # 3.037, 0.741, the same values FRANKA_PANDA_CFG ships and this task never overrides) via
        # Franka's published modified-DH parameters places panda_hand, plus the 0.107 m
        # hand->TCP offset already used by this task's arm_action, at (0.389, 0, 0.458) relative to
        # panda_link0. FrankaFixedMountReachEnvCfg's rot=(0,1,0,0) (180 deg about local X) flips that
        # to a world-frame offset of (0.389, 0, -0.458) from the base. So at lift_joint=0 with the
        # arm at its default pose, the gripper TCP sits ~0.458 m below whatever pos.z is set here --
        # solving pos.z - 0.458 = 0.90 (i.e. ~0.15 m of clearance above the box's Z=0.75 top surface,
        # so nothing clips through the lid at spawn) gives pos.z = 1.358, rounded to 1.36.
        #
        # x, y stay at 0.0 (unchanged): the box's hole (see fr3_box below) is close enough to
        # world (0, 0) -- within the reach of an IK-rel teleop nudge -- that shifting the whole base
        # to chase it isn't worth the extra deviation from the parent's pose.
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            pos=(0.0, 0.0, 1.36),
            joint_pos={**self.scene.robot.init_state.joint_pos, "lift_joint": 0.0},
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

        # --- FR3 transparent box (visual-only, no physics) ---
        # FR3_v2.usd is Y-up. The OBJ source was in mm, exported at 1/1000, so raw
        # USD coordinates (e.g. ±1 in X) already represent meters in real space —
        # scale=(1, 1, 1) gives the correct ~2 m × 3 m × 0.75 m box.
        # rot=(0.7071, 0.7071, 0, 0) is +90° around X, converting Y-up → Z-up so
        # the box stands upright with its opening facing +Z.
        # The Box_v2 child inside the USD has an internal translate of (-1.35, 0, 0.195)
        # in USD coords; after the rotation that shifts the geometry ~1.35 m in -X from
        # the root, so pos=(1.5, 0, 0) centres the box at roughly X ≈ 0.15 m.
        # Tune scale and pos visually after launch.
        self.scene.fr3_box = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FR3Box",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/exx/Tahsin/tahsin/FR3_v2.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(1.5, 0.0, 0.0),
                rot=(0.7071, 0.7071, 0.0, 0.0),
            ),
        )

        # --- cube, replacing the parent's tabletop placeholder ---
        # FrankaGripperTeleopReachEnvCfg's self.scene.object (inherited via super().__post_init__()
        # above) sits at pos=[-0.118, 0.013, 0.064] -- a spot tuned for grasping off a flat table at
        # the old pos=(0,0,0.6) mount height, not for sitting on FR3Box's internal platform. Overriding
        # it here rather than editing that parent file, same reasoning as the ``table`` removal above.
        #
        # Position found by opening FR3_v2.usd directly (pxr.Usd) and inspecting Box_v2/body's mesh
        # points rather than by guessing: most vertices cluster at local (Y-up) Y=0 (floor) and
        # Y=0.74-0.75 (lid), but a distinct plateau of 24 points sits at Y=0.32-0.34, spanning local
        # X=[-1.902,-0.806], Z=[-0.7225,-0.4976] -- the internal platform, with its top surface at
        # Y=0.34. Running that through the same transform fr3_box applies (rot=(0.7071,0.7071,0,0),
        # i.e. world_x=local_x, world_y=-local_z, world_z=local_y, then + pos=(1.5,0,0)) gives a
        # platform top surface centered at world (0.146, 0.61, 0.34). Cube center is that plus half of
        # its ~0.05 m side (0.025 m) so it rests on the surface rather than clipping into it.
        #
        # Note: the box's hole in the lid (also found via mesh inspection, a circular cut ~0.4 m
        # across centered near world (0.146, -0.195)) sits about 0.8 m away from this platform in Y
        # and ~0.4 m above it in Z -- deliberately not directly below the hole, matching the
        # "look-through perception" premise (navigate to hidden structure, not just drop straight
        # down). That combined reach is close to the Franka's ~0.855 m envelope even using the full
        # lift_joint travel; verify reachability in teleop and nudge fr3_box / this position if it
        # turns out to be just out of reach.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.146, 0.61, 0.365], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

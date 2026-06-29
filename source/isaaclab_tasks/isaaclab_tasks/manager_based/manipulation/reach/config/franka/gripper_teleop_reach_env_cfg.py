# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.devices import DevicesCfg, Se3GamepadCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .fixed_mount_reach_env_cfg import FrankaFixedMountReachEnvCfg


@configclass
class FrankaGripperTeleopReachEnvCfg(FrankaFixedMountReachEnvCfg):
    """Fixed-mount Franka reach task with real gripper open/close control for teleop.

    Adds the same binary gripper action term used by Isaac-Lift-Cube-Franka-IK-Rel-v0
    (the Reach task lineage never wires one up) plus a graspable cube, positioned at the
    gripper's deterministic spawn pose so a teleop user can immediately test open/close.
    Everything inherited from the fixed-mount reach config (arm action, base pose) is
    left untouched.
    """

    def __post_init__(self):
        # post init of parent (fixed-mount base pose + inherited IK-rel arm action)
        super().__post_init__()

        # Add gripper action, same term/joints/open-close values as Isaac-Lift-Cube-Franka-IK-Rel-v0
        # (source/isaaclab_tasks/.../manipulation/lift/config/franka/joint_pos_env_cfg.py).
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Add a graspable cube. The Reach scene has no rigid object by default, so this is new.
        # Same asset/scale/physics as Lift-Cube's cube. Position was found by probing headlessly:
        # spawning near the gripper's TCP overlapped the open fingers and PhysX depenetration
        # consistently shoved the cube to this same resting spot regardless of starting height,
        # so it's used directly as the spawn point (verified stable, no further drift).
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.118, 0.013, 0.064], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
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

        # ReachEnvCfg.__post_init__ hardcodes gripper_term=False for all teleop devices, since the
        # base Reach task has no gripper to control. That override is inherited verbatim through
        # FrankaReachEnvCfg -> FrankaFixedMountReachEnvCfg -> here, so it's now stale: it would
        # silently truncate Se3Keyboard/Se3Gamepad/Se3SpaceMouse.advance() to 6 elements regardless
        # of the K toggle, while this class's action space expects 7 (arm + gripper). Re-override it
        # here, scoped to this class only, so teleop devices include the gripper bit again.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=True,
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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 1: bare Franka robot with keyboard-teleoperated relative IK control.

Minimal scene -- ground plane, one light, one Franka arm. No cube, no table, no other
scene objects. Every state change (keyboard command received, EE target change, gripper
toggle, errors) is printed to the console; nothing relies on the GUI alone.

.. code-block:: bash

    ./isaaclab.sh -p scripts/motion_planning/step1_bare_robot_teleop.py --device cuda:1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 1: bare Franka robot, keyboard-teleoperated relative IK.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor for keyboard deltas.")
AppLauncher.add_app_launcher_args(parser)
# this workstation reserves cuda:0 for other users -- default to cuda:1 unless overridden
parser.set_defaults(device="cuda:1")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0


@configclass
class BareRobotSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane, light, single Franka. No other objects."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, sensitivity: float) -> None:
    """Main teleoperation loop with explicit console confirmation of every state change."""
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    # differential IK controller, relative mode (delta pose commands)
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)
    print("[INFO] DifferentialIKController ready (pose, relative mode, dls).")

    # keyboard device -- same mechanism as teleop_se3_agent.py's fallback keyboard path
    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
    )
    print(str(teleop_interface))
    print("[INFO] Keyboard teleop ready. Press 'L' to reset the teleop device state.")

    # markers to visualize current EE frame and commanded goal frame
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # initialize the controller's target at the current EE pose (zero delta = hold position)
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    zero_command = torch.zeros(scene.num_envs, ik_controller.action_dim, device=sim.device)
    ik_controller.set_command(zero_command, ee_pos_b, ee_quat_b)
    print(f"[INFO] Initial EE target (base frame): pos={ee_pos_b[0].tolist()}, quat={ee_quat_b[0].tolist()}")

    gripper_closed = False
    last_ee_pos_des = ik_controller.ee_pos_des.clone()
    sim_dt = sim.get_physics_dt()

    print("[INFO] Entering simulation loop. Move the mouse over the viewport for keyboard focus.")

    while simulation_app.is_running():
        try:
            command = teleop_interface.advance().to(sim.device)
            delta_pose = command[0:6].unsqueeze(0)
            gripper_cmd = command[6].item()

            command_is_active = bool(torch.any(torch.abs(delta_pose) > 1e-6))
            if command_is_active:
                print(
                    f"[TELEOP] Command received: delta_pos={delta_pose[0, 0:3].tolist()},"
                    f" delta_rot={delta_pose[0, 3:6].tolist()}"
                )

            # current EE pose in the robot base frame
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            ik_controller.set_command(delta_pose, ee_pos_b, ee_quat_b)
            if command_is_active and not torch.allclose(ik_controller.ee_pos_des, last_ee_pos_des, atol=1e-6):
                print(
                    f"[TARGET] New EE target (base frame): pos={ik_controller.ee_pos_des[0].tolist()},"
                    f" quat={ik_controller.ee_quat_des[0].tolist()}"
                )
                last_ee_pos_des = ik_controller.ee_pos_des.clone()

            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

            # gripper: binary open/close, set directly since there is no env action manager here
            is_close_cmd = gripper_cmd < 0.0
            if is_close_cmd != gripper_closed:
                gripper_closed = is_close_cmd
                state = "CLOSE" if gripper_closed else "OPEN"
                print(f"[GRIPPER] Toggled: {state}")
            gripper_target_val = GRIPPER_CLOSED if gripper_closed else GRIPPER_OPEN
            gripper_targets = torch.full((scene.num_envs, len(gripper_joint_ids)), gripper_target_val, device=sim.device)
            robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_pos_w = ik_controller.ee_pos_des + scene.env_origins
            goal_marker.visualize(goal_pos_w, ik_controller.ee_quat_des)

        except Exception as e:
            print(f"[ERROR] Simulation step failed: {e}")
            break

    print("[INFO] Simulation loop exited.")


def main() -> None:
    try:
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.3])

        scene_cfg = BareRobotSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        print("[INFO] Setup complete. Bare Franka scene created (no cube, no table, no extra objects).")
    except Exception as e:
        print(f"[ERROR] Failed to set up simulation/scene: {e}")
        simulation_app.close()
        return

    run_simulator(sim, scene, args_cli.sensitivity)


if __name__ == "__main__":
    main()
    simulation_app.close()

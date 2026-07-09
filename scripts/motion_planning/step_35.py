# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 3b: bare Franka robot, LIVE MOUSE-SLIDER teleop (Isaac Lab port of the
PyBullet `mouse_control()` tool), replacing the cuRobo planned-target tests with an interactive
omni.ui panel.

Same minimal scene and safety scaffolding as step3_ground_collision_test.py (ground plane, one
light, one Franka arm; default-pose reset fix; CUDA_VISIBLE_DEVICES=1 isolation guard) but instead
of planning to a fixed target with cuRobo, this opens a live omni.ui window with 6 pose sliders +
1 grasp slider (mirroring the professor's original PyBullet layout: X, Y, Z, Roll, Pitch, Yaw,
grasp). Every physics step, the sliders are read, converted into an IK-relative delta command
exactly like the PyBullet version, and solved with Isaac Lab's DifferentialIKController -- since
this bare scene has no ActionManager/env wrapper to route an "arm_action" term through, the IK
solve and joint targeting are done directly against the robot articulation, matching this script's
existing low-level `set_joint_position_target` style.

Controls:
  - Drag any of the 6 pose sliders in the "Mouse Control (Franka Teleop)" window to move the
    live target pose. The arm continuously IK-solves toward whatever the sliders currently read.
  - Drag "grasp" above 0.5 to close the gripper, below to open it.
  - Close the viewer window (or Ctrl+C in the terminal) to exit. On exit, prints the final
    delta-from-start action tuple (translate, rotate, grasp) the same way the PyBullet tool did
    when you broke out of its loop.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step3_mouse_control_teleop.py \
        --device cuda:0

"""

"""Launch Isaac Sim Simulator first."""

import os
import sys

# this workstation reserves physical GPU0 for Hangong's work -- see step2/step3's docstrings for
# the full investigation. Check this before anything touches CUDA.
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    print(
        "[ERROR] This script requires CUDA_VISIBLE_DEVICES=1 (lab policy: physical GPU0 is"
        " reserved for Hangong's work). Run as:\n"
        "    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p"
        " scripts/motion_planning/step3_mouse_control_teleop.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 3b: bare Franka robot, live mouse-slider teleop.")
parser.add_argument("--back_step", type=int, default=1, help="Reserved for parity with the original tool's back_step arg (unused for reset here; kept for interface compatibility).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni.ui as ui

# defense in depth: same double-check as step2/step3.
if torch.cuda.device_count() != 1:
    print(
        f"[ERROR] Expected exactly 1 visible CUDA device under CUDA_VISIBLE_DEVICES=1, found"
        f" {torch.cuda.device_count()}. Refusing to proceed."
    )
    simulation_app.close()
    sys.exit(1)

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# Slider ranges mirror the original PyBullet mouse_control() tool's
# addUserDebugParameter bounds exactly.
POS_X_RANGE = (0.0, 1.5)
POS_Y_RANGE = (-1.0, 1.0)
POS_Z_RANGE = (0.0, 1.15)
EULER_RANGE = (-3.15, 3.15)


@configclass
class BareRobotSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane, light, single Franka. No other objects."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def reset_to_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot) -> None:
    """Explicitly move the robot to its configured default ("ready") pose.

    sim.reset() alone leaves the articulation at the raw USD spawn pose, not
    default_joint_pos -- same pattern used in step2/step3.
    """
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()
    sim.step()
    scene.update(sim_dt)


class MouseControlWindow:
    """
    Isaac Lab port of the PyBullet mouse_control() teleop tool.

    Builds a live omni.ui panel with 6-DOF pose sliders + grasp, reads them every physics
    step, and returns an IK-relative delta action -- same shape and math as the original:
    TransEE (dx,dy,dz), RotEE (droll,dpitch,dyaw as euler), grasp.
    """

    def __init__(self, robot, ee_body_idx: int, root_body_idx: int, env_id: int = 0):
        self.robot = robot
        self.ee_body_idx = ee_body_idx
        self.root_body_idx = root_body_idx
        self.env_id = env_id
        self.device = robot.data.root_state_w.device

        self.window = None
        self.sliders = {}

        # Cache the EE pose at build time in the robot's own base frame, mirroring
        # actualEndEffectorPos / actualEndEffectorOrn from the original tool.
        self.start_pos_b, self.start_quat_b = self._get_ee_pose_in_base_frame()
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(self.start_quat_b.unsqueeze(0))
        self.start_euler_b = torch.cat([roll, pitch, yaw])

        print(
            "Your current trans and rot: ",
            self.start_pos_b.cpu().numpy().tolist(),
            torch.rad2deg(self.start_euler_b).cpu().numpy().tolist(),
        )

    # ------------------------------------------------------------------
    def _get_ee_pose_in_base_frame(self):
        """EE pose expressed in the robot's own root/base frame (world pose, transformed)."""
        ee_state_w = self.robot.data.body_state_w[self.env_id, self.ee_body_idx]
        root_state_w = self.robot.data.root_state_w[self.env_id]

        ee_pos_w = ee_state_w[0:3]
        ee_quat_w = ee_state_w[3:7]
        root_pos_w = root_state_w[0:3]
        root_quat_w = root_state_w[3:7]

        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos_w.unsqueeze(0), root_quat_w.unsqueeze(0),
            ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0),
        )
        return pos_b.squeeze(0), quat_b.squeeze(0)

    # ------------------------------------------------------------------
    def build_ui(self):
        """Isaac Lab equivalent of the p.addUserDebugParameter(...) block."""
        self.window = ui.Window("Mouse Control (Franka Teleop)", width=420, height=440)

        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label(
                    "Drag sliders to set target EE pose (robot base frame). "
                    "grasp > 0.5 closes gripper.",
                    word_wrap=True,
                    height=36,
                )
                self.sliders["x"] = self._add_slider("targetPosX", *POS_X_RANGE, float(self.start_pos_b[0]))
                self.sliders["y"] = self._add_slider("targetPosY", *POS_Y_RANGE, float(self.start_pos_b[1]))
                self.sliders["z"] = self._add_slider("targetPosZ", *POS_Z_RANGE, float(self.start_pos_b[2]))
                self.sliders["roll"] = self._add_slider("targetOriRoll", *EULER_RANGE, float(self.start_euler_b[0]))
                self.sliders["pitch"] = self._add_slider("targetOriPitch", *EULER_RANGE, float(self.start_euler_b[1]))
                self.sliders["yaw"] = self._add_slider("targetOriYaw", *EULER_RANGE, float(self.start_euler_b[2]))
                self.sliders["grasp"] = self._add_slider("grasp", 0.0, 1.0, 0.0)

    def _add_slider(self, label: str, lo: float, hi: float, default: float):
        with ui.HStack(height=24):
            ui.Label(label, width=120)
            model = ui.SimpleFloatModel(default)
            ui.FloatSlider(model=model, min=lo, max=hi, step=0.001)
        return model

    # ------------------------------------------------------------------
    def read_sliders(self):
        """Isaac Lab equivalent of the p.readUserDebugParameter(...) block."""
        target_pos = torch.tensor(
            [
                self.sliders["x"].get_value_as_float(),
                self.sliders["y"].get_value_as_float(),
                self.sliders["z"].get_value_as_float(),
            ],
            device=self.device,
        )
        target_euler = torch.tensor(
            [
                self.sliders["roll"].get_value_as_float(),
                self.sliders["pitch"].get_value_as_float(),
                self.sliders["yaw"].get_value_as_float(),
            ],
            device=self.device,
        )
        grasp = self.sliders["grasp"].get_value_as_float()
        finger_target = GRIPPER_CLOSED if grasp > 0.5 else GRIPPER_OPEN
        return target_pos, target_euler, grasp, finger_target

    # ------------------------------------------------------------------
    def get_target_pose_b(self):
        """Current slider target pose (pos, quat) in the robot base frame."""
        target_pos, target_euler, grasp, finger_target = self.read_sliders()
        target_quat = math_utils.quat_from_euler_xyz(
            target_euler[0:1], target_euler[1:2], target_euler[2:3]
        ).squeeze(0)
        return target_pos, target_quat, grasp, finger_target

    # ------------------------------------------------------------------
    def compute_delta_from_start(self):
        """
        Isaac Lab equivalent of the original tool's post-loop delta calculation:
        TransEE = target - actual_start
        RotEE   = euler(diff * inverse(start_quat))   where diff * q1 = q2
        Returns (TransEE, RotEE_euler, grasp) exactly like the PyBullet version's
        final np.concatenate((TransEE, np.rad2deg(RotEE), [grasp])) printout.
        """
        target_pos, target_quat, grasp, _ = self.get_target_pose_b()

        trans_ee = target_pos - self.start_pos_b
        diff_quat = math_utils.quat_mul(
            target_quat.unsqueeze(0), math_utils.quat_inv(self.start_quat_b.unsqueeze(0))
        )
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(diff_quat)
        rot_ee = torch.cat([roll, pitch, yaw])
        return trans_ee, rot_ee, grasp

    # ------------------------------------------------------------------
    def close(self):
        """Isaac Lab equivalent of p.removeAllUserParameters()."""
        if self.window is not None:
            self.window.visible = False
            self.window = None


def run_mouse_teleop(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Live mouse-slider teleop loop, replacing the cuRobo plan-to-fixed-target tests."""
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    ee_body_idx = robot_entity_cfg.body_ids[0]
    root_body_idx = 0  # root/base link, index 0 in body_state_w for a fixed-base articulation

    reset_to_default_pose(sim, scene, robot)
    print("[INFO] Robot set to default pose.")

    # DifferentialIKController solves for joint deltas toward an absolute target pose
    # expressed in the robot base frame -- this is the "no ActionManager" substitute for
    # the IK-relative arm_action term your box-world env normally routes this through.
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)
    ik_controller.reset()

    mouse_control = MouseControlWindow(robot, ee_body_idx=ee_body_idx, root_body_idx=root_body_idx)
    mouse_control.build_ui()

    jacobi_body_idx = ee_body_idx - 1 if robot.is_fixed_base else ee_body_idx
    jacobi_joint_ids = robot_entity_cfg.joint_ids

    sim_dt = sim.get_physics_dt()
    print("[INFO] Live mouse-slider teleop running. Drag sliders in the 'Mouse Control (Franka Teleop)' window.")
    print("[INFO] Close the viewer window or Ctrl+C in the terminal to exit.")

    try:
        while simulation_app.is_running():
            target_pos_b, target_quat_b, grasp, finger_target = mouse_control.get_target_pose_b()
            ik_controller.set_command(
                torch.cat([target_pos_b, target_quat_b]).unsqueeze(0)
            )

            jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
            ee_pose_b, ee_quat_b = mouse_control._get_ee_pose_in_base_frame()
            joint_pos_cur = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            joint_pos_des = ik_controller.compute(
                ee_pose_b.unsqueeze(0), ee_quat_b.unsqueeze(0), jacobian, joint_pos_cur
            )

            gripper_targets = torch.full(
                (scene.num_envs, len(gripper_joint_ids)), finger_target, device=sim.device
            )
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
    except KeyboardInterrupt:
        pass

    trans_ee, rot_ee, grasp = mouse_control.compute_delta_from_start()
    print(
        "Your current action - delta_trans, delta_rot, grasp: \n",
        torch.cat([trans_ee, torch.rad2deg(rot_ee), torch.tensor([grasp], device=trans_ee.device)])
        .cpu()
        .numpy(),
    )
    mouse_control.close()


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

    try:
        run_mouse_teleop(sim, scene)
    except Exception as e:
        print(f"[ERROR] Mouse-control teleop failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal 2-control slider teleop for Isaac-Lift-Cube-Franka-Box-v0: Reach/Descend (EE world Z)
and Gripper open/close. Rebuilt on branch official-task-upside-down-test after the original file
was lost; this task's action space is (1, 7) -- arm_action(6) + gripper_action(1), no lift_joint
exists on this branch's robot asset, so the Base Up/Down control from the earlier version is
dropped entirely rather than driving a joint that doesn't exist.

Reach/Descend reuses the same proven pattern verified via headless test on the other (lift_joint)
task: world-frame position error -> quat_apply_inverse into the robot's base frame (root_quat_w
read live, not a hardcoded sign flip) -> clip -> fed into the IK-relative arm_action. Only the Z
component of the error is ever non-zero.

Gripper is unchanged: raw pass-through, no delta-tracking.

Z slider range is grounded in a fresh headless FK check on THIS task specifically (not reused
from the other, differently-scaled task): post-reset EE Z measured at 0.2993 m, box outer top at
z=0.5, cube at z=0.28, fixed base at z=0.783. Range 0.2-0.75 covers descending toward the cube up
through comfortable retraction below the fixed base, without asking for an unreachable target
above it.

A "Loop tick" counter is included so a frozen main loop is immediately distinguishable from an
unresponsive slider widget.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal 2-control (Z / gripper) slider teleop for the box task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Box-v0", help="Task name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import omni.ui as ui

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse

MAX_Z_DELTA_PER_STEP = 0.02  # meters/frame, EE Z proportional-control clip
GRIPPER_OPEN_VALUE = 1.0

Z_MIN, Z_MAX, Z_STEP = 0.2, 0.75, 0.01  # grounded in a fresh headless FK check on this task


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    ee_frame = env.scene["ee_frame"]
    robot = env.scene["robot"]

    current_z = ee_frame.data.target_pos_w[0, 0, 2].item()

    window = ui.Window("Slider Test - 2 Axis (Box Task)", width=360, height=200)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label("Reach/Descend (EE world Z, m)")
            with ui.HStack():
                ui.Label("Target Z", width=90)
                z_slider = ui.FloatSlider(min=Z_MIN, max=Z_MAX, step=Z_STEP)
                z_slider.model.set_value(current_z)
            z_readout = ui.Label(f"Current Z: {current_z:.4f}")

            ui.Label("Gripper (raw, + open / - close)")
            gripper_slider = ui.FloatSlider(min=-1.0, max=1.0, step=0.1)
            gripper_slider.model.set_value(GRIPPER_OPEN_VALUE)

            tick_label = ui.Label("Loop tick: 0")

    env.reset()

    tick = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            root_quat_w = robot.data.root_quat_w[0:1]

            # --- Reach/Descend: world-frame Z error -> base-frame delta, clipped ---
            current_pos = ee_frame.data.target_pos_w[0, 0]
            target_z = z_slider.model.get_value_as_float()
            world_z_err = torch.zeros(1, 3, device=env.device)
            world_z_err[0, 2] = target_z - current_pos[2].item()
            base_z_err = quat_apply_inverse(root_quat_w, world_z_err)[0]
            z_delta = torch.clamp(base_z_err[2], -MAX_Z_DELTA_PER_STEP, MAX_Z_DELTA_PER_STEP)

            # --- Gripper: raw pass-through ---
            gripper_val = gripper_slider.model.get_value_as_float()

            action = torch.zeros(env.num_envs, 7, device=env.device)
            action[:, 2] = z_delta
            action[:, 6] = gripper_val

            env.step(action)

            z_readout.text = f"Current Z: {current_pos[2].item():.4f}"
            tick += 1
            tick_label.text = f"Loop tick: {tick}"

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

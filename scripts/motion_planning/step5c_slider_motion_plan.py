# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 5c: slider target picking + cuRobo motion planning, combining the two
confirmed-working pieces from step5b_slider_no_live_move.py and step2_single_target_motion_plan.py.

Same bare, ground-mounted robot scene as every prior step (ground plane, one light, one Franka arm
-- no cube, no table, no upside-down mounting). Per the advisor's "simplest domain first"
instruction, this step deliberately stays on the simplest possible scene while adding the one new
thing being tested: feeding a slider-picked target into cuRobo.

Ordering discipline (the advisor's explicit instruction, confirmed before implementing): on
"Confirm Target", the saved baseline state is RESTORED FIRST via scene.reset_to() + sim.forward(),
and ONLY THEN is the cuRobo planning start_state read from robot.data.joint_pos -- freshly, post-
restore. The plan is computed from the restored baseline, never from whatever state existed right
before the click. Since step5b already proved the robot cannot drift while the sliders are up, the
restored state and the pre-confirm state are numerically identical -- but the code path enforces
this ordering regardless, rather than assuming it.

Combines:
  1. From step5b_slider_no_live_move.py: SliderTargetWindow (X/Y/Z sliders, zero side effects on
     the robot, "Confirm Target" button) and the save/restore mechanism
     (scene.get_state() / scene.reset_to() + sim.forward()).
  2. From step2_single_target_motion_plan.py: the CUDA_VISIBLE_DEVICES=1 guard, cuRobo's MotionGen
     pipeline (franka.yml + dummy far-away obstacle), plan_single(), the [PLAN] SUCCESS/FAILED
     discipline, and automatic waypoint execution.

Flow:
  1. Reset to default pose, save this state (baseline). Build + warm up cuRobo's MotionGen (a
     one-time, robot-state-independent construction -- doing this before the slider loop does not
     violate the reset-before-plan ordering, since no planning call happens here).
  2. Show sliders. User drags to pick a target position -- the robot stays frozen the whole time
     (5b's zero-side-effect guarantee); every physics step verifies no drift against the baseline.
  3. User clicks "Confirm Target".
  4. On confirm: print the confirmed target, restore the saved baseline
     (scene.reset_to(saved_state) + sim.forward()), THEN read the post-restore joint state as
     cuRobo's start_state, THEN plan to the confirmed target.
  5. Print [PLAN] SUCCESS/FAILED. If successful, execute the planned trajectory automatically
     (no manual driving) -- same waypoint-stepping pattern as step2.
  6. After execution, print the final achieved EE position and error vs. the requested target.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step5c_slider_motion_plan.py \
        --device cuda:0

"""

"""Launch Isaac Sim Simulator first."""

import os
import sys

# this workstation reserves physical GPU0 for Hangong's work -- cuRobo has an internal
# device-threading gap (see step2_single_target_motion_plan.py's docstring for the full
# investigation) that we work around via CUDA_VISIBLE_DEVICES isolation rather than by
# patching cuRobo internals. Check this before anything touches CUDA (Isaac Sim boot is
# ~20s; fail fast instead of discovering this mid-warmup).
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    print(
        "[ERROR] This script requires CUDA_VISIBLE_DEVICES=1 (lab policy: physical GPU0 is"
        " reserved for Hangong's work, and cuRobo's internal device handling only works"
        " reliably here under single-GPU process isolation). Run as:\n"
        "    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p"
        " scripts/motion_planning/step5c_slider_motion_plan.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 5c: slider target picking + cuRobo motion planning.")
AppLauncher.add_app_launcher_args(parser)
# under the CUDA_VISIBLE_DEVICES=1 isolation enforced above, physical GPU1 (cuda:1) is
# remapped to index 0 for this process -- so cuda:0 here is the correct, safe target
parser.set_defaults(device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni.ui as ui

# defense in depth: the os.environ check above catches a missing/wrong env var, this
# catches the case where CUDA_VISIBLE_DEVICES=1 was set but somehow more than one GPU is
# still visible to torch -- either way, refuse rather than risk touching physical GPU0.
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from curobo.types.base import TensorDeviceType  # isort:skip
from curobo.types.math import Pose  # isort:skip
from curobo.types.state import JointState  # isort:skip
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # isort:skip

GRIPPER_OPEN = 0.04

# Slider ranges mirror step5b/step1_mouse_control_teleop.py's (in turn mirroring the original
# PyBullet mouse_control() tool's addUserDebugParameter bounds).
POS_X_RANGE = (0.0, 1.5)
POS_Y_RANGE = (-1.0, 1.0)
POS_Z_RANGE = (0.0, 1.15)

# No orientation slider yet (simplest domain first) -- fixed downward-facing gripper orientation,
# same convention used throughout step2/step3's fixed target quats.
DEFAULT_TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]

# Drift tolerance for the "robot hasn't moved while waiting for confirmation" check -- loose
# enough to absorb PD/physics settling jitter at rest, tight enough to catch a real bug.
DRIFT_TOL_RAD = 1e-3

# cuRobo's collision checker requires at least one registered primitive obstacle -- a genuinely
# empty world dict raises "Primitive Collision has no obstacles". This dummy cuboid is placed
# 100m away, far outside the Franka's ~0.85m reach: functionally an obstacle-free world.
DUMMY_FAR_AWAY_WORLD = {
    "cuboid": {
        "dummy_far_away": {
            "dims": [0.01, 0.01, 0.01],
            "pose": [100.0, 100.0, 100.0, 1, 0, 0, 0],
        }
    }
}


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
    default_joint_pos -- same pattern used by the other motion_planning steps.
    """
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()
    sim.step()
    scene.update(sim_dt)


class SliderTargetWindow:
    """
    Slider UI that only ever picks a target -- it never drives the robot.

    Copied unchanged from step5b_slider_no_live_move.py: there is no reference to the robot's
    write_*/set_*_target methods anywhere in this class. Reading the sliders (read_target_pos) is
    a pure UI read; the only things it feeds are the displayed label (update_label) and the
    confirmed-target snapshot taken on button click (_on_confirm). Neither touches the simulation.
    """

    def __init__(self, robot, ee_body_idx: int, env_id: int = 0):
        self.device = robot.data.root_state_w.device
        self.window = None
        self.sliders = {}
        self.target_label = None
        self.confirmed = False
        self.confirmed_target_pos = None

        # Default the sliders to the robot's actual current EE position in base frame.
        ee_state_w = robot.data.body_state_w[env_id, ee_body_idx]
        root_state_w = robot.data.root_state_w[env_id]
        start_pos_b, _ = math_utils.subtract_frame_transforms(
            root_state_w[0:3].unsqueeze(0), root_state_w[3:7].unsqueeze(0),
            ee_state_w[0:3].unsqueeze(0), ee_state_w[3:7].unsqueeze(0),
        )
        self.start_pos_b = start_pos_b.squeeze(0)

    # ------------------------------------------------------------------
    def build_ui(self):
        self.window = ui.Window("Pick Target (Slider + cuRobo Plan)", width=420, height=280)

        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label(
                    "Drag sliders to pick a target position. The robot will NOT move --"
                    " only the label below updates. Click 'Confirm Target' to plan to it.",
                    word_wrap=True,
                    height=48,
                )
                self.sliders["x"] = self._add_slider("targetPosX", *POS_X_RANGE, float(self.start_pos_b[0]))
                self.sliders["y"] = self._add_slider("targetPosY", *POS_Y_RANGE, float(self.start_pos_b[1]))
                self.sliders["z"] = self._add_slider("targetPosZ", *POS_Z_RANGE, float(self.start_pos_b[2]))
                self.target_label = ui.Label("Current target: (not yet read)", height=24)
                ui.Button("Confirm Target", height=32, clicked_fn=self._on_confirm)

    def _add_slider(self, label: str, lo: float, hi: float, default: float):
        with ui.HStack(height=24):
            ui.Label(label, width=120)
            model = ui.SimpleFloatModel(default)
            ui.FloatSlider(model=model, min=lo, max=hi, step=0.001)
        return model

    # ------------------------------------------------------------------
    def read_target_pos(self):
        """Pure read of the current slider values. No side effects on the robot."""
        return torch.tensor(
            [
                self.sliders["x"].get_value_as_float(),
                self.sliders["y"].get_value_as_float(),
                self.sliders["z"].get_value_as_float(),
            ],
            device=self.device,
        )

    def update_label(self):
        """UI-only: refreshes the displayed target string from the sliders. Never touches robot."""
        pos = self.read_target_pos()
        self.target_label.text = f"Current target: X={pos[0]:.3f}  Y={pos[1]:.3f}  Z={pos[2]:.3f}"

    def _on_confirm(self):
        """Button callback: snapshots the slider values and raises the confirmed flag.

        Still never touches the robot -- only the main loop, on seeing self.confirmed, decides
        what to do next.
        """
        self.confirmed_target_pos = self.read_target_pos()
        self.confirmed = True
        print(
            "[INFO] 'Confirm Target' pressed. Target snapshot: "
            f"{self.confirmed_target_pos.cpu().numpy().tolist()}"
        )

    def close(self):
        if self.window is not None:
            self.window.visible = False
            self.window = None


def run_test(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_idx = robot_entity_cfg.body_ids[0]
    sim_dt = sim.get_physics_dt()

    # ------------------------------------------------------------------
    # STEP 1: reset to default pose, save baseline, build + warm up cuRobo's planner.
    # ------------------------------------------------------------------
    reset_to_default_pose(sim, scene, robot)
    print("\n[STEP 1] Robot reset to default pose.")
    saved_state = scene.get_state(is_relative=False)
    saved_joint_pos = saved_state["articulation"]["robot"]["joint_position"][0, arm_joint_ids].clone()
    print(f"[STEP 1] Saved baseline joint_pos: {saved_joint_pos.cpu().numpy().tolist()}")

    tensor_args = TensorDeviceType(device=torch.device(sim.device))
    print("[INFO] Building cuRobo MotionGen (franka.yml, no obstacles within reach)...")
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        DUMMY_FAR_AWAY_WORLD,
        tensor_args=tensor_args,
        interpolation_dt=0.02,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("[INFO] Warming up cuRobo MotionGen (this takes a few seconds)...")
    motion_gen.warmup(enable_graph=True)
    print(f"[INFO] cuRobo MotionGen ready. Active joints: {motion_gen.kinematics.joint_names}")

    # ------------------------------------------------------------------
    # STEP 2/3: show sliders; while waiting for confirmation, update ONLY the label each step
    # and independently verify the robot hasn't drifted from the saved baseline.
    # ------------------------------------------------------------------
    window = SliderTargetWindow(robot, ee_body_idx=ee_body_idx)
    window.build_ui()
    print("\n[STEP 2] Sliders shown. Drag to pick a target -- the robot will not move.")
    print("[STEP 3] Verifying every physics step that the robot stays at the saved baseline...")

    while simulation_app.is_running() and not window.confirmed:
        window.update_label()
        sim.step()
        scene.update(sim_dt)

        current_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
        drift = (current_joint_pos - saved_joint_pos).abs().max().item()
        if drift > DRIFT_TOL_RAD:
            print(
                f"[ERROR] Robot drifted while waiting for confirmation! max abs diff = {drift:.6f} rad"
                f" (tolerance {DRIFT_TOL_RAD}). This should be impossible with no IK/target writes"
                " in the slider path -- investigate."
            )

    if not simulation_app.is_running():
        return

    # ------------------------------------------------------------------
    # STEP 4: confirmed -- print the target, RESTORE FIRST, THEN read the post-restore state
    # as cuRobo's planning start state, THEN plan. Restore must happen before planning.
    # ------------------------------------------------------------------
    target_pos_t = window.confirmed_target_pos.clone()
    print(f"\n[STEP 4] Final confirmed target pose (base frame): {target_pos_t.cpu().numpy().tolist()}")

    scene.reset_to(saved_state)
    sim.forward()  # kinematics-only update, no physics step -- same pattern as step5a/5b
    restored_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
    restore_diff = (restored_joint_pos - saved_joint_pos).abs().max().item()
    print(f"[STEP 4] Restored baseline. Post-restore joint_pos: {restored_joint_pos.cpu().numpy().tolist()}")
    print(f"[STEP 4] Max abs diff from saved baseline: {restore_diff:.6f} rad")

    # cuRobo's planning start state is read from the robot AFTER the restore above, not from any
    # value cached before the click -- the plan is computed from the restored baseline.
    arm_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
    start_state = JointState.from_position(
        arm_joint_pos.to(device=tensor_args.device, dtype=tensor_args.dtype),
        joint_names=motion_gen.kinematics.joint_names,
    )
    print(f"[STEP 4] cuRobo start state (post-restore, rad): {arm_joint_pos[0].tolist()}")

    target_quat_t = torch.tensor(DEFAULT_TARGET_QUAT, dtype=tensor_args.dtype, device=tensor_args.device)
    goal_pose = Pose(
        position=target_pos_t.to(dtype=tensor_args.dtype, device=tensor_args.device),
        quaternion=target_quat_t,
    )
    plan_config = MotionGenPlanConfig(max_attempts=10, enable_graph=True)
    print("[INFO] Planning...")
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    # ------------------------------------------------------------------
    # STEP 5: report and execute.
    # ------------------------------------------------------------------
    success = bool(result.success.item())
    if not success:
        print(f"[PLAN] FAILED. Status: {result.status}")
        print("[ERROR] No valid trajectory found for the confirmed target. Robot will remain at its restored pose.")
        window.close()
        print("[INFO] Leaving the viewer open. Close the window to exit.")
        while simulation_app.is_running():
            sim.render()
        return

    plan = result.interpolated_plan
    n_waypoints = len(plan.position)
    print(f"[PLAN] SUCCESS. Waypoints: {n_waypoints}, total planning time: {result.total_time:.3f}s")

    print(f"[EXEC] Starting trajectory execution ({n_waypoints} waypoints)...")
    gripper_targets = torch.full((scene.num_envs, len(gripper_joint_ids)), GRIPPER_OPEN, device=sim.device)
    for waypoint in plan.position:
        joint_pos_des = waypoint.unsqueeze(0).to(device=sim.device)
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    print("[EXEC] Trajectory execution complete.")

    # ------------------------------------------------------------------
    # STEP 6: verify the achieved pose against the requested target, with actual numbers.
    # ------------------------------------------------------------------
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
    ee_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    pos_error = torch.norm(ee_pos_b[0] - target_pos_t.to(sim.device)).item()
    print(f"[RESULT] Final EE position (base frame): {ee_pos_b[0].tolist()}, position error: {pos_error:.4f} m")

    window.close()
    print("\n[INFO] Test complete. Holding final pose. Close the viewer window to exit.")
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


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
        run_test(sim, scene)
    except Exception as e:
        print(f"[ERROR] Slider motion-plan test failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

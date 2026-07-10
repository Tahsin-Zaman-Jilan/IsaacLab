# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 6c: REPLAY a trajectory saved by step6a_save_trajectory.py, driving the
robot through the exact saved waypoints, WITHOUT calling cuRobo's planner again.

This is the actual "trajectory playback" milestone: step6a proved a successful plan can be
recorded to disk, step6b proved the recorded JSON reads back correctly. This script proves the
recorded data alone -- no MotionGen, no re-planning -- is enough to reproduce the robot's motion.

Setup is the same bare, ground-mounted robot scene as step6a (ground + light + single Franka, no
other objects, no sliders). Loading uses the same JSON format and file-discovery pattern already
proven in step6b (find_most_recent_trajectory / --file).

No cuRobo import anywhere in this file, so none of the multi-GPU device-threading workaround that
step2/step6a require applies here (that workaround exists specifically for cuRobo's internal
device handling -- see step2_single_target_motion_plan.py's docstring). This script follows the
same pattern as step1_bare_robot_teleop.py, the other non-cuRobo script in this series: it defaults
to --device cuda:1 (physical GPU1, the safe device on this workstation) and needs no
CUDA_VISIBLE_DEVICES env var at all.

.. code-block:: bash

    # replay the most recently saved trajectory
    ./isaaclab.sh -p scripts/motion_planning/step6c_replay_trajectory.py

    # replay a specific file
    ./isaaclab.sh -p scripts/motion_planning/step6c_replay_trajectory.py \
        --file scripts/motion_planning/trajectories/step6a_trajectory_20260710_161616.json

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 6c: replay a saved trajectory (no re-planning).")
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Path to a trajectory .json file (from step6a). Defaults to the most recently saved file "
    "in scripts/motion_planning/trajectories/.",
)
AppLauncher.add_app_launcher_args(parser)
# No cuRobo in this script -- physical GPU1 (cuda:1) directly, same as step1_bare_robot_teleop.py.
# The cuda:0-under-CUDA_VISIBLE_DEVICES=1 dance in step2/step6a is a cuRobo-specific workaround and
# does not apply here.
parser.set_defaults(device="cuda:1")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

GRIPPER_OPEN = 0.04

TRAJECTORY_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")


@configclass
class BareRobotSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane, light, single Franka. No other objects. Same as step6a."""

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


def find_most_recent_trajectory(directory: str) -> str:
    """Return the most recently modified .json trajectory file in directory. Same as step6b."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"[ERROR] No .json trajectory files found in {directory}. Run step6a_save_trajectory.py first.")
        sys.exit(1)
    return max(json_files, key=os.path.getmtime)


def run_test(sim: sim_utils.SimulationContext, scene: InteractiveScene, file_path: str) -> None:
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    # ------------------------------------------------------------------
    # STEP 1: load the saved trajectory (pure data load, same fields as step6b).
    # ------------------------------------------------------------------
    print(f"\n[STEP 1] Loading trajectory from: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    joint_names = data["joint_names"]
    position = data["position"]
    interpolation_dt = float(data["interpolation_dt"])
    n_waypoints = len(position)
    print(f"[STEP 1] joint_names: {joint_names}")
    print(f"[STEP 1] Number of waypoints: {n_waypoints}")
    print(f"[STEP 1] interpolation_dt: {interpolation_dt}")
    print(
        "[STEP 1] Saved velocity data present but unused -- this replay drives position targets "
        "only, same actuation pattern as every prior motion_planning step."
    )

    # Resolve joint indices by NAME rather than assuming any particular index order matches the
    # saved data -- this script has no MotionGen object to cross-check joint order against, unlike
    # step6a, so it must not assume cuRobo's arm-joint order lines up with any locally-resolved
    # joint_ids without checking.
    arm_joint_ids, resolved_names = robot.find_joints(joint_names)
    if resolved_names != joint_names:
        print(
            f"[ERROR] Resolved joint order {resolved_names} does not match saved joint_names "
            f"{joint_names} -- refusing to replay with a potentially wrong joint mapping."
        )
        sys.exit(1)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])

    position_t = torch.tensor(position, device=sim.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # STEP 2: reset to default pose, compare against the trajectory's first waypoint.
    # ------------------------------------------------------------------
    reset_to_default_pose(sim, scene, robot)
    sim_dt = sim.get_physics_dt()
    start_joint_pos = robot.data.joint_pos[0, arm_joint_ids].clone()
    first_waypoint = position_t[0]
    start_diff = (start_joint_pos - first_waypoint).abs().max().item()
    print(f"\n[STEP 2] Robot reset to default pose. Starting joint_pos: {start_joint_pos.cpu().numpy().tolist()}")
    print(f"[STEP 2] Trajectory's first waypoint:                     {first_waypoint.cpu().numpy().tolist()}")
    print(
        f"[STEP 2] Max abs diff (start vs. first waypoint): {start_diff:.6f} rad -- this is EXPECTED "
        "to be nonzero, not a bug: the original step6a run planned from wherever its own reset left "
        "the arm in that session, which need not match this session's reset state."
    )

    # ------------------------------------------------------------------
    # STEP 3: playback -- drive the robot through the saved waypoints in order, no re-planning.
    # ------------------------------------------------------------------
    steps_per_waypoint = max(1, round(interpolation_dt / sim_dt))
    print(
        f"\n[STEP 3] sim physics dt = {sim_dt}, trajectory interpolation_dt = {interpolation_dt} "
        f"-> holding each waypoint for {steps_per_waypoint} physics step(s)."
    )
    gripper_targets = torch.full((scene.num_envs, len(gripper_joint_ids)), GRIPPER_OPEN, device=sim.device)
    print(f"[STEP 3] Replaying {n_waypoints} waypoints (no MotionGen, no cuRobo import in this script)...")
    for waypoint in position_t:
        joint_pos_des = waypoint.unsqueeze(0)
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
        scene.write_data_to_sim()
        for _ in range(steps_per_waypoint):
            sim.step()
            scene.update(sim_dt)
    print("[STEP 3] Replay complete.")

    # ------------------------------------------------------------------
    # STEP 4: compare final achieved joint state against the trajectory's last waypoint.
    # ------------------------------------------------------------------
    final_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
    last_waypoint = position_t[-1]
    final_diff = (final_joint_pos - last_waypoint).abs().max().item()
    print(f"\n[STEP 4] Final achieved joint_pos: {final_joint_pos.cpu().numpy().tolist()}")
    print(f"[STEP 4] Trajectory's last waypoint: {last_waypoint.cpu().numpy().tolist()}")
    print(f"[STEP 4] Max abs diff (final vs. last waypoint): {final_diff:.6f} rad -- small means replay was faithful.")

    # ------------------------------------------------------------------
    # STEP 5: report achieved end-effector position (forward-kinematics-based), same style as
    # step6a's [RESULT] line.
    # ------------------------------------------------------------------
    robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    ee_body_idx = robot_entity_cfg.body_ids[0]
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
    ee_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    print(f"[RESULT] Final EE position (base frame): {ee_pos_b[0].tolist()}")
    print(f"[RESULT] Final EE position (world frame): {ee_pose_w[0, 0:3].tolist()}")

    print("\n[INFO] Test complete. Holding final pose. Close the viewer window to exit.")
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


def main() -> None:
    file_path = args_cli.file if args_cli.file is not None else find_most_recent_trajectory(TRAJECTORY_SAVE_DIR)
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        print(f"[ERROR] Trajectory file does not exist: {file_path}")
        simulation_app.close()
        sys.exit(1)

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
        run_test(sim, scene, file_path)
    except Exception as e:
        print(f"[ERROR] Trajectory replay test failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

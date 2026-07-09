# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 4: bare Franka robot mounted UPSIDE DOWN, single-target cuRobo reach.

Same minimal scene and cuRobo pattern as Step 2 (ground plane, one light, one Franka arm;
CUDA_VISIBLE_DEVICES=1 isolation guard; MotionGenConfig/plan_single/execute flow; explicit
[PLAN] SUCCESS/FAILED printing). The one new variable under test: the robot's root pose is
now rot=(0.0, 1.0, 0.0, 0.0) (180 degrees about local X, this project's established
upside-down convention) at an elevation measured fresh for this scene (see below) -- not
Step 2's identity pose at the origin.

cuRobo itself has ZERO built-in notion of the robot's world-frame root pose (confirmed by
inspecting cuda_robot_generator.py/robot.py: no root_pose/world_pose/gravity concept
anywhere in the kinematics-building code, and zero references to root_pos_w/root_quat_w in
cuRobo's own motion_gen.py -- those fields only appear in IsaacLab's separate wrapper code).
cuRobo plans entirely in the robot's own base_link frame, so the cuRobo target below is
reused VERBATIM from Step 2 (0.5, 0.0, 0.5) -- the same base-frame problem, unchanged by the
mount. The only new logic this script adds is converting between that base frame and the
world frame, using the same isaaclab.utils.math utilities Step 1's IK-relative teleop
already relies on (subtract_frame_transforms for world->base, combine_frame_transforms for
base->world) -- exercising the frame-conversion plumbing that Steps 2-3 never needed
(root_pos_w was (0,0,0) and root_quat_w was identity there, so world frame and base frame
coincided trivially).

Elevation (0.9m): NOT reused from hangongchen's vertical-lift branch (tied to a different,
box-based scene with an unresolved +/-0.13m measurement inconsistency -- see conversation).
Instead, measured fresh via cuRobo FK on this exact bare scene: the worst-case whole-body
collision-sphere extent along the arm's local +Z axis across the planned path (default pose
-> target) is 0.7699m -- this becomes the lowest point below the base once flipped 180 about
X. 0.9m gives ~0.13m margin. This script also measures the REAL live clearance at runtime
(from the actual simulated joint state, transformed through the actual root_pos_w/
root_quat_w) rather than trusting that offline number, and predicts the target's world-frame
landing point before planning, then compares it against the actual simulated result after
execution -- so the frame conversion is verified by comparison, not assumed.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step4_upside_down_reach.py \
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
        " scripts/motion_planning/step4_upside_down_reach.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 4: upside-down Franka mount, single-target cuRobo reach.")
AppLauncher.add_app_launcher_args(parser)
# under the CUDA_VISIBLE_DEVICES=1 isolation enforced above, physical GPU1 (cuda:1) is
# remapped to index 0 for this process -- so cuda:0 here is the correct, safe target
parser.set_defaults(device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

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
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from curobo.types.base import TensorDeviceType  # isort:skip
from curobo.types.math import Pose  # isort:skip
from curobo.types.state import JointState  # isort:skip
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # isort:skip

GRIPPER_OPEN = 0.04

# Upside-down mount: 180 degrees about local X, elevated 0.9m -- see module docstring for
# how 0.9 was measured (worst-case whole-body FK extent 0.7699m + ~0.13m margin).
ROBOT_ROOT_POS = (0.0, 0.0, 0.9)
ROBOT_ROOT_ROT = (0.0, 1.0, 0.0, 0.0)

# cuRobo target, reused VERBATIM from Step 2 -- same base-frame problem, unchanged by the
# mount, since cuRobo has no notion of the base's world placement.
TARGET_POS_BASE = [0.5, 0.0, 0.5]
TARGET_QUAT_BASE = [0.0, 1.0, 0.0, 0.0]

# satisfies cuRobo's "at least one registered primitive" requirement without constraining
# planning -- this step tests the mount/frame conversion, not ground collision (Step 3 did).
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
    """Minimal scene: ground plane, light, single Franka mounted upside down. No other objects."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=ROBOT_ROOT_POS, rot=ROBOT_ROOT_ROT),
    )


def reset_to_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot) -> None:
    """Explicitly move the robot to its configured default root pose AND joint pose.

    sim.reset() alone does not reliably apply either (verified in Step 2: the articulation
    was left at the raw USD spawn joint pose, not default_joint_pos, until explicitly
    written). Root pose is spawn-time USD placement and likely applies automatically, but
    given that exact lesson, this script writes both explicitly rather than assuming.
    """
    sim_dt = sim.get_physics_dt()
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    robot.write_root_state_to_sim(default_root_state)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()
    sim.step()
    scene.update(sim_dt)


def min_world_clearance(motion_gen: MotionGen, joint_pos: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> float:
    """Min ground clearance (world Z) of every collision sphere at this joint config.

    cuRobo's spheres are computed in the robot's base frame; this transforms each sphere
    center into world frame via the robot's actual root pose before comparing to the
    ground at world Z=0 -- the mount-orientation-aware version of Step 3's check.
    """
    spheres = motion_gen.kinematics.get_robot_as_spheres(joint_pos.unsqueeze(0))[0]
    min_z = float("inf")
    for s in spheres:
        pos_b = torch.tensor(s.position, dtype=root_pos_w.dtype, device=root_pos_w.device)
        pos_w = root_pos_w + quat_apply(root_quat_w, pos_b)
        min_z = min(min_z, pos_w[2].item() - s.radius)
    return min_z


def run_test(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Plan and execute a single-target motion on the upside-down mounted robot."""
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])

    reset_to_default_pose(sim, scene, robot)
    root_pos_w = robot.data.root_pos_w.clone()
    root_quat_w = robot.data.root_quat_w.clone()
    print(f"[INFO] Robot set to default pose. Root pos_w={root_pos_w[0].tolist()}, quat_w={root_quat_w[0].tolist()}")

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

    # live clearance check at the actual current (default) joint state -- measured, not assumed
    start_joint_pos = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].clone().to(tensor_args.device)
    start_clearance = min_world_clearance(motion_gen, start_joint_pos, root_pos_w[0].to(tensor_args.device), root_quat_w[0].to(tensor_args.device))
    print(f"[INFO] Live measured ground clearance at start pose: {start_clearance:.4f} m")
    if start_clearance < 0.0:
        print("[ERROR] Start pose already clips the ground -- the 0.9m elevation is insufficient in practice. Aborting.")
        while simulation_app.is_running():
            sim.render()
        return

    # predicted world-frame landing point for the base-frame target, BEFORE planning
    target_pos_b = torch.tensor(TARGET_POS_BASE, dtype=tensor_args.dtype, device=tensor_args.device)
    target_quat_b = torch.tensor(TARGET_QUAT_BASE, dtype=tensor_args.dtype, device=tensor_args.device)
    predicted_pos_w, predicted_quat_w = combine_frame_transforms(
        root_pos_w[0].to(tensor_args.device), root_quat_w[0].to(tensor_args.device), target_pos_b, target_quat_b
    )
    print(f"[TARGET] Base-frame target (reused from Step 2): pos={TARGET_POS_BASE}, quat={TARGET_QUAT_BASE}")
    print(f"[PREDICT] Expected world-frame landing point: pos={predicted_pos_w.tolist()}, quat={predicted_quat_w.tolist()}")

    start_state = JointState.from_position(
        start_joint_pos.unsqueeze(0), joint_names=motion_gen.kinematics.joint_names
    )
    goal_pose = Pose(position=target_pos_b, quaternion=target_quat_b)
    plan_config = MotionGenPlanConfig(max_attempts=10, enable_graph=True)
    print("[INFO] Planning...")
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    success = bool(result.success.item())
    if not success:
        print(f"[PLAN] FAILED. Status: {result.status}")
        print("[ERROR] No valid trajectory found. Robot will remain at its start pose.")
        print("[INFO] Leaving the viewer open. Close the window to exit.")
        while simulation_app.is_running():
            sim.render()
        return

    plan = result.interpolated_plan
    n_waypoints = len(plan.position)
    print(f"[PLAN] SUCCESS. Waypoints: {n_waypoints}, total planning time: {result.total_time:.3f}s")

    print(f"[INFO] Checking ground clearance across all {n_waypoints} waypoints (mount-aware)...")
    worst_clearance = float("inf")
    for q in plan.position:
        c = min_world_clearance(motion_gen, q, root_pos_w[0].to(tensor_args.device), root_quat_w[0].to(tensor_args.device))
        worst_clearance = min(worst_clearance, c)
    print(f"[RESULT] Minimum clearance above ground across full path: {worst_clearance:.4f} m")
    if worst_clearance < 0.0:
        print("[ERROR] A waypoint dips below the ground despite a successful plan -- investigate.")

    print(f"[EXEC] Starting trajectory execution ({n_waypoints} waypoints)...")
    sim_dt = sim.get_physics_dt()
    gripper_targets = torch.full((scene.num_envs, len(gripper_joint_ids)), GRIPPER_OPEN, device=sim.device)
    for waypoint in plan.position:
        joint_pos_des = waypoint.unsqueeze(0).to(device=sim.device)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    print("[EXEC] Trajectory execution complete.")

    # actual vs. predicted world-frame pose -- verifying the frame conversion, not assuming it
    root_pose_w_now = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    print(f"[RESULT] Actual world-frame EE pose: pos={ee_pose_w[0, 0:3].tolist()}, quat={ee_pose_w[0, 3:7].tolist()}")
    pos_error = torch.norm(ee_pose_w[0, 0:3] - predicted_pos_w).item()
    print(f"[RESULT] Predicted-vs-actual world-frame position error: {pos_error:.4f} m")

    # and the reverse direction: actual world pose converted back to base frame, vs. the
    # base-frame target we actually asked cuRobo for
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w_now[:, 0:3], root_pose_w_now[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    base_pos_error = torch.norm(ee_pos_b[0] - target_pos_b).item()
    print(f"[RESULT] Actual EE pose converted back to base frame: pos={ee_pos_b[0].tolist()}")
    print(f"[RESULT] Base-frame position error vs. requested cuRobo target: {base_pos_error:.4f} m")

    print("[INFO] Holding final pose. Close the viewer window to exit.")
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


def main() -> None:
    try:
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])

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
        print(f"[ERROR] Motion planning/execution failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

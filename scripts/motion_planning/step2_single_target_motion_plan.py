# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 2: bare Franka robot, single-target motion planning via cuRobo.

Same minimal scene as Step 1 (ground plane, one light, one Franka arm -- no cube, no
table, no obstacles). Instead of manual keyboard deltas, a single end-effector target
pose is handed to cuRobo's MotionGen planner, which solves for and executes the full
joint-space trajectory in one shot. This proves the motion-planning pipeline works in
isolation before any obstacles or complex scenes are introduced.

Uses cuRobo's native MotionGen API directly (not isaaclab_mimic's CuroboPlanner, which
is coupled to ManagerBasedEnv, downloads its robot URDF from Nucleus, and defaults to a
world with a collision table -- all inappropriate for this obstacle-free, offline test).
cuRobo ships its own local "franka.yml" config with a bundled URDF, used here with a
world containing a single dummy obstacle placed far outside the Franka's reach (cuRobo's
collision checker requires at least one registered primitive; a genuinely empty world
dict raises "Primitive Collision has no obstacles"). The dummy obstacle is functionally
irrelevant to planning -- this remains an obstacle-free test in every way that matters.

Requires CUDA_VISIBLE_DEVICES=1 (see the in-script guard below): cuRobo has an internal
bug where part of its rollout/cost config (used by BoundCost for retract_config) falls
back to a hardcoded default device ("cuda:0") instead of honoring the tensor_args device
we pass to MotionGenConfig -- confirmed by inspecting mg.rollout_fn.tensor_args.device
directly. Isolating the process to the one GPU we're allowed to use (cuda:1, remapped to
index 0) makes cuRobo's hardcoded default correct instead of chasing the gap inside
cuRobo's own source.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step2_single_target_motion_plan.py \
        --device cuda:0

    # test the failure path with an unreachable target
    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step2_single_target_motion_plan.py \
        --device cuda:0 --target-pos 5.0 5.0 5.0

"""

"""Launch Isaac Sim Simulator first."""

import os
import sys

# this workstation reserves physical GPU0 for Hangong's work -- cuRobo has an internal
# device-threading gap (see module docstring) that we work around via CUDA_VISIBLE_DEVICES
# isolation rather than by patching cuRobo internals. Check this before anything touches
# CUDA (Isaac Sim boot is ~20s; fail fast instead of discovering this mid-warmup).
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    print(
        "[ERROR] This script requires CUDA_VISIBLE_DEVICES=1 (lab policy: physical GPU0 is"
        " reserved for Hangong's work, and cuRobo's internal device handling only works"
        " reliably here under single-GPU process isolation). Run as:\n"
        "    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p"
        " scripts/motion_planning/step2_single_target_motion_plan.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 2: bare Franka robot, single-target cuRobo motion planning.")
parser.add_argument(
    "--target-pos",
    type=float,
    nargs=3,
    default=[0.5, 0.0, 0.5],
    metavar=("X", "Y", "Z"),
    help="Target end-effector position in the robot base frame (meters).",
)
parser.add_argument(
    "--target-quat",
    type=float,
    nargs=4,
    default=[0.0, 1.0, 0.0, 0.0],
    metavar=("QW", "QX", "QY", "QZ"),
    help="Target end-effector orientation in the robot base frame (w, x, y, z).",
)
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
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from curobo.types.base import TensorDeviceType  # isort:skip
from curobo.types.math import Pose  # isort:skip
from curobo.types.state import JointState  # isort:skip
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # isort:skip

GRIPPER_OPEN = 0.04

# cuRobo's collision checker requires at least one registered primitive obstacle -- a
# genuinely empty world dict raises "Primitive Collision has no obstacles". This dummy
# cuboid is placed 100m away, far outside the Franka's ~0.85m reach, so it never
# influences planning: functionally an obstacle-free world for this test.
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


def run_planner(sim: sim_utils.SimulationContext, scene: InteractiveScene, target_pos, target_quat) -> None:
    """Plan and execute a single-target motion, with explicit console confirmation throughout."""
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])

    # Explicitly move the robot to its configured default ("ready") pose before planning.
    # sim.reset() alone leaves the articulation at the raw USD spawn pose, not
    # default_joint_pos -- same pattern run_diff_ik.py uses on every reset.
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()
    sim.step()
    scene.update(sim_dt)
    print(f"[INFO] Robot set to default pose. Arm joints (rad): {default_joint_pos[0, robot_entity_cfg.joint_ids].tolist()}")

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

    # current arm joint positions as the planning start state
    arm_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
    start_state = JointState.from_position(
        arm_joint_pos.to(device=tensor_args.device, dtype=tensor_args.dtype),
        joint_names=motion_gen.kinematics.joint_names,
    )
    print(f"[INFO] Start joint state (rad): {arm_joint_pos[0].tolist()}")

    target_pos_t = torch.tensor(target_pos, dtype=tensor_args.dtype, device=tensor_args.device)
    target_quat_t = torch.tensor(target_quat, dtype=tensor_args.dtype, device=tensor_args.device)
    print(f"[TARGET] Requested EE goal (base frame): pos={target_pos}, quat={target_quat}")
    goal_pose = Pose(position=target_pos_t, quaternion=target_quat_t)

    plan_config = MotionGenPlanConfig(max_attempts=10, enable_graph=True)
    print("[INFO] Planning...")
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    success = bool(result.success.item())
    if not success:
        print(f"[PLAN] FAILED. Status: {result.status}")
        print("[ERROR] No valid trajectory found for the requested target. Robot will remain at its start pose.")
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
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    print("[EXEC] Trajectory execution complete.")

    # confirm the achieved pose against the requested target, not just the GUI
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    pos_error = torch.norm(ee_pos_b[0] - target_pos_t.to(sim.device)).item()
    print(f"[RESULT] Final EE position (base frame): {ee_pos_b[0].tolist()}, position error: {pos_error:.4f} m")

    print("[INFO] Holding final pose. Close the viewer window to exit.")
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
        run_planner(sim, scene, args_cli.target_pos, args_cli.target_quat)
    except Exception as e:
        print(f"[ERROR] Motion planning/execution failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

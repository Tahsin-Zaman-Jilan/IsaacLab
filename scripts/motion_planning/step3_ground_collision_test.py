# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 3: bare Franka robot, cuRobo motion planning against a REAL ground obstacle.

Same minimal scene and setup as Step 2 (ground plane, one light, one Franka arm; default-pose
reset fix; CUDA_VISIBLE_DEVICES=1 isolation guard) but this time the ground is registered as an
actual collision obstacle in cuRobo's world model, via cuRobo's own bundled "collision_table.yml"
(a 5x5x0.2m slab, top surface at z=0 -- matching IsaacLab's default GroundPlaneCfg position
exactly, so cuRobo's obstacle and Isaac Sim's real PhysX ground collider agree).

Two tests:

  TEST 1 (expect FAILURE): target pos=(0.5, 0.0, -0.1) is clearly inside/below the floor slab.
  Verified standalone beforehand: this fails with MotionGenStatus.IK_FAIL when the ground is
  registered, and the SAME exact pose succeeds in an obstacle-free world -- confirming the
  rejection is caused by the ground collision specifically, not mere unreachability.

  TEST 2 (expect SUCCESS, full-path clearance check): target pos=(0.6, 0.0, 0.2) is close to the
  floor (small margin) and reachable. Rather than trusting the single success/fail bit, this
  script walks EVERY interpolated waypoint of the resulting plan (not just the final pose) and
  computes the whole-body collision-sphere clearance above the ground at each one, reporting the
  minimum found across the entire path -- split into whole-body (including the fixed base plate,
  which never moves) and arm-only (excluding the base, which is the actually meaningful number
  since it changes as the arm moves). This demonstrates that cuRobo's trajectory optimizer
  enforces collision constraints across the full path, not just at the endpoints.

  Note: an attempt was made (see conversation) to also engineer a target where an uninstrumented
  path would dip through the floor mid-transit even though both endpoints are valid, to make the
  point via visible contrast. That could not be constructed for this single-obstacle (floor-only)
  scenario -- a floor only constrains the lower z bound, and Franka's redundant kinematics from
  its default pose don't naturally produce a dip-then-recover height excursion for reasonable
  nearby targets. Genuine forced detours need a blocking obstacle in front of the path, not merely
  a floor beneath. The full-waypoint clearance check above is the honest, still-rigorous
  substitute: it proves path-wide (not endpoint-only) checking regardless.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step3_ground_collision_test.py \
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
        " scripts/motion_planning/step3_ground_collision_test.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 3: bare Franka robot, cuRobo motion planning against real ground.")
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

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from curobo.types.base import TensorDeviceType  # isort:skip
from curobo.types.math import Pose  # isort:skip
from curobo.types.state import JointState  # isort:skip
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # isort:skip

GRIPPER_OPEN = 0.04

# panda_link0 (the fixed base plate) is always the first 2 spheres returned by
# get_robot_as_spheres(), verified standalone. Everything after that is the moving
# arm/hand/fingers -- the group whose clearance is actually meaningful evidence of
# path-wide avoidance, since the base spheres never move regardless of arm motion.
NUM_BASE_SPHERES = 2

# Test 1: clearly inside/below the real floor -- expect cuRobo to refuse this.
TEST1_TARGET_POS = [0.5, 0.0, -0.1]
TEST1_TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]

# Test 2: close to the floor but reachable -- expect success, then verify every
# waypoint (not just this final pose) keeps clear of the ground.
TEST2_TARGET_POS = [0.6, 0.0, 0.2]
TEST2_TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]


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
    default_joint_pos -- same pattern run_diff_ik.py uses on every reset.
    """
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()
    sim.step()
    scene.update(sim_dt)


def min_clearance_for_waypoints(motion_gen: MotionGen, joint_positions: torch.Tensor) -> tuple[float, float]:
    """Return (min_whole_body_clearance, min_arm_only_clearance) across all given waypoints.

    Clearance is each sphere's bottom (center z - radius) above the ground (z=0). Arm-only
    excludes the fixed base-link spheres, which never move regardless of arm motion.
    """
    min_whole_body = float("inf")
    min_arm_only = float("inf")
    for q in joint_positions:
        spheres = motion_gen.kinematics.get_robot_as_spheres(q.unsqueeze(0))[0]
        bottoms = [s.position[2] - s.radius for s in spheres]
        min_whole_body = min(min_whole_body, min(bottoms))
        min_arm_only = min(min_arm_only, min(bottoms[NUM_BASE_SPHERES:]))
    return min_whole_body, min_arm_only


def run_tests(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Run the below-ground failure test and the near-ground full-path clearance test."""
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])

    reset_to_default_pose(sim, scene, robot)
    print("[INFO] Robot set to default pose.")

    tensor_args = TensorDeviceType(device=torch.device(sim.device))

    print("[INFO] Building cuRobo MotionGen (franka.yml, REAL ground obstacle via collision_table.yml)...")
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args=tensor_args,
        interpolation_dt=0.02,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("[INFO] Warming up cuRobo MotionGen (this takes a few seconds)...")
    motion_gen.warmup(enable_graph=True)
    print(f"[INFO] cuRobo MotionGen ready. Active joints: {motion_gen.kinematics.joint_names}")

    def read_start_state() -> JointState:
        arm_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        return JointState.from_position(
            arm_joint_pos.to(device=tensor_args.device, dtype=tensor_args.dtype),
            joint_names=motion_gen.kinematics.joint_names,
        )

    # ---------------------------------------------------------------
    # TEST 1: target below the floor -- expect a collision-caused failure
    # ---------------------------------------------------------------
    print("\n[TEST 1] Target below the floor -- expecting cuRobo to refuse this plan.")
    print(f"[TARGET] Requested EE goal (base frame): pos={TEST1_TARGET_POS}, quat={TEST1_TARGET_QUAT}")
    start_state = read_start_state()
    goal_pose = Pose(
        position=torch.tensor(TEST1_TARGET_POS, dtype=tensor_args.dtype, device=tensor_args.device),
        quaternion=torch.tensor(TEST1_TARGET_QUAT, dtype=tensor_args.dtype, device=tensor_args.device),
    )
    plan_config = MotionGenPlanConfig(max_attempts=10, enable_graph=True)
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)
    success = bool(result.success.item())
    if success:
        print(
            "[ERROR] TEST 1 UNEXPECTEDLY SUCCEEDED. A target below the floor should have been"
            " rejected -- the ground obstacle is NOT being respected. Investigate before trusting"
            " any other result from this planner."
        )
    else:
        print(f"[PLAN] FAILED as expected. Status: {result.status}")
        print("[TEST 1] PASSED: the ground obstacle correctly blocked an unreachable-through-the-floor target.")

    # ---------------------------------------------------------------
    # TEST 2: target close to the floor -- expect success, then verify EVERY
    # waypoint (not just the final pose) keeps clear of the ground
    # ---------------------------------------------------------------
    reset_to_default_pose(sim, scene, robot)
    print("\n[TEST 2] Target close to the floor -- expecting success, then checking full-path clearance.")
    print(f"[TARGET] Requested EE goal (base frame): pos={TEST2_TARGET_POS}, quat={TEST2_TARGET_QUAT}")
    start_state = read_start_state()
    goal_pose = Pose(
        position=torch.tensor(TEST2_TARGET_POS, dtype=tensor_args.dtype, device=tensor_args.device),
        quaternion=torch.tensor(TEST2_TARGET_QUAT, dtype=tensor_args.dtype, device=tensor_args.device),
    )
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)
    success = bool(result.success.item())
    if not success:
        print(f"[PLAN] FAILED. Status: {result.status}")
        print("[ERROR] TEST 2 target could not be planned to. Nothing to execute; leaving viewer open.")
        while simulation_app.is_running():
            sim.render()
        return

    plan = result.interpolated_plan
    n_waypoints = len(plan.position)
    print(f"[PLAN] SUCCESS. Waypoints: {n_waypoints}, total planning time: {result.total_time:.3f}s")

    print(f"[INFO] Checking ground clearance across all {n_waypoints} waypoints (not just the final pose)...")
    min_whole_body, min_arm_only = min_clearance_for_waypoints(motion_gen, plan.position)
    print(f"[RESULT] Minimum whole-body clearance above ground across full path: {min_whole_body:.4f} m")
    print(f"[RESULT] Minimum arm-only clearance above ground across full path:   {min_arm_only:.4f} m")
    if min_whole_body < 0.0 or min_arm_only < 0.0:
        print("[ERROR] A waypoint dips below the ground despite a successful plan -- investigate.")
    else:
        print("[TEST 2] PASSED: every waypoint along the full path stayed clear of the ground, not just the endpoint.")

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
        run_tests(sim, scene)
    except Exception as e:
        print(f"[ERROR] Motion planning/execution failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

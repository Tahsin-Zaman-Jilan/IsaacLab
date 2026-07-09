# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 5a: save/restore correctness test, in isolation.

Same minimal scene as step1_bare_robot_teleop.py (ground plane, one light, one Franka arm --
no cube, no table, no sliders, no motion planning). This script proves IsaacLab's actual
save/restore mechanism works before it gets combined with anything else.

The mechanism under test is IsaacLab's OFFICIAL scene-state API -- not a hand-rolled
joint_pos/joint_vel clone -- confirmed by reading the local source
(source/isaaclab/isaaclab/scene/interactive_scene.py):

  - InteractiveScene.get_state(is_relative=False) returns a nested dict keyed by asset type/name,
    e.g. state["articulation"]["robot"] = {root_pose, root_velocity, joint_position, joint_velocity},
    all as CLONED tensors.
  - InteractiveScene.reset_to(state, env_ids=None, is_relative=False) writes root pose/velocity and
    joint state back via write_joint_state_to_sim(position, velocity), and -- critically -- also
    re-arms set_joint_position_target/set_joint_velocity_target from the restored values, so the PD
    controller doesn't immediately fight the restore on the next physics step. It then calls
    write_data_to_sim() internally.
  - ManagerBasedEnv.reset_to() (used for Isaac Lab Mimic demo replay) is a thin wrapper around
    exactly this: scene.reset_to(...) followed by sim.forward() -- a kinematics-only update (no
    physics step) so robot.data reflects the restored state immediately. This script mirrors that
    same sim.forward() call, since it uses InteractiveScene directly with no ManagerBasedEnv.

Flow:
  1. Reset to default pose, print joint state -- this is "state A" (the original).
  2. Save state A via scene.get_state().
  3. Manually write a clearly different joint pose directly into the sim (write_joint_state_to_sim
     + re-armed targets), step a few times, print the new state to confirm it actually moved.
  4. Restore the saved state via scene.reset_to() + sim.forward().
  5. Print the restored joint state and explicitly compare it to state A -- print
     [RESULT] MATCH or [RESULT] MISMATCH with the actual max-abs-diff numbers.

.. code-block:: bash

    ./isaaclab.sh -p scripts/motion_planning/step5a_save_restore_test.py --device cuda:1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 5a: save/restore correctness test, in isolation.")
AppLauncher.add_app_launcher_args(parser)
# this workstation reserves cuda:0 for other users -- default to cuda:1 unless overridden,
# same policy as step1_bare_robot_teleop.py
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

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

# how far (rad) to manually push each arm joint away from its default pose in step 3 -- large
# enough to be unmistakably different, small enough to stay within Franka's joint range after
# clamping to soft_joint_pos_limits below.
PERTURBATION_RAD = 0.5
SETTLE_STEPS = 30


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


def print_joint_state(label: str, robot, joint_ids) -> None:
    pos = robot.data.joint_pos[0, joint_ids]
    vel = robot.data.joint_vel[0, joint_ids]
    print(f"[{label}] joint_pos (rad): {pos.cpu().numpy().tolist()}")
    print(f"[{label}] joint_vel (rad/s): {vel.cpu().numpy().tolist()}")


def run_test(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot '{robot.cfg.prim_path}' spawned with {robot.num_joints} joints.")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    arm_joint_ids = robot_entity_cfg.joint_ids
    sim_dt = sim.get_physics_dt()

    # ------------------------------------------------------------------
    # STEP 1: reset to default pose, read + print state A (the original).
    # ------------------------------------------------------------------
    reset_to_default_pose(sim, scene, robot)
    print("\n[STEP 1] Robot reset to default pose.")
    print_joint_state("STATE A (original)", robot, arm_joint_ids)
    state_a_pos = robot.data.joint_pos[0, arm_joint_ids].clone()
    state_a_vel = robot.data.joint_vel[0, arm_joint_ids].clone()

    # ------------------------------------------------------------------
    # STEP 2: save state A via IsaacLab's official scene-state API.
    # ------------------------------------------------------------------
    saved_state = scene.get_state(is_relative=False)
    print("\n[STEP 2] Saved state via scene.get_state().")
    print(f"[STEP 2] Saved joint_position: {saved_state['articulation']['robot']['joint_position'][0].cpu().numpy().tolist()}")

    # ------------------------------------------------------------------
    # STEP 3: manually drive to a clearly different pose, step, confirm it moved.
    # ------------------------------------------------------------------
    soft_limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids]  # (num_arm_joints, 2)
    perturbed_pos = robot.data.joint_pos.clone()
    perturbed_pos[0, arm_joint_ids] = torch.clamp(
        state_a_pos + PERTURBATION_RAD, soft_limits[:, 0], soft_limits[:, 1]
    )
    perturbed_vel = torch.zeros_like(robot.data.joint_vel)

    robot.write_joint_state_to_sim(perturbed_pos, perturbed_vel)
    # re-arm the PD targets to match, exactly like scene.reset_to() does -- otherwise the
    # controller immediately starts dragging the arm back toward the stale target.
    robot.set_joint_position_target(perturbed_pos)
    robot.set_joint_velocity_target(perturbed_vel)
    robot.write_data_to_sim()
    for _ in range(SETTLE_STEPS):
        sim.step()
        scene.update(sim_dt)

    print(f"\n[STEP 3] Manually drove to a different pose and stepped {SETTLE_STEPS} times.")
    print_joint_state("STATE B (perturbed)", robot, arm_joint_ids)
    moved_diff = (robot.data.joint_pos[0, arm_joint_ids] - state_a_pos).abs().max().item()
    print(f"[STEP 3] Max abs diff from state A: {moved_diff:.4f} rad -- confirms the robot actually moved.")

    # ------------------------------------------------------------------
    # STEP 4: restore the saved state.
    # ------------------------------------------------------------------
    scene.reset_to(saved_state)
    sim.forward()  # kinematics-only update, no physics step -- matches ManagerBasedEnv.reset_to()
    print("\n[STEP 4] Restored saved state via scene.reset_to() + sim.forward().")

    # ------------------------------------------------------------------
    # STEP 5: print restored state, compare explicitly against state A.
    # ------------------------------------------------------------------
    print_joint_state("STATE C (restored)", robot, arm_joint_ids)
    restored_pos = robot.data.joint_pos[0, arm_joint_ids]
    restored_vel = robot.data.joint_vel[0, arm_joint_ids]
    pos_diff = (restored_pos - state_a_pos).abs()
    vel_diff = (restored_vel - state_a_vel).abs()
    max_pos_diff = pos_diff.max().item()
    max_vel_diff = vel_diff.max().item()

    print(f"\n[COMPARE] state A joint_pos: {state_a_pos.cpu().numpy().tolist()}")
    print(f"[COMPARE] state C joint_pos: {restored_pos.cpu().numpy().tolist()}")
    print(f"[COMPARE] max abs pos diff: {max_pos_diff:.6f} rad")
    print(f"[COMPARE] max abs vel diff: {max_vel_diff:.6f} rad/s")

    tol = 1e-4
    if max_pos_diff < tol and max_vel_diff < tol:
        print(f"\n[RESULT] MATCH -- restored state equals state A within tolerance ({tol}).")
    else:
        print(f"\n[RESULT] MISMATCH -- restored state differs from state A beyond tolerance ({tol}).")

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
        print(f"[ERROR] Save/restore test failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

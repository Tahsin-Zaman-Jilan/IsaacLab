# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 5b: slider target picking with NO live robot movement, combined with the
confirmed-working save/restore mechanism from step5a_save_restore_test.py.

Same bare, ground-mounted robot scene as step1/step5a (ground plane, one light, one Franka arm --
no cube, no table). This step combines two already-proven pieces:

  1. Save/restore: scene.get_state() / scene.reset_to() + sim.forward() -- exactly the pattern
     confirmed correct in step5a_save_restore_test.py.
  2. Slider UI: omni.ui.FloatSlider setup reused from step1_mouse_control_teleop.py.

Critical behavior change from step1_mouse_control_teleop.py: dragging the sliders here does NOT
drive the robot live. There is no IK controller in this script at all -- structurally, there is no
code path from a slider's value to any robot.write_*/set_*_target call. Dragging only updates a
displayed "current target" label. The only two places the robot is ever written to are
reset_to_default_pose() at the very start and scene.reset_to(saved_state) at the very end (via the
"Confirm Target" button). While waiting for that button, every physics step explicitly re-verifies
(against the saved state) that the robot's joint state has not drifted, and prints a loud [ERROR]
if it ever has -- rather than silently assuming the UI-only label update left it alone.

No cuRobo yet. This step only proves "slider picks a target without moving anything, and restore
still works correctly around it."

Controls:
  - Drag the X/Y/Z sliders in the "Pick Target (No Live Move)" window. The robot does not move;
    only the "Current target" label updates.
  - Click "Confirm Target" to lock in the displayed target, print it, and restore the robot to its
    saved (default) pose -- proving the restore still works cleanly even though nothing should have
    moved in the first place.

.. code-block:: bash

    ./isaaclab.sh -p scripts/motion_planning/step5b_slider_no_live_move.py --device cuda:1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 5b: slider target picking, no live robot movement.")
AppLauncher.add_app_launcher_args(parser)
# this workstation reserves cuda:0 for other users -- default to cuda:1 unless overridden,
# same policy as step1_bare_robot_teleop.py
parser.set_defaults(device="cuda:1")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni.ui as ui

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

# Slider ranges mirror step1_mouse_control_teleop.py's (in turn mirroring the original PyBullet
# mouse_control() tool's addUserDebugParameter bounds).
POS_X_RANGE = (0.0, 1.5)
POS_Y_RANGE = (-1.0, 1.0)
POS_Z_RANGE = (0.0, 1.15)

# Drift tolerance for the "robot hasn't moved while waiting for confirmation" check -- loose
# enough to absorb PD/physics settling jitter at rest, tight enough to catch a real bug.
DRIFT_TOL_RAD = 1e-3


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

    There is no reference to the robot's write_*/set_*_target methods anywhere in this class.
    Reading the sliders (read_target_pos) is a pure UI read; the only things it feeds are the
    displayed label (update_label) and the confirmed-target snapshot taken on button click
    (_on_confirm). Neither touches the simulation.
    """

    def __init__(self, robot, ee_body_idx: int, env_id: int = 0):
        self.device = robot.data.root_state_w.device
        self.window = None
        self.sliders = {}
        self.target_label = None
        self.confirmed = False
        self.confirmed_target_pos = None

        # Default the sliders to the robot's actual current EE position in base frame, same
        # nicety as step1_mouse_control_teleop.py's initial slider values.
        ee_state_w = robot.data.body_state_w[env_id, ee_body_idx]
        root_state_w = robot.data.root_state_w[env_id]
        start_pos_b, _ = math_utils.subtract_frame_transforms(
            root_state_w[0:3].unsqueeze(0), root_state_w[3:7].unsqueeze(0),
            ee_state_w[0:3].unsqueeze(0), ee_state_w[3:7].unsqueeze(0),
        )
        self.start_pos_b = start_pos_b.squeeze(0)

    # ------------------------------------------------------------------
    def build_ui(self):
        self.window = ui.Window("Pick Target (No Live Move)", width=420, height=280)

        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label(
                    "Drag sliders to pick a target position. The robot will NOT move --"
                    " only the label below updates. Click 'Confirm Target' to lock it in.",
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
    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_idx = robot_entity_cfg.body_ids[0]
    sim_dt = sim.get_physics_dt()

    # ------------------------------------------------------------------
    # STEP 1: reset to default pose, save it -- this is the baseline to protect.
    # ------------------------------------------------------------------
    reset_to_default_pose(sim, scene, robot)
    print("\n[STEP 1] Robot reset to default pose.")
    saved_state = scene.get_state(is_relative=False)
    saved_joint_pos = saved_state["articulation"]["robot"]["joint_position"][0, arm_joint_ids].clone()
    print(f"[STEP 1] Saved baseline joint_pos: {saved_joint_pos.cpu().numpy().tolist()}")

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
    # STEP 4: confirmed -- print the final target, restore the saved state, verify the restore.
    # ------------------------------------------------------------------
    print(f"\n[STEP 4] Final confirmed target pose (base frame): {window.confirmed_target_pos.cpu().numpy().tolist()}")
    scene.reset_to(saved_state)
    sim.forward()  # kinematics-only update, no physics step -- same pattern as step5a

    restored_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
    restore_diff = (restored_joint_pos - saved_joint_pos).abs().max().item()
    print(f"[STEP 4] Post-restore joint_pos: {restored_joint_pos.cpu().numpy().tolist()}")
    print(f"[STEP 4] Max abs diff from saved baseline: {restore_diff:.6f} rad")

    print("\n[RESULT] target confirmed and state restored.")

    print("\n[INFO] Test complete. Holding final pose. Close the viewer window to exit.")
    window.close()
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
        print(f"[ERROR] Slider-no-live-move test failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

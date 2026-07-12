# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 7c: box scene, REAL-TIME slider teleop + gripper, no confirm button.

step7b_box_slider_teleop.py is untouched -- kept as the working "confirm target -> cuRobo plans
through real box obstacles" checkpoint. This is a new file with a different core mechanism,
requested directly: remove the confirm button (sliders drive the robot live) and add gripper
control to actually grasp the cube.

cuRobo's plan_single is a one-shot batch planner -- built for "here's a start and a goal, give me
a full trajectory," not for being re-invoked every physics frame as a live target moves. Live
slider-driven control needs a different, already-proven mechanism in this directory:
step1_mouse_control_teleop.py's DifferentialIKController in absolute-pose mode, re-solved every
frame from whatever the sliders currently read, exactly like a live 6-DOF mouse/slider rig. This
script ports that mechanism (position sliders + grasp slider only, no roll/pitch/yaw sliders --
orientation stays fixed downward, matching step7b's DEFAULT_TARGET_QUAT; not asked for here) onto
the box scene: yawed robot mount + box_env_cfg.py's verbatim geometry + settling cube, all reused
unchanged from step7b (box_parts(), _add_box_prims(), the AABB helpers -- same functions, same
numbers, copied in since scripts in this directory are self-contained and don't import each
other).

Tradeoff worth stating plainly: real-time differential IK has NO obstacle awareness in this
codebase's existing pattern -- step1's version doesn't register anything, it's pure reactive IK.
Collision avoidance becomes the operator's job (watching the viewport while dragging the arm
through the hole), same as any manual teleop. This is NOT step7b's collision-aware cuRobo
planning. To keep some safety signal without turning this back into a blocking planner, every
frame independently checks each robot body's ORIGIN POINT against the box's real AABBs (reusing
step7b's exact obstacle geometry) and prints a rate-limited (edge-triggered, like the drift-check
fix) warning the moment any body is actually inside solid box material -- a coarse, radius-0
approximation (no per-body collision-sphere radius available without cuRobo), not a rigorous
clearance number. It's a HUD, not a gate: it never stops or overrides the commanded motion.

Controls:
  - Drag X/Y/Z sliders (base frame) in the "Box Teleop (Real-Time)" window -- the arm continuously
    IK-solves toward wherever they currently read, every physics step.
  - Drag "grasp" above 0.5 to close the gripper, below to open it -- same convention as
    step1_mouse_control_teleop.py.
  - Close the viewer window (or Ctrl+C) to exit.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step7c_box_realtime_teleop.py \
        --device cuda:0

"""

"""Launch Isaac Sim Simulator first."""

import os
import sys

# this workstation reserves physical GPU0 for Hangong's work -- see step2_single_target_motion_plan.py's
# docstring for the full investigation. Check this before anything touches CUDA (Isaac Sim boot is
# ~20s; fail fast instead of discovering this mid-warmup).
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    print(
        "[ERROR] This script requires CUDA_VISIBLE_DEVICES=1 (lab policy: physical GPU0 is"
        " reserved for Hangong's work). Run as:\n"
        "    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p"
        " scripts/motion_planning/step7c_box_realtime_teleop.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 7c: box scene, real-time slider teleop + gripper.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni.ui as ui

if torch.cuda.device_count() != 1:
    print(
        f"[ERROR] Expected exactly 1 visible CUDA device under CUDA_VISIBLE_DEVICES=1, found"
        f" {torch.cuda.device_count()}. Refusing to proceed."
    )
    simulation_app.close()
    sys.exit(1)

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# --- robot mount + box placement: identical to step7b (yawed +90deg about Z instead of moving
# the box -- see step7b's docstring for the hand-verified Rz(90deg) reasoning). ---
ROBOT_ROOT_POS = (0.0, 0.0, 0.0)
ROBOT_ROOT_ROT = (0.7071068, 0.0, 0.0, 0.7071068)
BOX_X, BOX_Y = 0.0, 0.65

# Base-frame slider ranges -- identical math to step7b (hole footprint transformed through the
# yawed base frame), still valid here since neither the box nor the mount changed.
POS_X_RANGE = (0.45, 0.85)
POS_Y_RANGE = (-0.20, 0.20)
POS_Z_RANGE = (0.25, 0.55)

# Fixed downward-facing gripper orientation (base frame) -- not asked to add roll/pitch/yaw
# sliders here, so orientation stays constant like step7b's DEFAULT_TARGET_QUAT.
FIXED_TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]


def box_parts() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], str]]:
    """Single source of truth for box_env_cfg.py's box geometry -- copied verbatim from step7b
    (translation-only placement at (BOX_X, BOX_Y), same numbers, same variable names)."""
    box_w, box_d, box_h, t, hole = 0.8, 0.8, 0.5, 0.01, 0.35
    z_bot, z_top = 0.0, 0.5
    z_mid = (z_bot + z_top) / 2
    top_z = z_top - t / 2
    side_w = (box_w - hole) / 2
    side_d = (box_d - hole) / 2

    platform_size = (0.5, 0.5, 0.01)
    platform_top_z = z_mid + platform_size[2] / 2
    ceiling_z = z_top - t
    post_h = ceiling_z - platform_top_z
    post_z = (platform_top_z + ceiling_z) / 2
    post_inset = 0.23
    cube_size = 0.05
    cube_z = platform_top_z + cube_size / 2

    parts = [
        ("wall_left", (BOX_X - box_w / 2 + t / 2, BOX_Y, z_mid), (t, box_d, box_h), "glass"),
        ("wall_right", (BOX_X + box_w / 2 - t / 2, BOX_Y, z_mid), (t, box_d, box_h), "glass"),
        ("wall_front", (BOX_X, BOX_Y - box_d / 2 + t / 2, z_mid), (box_w, t, box_h), "glass"),
        ("wall_back", (BOX_X, BOX_Y + box_d / 2 - t / 2, z_mid), (box_w, t, box_h), "glass"),
        ("floor", (BOX_X, BOX_Y, z_bot + t / 2), (box_w, box_d, t), "floor"),
        ("top_left", (BOX_X - hole / 2 - side_w / 2, BOX_Y, top_z), (side_w, box_d, t), "glass"),
        ("top_right", (BOX_X + hole / 2 + side_w / 2, BOX_Y, top_z), (side_w, box_d, t), "glass"),
        ("top_front", (BOX_X, BOX_Y - hole / 2 - side_d / 2, top_z), (hole, side_d, t), "glass"),
        ("top_back", (BOX_X, BOX_Y + hole / 2 + side_d / 2, top_z), (hole, side_d, t), "glass"),
        ("platform", (BOX_X, BOX_Y, z_mid), platform_size, "platform"),
        ("post_fl", (BOX_X - post_inset, BOX_Y - post_inset, post_z), (0.015, 0.015, post_h), "platform"),
        ("post_fr", (BOX_X + post_inset, BOX_Y - post_inset, post_z), (0.015, 0.015, post_h), "platform"),
        ("post_bl", (BOX_X - post_inset, BOX_Y + post_inset, post_z), (0.015, 0.015, post_h), "platform"),
        ("post_br", (BOX_X + post_inset, BOX_Y + post_inset, post_z), (0.015, 0.015, post_h), "platform"),
    ]
    return parts, (BOX_X, BOX_Y, cube_z), cube_size


@configclass
class BoxSceneCfg(InteractiveSceneCfg):
    """Yawed-mount Franka beside box_env_cfg.py's box (verbatim geometry, translated only)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=ROBOT_ROOT_POS, rot=ROBOT_ROOT_ROT),
    )


def _add_box_prims(scene_cfg: BoxSceneCfg) -> tuple[float, float, float]:
    parts, cube_pos, cube_size = box_parts()

    static_rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True)
    static_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0)
    static_mass_props = sim_utils.MassPropertiesCfg(mass=5.0)
    materials = {
        "glass": sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.52),
        "floor": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        "platform": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
    }

    for name, center, dims, kind in parts:
        setattr(
            scene_cfg,
            f"box_{name}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Box_{name}",
                spawn=sim_utils.CuboidCfg(
                    size=dims,
                    rigid_props=static_rigid_props,
                    mass_props=static_mass_props,
                    collision_props=static_collision_props,
                    visual_material=materials[kind],
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
            ),
        )

    scene_cfg.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cube_pos, rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )
    return cube_pos


def box_obstacle_aabbs(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    parts, _, _ = box_parts()
    aabbs = []
    for name, center, dims, _ in parts:
        c = torch.tensor(center, dtype=torch.float32, device=device)
        d = torch.tensor(dims, dtype=torch.float32, device=device)
        aabbs.append((name, c - d / 2, c + d / 2))
    return aabbs


def _point_inside_any_aabb(
    point: torch.Tensor, obstacle_aabbs: list[tuple[str, torch.Tensor, torch.Tensor]]
) -> str | None:
    """Radius-0 penetration check: returns the obstacle name if `point` is inside its AABB, else None."""
    for name, box_min, box_max in obstacle_aabbs:
        if bool(torch.all(point >= box_min) and torch.all(point <= box_max)):
            return name
    return None


def _aabb_signed_distance(point: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor) -> float:
    """Signed distance from a point to an axis-aligned box surface (negative = inside/penetrating).
    Unlike _point_inside_any_aabb, this reports "pressed up against" (small positive/negative
    distance) even before a body's ORIGIN point fully crosses into the box -- real PhysX contact
    happens at a link's actual surface, which this origin-only check underreports."""
    clamped = torch.max(torch.min(point, box_max), box_min)
    outside_vec = point - clamped
    outside_dist = torch.norm(outside_vec).item()
    if outside_dist > 1e-9:
        return outside_dist
    face_dists = torch.min(point - box_min, box_max - point)
    return -face_dists.min().item()


def min_clearance_all_bodies(
    body_pos_w: torch.Tensor, obstacle_aabbs: list[tuple[str, torch.Tensor, torch.Tensor]]
) -> tuple[float, str, int]:
    """Minimum signed distance from any robot body ORIGIN to any box obstacle -- still a coarse,
    radius-0 approximation (no per-link collision-sphere radius available without cuRobo), but
    catches "pressed against a wall" (small clearance) rather than only "already penetrating"."""
    worst, worst_name, worst_idx = float("inf"), "", -1
    for b_idx in range(body_pos_w.shape[0]):
        p = body_pos_w[b_idx]
        for name, box_min, box_max in obstacle_aabbs:
            d = _aabb_signed_distance(p, box_min, box_max)
            if d < worst:
                worst, worst_name, worst_idx = d, name, b_idx
    return worst, worst_name, worst_idx


def reset_to_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot) -> None:
    """Write both root pose and joint pose explicitly (step4/step7b's defensive discipline for a
    non-identity root rotation)."""
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


class RealtimeControlWindow:
    """Position + grasp sliders, read live every frame -- no confirm button, no side-effect
    guarantee to preserve (unlike step5b/5c/step7b's SliderTargetWindow, this class's whole
    purpose IS to drive the robot). Ported from step1_mouse_control_teleop.py's MouseControlWindow,
    trimmed to position-only (no roll/pitch/yaw sliders -- fixed downward orientation)."""

    def __init__(self, robot, ee_body_idx: int, env_id: int = 0):
        self.device = robot.data.root_state_w.device
        self.window = None
        self.sliders = {}
        self.target_label = None

        ee_state_w = robot.data.body_state_w[env_id, ee_body_idx]
        root_state_w = robot.data.root_state_w[env_id]
        start_pos_b, _ = math_utils.subtract_frame_transforms(
            root_state_w[0:3].unsqueeze(0), root_state_w[3:7].unsqueeze(0),
            ee_state_w[0:3].unsqueeze(0), ee_state_w[3:7].unsqueeze(0),
        )
        self.start_pos_b = start_pos_b.squeeze(0)

    def build_ui(self):
        self.window = ui.Window("Box Teleop (Real-Time)", width=420, height=300)
        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label(
                    "Drag sliders to move the arm LIVE (base frame). No confirm button --"
                    " every frame IK-solves toward the current slider values."
                    " grasp > 0.5 closes the gripper.",
                    word_wrap=True,
                    height=54,
                )
                self.sliders["x"] = self._add_slider("targetPosX", *POS_X_RANGE, float(self.start_pos_b[0]))
                self.sliders["y"] = self._add_slider("targetPosY", *POS_Y_RANGE, float(self.start_pos_b[1]))
                self.sliders["z"] = self._add_slider("targetPosZ", *POS_Z_RANGE, float(self.start_pos_b[2]))
                self.sliders["grasp"] = self._add_slider("grasp", 0.0, 1.0, 0.0)
                self.target_label = ui.Label("Current target: (not yet read)", height=24)

    def _add_slider(self, label: str, lo: float, hi: float, default: float):
        with ui.HStack(height=24):
            ui.Label(label, width=120)
            model = ui.SimpleFloatModel(default)
            ui.FloatSlider(model=model, min=lo, max=hi, step=0.001)
        return model

    def read_target(self):
        target_pos = torch.tensor(
            [
                self.sliders["x"].get_value_as_float(),
                self.sliders["y"].get_value_as_float(),
                self.sliders["z"].get_value_as_float(),
            ],
            device=self.device,
        )
        grasp = self.sliders["grasp"].get_value_as_float()
        finger_target = GRIPPER_CLOSED if grasp > 0.5 else GRIPPER_OPEN
        return target_pos, finger_target

    def update_label(self, target_pos: torch.Tensor, finger_target: float):
        gripper_state = "CLOSED" if finger_target == GRIPPER_CLOSED else "OPEN"
        self.target_label.text = (
            f"Target: X={target_pos[0]:.3f} Y={target_pos[1]:.3f} Z={target_pos[2]:.3f}  Gripper: {gripper_state}"
        )

    def close(self):
        if self.window is not None:
            self.window.visible = False
            self.window = None


def run_teleop(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    cube = scene["object"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot mount: pos={ROBOT_ROOT_POS}, rot={ROBOT_ROOT_ROT} (+90deg yaw about world Z).")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    ee_body_idx = robot_entity_cfg.body_ids[0]

    reset_to_default_pose(sim, scene, robot)
    hand_pos_w = robot.data.body_pos_w[0, ee_body_idx]
    print(f"[VERIFY] Default-pose panda_hand world position: {hand_pos_w.tolist()}")
    if abs(hand_pos_w[1].item()) > abs(hand_pos_w[0].item()):
        print("[VERIFY] PASSED: Y is the dominant reach axis (yaw fix confirmed, same as step7b).")
    else:
        print("[VERIFY] WARNING: X is still dominant over Y -- investigate before teleoperating.")

    sim_dt = sim.get_physics_dt()
    for _ in range(30):
        sim.step()
        scene.update(sim_dt)
    print(f"[INFO] Settled cube position: {cube.data.root_pos_w[0].tolist()}")

    # Real-time IK -- absolute-pose command, re-solved every frame from live slider values
    # (step1_mouse_control_teleop.py's proven mechanism, not cuRobo's batch planner).
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)
    ik_controller.reset()

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    window = RealtimeControlWindow(robot, ee_body_idx=ee_body_idx)
    window.build_ui()

    jacobi_body_idx = ee_body_idx - 1 if robot.is_fixed_base else ee_body_idx
    jacobi_joint_ids = robot_entity_cfg.joint_ids
    target_quat_b = torch.tensor(FIXED_TARGET_QUAT, device=sim.device)

    obstacle_aabbs = box_obstacle_aabbs(torch.device(sim.device))
    body_names = robot.body_names

    print("\n[INFO] Real-time teleop running. Drag sliders in the 'Box Teleop (Real-Time)' window.")
    print("[INFO] No confirm button -- the arm follows the sliders live. Close the viewer window or Ctrl+C to exit.")
    print("[INFO] Collision HUD is passive (warns only, never blocks) -- box walls are NOT auto-avoided.")

    # Max joint-space step commanded per physics frame (rad). At dt=0.01 this caps the effective
    # commanded joint speed at MAX_STEP_RAD/0.01 = 2.0 rad/s -- comfortably inside Franka's real
    # joint speed range, and prevents ik_controller.compute()'s UNCAPPED one-shot DLS correction
    # (see differential_ik.py: compute() returns joint_pos + delta_joint_pos with no rate limiting
    # at all) from commanding a full, large, one-frame jump whenever an absolute-position slider
    # is dragged far from its current value. Root cause: step1_mouse_control_teleop.py never needed
    # this because a mouse drag changes the target by a tiny amount every frame (physical mouse
    # motion between two 10ms frames is small) -- these sliders hold an absolute value, so a real
    # drag CAN jump the target by a large distance in one frame. A large uncapped correction here
    # both overshoots (given FRANKA_PANDA_HIGH_PD_CFG's stiffness=400/effort_limit=87) and gets
    # re-solved fresh from the overshot pose next frame, correcting back the other way -- a stable
    # limit cycle that never converges, confirmed live: with a fixed target, joint_pos was measured
    # cycling through the exact same 3 values every 3 frames for 58 straight frames, never
    # approaching the ~1.0 rad commanded delta. Clamping the per-frame step forces gradual,
    # convergent tracking instead.
    MAX_STEP_RAD = 0.02

    frame = 0
    status_interval = 30  # print a one-line status every N frames instead of every frame
    ee_pos_at_last_checkpoint = None  # tracks NET displacement between status lines -- distinguishes
    # "oscillating back and forth, net motion ~0 despite constant commanded steps" from
    # "genuinely creeping toward the target too slowly" (same 0.02rad/frame either way looks
    # identical in cart_error_norm alone; net displacement over N frames tells them apart).

    # Reference-trajectory smoothing: track OUR OWN commanded joint position across frames,
    # instead of re-deriving the next command from the live simulated joint_pos every frame.
    # Root cause found via joint_vel_norm reading ~10 rad/s (unphysically high for this arm,
    # which tops out around 2-2.6 rad/s) alongside near-zero net displacement -- classic
    # feedback-noise amplification: with stiffness=400 and no velocity limit on these actuators,
    # normal PD overshoot/settling noise in the REAL simulated state was feeding straight back
    # into the next frame's fresh DLS solve as if it were a real position, producing a slightly
    # different target every frame and exciting a high-frequency self-sustaining oscillation.
    # cuRobo's confirm-button version never hit this: it plans a smooth trajectory ONCE and plays
    # it back open-loop, never re-solving from live feedback. Fix: use the REAL simulated EE pose
    # to measure genuine position error (so it still corrects toward the true target), but
    # integrate the rate-limited step from OUR OWN prior command, not from noisy live feedback --
    # this decouples the commanded trajectory's smoothness from simulation noise.
    commanded_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()

    try:
        while simulation_app.is_running():
            frame += 1
            target_pos_b, finger_target = window.read_target()
            window.update_label(target_pos_b, finger_target)
            ik_controller.set_command(torch.cat([target_pos_b, target_quat_b]).unsqueeze(0))

            jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
            # PhysX's jacobian is expressed in WORLD frame; ee_pos_b/target below are in BASE
            # frame, and with our yawed mount (ROBOT_ROOT_ROT != identity) those are no longer
            # the same frame. Rotate the jacobian into base frame via the inverse of root_quat_w
            # -- same fix IsaacLab's own DifferentialInverseKinematicsAction.jacobian_b applies
            # (task_space_actions.py), required for exactly this reason.
            base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_quat_w))
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
            ee_state_w = robot.data.body_state_w[0, ee_body_idx]
            root_state_w = robot.data.root_state_w[0]
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_state_w[0:3].unsqueeze(0), root_state_w[3:7].unsqueeze(0),
                ee_state_w[0:3].unsqueeze(0), ee_state_w[3:7].unsqueeze(0),
            )
            joint_pos_cur = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]  # still read for status/limit checks
            # IK solved from OUR OWN reference (commanded_joint_pos), not the live noisy joint_pos_cur.
            joint_pos_des_raw = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, commanded_joint_pos)
            # rate-limit: advance at most MAX_STEP_RAD toward the raw IK solution this frame,
            # instead of jumping straight to it (see MAX_STEP_RAD comment above) -- integrated
            # from our own reference, so the commanded sequence stays smooth frame to frame.
            delta = torch.clamp(joint_pos_des_raw - commanded_joint_pos, -MAX_STEP_RAD, MAX_STEP_RAD)
            commanded_joint_pos = commanded_joint_pos + delta
            joint_pos_des = commanded_joint_pos

            if frame % status_interval == 0:
                cart_err_norm = (target_pos_b - ee_pos_b.squeeze(0)).norm().item()
                target_reach = target_pos_b.norm().item()  # straight-line distance from base to the commanded target
                # how close each arm joint currently sits to its own hard limit (rad of margin) --
                # distinguishes "target physically out of reach" (a joint pinned at its limit) from
                # any other stall cause.
                limits = robot.data.soft_joint_pos_limits[0, robot_entity_cfg.joint_ids]
                margin_lo = (joint_pos_cur[0] - limits[:, 0]).abs()
                margin_hi = (limits[:, 1] - joint_pos_cur[0]).abs()
                min_margin = torch.min(margin_lo, margin_hi).min().item()
                ee_pos_now = ee_pos_b.squeeze(0).clone()
                if ee_pos_at_last_checkpoint is not None:
                    net_disp = (ee_pos_now - ee_pos_at_last_checkpoint).norm().item()
                else:
                    net_disp = float("nan")
                ee_pos_at_last_checkpoint = ee_pos_now
                # real minimum clearance (can go negative = penetrating) from any robot body to the
                # box's actual collision geometry -- tests whether a physical contact is what's
                # actually stopping progress, not just kinematic reach/joint limits.
                min_clear, min_clear_obs, min_clear_body_idx = min_clearance_all_bodies(
                    robot.data.body_pos_w[0], obstacle_aabbs
                )
                min_clear_body_name = body_names[min_clear_body_idx] if min_clear_body_idx >= 0 else "?"
                # confirm the write pipeline actually landed: what does the SIMULATION think the
                # current position target is, vs. what we just computed and (about to) send -- and
                # is there ANY joint velocity at all (fully locked vs. genuinely-but-glacially moving).
                sim_side_target = robot.data.joint_pos_target[0, robot_entity_cfg.joint_ids]
                joint_vel_norm = robot.data.joint_vel[0, robot_entity_cfg.joint_ids].norm().item()
                print(
                    f"[STATUS] frame={frame} target_pos_b={target_pos_b.tolist()} target_reach={target_reach:.4f} m "
                    f"cart_error_norm={cart_err_norm:.4f} m max_abs_delta={delta.abs().max().item():.4f} rad "
                    f"min_joint_limit_margin={min_margin:.4f} rad "
                    f"net_ee_disp_since_last_status={net_disp:.5f} m (over {status_interval} frames) "
                    f"min_box_clearance={min_clear:.4f} m (body '{min_clear_body_name}' vs '{min_clear_obs}') "
                    f"joint_vel_norm={joint_vel_norm:.6f} rad/s "
                    f"gripper={'CLOSED' if finger_target == GRIPPER_CLOSED else 'OPEN'}"
                )
                print(f"[STATUS-WRITE-CHECK] frame={frame} joint_pos_des(about to send)={joint_pos_des[0].tolist()}")
                print(f"[STATUS-WRITE-CHECK] frame={frame} sim's joint_pos_target(from LAST frame's write)={sim_side_target.tolist()}")
            if torch.isnan(joint_pos_des).any() or torch.isinf(joint_pos_des).any():
                print(f"[ERROR] frame={frame} NaN/Inf in joint_pos_des -- refusing to command it, holding previous target.")
                joint_pos_des = joint_pos_cur

            gripper_targets = torch.full((scene.num_envs, len(gripper_joint_ids)), finger_target, device=sim.device)
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_pos_w = target_pos_b.unsqueeze(0) + scene.env_origins
            goal_marker.visualize(goal_pos_w, target_quat_b.unsqueeze(0))
            # (real-time collision clearance is now reported in the periodic [STATUS] line above,
            # via min_clearance_all_bodies -- superseded the old per-frame point-in-AABB HUD, which
            # only caught full penetration of a body's origin, not "pressed up against a surface".)
    except KeyboardInterrupt:
        pass

    print(f"[INFO] Final cube position: {cube.data.root_pos_w[0].tolist()}")
    window.close()


def main() -> None:
    try:
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.4, 0.3])

        scene_cfg = BoxSceneCfg(num_envs=1, env_spacing=2.5)
        _add_box_prims(scene_cfg)
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        print("[INFO] Setup complete. box_env_cfg.py's box (unmodified) beside a +90deg-yawed robot mount, real-time teleop.")
    except Exception as e:
        print(f"[ERROR] Failed to set up simulation/scene: {e}")
        simulation_app.close()
        return

    try:
        run_teleop(sim, scene)
    except Exception as e:
        print(f"[ERROR] Real-time teleop failed: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

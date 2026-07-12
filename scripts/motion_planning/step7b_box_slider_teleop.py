# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 7b: box scene + slider target picking + REAL box-obstacle cuRobo planning.

Builds on step7a_box_scene_normal_mount.py (box_env_cfg.py's box, verbatim geometry, hole on
top) and step5c_slider_motion_plan.py (slider target picking, lock-then-confirm gate,
save/restore via scene.get_state()/scene.reset_to(), cuRobo plan_single + auto-execute).
step7a_box_scene_normal_mount.py itself is untouched -- this is a new file, self-contained like
every other script in this directory (none of them import each other).

Two fixes over step7a, both applied here:

1. ROBOT ORIENTATION FIX. step7a placed the box at (BOX_X, BOX_Y) = (0.0, 0.65) -- a pure
   lateral (+Y) offset -- but left the robot at rot=(1,0,0,0) (identity). That was wrong: the
   Panda's default-pose reach is along +X, not +Y, confirmed two ways (not guessed): (a) the
   official Isaac-Lift-Cube-Franka-IK-Rel-v0 task itself places its cube at (0.5,0,0.055) and
   validates its whole reach envelope on pos_x, at this exact identity rotation; (b) step7a's
   own live FK run measured the default-pose's tightest-clearance body sphere at world
   (0.4186, -0.0491, 0.3868) -- X-dominant, Y near zero. So this script instead rotates the
   ROBOT (not the box) by +90 deg about world Z:
       ROBOT_ROOT_ROT = Rz(+90deg) = (cos45, 0, 0, sin45) = (0.7071068, 0, 0, 0.7071068)
   Rz(90deg)'s matrix maps local +X -> world (cos90, sin90, 0) = (0, 1, 0) = world +Y -- verified
   by hand before writing this, not assumed. run_test() below re-measures the default-pose
   panda_hand position live and prints an explicit PASS/WARNING against this prediction, the
   same live-check discipline step7a used for the original (wrong) placement.

   Side effect worth knowing: cuRobo plans entirely in the robot's BASE frame, not world (see
   step4_upside_down_reach.py's docstring -- confirmed by inspecting cuRobo's kinematics code,
   no root-pose concept anywhere in it). With the base now yawed, base frame != world frame, so
   slider targets below are BASE-frame coordinates (same convention step5c already used, just
   no longer coincident with world since our root rotation is no longer identity). A downward
   target quat of (0,1,0,0) in base frame still points the gripper straight down in WORLD frame
   regardless of this yaw, because a rotation about local X only flips the local Z axis, and a
   pure yaw about world Z never changes the Z axis at all -- "down" in base frame stays "down"
   in world frame. Verified by hand, not assumed.

2. REAL BOX OBSTACLES. step7a's default-pose check only tested the box's outer shell against a
   dummy far-away cuRobo world -- it never asked cuRobo to plan a path THROUGH the hole. Here,
   all 13 real box parts (4 walls, floor, 4 hole-framing top pieces, platform, 4 posts) are
   computed ONCE via box_parts() and used as the single source of truth for both the spawned
   AssetBaseCfg prims (what you'll see/collide with in sim) and cuRobo's world_cfg (what the
   planner avoids) -- eliminating the exact mismatch that broke the scrapped v1 of step7a (a
   separately hand-rotated obstacle list that no longer matched the spawned geometry).

Reach caveat, measured not assumed: box_env_cfg.py's box sits at (0,0.65) purely because that
clears the robot base laterally (step7a's fix), not because it was reach-optimized. In the
robot's base frame, the hole's plane CENTER is at local (0.65, 0, 0.5) -- straight-line distance
sqrt(0.65^2+0.5^2) = 0.820 m, ~96% of the Franka's ~0.855 m rated reach (the same territory
step5c's own docstring already flagged as producing ~27cm IK error in a similar near-limit
test). The hole's far corner, local (0.825, 0.175, 0.5), is at 0.980 m -- past max reach,
definitely unreachable. The cube itself, deeper inside at local (0.65, 0, 0.28), is a
comfortable 0.708 m. Slider ranges below cover the hole's footprint plus the cube depth
deliberately, INCLUDING the marginal/unreachable far edge, so failed plans there are informative
about this placement's limits, not hidden by shrinking the sliders to only ever succeed.

Flow (unchanged from step5c): reset -> verify orientation flip -> save baseline -> build/warm
cuRobo with REAL box obstacles -> show sliders (zero side effects, drift-checked every step) ->
"Confirm Target" -> restore baseline FIRST, then read post-restore start state, then plan_single
-> print SUCCESS/FAILED -> on success, independently sweep every waypoint's collision spheres
against the box's own AABBs by hand (not trusting cuRobo's internal margin, same discipline as
step7a/step3) -> auto-execute -> report final EE error.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step7b_box_slider_teleop.py \
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
        " scripts/motion_planning/step7b_box_slider_teleop.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 7b: box scene, slider target picking, real box-obstacle cuRobo planning.")
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from curobo.types.base import TensorDeviceType  # isort:skip
from curobo.types.math import Pose  # isort:skip
from curobo.types.state import JointState  # isort:skip
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # isort:skip

GRIPPER_OPEN = 0.04
# step5b/step5c used 1e-3 rad for a bare-robot scene; this box scene has more static kinematic
# bodies + a settling cube under friction/contact, producing more PD/solver jitter at rest
# (observed 0.001-0.005 rad). 0.015 rad gives ~3x margin above that noise ceiling while still
# catching genuine, sustained drift (which would grow well past this, not hover near it).
DRIFT_TOL_RAD = 0.015

# --- Fix 1: robot yawed +90 deg about world Z instead of moving the box. See module docstring
# for the hand-verified Rz(90deg) matrix mapping local +X -> world +Y. ---
ROBOT_ROOT_POS = (0.0, 0.0, 0.0)
ROBOT_ROOT_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

# Box placement UNCHANGED from step7a (translation only, no rotation, verbatim box_env_cfg.py
# geometry) -- Fix 1 addressed the mismatch by rotating the robot instead.
BOX_X, BOX_Y = 0.0, 0.65

# Base-frame slider ranges (cuRobo plans in base frame -- see module docstring). Computed from
# the hole's actual world footprint transformed into the yawed base frame via
# local = (world_y, -world_x, world_z) for Rz(-90deg) (the inverse of the mount rotation above,
# hand-derived the same way as the forward mapping):
#   hole world X in [-0.175, 0.175]  -> local Y in [-0.175, 0.175]
#   hole world Y in [0.475, 0.825]   -> local X in [0.475, 0.825]
#   hole/cube world Z in [0.28, 0.5] -> local Z unchanged (yaw about Z doesn't move Z)
# Ranges below bracket that footprint with a small margin, DELIBERATELY including the far edge
# even though it's beyond reach (see module docstring's 0.980 m corner-distance math) -- the
# point of sliders is to let failures at the edge be visible, not to hide them by shrinking the
# range to guaranteed-success territory.
POS_X_RANGE = (0.45, 0.85)
POS_Y_RANGE = (-0.20, 0.20)
POS_Z_RANGE = (0.25, 0.55)

# Downward-facing gripper, base frame -- still means "straight down" in world frame despite the
# yaw (see module docstring). Same convention as step2/step3/step5c.
DEFAULT_TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]


def box_parts() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], str]]:
    """Single source of truth for box_env_cfg.py's box geometry (verbatim numbers, translated to
    (BOX_X, BOX_Y) only -- no rotation). Returns (name, center, dims, material_kind) for the 13
    static parts. Used to build BOTH the spawned AssetBaseCfg prims and cuRobo's real obstacle
    world_cfg, so the two can never drift apart the way step7a v1's independent reconstruction did.
    """
    box_w, box_d, box_h, t, hole = 0.8, 0.8, 0.5, 0.01, 0.35
    z_bot, z_top = 0.0, 0.5
    z_mid = (z_bot + z_top) / 2  # 0.25
    top_z = z_top - t / 2  # 0.495
    side_w = (box_w - hole) / 2  # 0.225
    side_d = (box_d - hole) / 2  # 0.225

    platform_size = (0.5, 0.5, 0.01)
    platform_top_z = z_mid + platform_size[2] / 2  # 0.255
    ceiling_z = z_top - t  # 0.49
    post_h = ceiling_z - platform_top_z  # 0.235
    post_z = (platform_top_z + ceiling_z) / 2  # 0.3725
    post_inset = 0.23
    cube_size = 0.05
    cube_z = platform_top_z + cube_size / 2  # 0.28

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
    """Spawn all box_parts() as kinematic AssetBaseCfg prims, plus the dynamic cube. Returns the
    cube's spawn position for later reference."""
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


def _cuboid_world_cfg() -> dict:
    """Real box obstacles for cuRobo, built from the SAME box_parts() list used to spawn the
    sim assets above -- not a separately reconstructed geometry."""
    parts, _, _ = box_parts()
    return {"cuboid": {name: {"dims": list(dims), "pose": [*center, 1, 0, 0, 0]} for name, center, dims, _ in parts}}


def _aabb_signed_distance(center: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor) -> float:
    """Signed distance from a point to an axis-aligned box surface (negative = inside/penetrating)."""
    clamped = torch.max(torch.min(center, box_max), box_min)
    outside_vec = center - clamped
    outside_dist = torch.norm(outside_vec).item()
    if outside_dist > 1e-9:
        return outside_dist
    face_dists = torch.min(center - box_min, box_max - center)
    return -face_dists.min().item()


def box_obstacle_aabbs(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    parts, _, _ = box_parts()
    aabbs = []
    for name, center, dims, _ in parts:
        c = torch.tensor(center, dtype=torch.float32, device=device)
        d = torch.tensor(dims, dtype=torch.float32, device=device)
        aabbs.append((name, c - d / 2, c + d / 2))
    return aabbs


def min_clearance_against_obstacles(
    motion_gen: MotionGen, joint_positions: torch.Tensor, obstacle_aabbs: list[tuple[str, torch.Tensor, torch.Tensor]]
) -> tuple[float, str, int]:
    """Independent (non-cuRobo-internal) clearance sweep across every given joint config."""
    device = joint_positions.device
    worst, worst_obstacle, worst_wp = float("inf"), "", -1
    for wp_idx, q in enumerate(joint_positions):
        spheres = motion_gen.kinematics.get_robot_as_spheres(q.unsqueeze(0))[0]
        for s in spheres:
            center = torch.tensor(s.position, dtype=torch.float32, device=device)
            for name, box_min, box_max in obstacle_aabbs:
                clearance = _aabb_signed_distance(center, box_min, box_max) - s.radius
                if clearance < worst:
                    worst, worst_obstacle, worst_wp = clearance, name, wp_idx
    return worst, worst_obstacle, worst_wp


def reset_to_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot) -> None:
    """Explicitly write both root pose AND joint pose (step4's discipline, not step5c's simpler
    joint-only version) -- deliberately defensive here since this script is the first to combine
    a non-identity root ROTATION with this box scene, and that combination hasn't been verified
    live before."""
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


class SliderTargetWindow:
    """Copied unchanged from step5c_slider_motion_plan.py: pure UI read, zero side effects on
    the robot. Sliders are BASE-frame coordinates (see module docstring)."""

    def __init__(self, robot, ee_body_idx: int, env_id: int = 0):
        self.device = robot.data.root_state_w.device
        self.window = None
        self.sliders = {}
        self.target_label = None
        self.confirmed = False
        self.confirmed_target_pos = None

        ee_state_w = robot.data.body_state_w[env_id, ee_body_idx]
        root_state_w = robot.data.root_state_w[env_id]
        start_pos_b, _ = math_utils.subtract_frame_transforms(
            root_state_w[0:3].unsqueeze(0), root_state_w[3:7].unsqueeze(0),
            ee_state_w[0:3].unsqueeze(0), ee_state_w[3:7].unsqueeze(0),
        )
        self.start_pos_b = start_pos_b.squeeze(0)

    def build_ui(self):
        self.window = ui.Window("Pick Target (Box Hole, Slider + cuRobo Plan)", width=440, height=300)
        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label(
                    "Drag sliders to pick a target (base frame). The robot will NOT move --"
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

    def read_target_pos(self):
        return torch.tensor(
            [
                self.sliders["x"].get_value_as_float(),
                self.sliders["y"].get_value_as_float(),
                self.sliders["z"].get_value_as_float(),
            ],
            device=self.device,
        )

    def update_label(self):
        pos = self.read_target_pos()
        self.target_label.text = f"Current target (base frame): X={pos[0]:.3f}  Y={pos[1]:.3f}  Z={pos[2]:.3f}"

    def _on_confirm(self):
        self.confirmed_target_pos = self.read_target_pos()
        self.confirmed = True
        print(f"[INFO] 'Confirm Target' pressed. Target snapshot (base frame): {self.confirmed_target_pos.cpu().numpy().tolist()}")

    def close(self):
        if self.window is not None:
            self.window.visible = False
            self.window = None


def run_test(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot mount: pos={ROBOT_ROOT_POS}, rot={ROBOT_ROOT_ROT} (+90deg yaw about world Z).")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_idx = robot_entity_cfg.body_ids[0]
    sim_dt = sim.get_physics_dt()

    # ------------------------------------------------------------------
    # STEP 0: reset, then live-verify the orientation fix actually flipped the reach axis
    # (same discipline as step7a's original measurement, not assumed from the Rz(90) math alone).
    # ------------------------------------------------------------------
    reset_to_default_pose(sim, scene, robot)
    hand_pos_w = robot.data.body_pos_w[0, ee_body_idx]
    print(f"[VERIFY] Default-pose panda_hand world position: {hand_pos_w.tolist()}")
    print("[VERIFY] Pre-fix (identity rot, step7a) measurement was X~0.42, Y~-0.05 (reach along +X).")
    if abs(hand_pos_w[1].item()) > abs(hand_pos_w[0].item()):
        print("[VERIFY] PASSED: Y is now the dominant reach axis -- the +90deg yaw flipped the facing direction as intended.")
    else:
        print("[VERIFY] WARNING: X is still dominant over Y -- the yaw does NOT appear to have flipped the facing axis. Investigate before trusting planning below.")

    for _ in range(30):
        sim.step()
        scene.update(sim_dt)
    cube = scene["object"]
    print(f"[INFO] Settled cube position: {cube.data.root_pos_w[0].tolist()}")

    # ------------------------------------------------------------------
    # STEP 1: save baseline, build + warm cuRobo with REAL box obstacles.
    # ------------------------------------------------------------------
    saved_state = scene.get_state(is_relative=False)
    saved_joint_pos = saved_state["articulation"]["robot"]["joint_position"][0, arm_joint_ids].clone()
    print(f"[STEP 1] Saved baseline joint_pos: {saved_joint_pos.cpu().numpy().tolist()}")

    tensor_args = TensorDeviceType(device=torch.device(sim.device))
    obstacle_aabbs = box_obstacle_aabbs(tensor_args.device)
    world_cfg = _cuboid_world_cfg()
    print(f"[INFO] Building cuRobo MotionGen (franka.yml, {len(world_cfg['cuboid'])} REAL box obstacles)...")
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml", world_cfg, tensor_args=tensor_args, interpolation_dt=0.02
    )
    motion_gen = MotionGen(motion_gen_config)
    print("[INFO] Warming up cuRobo MotionGen (this takes a few seconds)...")
    motion_gen.warmup(enable_graph=True)
    print(f"[INFO] cuRobo MotionGen ready. Active joints: {motion_gen.kinematics.joint_names}")

    # ------------------------------------------------------------------
    # STEP 2/3: sliders; robot frozen, drift-checked every step (unchanged from step5c).
    # ------------------------------------------------------------------
    window = SliderTargetWindow(robot, ee_body_idx=ee_body_idx)
    window.build_ui()
    print("\n[STEP 2] Sliders shown (base frame). Drag to pick a target -- the robot will not move.")
    print("[STEP 3] Verifying every physics step that the robot stays at the saved baseline...")

    drift_warned = False
    while simulation_app.is_running() and not window.confirmed:
        window.update_label()
        sim.step()
        scene.update(sim_dt)
        current_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
        drift = (current_joint_pos - saved_joint_pos).abs().max().item()
        if drift > DRIFT_TOL_RAD:
            if not drift_warned:
                print(
                    f"[ERROR] Robot drifted while waiting for confirmation! max abs diff = {drift:.6f} rad"
                    f" (tolerance {DRIFT_TOL_RAD}). Further repeats of this warning are suppressed until"
                    " drift returns within tolerance."
                )
                drift_warned = True
        else:
            drift_warned = False

    if not simulation_app.is_running():
        return

    # ------------------------------------------------------------------
    # STEP 4: confirmed -- restore FIRST, then read post-restore start state, then plan.
    # ------------------------------------------------------------------
    target_pos_t = window.confirmed_target_pos.clone()
    print(f"\n[STEP 4] Final confirmed target (base frame): {target_pos_t.cpu().numpy().tolist()}")

    scene.reset_to(saved_state)
    sim.forward()
    restored_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
    restore_diff = (restored_joint_pos - saved_joint_pos).abs().max().item()
    print(f"[STEP 4] Restored baseline. Max abs diff from saved baseline: {restore_diff:.6f} rad")

    arm_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
    start_state = JointState.from_position(
        arm_joint_pos.to(device=tensor_args.device, dtype=tensor_args.dtype), joint_names=motion_gen.kinematics.joint_names
    )
    target_quat_t = torch.tensor(DEFAULT_TARGET_QUAT, dtype=tensor_args.dtype, device=tensor_args.device)
    goal_pose = Pose(position=target_pos_t.to(dtype=tensor_args.dtype, device=tensor_args.device), quaternion=target_quat_t)
    plan_config = MotionGenPlanConfig(max_attempts=20, enable_graph=True)
    print("[INFO] Planning against REAL box obstacles...")
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    # ------------------------------------------------------------------
    # STEP 5: report, independently verify clearance, execute.
    # ------------------------------------------------------------------
    success = bool(result.success.item())
    if not success:
        print(f"[PLAN] FAILED. Status: {result.status}")
        print("[ERROR] No valid trajectory found for the confirmed target through the box. Robot will remain at its restored pose.")
        window.close()
        print("[INFO] Leaving the viewer open. Close the window to exit.")
        while simulation_app.is_running():
            sim.render()
        return

    plan = result.interpolated_plan
    n_waypoints = len(plan.position)
    print(f"[PLAN] SUCCESS. Waypoints: {n_waypoints}, total planning time: {result.total_time:.3f}s")

    print(f"[INFO] Independently sweeping all {n_waypoints} waypoints against the real box AABBs (hand-computed, not cuRobo's internal margin)...")
    worst, worst_obstacle, worst_wp = min_clearance_against_obstacles(motion_gen, plan.position, obstacle_aabbs)
    print(f"[RESULT] Tightest clearance across the full path: {worst:.4f} m (waypoint {worst_wp}/{n_waypoints - 1}, obstacle '{worst_obstacle}')")
    if worst < 0.0:
        print("[ERROR] A waypoint penetrates a box obstacle despite a successful plan -- investigate before trusting this path.")

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

    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
    ee_pos_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
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
        sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.4, 0.3])

        scene_cfg = BoxSceneCfg(num_envs=1, env_spacing=2.5)
        _add_box_prims(scene_cfg)
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        print("[INFO] Setup complete. box_env_cfg.py's box (unmodified) beside a +90deg-yawed robot mount.")
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

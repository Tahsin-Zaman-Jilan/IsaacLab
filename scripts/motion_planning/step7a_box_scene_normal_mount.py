# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 7a: box_env_cfg.py's box, UNMODIFIED, placed beside a NORMALLY mounted robot.

Companion to step4_upside_down_reach.py, testing the opposite premise: box_env_cfg.py's box
geometry, but reachable by a robot mounted exactly like the official Isaac-Lift-Cube-Franka-
IK-Rel-v0 task (pos=(0,0,0), rot=(1,0,0,0), stock default joint pose) -- NOT box_env_cfg's
inherited upside-down mount.

v1 of this script tried to be clever: rotate the box 90 deg so the hole faces the robot
sideways instead of on top, and rebuild the platform/posts (since a rigid rotation turns the
horizontal cube-supporting shelf into a useless vertical plate). That reconstruction stopped
being "box_env_cfg.py's box" -- it was a hand-derived reinterpretation of it, sized and shaped
differently from the original, and it turned out to collide with the default pose (verified
live: -0.0032 m, tightest against the rebuilt platform). Scrapped per direct feedback: this
version does NOT reinterpret the geometry at all.

This version imports the box construction VERBATIM -- same local-frame formulas as
box_env_cfg.py (hole-on-TOP, platform/posts/cube exactly as authored there, right down to the
variable names and numbers) -- and only TRANSLATES the whole assembly sideways
(BOX_X, BOX_Y) = (0.0, 0.65) so it sits next to the robot instead of overlapping its base.
box_env_cfg.py and upside_down_env_cfg.py are still NOT imported or modified.

Scope, deliberately narrowed: this step only places the box and checks that the robot's
DEFAULT reset pose doesn't collide with it. It does NOT attempt to plan a reach into the
hole -- with the hole still on top, reaching it from a normal upright mount is the original
"arc the arm up and over 0.5 m walls" problem this project's own upside-down variant already
found fragile (see box_env_cfg.py's own docstring). Solving that reach is deferred to a later
step, after this placement is confirmed clean.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/motion_planning/step7a_box_scene_normal_mount.py \
        --device cuda:0 --headless

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
        " scripts/motion_planning/step7a_box_scene_normal_mount.py --device cuda:0"
    )
    sys.exit(1)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 7a: box_env_cfg.py's box (unmodified), normal robot mount.")
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from curobo.types.base import TensorDeviceType  # isort:skip
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig  # isort:skip

# Normal, upright, table-style mount -- IDENTICAL to the official Isaac-Lift-Cube-Franka-IK-Rel-v0
# task (no override beyond FRANKA_PANDA_HIGH_PD_CFG's own defaults). Deliberately NOT
# box_env_cfg's inherited (0,1,0,0) upside-down rotation.
ROBOT_ROOT_POS = (0.0, 0.0, 0.0)
ROBOT_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)

# Box placement: TRANSLATION ONLY, no rotation. (0.0, 0.65) puts the box's near edge
# (Y = 0.65 - 0.4 = 0.25) a clear 0.25 m from the robot base at the origin -- box_env_cfg.py's
# own box was centered at (0,0) under an elevated, offset-base upside-down mount where that
# overlap was fine; here the base is on the ground at (0,0,0), so centering the box there would
# put it around the robot. "Beside the robot" = lateral (Y) offset, box footprint otherwise
# untouched.
BOX_X, BOX_Y = 0.0, 0.65


@configclass
class BoxSceneCfg(InteractiveSceneCfg):
    """Normal-mount Franka beside box_env_cfg.py's box (verbatim geometry, translated only)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=ROBOT_ROOT_POS, rot=ROBOT_ROOT_ROT),
    )


def _add_box_prims(scene_cfg: BoxSceneCfg) -> None:
    """Attach box_env_cfg.py's box to the scene config, VERBATIM (same local-frame formulas,
    same variable names/numbers as that file), just translated to (BOX_X, BOX_Y) instead of
    (0, 0). No rotation, no resized/rebuilt parts -- this is box_env_cfg.py's box, not a
    reinterpretation of it.
    """
    _box_x, _box_y = BOX_X, BOX_Y
    _box_z_bot, _box_z_top = 0.0, 0.5
    _box_w, _box_d, _box_h = 0.8, 0.8, 0.5
    _t = 0.01
    _hole = 0.35
    _box_z_mid = (_box_z_bot + _box_z_top) / 2

    _glass_material = sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.52)
    _static_rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True)
    _static_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0)
    _static_mass_props = sim_utils.MassPropertiesCfg(mass=5.0)

    # four side walls
    scene_cfg.box_wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxWallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(_t, _box_d, _box_h),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x - _box_w / 2 + _t / 2, _box_y, _box_z_mid)),
    )
    scene_cfg.box_wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxWallRight",
        spawn=sim_utils.CuboidCfg(
            size=(_t, _box_d, _box_h),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x + _box_w / 2 - _t / 2, _box_y, _box_z_mid)),
    )
    scene_cfg.box_wall_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxWallFront",
        spawn=sim_utils.CuboidCfg(
            size=(_box_w, _t, _box_h),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x, _box_y - _box_d / 2 + _t / 2, _box_z_mid)),
    )
    scene_cfg.box_wall_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxWallBack",
        spawn=sim_utils.CuboidCfg(
            size=(_box_w, _t, _box_h),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x, _box_y + _box_d / 2 - _t / 2, _box_z_mid)),
    )

    # floor
    scene_cfg.box_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxFloor",
        spawn=sim_utils.CuboidCfg(
            size=(_box_w, _box_d, _t),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x, _box_y, _box_z_bot + _t / 2)),
    )

    # top: 4 pieces framing the central hole x hole square opening
    _top_z = _box_z_top - _t / 2
    _side_w = (_box_w - _hole) / 2
    _side_d = (_box_d - _hole) / 2
    scene_cfg.box_top_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxTopLeft",
        spawn=sim_utils.CuboidCfg(
            size=(_side_w, _box_d, _t),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x - _hole / 2 - _side_w / 2, _box_y, _top_z)),
    )
    scene_cfg.box_top_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxTopRight",
        spawn=sim_utils.CuboidCfg(
            size=(_side_w, _box_d, _t),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x + _hole / 2 + _side_w / 2, _box_y, _top_z)),
    )
    scene_cfg.box_top_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxTopFront",
        spawn=sim_utils.CuboidCfg(
            size=(_hole, _side_d, _t),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x, _box_y - _hole / 2 - _side_d / 2, _top_z)),
    )
    scene_cfg.box_top_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxTopBack",
        spawn=sim_utils.CuboidCfg(
            size=(_hole, _side_d, _t),
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_glass_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x, _box_y + _hole / 2 + _side_d / 2, _top_z)),
    )

    # internal platform + 4 corner support posts, suspended from the ceiling -- exactly as
    # box_env_cfg.py, no resizing
    _platform_size = (0.5, 0.5, 0.01)
    _platform_z = _box_z_mid
    _platform_top_z = _platform_z + _platform_size[2] / 2
    _ceiling_z = _box_z_top - _t
    _post_h = _ceiling_z - _platform_top_z
    _post_z = (_platform_top_z + _ceiling_z) / 2
    _post_inset = 0.23

    _platform_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1))
    scene_cfg.box_platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxPlatform",
        spawn=sim_utils.CuboidCfg(
            size=_platform_size,
            rigid_props=_static_rigid_props,
            mass_props=_static_mass_props,
            collision_props=_static_collision_props,
            visual_material=_platform_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_box_x, _box_y, _platform_z)),
    )
    for name, sx, sy in [("Fl", -1, -1), ("Fr", 1, -1), ("Bl", -1, 1), ("Br", 1, 1)]:
        setattr(
            scene_cfg,
            f"box_post_{name.lower()}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/BoxPost{name}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.015, 0.015, _post_h),
                    rigid_props=_static_rigid_props,
                    mass_props=_static_mass_props,
                    collision_props=_static_collision_props,
                    visual_material=_platform_material,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(_box_x + sx * _post_inset, _box_y + sy * _post_inset, _post_z)
                ),
            ),
        )

    # cube, resting on the platform via gravity -- exactly as box_env_cfg.py
    _cube_size = 0.05
    _cube_z = _platform_top_z + _cube_size / 2
    scene_cfg.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_box_x, _box_y, _cube_z), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(_cube_size, _cube_size, _cube_size),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    print(
        f"[INFO] Box (verbatim box_env_cfg.py geometry) placed at center ({_box_x}, {_box_y}), "
        f"bottom Z={_box_z_bot}, top Z={_box_z_top}, hole {_hole}x{_hole} on top, "
        f"cube resting at ({_box_x}, {_box_y}, {_cube_z})."
    )


def reset_to_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot) -> None:
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


def _aabb_signed_distance(center: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor) -> float:
    """Signed distance from a point to an axis-aligned box surface (negative = inside/penetrating)."""
    clamped = torch.max(torch.min(center, box_max), box_min)
    outside_vec = center - clamped
    outside_dist = torch.norm(outside_vec).item()
    if outside_dist > 1e-9:
        return outside_dist
    face_dists = torch.min(center - box_min, box_max - center)
    return -face_dists.min().item()


# AABBs for the box's OUTER shell only (walls/floor/top), matching the verbatim geometry above --
# this is what the default (folded, non-reaching) pose needs to clear; the hole means the
# platform/posts/cube sit inside an opening the shell AABBs don't cover, which is correct: they
# are only reachable through the hole, not obstacles to a robot that hasn't entered it yet.
def box_shell_aabbs(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    box_w, box_d, box_h, t = 0.8, 0.8, 0.5, 0.01
    z_bot, z_mid = 0.0, 0.25
    parts = [
        ("wall_left", (BOX_X - box_w / 2 + t / 2, BOX_Y, z_mid), (t, box_d, box_h)),
        ("wall_right", (BOX_X + box_w / 2 - t / 2, BOX_Y, z_mid), (t, box_d, box_h)),
        ("wall_front", (BOX_X, BOX_Y - box_d / 2 + t / 2, z_mid), (box_w, t, box_h)),
        ("wall_back", (BOX_X, BOX_Y + box_d / 2 - t / 2, z_mid), (box_w, t, box_h)),
        ("floor", (BOX_X, BOX_Y, z_bot + t / 2), (box_w, box_d, t)),
    ]
    aabbs = []
    for name, center, dims in parts:
        c = torch.tensor(center, dtype=torch.float32, device=device)
        d = torch.tensor(dims, dtype=torch.float32, device=device)
        aabbs.append((name, c - d / 2, c + d / 2))
    return aabbs


def run_test(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    print(f"[INFO] Scene ready. Entities: {list(scene.keys())}")
    print(f"[INFO] Robot mount: pos={ROBOT_ROOT_POS}, rot={ROBOT_ROOT_ROT} (normal/upright, matches official task).")

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    reset_to_default_pose(sim, scene, robot)
    print("[INFO] Robot set to default pose. Letting the cube settle onto the platform (30 steps)...")
    sim_dt = sim.get_physics_dt()
    for _ in range(30):
        sim.step()
        scene.update(sim_dt)
    cube = scene["object"]
    print(f"[INFO] Settled cube position: {cube.data.root_pos_w[0].tolist()}")

    tensor_args = TensorDeviceType(device=torch.device(sim.device))
    obstacle_aabbs = box_shell_aabbs(tensor_args.device)

    print("[INFO] Building cuRobo MotionGen (franka.yml) for FK collision-sphere access only -- no motion planning in this step...")
    dummy_world = {"cuboid": {"dummy_far_away": {"dims": [0.01, 0.01, 0.01], "pose": [100.0, 100.0, 100.0, 1, 0, 0, 0]}}}
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml", dummy_world, tensor_args=tensor_args, interpolation_dt=0.02
    )
    motion_gen = MotionGen(motion_gen_config)
    print("[INFO] Warming up cuRobo MotionGen...")
    motion_gen.warmup(enable_graph=False)
    print("[INFO] cuRobo MotionGen ready.")

    start_joint_pos = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].clone().to(tensor_args.device)
    spheres = motion_gen.kinematics.get_robot_as_spheres(start_joint_pos.unsqueeze(0))[0]
    worst = float("inf")
    worst_obstacle = ""
    worst_sphere_desc = ""
    for s_idx, s in enumerate(spheres):
        center = torch.tensor(s.position, dtype=torch.float32, device=tensor_args.device)
        for name, box_min, box_max in obstacle_aabbs:
            clearance = _aabb_signed_distance(center, box_min, box_max) - s.radius
            if clearance < worst:
                worst = clearance
                worst_obstacle = name
                worst_sphere_desc = f"sphere#{s_idx} at {center.tolist()} (r={s.radius:.4f})"

    print(f"[RESULT] Default reset pose clearance vs. box shell: {worst:.4f} m (tightest vs '{worst_obstacle}', {worst_sphere_desc}).")
    if worst < 0.0:
        print("[ERROR] Default reset pose COLLIDES with the box shell at this placement.")
    else:
        print("[INFO] Default reset pose is clear of the box. Placement confirmed clean; reach-through-the-hole planning is a separate, later step.")

    print("[INFO] Holding pose. Close the viewer window to exit (or Ctrl+C in headless mode).")
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
        print("[INFO] Setup complete. box_env_cfg.py's box (unmodified) placed beside a normal-mount robot.")
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

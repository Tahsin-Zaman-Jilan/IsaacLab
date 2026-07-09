# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices import DevicesCfg, Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from . import ik_rel_env_cfg


@configclass
class FrankaCubeLiftUpsideDownNoTableEnvCfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
    """Isaac-Lift-Cube-Franka-IK-Rel-v0 with the table removed (cube rests on the ground plane),
    the robot mount flipped 180 deg (base up, gripper down), the default reset pose straightened,
    and a keyboard-controllable prismatic lift joint added to the base itself (not IK-driven).
    Step 1 of the look-through-perception rebuild: confirm basic upside-down teleop + grasping
    works with no table in the way, before box/hole/platform geometry is added back in later
    steps.

    arm_action (6) + gripper_action (1) are inherited unchanged from the official IK-rel lift
    task. lift_action (1) is added by this class, making action_dim=8, not the official task's 7.
    scene.table, scene.plane, scene.object, scene.robot.spawn/init_state/actuators, self.actions,
    and self.teleop_devices are overridden/added here.

    All numbers below were measured via live sim FK/settle, not hand-derived -- see this env's
    development history for the probe methodology (a prior attempt at reusing a straight-pose
    recipe from a different mount configuration was caught and corrected this way).
    """

    def __post_init__(self):
        # post init of parent: official IK-rel arm action + gripper action + cube physics, untouched.
        super().__post_init__()

        # --- remove the table; cube rests on the ground plane instead ---
        self.scene.table = None

        # Ground plane was at pos.z=-1.05 only to hide it 1.05 m below the table's top surface
        # (both were near world Z=0). With the table gone, raise the ground plane to z=0.0 so it
        # sits at the same reference height the rest of this task's geometry (robot base, cube)
        # is measured from.
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)

        # Cube: same (x, y) anchor as the stock task (0.5, 0), spawned just above the relocated
        # ground plane. Real resting height measured via sim (dropped from z=0.05, physics
        # settled it to z=0.024) rather than assumed from the old table-relative z=0.055 -- spawn
        # z=0.03 here is just above that settle height so there's no large initial drop.
        # Note: the inherited reset_object_position event (lift_env_cfg.py) randomizes this cube's
        # x by +/-0.1 and y by +/-0.25 on every reset (including a teleop R-key reset); z stays
        # fixed. That randomization is accounted for in the base placement's reach margin below.
        self.scene.object.init_state.pos = [0.5, 0.0, 0.03]

        # --- inverted mount: base above the cube's nominal (x, y), elevated for a conservative,
        # generous ground clearance rather than a tightly-optimized reach distance.
        #
        # Earlier attempts to solve for base_z from a target reach distance ran into a real
        # kinematic fact (confirmed via live FK): under this straight pose (see below) + this
        # rotation, the hand sits ~0.57-0.63 m off to the side (large -y) of the base, and that
        # lateral offset is a fixed property of the joint angles -- it does NOT change with
        # base_z (translating the base only translates the whole chain in z, not x/y). So there
        # is no base_z that makes this fixed-joint-angle pose "hang centered under the base";
        # centering has to happen via teleop (W/S/A/D) instead, not via base placement. This
        # class's base_z is therefore chosen purely for a safe, generous ground clearance, err on
        # the side of "starts too high" -- teleop's Q/E lowers it from there.
        #
        # Reusing the full-body FK check already done for this exact pose at base_z=0.527 (see
        # development history): the lowest of all 11 body links (panda_link1/panda_link2, not the
        # hand/fingers) sat at world z=0.194 there, i.e. a fixed local offset from the base of
        # 0.194 - 0.527 = -0.333 m (hand/finger local offsets were consistently -0.21 to -0.24 m
        # across separate measurements that same night -- link1/2 is the binding constraint).
        # Solving for a generous ~0.22 m clearance on that lowest link:
        #   base_z = target_clearance - local_dz_min = 0.22 - (-0.333) = 0.55
        # giving a projected clearance of 0.55 - 0.333 = 0.217 m -- comfortably more than the
        # already-safe 0.194 m at the old base_z=0.527, with margin for the run-to-run variability
        # observed in these measurements (up to ~1.6 cm on the hand across repeated trials).
        # Raised further to base_z=1.87 per explicit request for a much more dramatic starting
        # elevation than the 0.55 m conservative-clearance value above -- prior bumps here
        # (0.527 -> 0.55) were too incremental/conservative given what was actually being asked
        # for.
        #
        # base_z=1.87 is back-solved from a live-FK-confirmed formula, not hand-derived:
        # panda_link0_z = lift_anchor_z - lift_joint (SUBTRACTION -- this joint's positive
        # direction moves panda_link0 DOWN, away from the anchor, not up, a consequence of the
        # 180 deg mount flip below inverting the joint's local Z axis relative to world Z). A
        # first attempt at this class assumed ADDITION (base_z + lift_joint) and landed on
        # base_z=1.5 with lift_joint's default at 0.5, "predicting" panda_link0 at ~2.0 m; live FK
        # after the box shell was added (step2a) instead measured it settling at world Z~1.13 m --
        # arm hanging down into where the box sits, not floating ~2 m above it as intended, caught
        # from a screenshot showing the arm at/inside the box rather than elevated above it. See
        # the lift_joint default note below for the corrected math and the real measured numbers
        # base_z=1.87 is solved from.
        self.scene.robot.init_state.pos = (0.5, 0.0, 1.87)
        self.scene.robot.init_state.rot = (0.0, 1.0, 0.0, 0.0)

        # --- straight/extended default reset pose (minimal elbow/wrist bend), replacing the
        # stock curled default (joint2=-0.569, joint4=-2.810, joint6=3.037, joint7=0.741).
        #
        # Ground-clearance verified via live FK on THIS mount before finalizing (not assumed from
        # a prior session's numbers for a different mount with extra lift/yaw joints, which turned
        # out not to transfer -- see class docstring). At base_z=0.527 (the height this pose was
        # directly measured at; base_z was since raised to 0.55 above for extra margin, which only
        # increases every one of these clearances further since local offsets are base_z-invariant
        # -- see reasoning above):
        #   - lowest of all 11 body links (panda_link1/2) at world Z ~= 0.19-0.33 m depending on
        #     measurement method -- comfortable clearance above the z=0 ground plane.
        #   - hand swings ~0.57-0.63 m sideways (large -y) rather than sitting directly under the
        #     base -- a real kinematic property of this pose+rotation combo, not a defect (this is
        #     the fact that motivated switching base_z from a solved reach distance to a
        #     conservative clearance target, see above). Three alternative "partial straightening"
        #     poses (leaving panda_joint6 closer to its stock value, to keep the hand better
        #     centered) were also measured: all three left the fingers only 2.5-3.8 cm above the
        #     ground/cube, too tight to trust for live teleop. This full-straight recipe was kept
        #     for its clearance margin despite the off-center start; the operator recenters via
        #     teleop (W/S/A/D) at the start of each episode.
        #
        # joint4=-0.1 and joint6=0.0 (this section's original values) are pinned right against the
        # Panda's published hard joint limits (joint4: -3.0718 to -0.0698; joint6: -0.0175 to
        # 3.7525) -- 0.03 rad and ~0 rad of headroom respectively. Sitting a joint at/against its
        # limit under gravity load reads as "not actually straight" in the GUI, and leaves the
        # differential-IK solver no room to satisfy further teleop deltas in some directions
        # (Q/E, etc. can look unresponsive because those joints have nowhere left to go). Backed
        # off to -0.5 / 0.5 in a first pass, which was itself confirmed too bent live in the GUI.
        # Straightened to -0.2 / 0.1 in a second pass (~4x the original pinned margin). Live testing
        # then confirmed the bend is caused by these joint values themselves, not by base height --
        # raising the base via the lift joint (below) does not loosen the elbow/wrist shape at all,
        # since lift_joint only translates the whole chain, it doesn't change joint angles. Pushed
        # closer to straight again here, to -0.13 / 0.03: 0.0602 rad (~3.45 deg) of margin from
        # joint4's -0.0698 limit (~2.0x the original ~0.03 rad pinned-failure margin), 0.0475 rad
        # (~2.72 deg) from joint6's -0.0175 limit (~2.7x its ~0.0175 rad pinned-failure margin) --
        # clearly off both hard stops, but with less slack than the -0.2/0.1 pass.
        self.scene.robot.init_state.joint_pos.update(
            {"panda_joint2": 0.0, "panda_joint4": -0.13, "panda_joint6": 0.03, "panda_joint7": 0.0}
        )

        # --- prismatic lift joint on the base itself (a real joint, not an IK-relative EE move) ---
        # Uses panda_instanceable_vertical_lift_wide.usd, a task-specific clone of
        # panda_instanceable_vertical_lift.usd (the original is gitignored and shared, as-is,
        # across several other branches -- cloning rather than editing it in place avoids
        # changing lift_joint's travel for any of those). It splices a fixed-to-world anchor and a
        # lift_joint (prismatic Z) in front of panda_link0, in place of the stock asset's direct
        # world->panda_link0 attachment. Only the usd_path changes; rigid_props/articulation_props
        # etc. from FRANKA_PANDA_HIGH_PD_CFG's spawn (set by the parent ik_rel class) are otherwise
        # untouched.
        #
        # lift_joint's physics:lowerLimit/upperLimit were widened on this clone only (read/written
        # via a standalone pxr.Usd stage edit, verified by re-opening the saved file): from the
        # original's symmetric -0.3/+0.3 to -0.35/+0.6 here.
        #
        # NOTE on the asymmetry's original justification: this was reasoned as "upward has no
        # collision risk, so give it the full double" -- that reasoning assumed lift_joint's
        # positive direction raises the arm. It's since been confirmed backwards (see the
        # panda_link0_z = lift_anchor_z - lift_joint note above and below): positive lift_joint
        # LOWERS the arm, toward the box, not away from it, so it's the UPPER limit (+0.6) that
        # carries collision risk against the box, and the LOWER limit (-0.35) that gives extra
        # ground/mount clearance. The USD-level limit values themselves are unchanged here (not
        # touched this pass); this note exists so the asymmetry isn't misread as intentionally
        # safety-biased in the direction it was originally reasoned to be.
        lift_usd_path = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Robots", "FrankaEmika", "panda_instanceable_vertical_lift_wide.usd"
        )
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=lift_usd_path)

        # lift_joint default start: init_state.pos above places the fixed anchor at base_z=1.87,
        # and lift_joint's value is panda_link0's offset from that anchor -- but as SUBTRACTION,
        # not addition: panda_link0_z = lift_anchor_z - lift_joint. Confirmed via two live-FK data
        # points on this exact asset/mount (step2a box-shell branch), both fitting the same linear
        # relation to float precision:
        #   commanded lift_joint=0.5 (pre-settle) -> panda_link0_z=1.0000 == 1.5 - 0.5
        #   settled   lift_joint=0.3701           -> panda_link0_z=1.1299 == 1.5 - 0.3701
        # A prior version of this comment assumed ADDITION (base_z + lift_joint = 1.5 + 0.5 =
        # ~2.0 m) -- that arithmetic was never true for this asset; the 180 deg mount flip above
        # inverts the joint's local Z axis relative to world Z, so its positive direction moves
        # panda_link0 DOWN (toward the box), not up. This was caught from a screenshot showing the
        # arm hanging at/inside the box instead of floating above it, not from the numbers alone.
        #
        # Gravity sag, separate from the sign bug: even holding lift_joint's commanded value
        # constant, the implicit PD actuator (gains below) doesn't perfectly cancel the constant
        # gravity/dynamic load along this axis, so the SETTLED value differs from the commanded
        # default by a fixed offset, confirmed reproducible across both data points above:
        #   0.5 (commanded) -> 0.3701 (settled) ... 0.5 - 0.3701 = 0.1299
        # Settled = commanded - ~0.13, independent of the commanded value itself (steady-state PD
        # tracking error against a constant load is load/stiffness, not a function of the target),
        # as long as the result stays clear of the -0.35/+0.6 hard limits noted above.
        #
        # Target: panda_link0 settling at world Z~2.0 m (the original "~2 m" intent, now hit for
        # real). Solving base_z - (commanded - 0.13) = 2.0 with commanded=0.0 (clean, centered,
        # matches box-world-v1's own lift_joint=0.0 default, and avoids sitting anywhere near
        # either widened hard limit): base_z = 2.0 - 0.13 = 1.87, giving settled lift_joint ~=
        # -0.13 -- comfortable margin from both -0.35 (0.22 m) and +0.6 (0.73 m).
        #
        # Downstream clearance above the box: the arm's lowest body in this straight pose
        # (panda_link5/6) sits a measured 1.03 m below panda_link0 (1.0297 m and 1.0300 m across
        # two independent box-free probes) -- a local offset invariant to both base_z and
        # lift_joint, since both are pure Z-translations upstream of panda_link0 and don't change
        # any joint angle (same invariance argument as the base_z reasoning above, extended to
        # lift_joint since it's the same category of transform). At the settled panda_link0~2.0 m
        # target, that puts the lowest link at ~0.97 m, i.e. ~0.47 m above the box's top (Z=0.5) --
        # floating clearly above it, not touching, matching the reference image this fix was
        # driven by. Real usable up/down range via U/J once elevated is for live GUI testing to
        # confirm, not asserted here.
        self.scene.robot.init_state.joint_pos.update({"lift_joint": 0.0})

        # Actuator gains are untuned placeholders (mirrors panda_hand's finger actuator, the only
        # other prismatic/meters-units joint on this asset), not load-bearing calculations -- same
        # values used for this exact joint on an earlier branch.
        self.scene.robot.actuators = {
            **self.scene.robot.actuators,
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=1.0,
                stiffness=1000.0,
                damping=100.0,
            ),
        }

        # --- lift_action, added AFTER arm_action/gripper_action (inherited above via
        # super().__post_init__()): action term insertion order must be
        # [arm_action, gripper_action, lift_action] to match Se3Keyboard.advance()'s fixed
        # concatenation order [pos+rot(6), gripper(1), lift(1)]. action_dim: 7 -> 8.
        self.actions.lift_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=1.0,
        )

        # --- rebuild teleop_devices with lift_term=True ---
        # Unset by the parent chain (falls through to teleop_se3_agent.py's default keyboard
        # construction), so it must be defined here for lift_term to reach the keyboard at all.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=True,
                    lift_term=True,
                    sim_device=self.sim.device,
                ),
            },
        )

        # --- box shell: 4 walls, floor, top-with-hole, ported from box-world-v1's
        # vertical_lift_gripper_teleop_reach_env_cfg.py (FrankaVerticalLiftGripperTeleopReachEnvCfg).
        # Shell geometry only -- that source class's internal suspended platform/posts, cube
        # override, gripper firmer-close tweak, and gamepad/spacemouse teleop devices are NOT
        # ported here; this step only adds the enclosure around the existing, confirmed-working
        # mount/pose/lift-joint/cube setup from step 1, untouched above.
        #
        # Recentered from box-world-v1's (0.0, 0.0) to (0.5, 0.0) to match this task's robot base
        # and cube anchor (both at x=0.5, y=0.0, see scene.robot.init_state.pos and
        # scene.object.init_state.pos above) -- box-world-v1's own center was likewise chosen to
        # line the hole up under its robot base, so this is the same placement rule applied to
        # this task's actual base position, not a reused literal coordinate.
        #
        # Box bottom (z=0.0) coincides with the ground plane relocated above (self.scene.plane at
        # z=0.0), and the box footprint's randomization-reachable interior contains the cube's
        # nominal (0.5, 0.0) anchor plus its +/-0.1 m x / +/-0.25 m y reset randomization.
        #
        # Clearance confirmed via live FK measurement on this exact mount/pose combination (not
        # assumed from box-world-v1's numbers, which used a different base height and pose): at
        # the settled default reset pose, the lowest body (panda_link5/6) sits at world
        # (x=0.615, y=-0.006, z=0.090) -- comfortably inside this hole's x/y span and 9 cm above
        # the floor -- and the two bodies passing closest to the top plate (panda_link3 at
        # z=0.474, panda_link4 at z=0.484) are also both within the hole's x/y footprint. No body
        # is positioned over solid wall/top material at that pose.
        _box_x, _box_y = 0.5, 0.0  # box center X, Y (world) -- matches robot base / cube anchor
        _box_z_bot, _box_z_top = 0.0, 0.5  # box bottom / top Z (world)
        _box_w, _box_d, _box_h = 0.8, 0.8, 0.5  # external size: X, Y, Z
        _t = 0.01  # wall thickness
        _hole = 0.35  # square hole side, centered on the box top
        _box_z_mid = (_box_z_bot + _box_z_top) / 2

        _glass_material = sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.52)
        _static_rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True)
        _static_collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.01, rest_offset=0.0
        )
        _static_mass_props = sim_utils.MassPropertiesCfg(mass=5.0)

        # four side walls
        self.scene.box_wall_left = AssetBaseCfg(
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
        self.scene.box_wall_right = AssetBaseCfg(
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
        self.scene.box_wall_front = AssetBaseCfg(
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
        self.scene.box_wall_back = AssetBaseCfg(
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

        # floor (opaque, so it reads clearly as the box's bottom through the glass walls)
        self.scene.box_floor = AssetBaseCfg(
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

        # top: 4 pieces framing the central _hole x _hole square opening, centered at (_box_x, _box_y)
        _top_z = _box_z_top - _t / 2
        _side_w = (_box_w - _hole) / 2  # width of the left/right top pieces
        _side_d = (_box_d - _hole) / 2  # depth of the front/back top pieces
        self.scene.box_top_left = AssetBaseCfg(
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
        self.scene.box_top_right = AssetBaseCfg(
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
        self.scene.box_top_front = AssetBaseCfg(
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
        self.scene.box_top_back = AssetBaseCfg(
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


@configclass
class FrankaCubeLiftUpsideDownNoTableEnvCfg_PLAY(FrankaCubeLiftUpsideDownNoTableEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

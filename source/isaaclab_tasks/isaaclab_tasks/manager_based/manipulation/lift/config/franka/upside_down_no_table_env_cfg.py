# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.actuators import ImplicitActuatorCfg
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
        # Raised further to base_z=1.5 per explicit request for a much more dramatic starting
        # elevation (~2 m combined with the lift_joint default below) -- prior bumps here
        # (0.527 -> 0.55) were too incremental/conservative given what was actually being asked
        # for. IMPORTANT CONSEQUENCE, flagged separately below: at this height the cube is no
        # longer reachable at lift_joint's default or even its prior -0.35 lower limit -- the lift
        # joint's downward range needs widening too, see the follow-up note after this block.
        self.scene.robot.init_state.pos = (0.5, 0.0, 1.5)
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
        # original's symmetric -0.3/+0.3 to -0.35/+0.6 here. Asymmetric on purpose, not a blind
        # double: at base_z=0.55, documented worst-case clearance for the lowest arm link is
        # ~0.19 m, so widening downward travel much past -0.35 starts eating into that margin (the
        # original -0.3 alone already uses most of it); there's no equivalent collision risk
        # raising the base, so upward got the full double instead.
        lift_usd_path = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Robots", "FrankaEmika", "panda_instanceable_vertical_lift_wide.usd"
        )
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=lift_usd_path)

        # lift_joint default start: init_state.pos above places the fixed anchor at base_z=1.5,
        # and lift_joint's value is panda_link0's offset from that anchor -- actual starting world
        # height is base_z + lift_joint = 1.5 + 0.5 = ~2.0 m, per explicit request.
        #
        # IMPORTANT: this default, combined with base_z=1.5, puts the cube out of reach at this
        # lift_joint value -- see the range-widening note below this block, still pending
        # confirmation, for the fix that keeps the cube reachable via J.
        self.scene.robot.init_state.joint_pos.update({"lift_joint": 0.5})

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

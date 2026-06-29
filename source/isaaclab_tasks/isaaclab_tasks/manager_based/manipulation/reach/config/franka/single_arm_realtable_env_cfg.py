# single_arm_realtable_env_cfg.py
#
# One arm + your real cuboid table. Official reward/command/termination
# completely untouched. Tests whether the table alone causes the noisy
# curves, independent of the second arm.

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg


@configclass
class FrankaSingleArmRealTableEnvCfg(FrankaReachEnvCfg):
    """Official single-arm reach, but on your real cuboid table instead of
    the Nucleus SeattleLabTable. Everything else is untouched."""

    def __post_init__(self):
        super().__post_init__()

        # swap official table for your cuboid
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.8, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.88)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )

        # move robot up to table height
        base_pos = self.scene.robot.init_state.pos
        self.scene.robot.init_state.pos = (base_pos[0], base_pos[1], 1.0)

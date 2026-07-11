"""PPO runner config for the stand-and-kick drill.

Inherits :class:`soccer_kick_amp.ppo_cfg.PPORunnerCfg` (same AMP-PPO settings)
and only

  * uses a **kick-only** AMP corpus (drops the ``walk_policy`` walk prior), and
  * sets a distinct ``experiment_name`` so its logs never collide with the
    ongoing ``soccer_kick_amp`` run.
"""
import glob
import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.ppo_cfg import (
    PPORunnerCfg as _BaseKickPPORunnerCfg,
)


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
_KICK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni", "kick", "walk_kick*.txt")))


@configclass
class PPORunnerCfg(_BaseKickPPORunnerCfg):
    experiment_name = "soccer_stand_kick_amp"
    # Kick-only AMP corpus: no walk prior. The stand-kick drill needs no walking
    # style — the ball is within an in-place kick swing.
    amp_motion_files = _KICK_FILES

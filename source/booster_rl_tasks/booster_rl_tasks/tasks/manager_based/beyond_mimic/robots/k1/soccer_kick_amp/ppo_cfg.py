"""PPO runner config for the soccer kick AMP task (V1).

Single-critic AMP PPO. Reuses the locomotion AMP corpus (walk / walk2run / run)
plus the new walk+kick GMR-retargeted clips for motion priors (legacy
``kick.txt`` was a compilation of non-kick motions and is left on disk as a
backup only). Multi-critic split (goal vs aux) is deferred to V1.2 — see
``docs/soccer_amp_roadmap.md``.
"""
import glob
import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BaseAMPAgentCfg,
)


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
_KICK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni", "kick", "walk_kick*.txt")))
# Walk style prior: rollouts of the frozen lower-body walk policy (built by
# ``scripts/build_amp_corpus.py --walk_rollout_dir ...``). Falls back to the
# legacy mocap ``walk.txt`` until those clips are generated.
_WALK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni", "walk_policy", "walk_policy_*.txt"))) or [
    os.path.join(_AMP_ROOT, "walk.txt")
]


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    experiment_name = "soccer_kick_amp"
    max_iterations = 50000

    # AMP corpus: locomotion priors (legacy 56-col) + new 56-col walk+kick
    # clips. All clips share the env-side ``AMPObsCfg`` width (joint+EE, no
    # root vel = 56 cols); see ``scripts/build_amp_corpus.py``.
    amp_reward_coef = 0.35
    amp_motion_files = [
        *_WALK_FILES,
        *_KICK_FILES,
    ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.65
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22

"""PPO runner config for the stand-and-kick drill.

Inherits :class:`soccer_kick_amp.ppo_cfg.PPORunnerCfg` (same AMP-PPO settings)
and only

  * uses a **kick-only, leg-only** AMP corpus (drops the ``walk_policy`` walk
    prior, and drops the upper body), and
  * sets a distinct ``experiment_name`` so its logs never collide with the
    ongoing ``soccer_kick_amp`` run.
"""
import glob
import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg

from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.ppo_cfg import (
    PPORunnerCfg as _BaseKickPPORunnerCfg,
)


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
# 30-col leg-only clips, sliced from the 56-col ``omni/kick`` corpus by
# ``scripts/make_leg_amp_corpus.py``. Must stay in lockstep with the 30-dim AMP
# observation group in ``env_cfg.FlatStandKickEnvCfg``.
#
# Renamed ``stand_kick*`` on the way out: the source clips are called
# ``walk_kick*`` but contain no walking gait — one leg swings through while the
# other stays planted, and neither foot lifts more than ~0.09 m. They are in-place
# kicks, which is exactly what this drill trains.
_KICK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni_legs", "kick", "stand_kick*.txt")))


@configclass
class PPORunnerCfg(_BaseKickPPORunnerCfg):
    experiment_name = "soccer_stand_kick_amp"
    # Kick-only AMP corpus: no walk prior. The stand-kick drill needs no walking
    # style — the ball is within an in-place kick swing.
    # Leg-only (30-col): the robot's arms are welded, so a corpus that still
    # carried the upper body would let the discriminator win on that constant
    # alone and collapse the style reward.
    amp_motion_files = _KICK_FILES


@configclass
class GRUPPORunnerCfg(PPORunnerCfg):
    """Same drill, but with a recurrent (GRU) actor/critic instead of the MLP.

    Only the policy net changes: the AMP discriminator still scores single
    ``(s, s')`` transitions from its own observation stream, so the style reward
    path is untouched and the corpus stays as-is.

    ``num_steps_per_env`` is the BPTT horizon the GRU is trained over — the
    memory cannot span more than one rollout. It is left at the base value so
    this run stays comparable to the MLP baseline; lengthen it if the kick swing
    turns out to need more context than one rollout carries.
    """

    experiment_name = "soccer_stand_kick_amp_gru"
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

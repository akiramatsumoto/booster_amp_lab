"""Stand-and-kick AMP drill (kick-only prior).

Registers ``Booster-Soccer-StandKick-AMP-v0``. This is a purely *additive*
variant of :mod:`soccer_kick_amp`: it inherits that task's config unchanged and
only

  * narrows the ball spawn to a standing-reach cone (0.3-0.4 m, ±20°), which the
    kick-only corpus can serve with an in-place swing + slight weight shift, and
  * drops the ``walk_policy`` walk prior from the AMP corpus (kick clips only).

Nothing under ``soccer_kick_amp`` is modified, so this task can be trained
alongside the existing kick task without interfering with it.
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-StandKick-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatStandKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

# Same env, GRU actor/critic instead of the MLP. Registered separately so the
# MLP baseline above stays runnable for comparison.
gym.register(
    id="Booster-Soccer-StandKick-AMP-GRU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatStandKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:GRUPPORunnerCfg",
    },
)

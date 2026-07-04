# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the commanded ("蹴りたい") vs. actual ("蹴った") kick direction over N episodes.

For each completed episode this logs, in the world frame:
  * target_deg  — commanded kick direction, latched on the step just before contact
                  (``SoccerKickCommand.target_dir_w`` re-derives itself from the live
                  ball position after the kick, so the pre-kick value is frozen).
  * kicked_deg  — direction of the ball's velocity at its peak XY speed after contact.
  * error_deg   — signed (kicked - target) wrapped to (-180, 180].

Results are printed and written to a CSV (default: kick_directions.csv next to the
checkpoint). Episodes with no detected kick are skipped by default.

Example:
    python scripts/rsl_rl/record_kick_directions.py \
        --task Booster-Soccer-Kick-AMP-v0 --headless \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt --num_episodes 10
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record commanded vs. actual kick direction.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Booster-Soccer-Kick-AMP-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of kick episodes to record.")
parser.add_argument(
    "--include_no_kick",
    action="store_true",
    default=False,
    help="Also record episodes where no kick was detected (kicked_deg left blank).",
)
parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: next to checkpoint).")
parser.add_argument("--max_steps", type=int, default=100000, help="Safety cap on total simulation steps.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import math
import os
import torch

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, TrackAdapterRunner

try:
    from rsl_rl.runners import EncoderMultiCriticAmpRunner, MultiCriticAmpOnPolicyRunner
except ImportError:
    EncoderMultiCriticAmpRunner = None
    MultiCriticAmpOnPolicyRunner = None

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401


def _split_obs(obs_td):
    groups = dict(obs_td.items()) if hasattr(obs_td, "items") else dict(obs_td)
    policy_obs = groups.pop("policy")
    return policy_obs, groups


class _UnwrappedProxy:
    def __init__(self, unwrapped):
        self.env = unwrapped


class _LegacyRslRlEnv:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def env(self):
        return _UnwrappedProxy(self._env.unwrapped)

    def get_observations(self):
        obs, others = _split_obs(self._env.get_observations())
        return obs, {"observations": others}

    def reset(self):
        obs_td, extras = self._env.reset()
        obs, others = _split_obs(obs_td)
        extras = dict(extras) if extras is not None else {}
        extras["observations"] = others
        return obs, extras

    def step(self, actions):
        obs_td, rew, dones, infos = self._env.step(actions)
        obs, others = _split_obs(obs_td)
        infos = dict(infos) if infos is not None else {}
        infos["observations"] = others
        return obs, rew, dones, infos

    def close(self):
        return self._env.close()


def _wrap_deg(deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        checkpoint = os.path.expanduser(args_cli.checkpoint)
        if os.path.exists(checkpoint):
            resume_path = retrieve_file_path(checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = _LegacyRslRlEnv(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner_class_name = getattr(agent_cfg, "runner_class_name", "OnPolicyRunner")
    if runner_class_name == "AmpOnPolicyRunner":
        ppo_runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "TrackAdapterRunner":
        ppo_runner = TrackAdapterRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "MultiCriticAmpOnPolicyRunner":
        ppo_runner = MultiCriticAmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "EncoderMultiCriticAmpRunner":
        ppo_runner = EncoderMultiCriticAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # soccer command term (holds target_dir_w / ball_vel_w / kick state)
    cmd = env.unwrapped.command_manager.get_term("soccer_kick")
    device = env.unwrapped.device
    N = env.num_envs

    # per-env episode accumulators
    latched_target_deg = torch.full((N,), float("nan"), device=device)  # commanded dir at kick
    prev_target_deg = torch.full((N,), float("nan"), device=device)      # commanded dir last step
    prev_ssk = torch.full((N,), -1, dtype=torch.long, device=device)     # steps_since_kick last step
    peak_speed = torch.zeros(N, device=device)                           # peak post-kick ball XY speed
    peak_kicked_deg = torch.full((N,), float("nan"), device=device)      # ball-vel dir at peak speed
    kicked_flag = torch.zeros(N, dtype=torch.bool, device=device)        # a kick happened this episode
    target_strength = torch.full((N,), float("nan"), device=device)      # commanded strength at kick

    records: list[dict] = []
    want = args_cli.num_episodes

    def _target_deg() -> torch.Tensor:
        d = cmd.target_dir_w[:, :2]
        return torch.rad2deg(torch.atan2(d[:, 1], d[:, 0]))

    obs, _ = env.get_observations()  # _LegacyRslRlEnv returns (policy_obs, extras)
    dones = None
    step = 0

    print(f"[INFO] Recording {want} kick episode(s) across {N} env(s)...")
    while len(records) < want and step < args_cli.max_steps and simulation_app.is_running():
        with torch.no_grad():
            if runner_class_name == "TrackAdapterRunner":
                actions = policy(obs, dones=dones)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        step += 1

        cur_deg = _target_deg()
        ssk = cmd.steps_since_kick.clone()

        # detect the kick transition (ssk: <0 -> >=0) and latch the pre-kick command
        kick_now = (prev_ssk < 0) & (ssk >= 0)
        if kick_now.any():
            idx = kick_now.nonzero(as_tuple=False).flatten()
            latched_target_deg[idx] = prev_target_deg[idx]
            target_strength[idx] = cmd.target_strength[idx]
            kicked_flag[idx] = True

        # track peak post-kick ball speed + its direction (the actual kicked dir)
        vel_xy = cmd.ball_vel_w[:, :2]
        speed = torch.linalg.norm(vel_xy, dim=-1)
        vel_deg = torch.rad2deg(torch.atan2(vel_xy[:, 1], vel_xy[:, 0]))
        improve = kicked_flag & (speed > peak_speed)
        if improve.any():
            jdx = improve.nonzero(as_tuple=False).flatten()
            peak_speed[jdx] = speed[jdx]
            peak_kicked_deg[jdx] = vel_deg[jdx]

        prev_target_deg = cur_deg.clone()
        prev_ssk = ssk

        # handle episode boundaries
        if dones is not None and dones.any():
            done_idx = dones.nonzero(as_tuple=False).flatten().tolist()
            for e in done_idx:
                had_kick = bool(kicked_flag[e].item())
                if had_kick or args_cli.include_no_kick:
                    if len(records) < want:
                        t_deg = float(latched_target_deg[e].item())
                        k_deg = float(peak_kicked_deg[e].item())
                        err = _wrap_deg(k_deg - t_deg) if (had_kick and not math.isnan(t_deg)) else float("nan")
                        rec = {
                            "episode": len(records) + 1,
                            "env": e,
                            "kicked": had_kick,
                            "target_deg": round(t_deg, 2) if not math.isnan(t_deg) else "",
                            "kicked_deg": round(k_deg, 2) if had_kick and not math.isnan(k_deg) else "",
                            "error_deg": round(err, 2) if not math.isnan(err) else "",
                            "target_strength": round(float(target_strength[e].item()), 2)
                            if not math.isnan(target_strength[e].item())
                            else "",
                            "peak_ball_speed": round(float(peak_speed[e].item()), 2),
                        }
                        records.append(rec)
                        print(
                            f"[EP {rec['episode']:>2}] env={e} kicked={had_kick} "
                            f"target={rec['target_deg']}deg kicked={rec['kicked_deg']}deg "
                            f"error={rec['error_deg']}deg strength={rec['target_strength']} "
                            f"peak_spd={rec['peak_ball_speed']}"
                        )
                # reset this env's accumulators (env auto-resets on done)
                latched_target_deg[e] = float("nan")
                prev_target_deg[e] = float("nan")
                prev_ssk[e] = -1
                peak_speed[e] = 0.0
                peak_kicked_deg[e] = float("nan")
                kicked_flag[e] = False
                target_strength[e] = float("nan")

    # write CSV
    out_path = args_cli.out or os.path.join(log_dir, "kick_directions.csv")
    fields = [
        "episode",
        "env",
        "kicked",
        "target_deg",
        "kicked_deg",
        "error_deg",
        "target_strength",
        "peak_ball_speed",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # summary
    errs = [r["error_deg"] for r in records if isinstance(r["error_deg"], (int, float))]
    print(f"\n[INFO] Recorded {len(records)} episode(s) -> {out_path}")
    if errs:
        abs_errs = [abs(e) for e in errs]
        mean_abs = sum(abs_errs) / len(abs_errs)
        print(
            f"[INFO] |error| mean={mean_abs:.1f}deg  min={min(abs_errs):.1f}deg  max={max(abs_errs):.1f}deg "
            f"(over {len(errs)} kicked episode(s))"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

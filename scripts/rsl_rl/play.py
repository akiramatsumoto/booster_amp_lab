# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--viser",
    action="store_true",
    default=False,
    help="Stream the policy to a viser web viewer (browser-based 3D view + joystick).",
)
parser.add_argument("--viser_host", type=str, default="0.0.0.0", help="Viser server bind host.")
parser.add_argument("--viser_port", type=int, default=8080, help="Viser server port.")
parser.add_argument(
    "--fixed_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Override base_velocity command every step during play.",
)
parser.add_argument("--disable_push", action="store_true", default=False, help="Disable interval push events for play.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, TrackAdapterRunner
try:
    from rsl_rl.runners import EncoderMultiCriticAmpRunner, MultiCriticAmpOnPolicyRunner
except ImportError:
    EncoderMultiCriticAmpRunner = None
    MultiCriticAmpOnPolicyRunner = None
# from rsl_rl.runners import  OnPolicyRunner


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx


def _split_obs(obs_td):
    groups = dict(obs_td.items()) if hasattr(obs_td, "items") else dict(obs_td)
    policy_obs = groups.pop("policy")
    return policy_obs, groups


class _UnwrappedProxy:
    """Holds the unwrapped base env so `proxy.env` returns it.

    The vendored rsl_rl runners reach into `self.env.env.env.step_dt`, counting on a
    specific wrapper depth. RecordVideo adds an extra wrapper when --video is on,
    which breaks that chain. This proxy normalizes it.
    """

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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # when recording, track the robot so it stays in frame regardless of its world position
    if args_cli.video:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (2.5, -2.5, 0.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_push and hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # enable command-term debug markers during play / recording (target / goal /
    # pass arrows + post-kick stop-mode sphere). They default to ``debug_vis=False``
    # so training stays fast; here we force them on so they show up in the viewer
    # and in recorded video.
    if getattr(env_cfg, "commands", None) is not None:
        for _term_name in vars(env_cfg.commands):
            _term = getattr(env_cfg.commands, _term_name)
            if hasattr(_term, "debug_vis"):
                _term.debug_vis = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        # Support both:
        #   --checkpoint /path/to/model.pt
        #   --load_run RUN --checkpoint model_5400.pt
        # IsaacLab's helper treats --checkpoint as a direct path here, so route
        # bare filenames through the experiment/run resolver.
        checkpoint = os.path.expanduser(args_cli.checkpoint)
        if os.path.exists(checkpoint):
            resume_path = retrieve_file_path(checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = _LegacyRslRlEnv(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_class_name = getattr(agent_cfg, "runner_class_name", "OnPolicyRunner")
    if runner_class_name == "AmpOnPolicyRunner":
        ppo_runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "TrackAdapterRunner":
        ppo_runner = TrackAdapterRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "MultiCriticAmpOnPolicyRunner":
        if MultiCriticAmpOnPolicyRunner is None:
            raise RuntimeError(
                "MultiCriticAmpOnPolicyRunner not available — install the vendored rsl_rl."
            )
        ppo_runner = MultiCriticAmpOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif runner_class_name == "EncoderMultiCriticAmpRunner":
        if EncoderMultiCriticAmpRunner is None:
            raise RuntimeError(
                "EncoderMultiCriticAmpRunner not available — install the vendored rsl_rl."
            )
        ppo_runner = EncoderMultiCriticAmpRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(ppo_runner, "obs_normalizer"):     # compatibility for older versions
        normalizer = ppo_runner.obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit. TrackAdapterActorCritic does not expose the
    # standard RSL-RL .actor/.student module. EncoderActorCritic exposes an
    # internal compressed actor input, so the generic exporter feeds the wrong
    # observation width. Skip export for visual play in these cases.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    run_name = os.path.basename(log_dir)
    if runner_class_name in ("TrackAdapterRunner", "EncoderMultiCriticAmpRunner"):
        print(f"[INFO] Skipping JIT/ONNX export for {runner_class_name} play.")
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir,
                             filename=f"{agent_cfg.experiment_name}_{run_name}.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=export_model_dir,
            filename=f"{agent_cfg.experiment_name}_{run_name}.onnx"
        )

    if args_cli.headless and not args_cli.video and not args_cli.viser:
        print("[INFO] Headless mode and no video recording. Exiting after model export.")
        env.close()
        return


    dt = env.unwrapped.step_dt

    # optional viser web viewer
    bridge = None
    cmd_term = None
    fixed_command = None
    if args_cli.fixed_command is not None:
        fixed_command = torch.tensor(args_cli.fixed_command, device=env.unwrapped.device, dtype=torch.float32)
        try:
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        except Exception:
            cmd_term = None
            print("[WARN] --fixed_command requested, but no base_velocity command term was found.")
    if args_cli.viser:
        from viser_bridge import BoosterViserBridge

        bridge = BoosterViserBridge(
            env.unwrapped, host=args_cli.viser_host, port=args_cli.viser_port,
        )
        try:
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        except Exception:
            cmd_term = None

    def _policy_obs_from_result(result):
        obs_result = result[0] if isinstance(result, tuple) else result
        if hasattr(obs_result, "items") and "policy" in obs_result:
            obs_result, _ = _split_obs(obs_result)
        return obs_result

    def _get_policy_obs():
        return _policy_obs_from_result(env.get_observations())

    def _zero_actions():
        return torch.zeros(
            (env.num_envs, env.num_actions),
            device=env.unwrapped.device,
            dtype=torch.float32,
        )

    def _force_episode_reset():
        # Do not call wrapper env.reset() from the live play loop. Force the
        # normal IsaacLab timeout/reset path instead, so reset behaves like an
        # episode boundary and keeps the app/viewer alive.
        base_env = env.unwrapped
        if hasattr(base_env, "episode_length_buf") and hasattr(base_env, "max_episode_length"):
            base_env.episode_length_buf[:] = int(base_env.max_episode_length)
            return env.step(_zero_actions())[0::2]

        reset_obs = _policy_obs_from_result(env.reset())
        reset_dones = torch.ones(env.num_envs, device=base_env.device, dtype=torch.long)
        return reset_obs, reset_dones

    # reset environment
    obs = _get_policy_obs()
    timestep = 0
    dones = None

    # Optional soccer-kick command term, used to log post-kick stop-mode state.
    soccer_cmd_term = None
    try:
        soccer_cmd_term = env.unwrapped.command_manager.get_term("soccer_kick")
    except Exception:
        soccer_cmd_term = None
    prev_ball_spd = None

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        if bridge is not None and bridge.consume_reset_request():
            with torch.no_grad():
                obs_result, dones = _force_episode_reset()
            obs = _policy_obs_from_result(obs_result)
            if bridge.apply_soccer_controls():
                obs = _get_policy_obs()
            bridge.update(force=True)
            sleep_time = dt - (time.time() - start_time)
            if (args_cli.real_time or args_cli.viser) and sleep_time > 0:
                time.sleep(sleep_time)
            continue

        if bridge is not None and bridge.paused:
            bridge.update()
            sleep_time = dt - (time.time() - start_time)
            if (args_cli.real_time or args_cli.viser) and sleep_time > 0:
                time.sleep(sleep_time)
            continue

        # Keep gradients off for policy play, but do not use inference_mode
        # around env.step(): reward/command terms mutate persistent tensors.
        with torch.no_grad():
            # apply joystick override before stepping so the policy sees it next step
            if bridge is not None and cmd_term is not None:
                bridge.apply_joystick(cmd_term)
            if bridge is not None and bridge.apply_soccer_controls():
                obs = _get_policy_obs()
            if fixed_command is not None and cmd_term is not None and hasattr(cmd_term, "vel_command_b"):
                cmd_term.vel_command_b[:] = fixed_command
                obs = _get_policy_obs()
            # agent stepping
            if runner_class_name == "TrackAdapterRunner":
                actions = policy(obs, dones=dones)
            else:
                actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if fixed_command is not None and cmd_term is not None and hasattr(cmd_term, "vel_command_b"):
                cmd_term.vel_command_b[:] = fixed_command
                obs = _get_policy_obs()
            if bridge is not None and bridge.apply_soccer_controls():
                obs = _get_policy_obs()
        # log post-kick stop-mode state + kick-contact diagnostics every step
        if soccer_cmd_term is not None and hasattr(soccer_cmd_term, "stop_mode"):
            stop_mode = soccer_cmd_term.stop_mode
            n_stop = int(stop_mode.sum().item())
            on_envs = stop_mode.nonzero(as_tuple=False).flatten().tolist()
            # diagnostics for env 0: why has contact (hence stop_mode) not latched?
            ball_speed = float(
                torch.linalg.norm(soccer_cmd_term.ball_vel_w[0, :2]).item()
            )
            foot_pos = soccer_cmd_term.robot.data.body_pos_w[0, soccer_cmd_term._foot_ids, :]
            min_fb = float(
                torch.linalg.norm(foot_pos - soccer_cmd_term.ball_pos_w[0], dim=-1).min().item()
            )
            contact = bool(soccer_cmd_term.kick_contact_awarded[0].item())
            ssk = int(soccer_cmd_term.steps_since_kick[0].item())
            # only log when the ball speed changes (rounded to 2 decimals)
            ball_spd_r = round(ball_speed, 2)
            if prev_ball_spd is None or ball_spd_r != prev_ball_spd:
                print(
                    f"[INFO] stop={n_stop}/{stop_mode.numel()} on={on_envs} | "
                    f"ball_spd={ball_speed:.2f} min_foot_ball={min_fb:.2f} "
                    f"contact={contact} ssk={ssk}"
                )
                prev_ball_spd = ball_spd_r

        if bridge is not None:
            bridge.update()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if (args_cli.real_time or args_cli.viser) and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if bridge is not None:
        bridge.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

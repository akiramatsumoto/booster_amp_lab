# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# experimental left-right symmetry + near-foot kicking (soccer kick task).
# Default ON for the soccer kick task; pass --no-symmetry / --no-near_foot_kick
# to disable. Both are gated to the soccer kick task, so they no-op elsewhere.
parser.add_argument(
    "--symmetry",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable left-right symmetry data augmentation (soccer kick task). Default on.",
)
parser.add_argument(
    "--near_foot_kick",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Reward kicking with the foot on the ball's spawn side (soccer kick task). Default on.",
)
parser.add_argument(
    "--near_foot_kick_weight",
    type=float,
    default=2.0,
    help="Reward weight for --near_foot_kick.",
)
# walk→kick transition: reset into sampled mid-walk states (soccer kick task)
parser.add_argument(
    "--walk_init",
    type=str,
    default=None,
    help="Path to a .pt walk-state dataset (from play.py --dump_states). When set, the "
    "soccer kick task resets a fraction of envs into a sampled mid-walk pose/velocity.",
)
parser.add_argument(
    "--walk_init_prob",
    type=float,
    default=1.0,
    help="Per-env probability of using a walk-init state vs the default standing pose.",
)
parser.add_argument(
    "--deploy_default_pose",
    action="store_true",
    default=False,
    help="Use the deploy DEFAULT_ANGLES pose as the standing reset pose (soccer kick task).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import faulthandler
import os
import signal
import torch
from datetime import datetime

import omni
from rsl_rl.runners import (
    AmpOnPolicyRunner,
    EncoderMultiCriticAmpRunner,
    MultiCriticAmpOnPolicyRunner,
    OnPolicyRunner,
    TrackAdapterRunner,
    WMPRunner,
)
# from rsl_rl.runners import  OnPolicyRunner


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg, RslRlVecEnvWrapper

# Import path (module:attr) of the soccer-kick left-right symmetry function.
_SYMMETRY_FUNC = (
    "booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.symmetry:compute_symmetric_states"
)


def _split_obs(obs_td):
    """Split a TensorDict of obs groups into (policy_tensor, observations_dict).

    The fork of rsl_rl used by this repo expects `env.get_observations()` to return
    `(obs, extras)` and `env.step()` to expose other groups under
    `infos["observations"]`. IsaacLab's modern RslRlVecEnvWrapper returns a
    single TensorDict of all groups, so adapt the shape here.
    """
    groups = dict(obs_td.items()) if hasattr(obs_td, "items") else dict(obs_td)
    policy_obs = groups.pop("policy")
    return policy_obs, groups


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


def _env_truthy_default(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _track_adapter_full_training_unlocked() -> bool:
    return (
        _env_truthy("BOOSTER_TRACK_ADAPTER_FULL_TRAINING")
        and _env_truthy("BOOSTER_TRACK_ADAPTER_SMOKE_PASSED")
        and _env_truthy("BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING")
    )


def _install_debug_signal_handlers():
    if not _env_truthy("BOOSTER_TRAIN_DEBUG_SIGNALS"):
        return
    faulthandler.enable(all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        print("[train-debug] SIGUSR1 will dump Python stacks.", flush=True)
    except RuntimeError as exc:
        print(f"[train-debug] Could not register SIGUSR1 faulthandler: {exc}", flush=True)


def _restore_curriculum_progress_from_runner(env, runner):
    """Restore stateless curriculum progress after loading a checkpoint.

    IsaacLab's ``common_step_counter`` belongs to the environment, not the
    RSL-RL checkpoint. Without this, resumed runs silently restart curricula
    from easy settings while the policy/iteration counter resumes from later
    training.
    """
    if not _env_truthy_default("BOOSTER_RESTORE_CURRICULUM_STEP", True):
        return

    base_env = getattr(env, "unwrapped", None)
    if base_env is None or not hasattr(base_env, "common_step_counter"):
        return

    loaded_iter = int(getattr(runner, "current_learning_iteration", 0) or 0)
    rollout_len = int(getattr(runner, "num_steps_per_env", 0) or 0)
    restored_step = loaded_iter * rollout_len
    if restored_step <= 0:
        return

    base_env.common_step_counter = restored_step
    print(
        "[INFO]: Restored environment curriculum step from checkpoint iteration: "
        f"iter={loaded_iter}, rollout_len={rollout_len}, common_step_counter={restored_step}"
    )

    if hasattr(base_env, "curriculum_manager"):
        base_env.curriculum_manager.compute(env_ids=None)
        try:
            active_terms = base_env.curriculum_manager.get_active_iterable_terms(0)
            print(f"[INFO]: Active curriculum after restore: {active_terms}")
        except Exception as exc:
            print(f"[WARN]: Could not print restored curriculum terms: {exc}")

    # The RSL-RL wrapper reset the env before checkpoint load, so its initial
    # command samples came from step 0 curriculum. Reset the full env once so
    # all managers, observations, command samples, scene state, and buffers are
    # consistent with the restored curriculum.
    env.reset()


class _LegacyRslRlEnv:
    """Adapter making the new RslRlVecEnvWrapper behave like the legacy one the fork expects."""

    def __init__(self, env: RslRlVecEnvWrapper):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    _install_debug_signal_handlers()
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if getattr(agent_cfg, "runner_class_name", None) == "TrackAdapterRunner":
        smoke_limit = int(getattr(agent_cfg, "max_iterations_without_full_training_unlock", 64))
        requested_iterations = int(getattr(agent_cfg, "gated_requested_max_iterations", agent_cfg.max_iterations))
        requested_iterations = max(requested_iterations, int(agent_cfg.max_iterations))
        full_unlocked = bool(
            getattr(agent_cfg, "allow_full_training", False)
        ) and _track_adapter_full_training_unlocked()
        if requested_iterations > smoke_limit and not full_unlocked:
            raise RuntimeError(
                "Refusing Track Adapter full training before gates are unlocked. "
                f"Requested {requested_iterations} iterations; smoke limit is {smoke_limit}. "
                "Run a bounded smoke first, then set BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1, "
                "BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1, and BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1 "
                "only after contact/residual/style/sudden-stop gates pass."
            )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # experimental toggles (soccer kick task) -------------------------------
    # Gate symmetry to the soccer kick task: ``_SYMMETRY_FUNC`` is soccer-specific,
    # so applying it to other tasks (now that the flag defaults on) would be wrong.
    _commands_cfg = getattr(env_cfg, "commands", None)
    _is_soccer_kick = _commands_cfg is not None and hasattr(_commands_cfg, "soccer_kick")
    if args_cli.symmetry and _is_soccer_kick:
        agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=_SYMMETRY_FUNC,
        )
        print("[INFO] Left-right symmetry data augmentation ENABLED.")
    elif args_cli.symmetry and not _is_soccer_kick:
        print("[INFO] --symmetry is soccer-kick-specific; skipping for this task.")
    if args_cli.near_foot_kick:
        rewards_cfg = getattr(env_cfg, "rewards", None)
        if rewards_cfg is not None and hasattr(rewards_cfg, "near_foot_kick"):
            rewards_cfg.near_foot_kick.weight = args_cli.near_foot_kick_weight
            print(
                f"[INFO] Near-foot kick reward ENABLED (weight={args_cli.near_foot_kick_weight})."
            )
        else:
            print("[WARN] --near_foot_kick set but env_cfg has no 'near_foot_kick' reward term; ignoring.")
    if args_cli.walk_init is not None:
        commands_cfg = getattr(env_cfg, "commands", None)
        soccer_cmd = getattr(commands_cfg, "soccer_kick", None) if commands_cfg is not None else None
        if soccer_cmd is not None and hasattr(soccer_cmd, "init_state_dataset_path"):
            soccer_cmd.init_state_dataset_path = args_cli.walk_init
            soccer_cmd.init_state_prob = args_cli.walk_init_prob
            print(
                f"[INFO] Walk-init ENABLED from {args_cli.walk_init!r} "
                f"(prob={args_cli.walk_init_prob})."
            )
        else:
            print("[WARN] --walk_init set but env_cfg has no 'soccer_kick' command term; ignoring.")
    if args_cli.deploy_default_pose:
        commands_cfg = getattr(env_cfg, "commands", None)
        soccer_cmd = getattr(commands_cfg, "soccer_kick", None) if commands_cfg is not None else None
        if soccer_cmd is not None and hasattr(soccer_cmd, "use_deploy_default_pose"):
            soccer_cmd.use_deploy_default_pose = True
            print("[INFO] Standing reset pose set to deploy DEFAULT_ANGLES.")
        else:
            print("[WARN] --deploy_default_pose set but env_cfg has no 'soccer_kick' command term; ignoring.")

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # adapt to legacy (obs, extras) API expected by the vendored rsl_rl fork
    env = _LegacyRslRlEnv(env)
    # create runner from rsl-rl
    runner_class_name = getattr(agent_cfg, "runner_class_name", "OnPolicyRunner")
    if runner_class_name == "AmpOnPolicyRunner":
        runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif runner_class_name == "MultiCriticAmpOnPolicyRunner":
        runner = MultiCriticAmpOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif runner_class_name == "EncoderMultiCriticAmpRunner":
        runner = EncoderMultiCriticAmpRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif runner_class_name == "TrackAdapterRunner":
        runner = TrackAdapterRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif runner_class_name == "WMPRunner":
        runner = WMPRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        load_optimizer = True
        if runner_class_name == "AmpOnPolicyRunner":
            load_optimizer = os.getenv("BOOSTER_AMP_LOAD_OPTIMIZER", "1").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not load_optimizer:
                print("[INFO]: Loading AMP checkpoint without optimizer state.")
        if runner_class_name == "TrackAdapterRunner":
            load_optimizer = os.getenv("BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER", "1").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not load_optimizer:
                print("[INFO]: Loading Track Adapter checkpoint without optimizer state.")
        if runner_class_name == "EncoderMultiCriticAmpRunner":
            load_optimizer = os.getenv("BOOSTER_ENCODER_LOAD_OPTIMIZER", "1").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not load_optimizer:
                print("[INFO]: Loading Encoder checkpoint without optimizer state.")
        runner.load(resume_path, load_optimizer=load_optimizer)
        _restore_curriculum_progress_from_runner(env, runner)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

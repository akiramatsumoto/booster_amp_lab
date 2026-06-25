import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "beyond_mimic"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class BaseLowFreqPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)

@configclass
class BaseAMPAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "run"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "run"
    wandb_project = "run"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = [os.path.join(_AMP_ROOT, "walk.txt")]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22


@configclass
class MultiCriticAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO-algorithm cfg with multi-critic extras (LVDRS §3.2)."""

    class_name: str = "MultiCriticAMPPPO"
    num_critic_groups: int = 2
    critic_group_names: tuple = ("goal", "aux")
    critic_group_weights: tuple = (2.0, 1.0)
    # Which critic group receives the AMP style reward. Defaults to "aux".
    amp_critic_group: str = "aux"


@configclass
class BaseMultiCriticAMPAgentCfg(BaseAMPAgentCfg):
    """Base runner cfg for multi-critic AMP-PPO. Inherits AMP setup; swaps in
    multi-critic actor-critic + algorithm + runner.
    """

    policy = RslRlPpoActorCriticCfg(
        class_name="MultiCriticActorCritic",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = MultiCriticAMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    runner_class_name = "MultiCriticAmpOnPolicyRunner"


@configclass
class EncoderCfg:
    """V4 Stage 5 encoder/decoder configuration.

    Slices and dims describe how :class:`EncoderActorCritic` projects the
    history segment of the actor/critic observation through a learned encoder
    and reconstructs a privileged target through a side-channel decoder.
    """

    # Contiguous half-open slice ``[start, end)`` into the actor obs that
    # contains the ball history (history_len * history_dim wide).
    history_slice: tuple = (0, 50)
    # Same slice expressed in critic-obs coords. ``None`` defaults to mirror
    # the actor-side slice.
    critic_history_slice: tuple | None = None
    history_len: int = 10
    history_dim: int = 5
    latent_dim: int = 64
    encoder_hidden_dims: tuple = (128, 64)
    # Width of the decoder output / privileged target slice.
    decoder_target_dim: int = 9
    decoder_hidden_dims: tuple = (64, 64)
    # Half-open slice into critic obs that the decoder must reconstruct.
    decoder_target_slice: tuple = (0, 9)
    decoder_loss_coef: float = 1.0


@configclass
class EncoderMultiCriticAMPAlgorithmCfg(MultiCriticAMPAlgorithmCfg):
    """Stage 5 algorithm cfg — adds decoder-aux-loss knobs."""

    class_name: str = "EncoderMultiCriticAMPPPO"


@configclass
class BaseEncoderMultiCriticAMPAgentCfg(BaseMultiCriticAMPAgentCfg):
    """Stage 5 base agent cfg — opt-in by inheriting from this class.

    The encoder is dormant until a task cfg subclasses this template and sets
    a task-specific ``encoder_cfg`` (history_slice etc.). Stage 1/2/3 task
    cfgs continue to inherit :class:`BaseMultiCriticAMPAgentCfg` and are not
    affected by this scaffolding.
    """

    policy = RslRlPpoActorCriticCfg(
        class_name="EncoderActorCritic",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = EncoderMultiCriticAMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    encoder_cfg: EncoderCfg = EncoderCfg()
    runner_class_name = "EncoderMultiCriticAmpRunner"
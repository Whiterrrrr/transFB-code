# pylint: disable=W0212

"""Module for holding abstract base classes for all agents."""

import abc
import numpy as np
import torch
import wandb
import dataclasses
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from utils import BASE_DIR
from os import makedirs

from agents.utils import TruncatedNormal, squashed_gaussian
from rewards import RewardFunctionConstructor


class AbstractAgent(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for all agents."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        name: str,
    ):
        super().__init__()
        self._observation_dimension = observation_length
        self._action_dimension = action_length
        self._name = name

    @property
    def observation_length(self) -> int:
        """Length of observation space used as input to agent."""
        return self._observation_dimension

    @property
    def action_length(self) -> int:
        """Length of action space used as input to agent."""
        return self._action_dimension

    @property
    def name(self) -> str:
        """
        Agent name.
        """
        return self._name

    @abc.abstractmethod
    def act(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns an action for a given input.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> Dict:
        """
        Updates parameters of model.
        """
        raise NotImplementedError

    def save(self, dir_path: Path) -> Path:
        """
        Saves a copy of the model in a format that can be loaded by load
        """
        dir_path.mkdir(exist_ok=True)
        save_path = dir_path / Path(str(self._name) + ".pickle")
        torch.save(self, save_path)

        return save_path

    @abc.abstractmethod
    def load(self, filepath: Path):
        pass


class AbstractMLP(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for all feedforward networks."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        preprocessor: bool = False,
        layernorm: bool = False,
    ):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._hidden_dimension = hidden_dimension
        self._hidden_layers = hidden_layers
        self._activation = activation
        self.device = device
        self._preprocessor = preprocessor
        self._layernorm = layernorm

        super().__init__()
        self.trunk = self._build()

    def _build(self) -> torch.nn.Sequential:
        """
        Creates MLP trunk.
        """
        if self.hidden_layers == 0:
            function = [torch.nn.Linear(self.input_dimension, self.output_dimension)]
        else:
            # first layer
            # ICLR paper uses layer norm and tanh for first layer of every network
            if self._layernorm:
                function = [
                    torch.nn.Linear(self.input_dimension, self.hidden_dimension),
                    torch.nn.LayerNorm(self.hidden_dimension),
                    torch.nn.Tanh(),
                ]
            else:
                function = [
                    torch.nn.Linear(self.input_dimension, self.hidden_dimension),
                    self.activation,
                ]

            # hidden layers
            for _ in range(self.hidden_layers - 1):
                function += [
                    torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
                    self.activation,
                ]

            # last layer
            function.append(
                torch.nn.Linear(self.hidden_dimension, self.output_dimension)
            )

        # add non-linearity to last layer for preprocessor
        if self.preprocessor:
            function.append(self.activation)

        trunk = torch.nn.Sequential(*function).to(self.device)

        return trunk

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    @property
    def hidden_dimension(self) -> int:
        return self._hidden_dimension

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    @property
    def activation(self) -> torch.nn:
        if self._activation == "relu":
            return torch.nn.ReLU()
        else:
            raise NotImplementedError(f"{self._activation} not implemented.")

    @property
    def preprocessor(self) -> bool:
        return self._preprocessor


class AbstractCritic(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract critic class."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = False,
    ):
        self._observation_length = observation_length
        self._action_length = action_length
        self._hidden_dimension = hidden_dimension
        self._hidden_layers = hidden_layers
        super().__init__(
            input_dimension=observation_length + action_length,
            output_dimension=int(1),
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            preprocessor=False,
            layernorm=layernorm,
        )

    def forward(self, observation_action: torch.Tensor) -> torch.Tensor:
        """
        Passes observation_action pair through network to predict q value
        Args:
            observation_action: tensor of shape
                                        [batch_dim, observation_length + action_length]

        Returns:
            q: q value tensor of shape [batch_dim, 1]
        """
        q = self.trunk(observation_action)  # pylint: disable=E1102

        return q


class DoubleQCritic(torch.nn.Module):
    """Critic network employing double Q learning."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = False,
    ):
        super().__init__()

        self.Q1 = AbstractCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )
        self.Q2 = AbstractCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )
        self.outputs = {}

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes obs-action pair through q functions.
        Args:
            observation: tensor of shape [batch_dimension, observation_length]
            action: tensor of shape [batch_dimension, action_length]

        Returns:
            q1: q value from first q function
            q2: q value from second q function
        """
        assert observation.size(0) == action.size(0)

        observation_action = torch.cat([observation, action], dim=-1)
        q1 = self.Q1.forward(observation_action)
        q2 = self.Q2.forward(observation_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2


class AbstractActor(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract actor that selects action given input."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=True,
        )

    def forward(
        self, observation: torch.Tensor, std: float
    ) -> torch.distributions.Distribution:
        """
        Passes input through network to predict action
        Args:
            observation: obs tensor of shape [batch_dim, input_length]
            std: standard deviation of action distribution
        Returns:
            action: action tensor of shape [batch_dim, action_length]
        """
        if observation.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Input shape {observation.shape} does not "
                f"match input dimension {self.input_dimension}"
            )

        mu = self.trunk(observation)  # pylint: disable=E1102
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)

        return dist


class AbstractGaussianActor(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract gaussian actor that selects action given input."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        log_std_bounds: Tuple[float] = (-5.0, 2.0),
    ):

        self.log_std_min = log_std_bounds[0]
        self.log_std_max = log_std_bounds[1]

        super().__init__(
            input_dimension=observation_length,
            output_dimension=action_length * 2,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=False,
        )

    def forward(self, observation: torch.Tensor, sample=True):
        """
        Takes observation and returns squashed normal distribution over action space.
        Args:
            observation: tensor of shape [batch_dim, observation_length]

        Returns:
            dist: SquashedNormal (multivariate Gaussian) dist over action space.

        """
        # mu, log_std = self.trunk(observation).chunk(2, dim=-1)  # pylint: disable=E1102
        output = self.trunk(observation)
        action, action_dict = squashed_gaussian(x=output, sample=sample)

        return action, action_dict


class AbstractLogger(metaclass=abc.ABCMeta):
    """
    Abstract class for collecting metrics from training
    / eval runs.
    """

    def __init__(
        self, agent_config: Dict, use_wandb: bool = False, wandb_tags: List[str] = None
    ):
        self._agent_config = agent_config
        self.metrics = {}  # overwritten in concrete class

        if use_wandb:
            wandb.init(
                config=agent_config,
                tags=wandb_tags,
                reinit=True,
            )

    def log(self, metrics: Dict[str, float]):
        """Adds metrics to logger."""

        for key, value in metrics.items():
            try:
                self.metrics[key].append(value)
            except KeyError:
                raise KeyError(  # pylint: disable=W0707
                    f"Metric {key} not in metrics dictionary."
                )  # pylint: disable=W0707

        if wandb.run is not None:
            wandb.log(metrics)


@dataclasses.dataclass
class Batch:
    """
    Dataclass for batches of offline data.

    Args:
        observations: observations from current step in trajectory
        next_observations: observations from next step in trajectory
        other_observations: observations from anywhere in the dataset
        future_observations: observations from an arbitrary *future* step in trajectory
        discounts: future state discounts
        actions: actions from current step in trajectory
        rewards: rewards from transition
        not_dones: not done flags from transition
        physics: dm_control physics parameters
        goals: goal at current step in trajectory
        next_goals: goal at current step in trajectory
        future_goals: goals from an arbitrary *future* step in trajectory
    """

    observations: torch.Tensor
    next_observations: torch.Tensor
    discounts: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    not_dones: torch.Tensor
    other_observations: Optional[torch.Tensor] = None
    future_observations: Optional[torch.Tensor] = None
    physics: Optional[torch.Tensor] = None
    goals: Optional[torch.Tensor] = None
    next_goals: Optional[torch.Tensor] = None
    future_goals: Optional[torch.Tensor] = None


class AbstractReplayBuffer(metaclass=abc.ABCMeta):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def __init__(self, device: torch.device):
        self.device = device

    @abc.abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size: int) -> Dict:
        raise NotImplementedError


class AbstractOnlineReplayBuffer(AbstractReplayBuffer, metaclass=abc.ABCMeta):
    """Abstract buffer for online RL algorithms."""

    def __init__(
        self,
        capacity: int,
        observation_length: int,
        action_length: int,
        device: torch.device,
    ):
        super().__init__(device=device)
        self.observations = NotImplementedError("observations array not defined.")
        self.next_observations = NotImplementedError(
            "next_observations array not defined."
        )
        self.actions = NotImplementedError("actions array not defined.")
        self.rewards = NotImplementedError("rewards array not defined.")
        self.dones = NotImplementedError("dones array not defined.")
        self.current_memory_index = NotImplementedError(
            "current memory index not defined."
        )
        self.full_memory = NotImplementedError("full memory flag not implemented.")

        # properties
        self._capacity = capacity
        self._observation_length = observation_length
        self._action_length = action_length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def observation_length(self) -> int:
        return self._observation_length

    @property
    def action_length(self) -> int:
        return self._action_length


class AbstractOfflineReplayBuffer(AbstractReplayBuffer, metaclass=abc.ABCMeta):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def __init__(self, device: torch.device, transitions: int):
        super().__init__(device)

        self._transitions = transitions
        self.storage = NotImplementedError("Storage not implemented in base class.")

    @abc.abstractmethod
    def load_offline_dataset(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @property
    def transitions(self) -> int:
        """Number of transitions to sample into buffer from dataset."""
        return self._transitions


class OfflineReplayBuffer(AbstractOfflineReplayBuffer):
    """Forward backward replay buffer."""

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        discount: float,
        device: torch.device,
        task: str,
        relabel: bool = True,
        transitions: int = None,
        action_condition: dict = None,
    ):
        super().__init__(device=device, transitions=transitions)

        self._discount = discount

        self.storage = {}

        # load dataset on init
        self.load_offline_dataset(
            reward_constructor=reward_constructor,
            dataset_path=dataset_path,
            relabel=relabel,
            task=task,
            action_condition=action_condition,
        )

    def load_offline_dataset(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        relabel: bool = True,
        task: str = None,
        action_condition: dict = None,
    ) -> None:
        """
        Load the offline dataset into the replay buffer.
        Args:
            reward_constructor: DMC environments (used for relabeling)
            dataset_path: path to the dataset
            relabel: whether to relabel the dataset
            task: task for reward relabeling
            action_condition: dict (action index: action value), we assume the
                            action index must always be higher than action value
        Returns:
            None
        """

        # load offline dataset in the form of episode paths
        episodes = np.load(dataset_path, allow_pickle=True)
        episodes = dict(episodes)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        discounts = []
        not_dones = []
        physics = []

        # load the episodes
        for _, episode in tqdm(episodes.items(), desc="Loading episodes from buffer"):

            episode = episode.item()

            # relabel the episode
            if relabel:
                episode = self._relabel_episode(reward_constructor, episode, task)

            # store in lists
            observations.append(
                torch.as_tensor(episode["observation"][:-1], device=self.device)
            )
            actions.append(torch.as_tensor(episode["action"][1:], device=self.device))
            rewards.append(torch.as_tensor(episode["reward"][1:], device=self.device))
            next_observations.append(
                torch.as_tensor(episode["observation"][1:], device=self.device)
            )
            discounts.append(
                torch.as_tensor(
                    episode["discount"][1:] * self._discount, device=self.device
                )
            )
            physics.append(np.array(episode["physics"][:-1]))
            # hack the dones (we know last transition is terminal)
            not_done = torch.ones_like(
                torch.tensor(episode["reward"]), device=self.device
            )
            not_done[-1] = 0
            not_dones.append(not_done)

        # the below creates a "local" random number generator with fixed seed that
        # always subsamples the same transitions from the dataset, even if the
        # global seed is changed
        rng = np.random.default_rng(42)
        dataset_length = sum(len(obs) for obs in observations)

        if self.transitions is None:
            logger.info(
                f"Sampling {dataset_length} transitions from"
                f" dataset of length {dataset_length}"
            )
            sample_indices = rng.choice(dataset_length, dataset_length, replace=False)
        else:
            logger.info(
                f"Sampling {self.transitions} transitions from"
                f" dataset of length {dataset_length}"
            )
            sample_indices = rng.choice(dataset_length, self.transitions, replace=False)

        # concatenate into storage
        self.storage["observations"] = torch.cat(observations)[sample_indices]
        rand_idx = torch.randperm(self.storage["observations"].size(0))
        self.storage["other_observations"] = self.storage["observations"][rand_idx]
        self.storage["actions"] = torch.cat(actions)[sample_indices]
        # self.storage["other_actions"] = self.storage["actions"][rand_idx]
        self.storage["rewards"] = torch.cat(rewards)[sample_indices]
        self.storage["next_observations"] = torch.cat(next_observations)[sample_indices]
        self.storage["discounts"] = torch.cat(discounts)[sample_indices]
        self.storage["physics"] = np.concatenate(physics)[sample_indices]
        self.storage["not_dones"] = torch.cat(not_dones)[sample_indices]

        # sub sample only the transitions that satisfy the action condition
        if action_condition is not None:
            for key, value in action_condition.items():
                action_condition_idxs = (
                    torch.where(self.storage["actions"][:, key] > value)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                break

            self.storage["observations"] = self.storage["observations"][
                action_condition_idxs
            ]
            self.storage["actions"] = self.storage["actions"][action_condition_idxs]
            self.storage["rewards"] = self.storage["rewards"][action_condition_idxs]
            self.storage["next_observations"] = self.storage["next_observations"][
                action_condition_idxs
            ]
            self.storage["discounts"] = self.storage["discounts"][action_condition_idxs]
            self.storage["physics"] = self.storage["physics"][action_condition_idxs]
            self.storage["not_dones"] = self.storage["not_dones"][action_condition_idxs]

    @staticmethod
    def _relabel_episode(
        reward_constructor: RewardFunctionConstructor,
        episode: Dict[str, np.ndarray],
        task: str,
    ) -> np.array:
        """
        Takes episode data and relabels rewards w.r.t. the task.
        Args:
            reward_constructor: DMC environments (used for relabeling)
            episode: episode data
            task: task for reward relabeling
        Returns
            episode: the relabeled episode
        """

        env = reward_constructor._env

        task_idx = reward_constructor.task_names.index(task)
        episode = deepcopy(episode)

        rewards = []
        states = episode["physics"]

        # cycle through the states and relabel
        for i in range(states.shape[0]):
            with env.physics.reset_context():
                env.physics.set_state(states[i])
            task_rewards = reward_constructor(env.physics)
            # print(task_rewards)
            reward = np.full((1,), task_rewards[task_idx], dtype=np.float32)
            rewards.append(reward)

        episode["reward"] = np.array(rewards, dtype=np.float32)

        return episode

    def sample(self, batch_size: int) -> Batch:
        """
        Samples OfflineBatch from the replay buffer.
        Args:
            batch_size: the batch size
        Returns:
            Batch: the batch of transitions
        """

        if len(self.storage) == 0:
            raise RuntimeError("The replay buffer is empty.")

        batch_indices = torch.randint(
            0, len(self.storage["observations"]), (batch_size,)
        )

        return Batch(
            observations=self.storage["observations"][batch_indices],
            other_observations = self.storage["other_observations"][batch_indices],
            actions=self.storage["actions"][batch_indices],
            # other_actions = self.storage["other_actions"][batch_indices],
            rewards=self.storage["rewards"][batch_indices],
            next_observations=self.storage["next_observations"][batch_indices],
            discounts=self.storage["discounts"][batch_indices],
            not_dones=self.storage["not_dones"][batch_indices],
            physics=self.storage["physics"][batch_indices],
        )

    def add(self, *args, **kwargs):
        pass


class AbstractWorkspace(metaclass=abc.ABCMeta):
    """
    Abstract workspace for training and evaluating agents
    in an environment.
    """

    def __init__(self, env, reward_functions):
        self.env = env
        self.reward_functions = reward_functions

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, *args, **kwargs):
        raise NotImplementedError

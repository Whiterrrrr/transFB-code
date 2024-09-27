"""Module that creates workspaces for training/evaling various agents."""

import wandb
import torch
import shutil
from os import makedirs
from loguru import logger
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional
from scipy import stats

from rewards import RewardFunctionConstructor
from custom_dmc_tasks.point_mass_maze import GOALS as point_mass_maze_goals

from agents.base import AbstractWorkspace
from agents.fb.agent import FB
from agents.fb.replay_buffer import FBReplayBuffer, OnlineFBReplayBuffer

from agents.sac.agent import SAC
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer

from agents.cql.agent import CQL
from agents.base import OfflineReplayBuffer
from agents.ifb.agent import IFB
from agents.cfb.agent import CFB
from agents.calfb.agent import CalFB
from agents.cexp.agent import CEXP
from agents.exp.agent import EXP
from agents.lolocexp.agent import LOLOCEXP

class OnlineRLWorkspace(AbstractWorkspace):
    """
    Trains/evals/rollouts online RL algorithm on one task
    """

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        seed_steps: int,
        wandb_logging: bool = True,
    ):
        super().__init__(
            env=reward_constructor._env,
            reward_functions=reward_constructor.reward_functions,
        )

        self.eval_frequency = eval_frequency  # how frequently to eval
        self.eval_rollouts = eval_rollouts  # how many rollouts per eval step
        self.model_dir = model_dir
        self.learning_steps = learning_steps
        self.seed_steps = seed_steps
        self.wandb_logging = wandb_logging

    def train(
        self,
        agent: SAC,
        task: str,
        agent_config: Dict,
        replay_buffer: SoftActorCriticReplayBuffer,
    ):
        """
        Trains online RL algorithm on one task.
        """
        if self.wandb_logging:
            run = wandb.init(
                entity='1155173723',
                project="zero-shot", 
                config=agent_config,
                tags=[agent.name],
                reinit=True,
            )

            model_path = self.model_dir / run.name
            makedirs(str(model_path))

        else:
            date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        logger.info(f"Training {agent.name}")
        best_eval_reward = -np.inf
        best_model_path = None
        done = True

        for i in tqdm(range(self.learning_steps + 1)):

            # reset env
            if done:
                timestep = self.env.reset()

            # sample actions uniformly for seed steps
            if i < self.seed_steps:
                action = np.random.uniform(
                    low=-1, high=1, size=(self.env.action_spec().shape[0],)
                )

            else:
                action = agent.act(
                    timestep.observation["observations"],
                    sample=True,
                    replay_buffer=replay_buffer,
                )

            observation = timestep.observation["observations"]
            timestep = self.env.step(action)
            reward = self.reward_functions[task](self.env.physics)
            done = timestep.last()

            replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=timestep.observation["observations"],
                done=done,
            )

            eval_metrics = {}
            if (i % self.eval_frequency == 0) & (i > 0):
                eval_metrics = self.eval(agent=agent, task=task)
                if eval_metrics["eval/episode_reward_iqm"] > best_eval_reward:

                    # delete current best model
                    if best_model_path is not None:
                        best_model_path.unlink(missing_ok=True)

                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/episode_reward_iqm']:.3f}."
                        f" Saving model."
                    )

                    name = f"{task}_{i}.pickle"
                    # save locally
                    best_model_path = agent.save(model_path / name)

                    best_eval_reward = eval_metrics["eval/episode_reward_iqm"]

                agent.train()

            train_metrics = {}
            if (i % agent.actor_update_frequency == 0) and (i > self.seed_steps):
                batch = replay_buffer.sample(agent.batch_size)
                train_metrics = agent.update(batch=batch, step=i)

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                run.log(metrics)

        if self.wandb_logging:
            # upload best model to wandb at end of training
            run.save(best_model_path.as_posix(), base_path=model_path.as_posix())
            run.finish()

        # delete local models
        shutil.rmtree(model_path)

    def eval(
        self,
        agent: SAC,
        task: str,
    ) -> Dict[str, float]:
        logger.info("Performing eval rollouts.")
        eval_rewards = []
        agent.eval()
        for _ in tqdm(range(self.eval_rollouts)):

            rollout_reward = 0.0
            timestep = self.env.reset()
            while not timestep.last():
                action = agent.act(
                    timestep.observation["observations"], sample=False, step=None
                )
                timestep = self.env.step(action)
                rollout_reward += self.reward_functions[task](self.env.physics)

            eval_rewards.append(rollout_reward)

        metrics = {
            "eval/episode_reward_iqm": float(stats.trim_mean(eval_rewards, 0.25))
        }

        return metrics


class OfflineRLWorkspace(AbstractWorkspace):
    """
    Trains/evals/rollouts an offline RL agent given
    """

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        device: torch.device,
        z_inference_steps: Optional[int] = None,  # FB only
        train_std: Optional[float] = None,  # FB only
        eval_std: Optional[float] = None,  # FB only
        project: str = "zero-shot",
    ):
        super().__init__(
            env=reward_constructor._env,
            reward_functions=reward_constructor.reward_functions,
        )

        self.eval_frequency = eval_frequency  # how frequently to eval
        self.eval_rollouts = eval_rollouts  # how many rollouts per eval step
        self.model_dir = model_dir
        self.learning_steps = learning_steps
        self.z_inference_steps = z_inference_steps
        self.train_std = train_std
        self.eval_std = eval_std
        self.observations_z = None
        self.rewards_z = None
        self.wandb_logging = wandb_logging
        self.domain_name = reward_constructor.domain_name
        self.device = device
        self.project = project

    def train(
        self,
        agent: Union[CQL, SAC, FB, CFB, CalFB, CEXP, EXP, LOLOCEXP],
        tasks: List[str],
        agent_config: Dict,
        replay_buffer: Union[OfflineReplayBuffer, FBReplayBuffer],
    ) -> None:
        """
        Trains an offline RL algorithm on one task.
        """
        date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
        if self.wandb_logging:
            run = wandb.init(
                config=agent_config,
                tags=[agent.name, "core"],
                reinit=True,
                entity='l_air',
                project="zero-shot", 
                name=agent.name,
            )
            model_path = Path(self.model_dir + '/' + run.name + date)
            makedirs(str(model_path))
        else:
            model_path = Path(self.model_dir +'/'+ f"local-run-{date}")
            makedirs(str(model_path))

        logger.info(f"Training {agent.name}.")
        best_mean_task_reward = -np.inf
        best_model_path = None

        # sample set transitions for z inference
        if isinstance(agent, FB) or isinstance(agent, CEXP) or isinstance(agent, EXP) or isinstance(agent, IFB) or isinstance(agent, LOLOCEXP):
            if self.domain_name == "point_mass_maze":
                self.goal_states = {}
                for task, goal_state in point_mass_maze_goals.items():
                    self.goal_states[task] = torch.tensor(
                        goal_state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
            else:
                (
                    self.observations_z,
                    self.rewards_z,
                ) = replay_buffer.sample_task_inference_transitions(
                    inference_steps=self.z_inference_steps
                )

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(agent.batch_size)
            if isinstance(agent, CEXP) or isinstance(agent, EXP) or isinstance(agent, LOLOCEXP):
                batch_rand = replay_buffer.sample(agent.batch_size)
                train_metrics = agent.update(batch=batch, batch_rand=batch_rand, step=i)
            else:
                train_metrics = agent.update(batch=batch, step=i)
            
            eval_metrics = {}

            if i % self.eval_frequency == 0:
                eval_metrics = self.eval(agent=agent, tasks=tasks)

                if eval_metrics["eval/task_reward_iqm"] > best_mean_task_reward:
                    logger.info(
                        f"New max IQM task reward: {best_mean_task_reward:.3f} -> "
                        f"{eval_metrics['eval/task_reward_iqm']:.3f}."
                        f" Saving model in {model_path}."
                    )

                    # delete current best model
                    if best_model_path is not None:
                        best_model_path.unlink(missing_ok=True)

                    agent._name = i  # pylint: disable=protected-access
                    # save locally
                    best_model_path = agent.save(model_path)

                    best_mean_task_reward = eval_metrics["eval/task_reward_iqm"]

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                run.log(metrics)

        if self.wandb_logging:
            # save to wandb_logging
            run.save(best_model_path.as_posix(), base_path=model_path.as_posix())
            run.finish()

        # delete local models
        shutil.rmtree(model_path)

    def eval(
        self,
        agent: Union[CQL, SAC, FB, CFB, CalFB],
        tasks: List[str],
    ) -> Dict[str, float]:
        """
        Performs eval rollouts.
        Args:
            agent: agent to evaluate
            tasks: tasks to evaluate on
        Returns:
            metrics: dict of metrics
        """

        if isinstance(agent, FB) or isinstance(agent, CEXP) or isinstance(agent, EXP) or isinstance(agent, IFB) or isinstance(agent, LOLOCEXP):
            zs = {}
            metrics = {}
            if self.domain_name == "point_mass_maze":
                for task, goal_state in self.goal_states.items():
                    zs[task] = agent.infer_z(goal_state)
                    metrics[f"train/{task}_infer_z"] = zs[task].mean().item()
            else:
                for task, rewards in self.rewards_z.items():
                    zs[task] = agent.infer_z(self.observations_z, rewards)
                    metrics[f"train/{task}_infer_z"] = zs[task].mean().item()

            agent.std_dev_schedule = self.eval_std

        logger.info("Performing eval rollouts.")
        eval_rewards = {}
        agent.eval()
        for _ in tqdm(range(self.eval_rollouts)):

            for task in tasks:
                task_rewards = 0.0

                timestep = self.env.reset()
                while not timestep.last():
                    if isinstance(agent, FB) or isinstance(agent, CEXP) or isinstance(agent, EXP) or isinstance(agent, IFB) or isinstance(agent, LOLOCEXP):
                        action, _ = agent.act(
                            timestep.observation["observations"],
                            task=zs[task],
                            step=None,
                            sample=False,
                        )
                    else:
                        action = agent.act(
                            timestep.observation["observations"],
                            sample=False,
                            step=None,
                        )
                    timestep = self.env.step(action)
                    task_rewards += self.reward_functions[task](self.env.physics)

                if task not in eval_rewards:
                    eval_rewards[task] = []
                eval_rewards[task].append(task_rewards)

        # average over rollouts for metrics
        mean_task_performance = 0.0
        for task, rewards in eval_rewards.items():
            mean_task_reward = stats.trim_mean(rewards, 0.25)  # IQM
            metrics[f"eval/{task}/episode_reward_iqm"] = mean_task_reward
            mean_task_performance += mean_task_reward

        # log mean task performance
        metrics["eval/task_reward_iqm"] = mean_task_performance / len(tasks)

        if isinstance(agent, FB) or isinstance(agent, CEXP) or isinstance(agent, EXP) or isinstance(agent, IFB) or isinstance(agent, LOLOCEXP):
            agent.std_dev_schedule = self.train_std

        return metrics


class FinetuningWorkspace(OfflineRLWorkspace):
    """
    Finetunes FB or CFB on one task.
    """

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        online: bool,
        critic_tuning: bool,
        device: torch.device,
        z_inference_steps: Optional[int] = None,  # FB only
        train_std: Optional[float] = None,  # FB only
        eval_std: Optional[float] = None,  # FB only
    ):
        super().__init__(
            reward_constructor=reward_constructor,
            learning_steps=learning_steps,
            model_dir=model_dir,
            eval_frequency=eval_frequency,
            eval_rollouts=eval_rollouts,
            wandb_logging=wandb_logging,
            device=device,
            z_inference_steps=z_inference_steps,
            train_std=train_std,
            eval_std=eval_std,
        )

        self.online = online
        self.critic_tuning = critic_tuning

    def train(
        self,
        agent: Union[FB, CFB, CalFB],
        tasks: List[str],
        agent_config: Dict,
        replay_buffer: Union[FBReplayBuffer, OnlineFBReplayBuffer],
        episodes: int = None,
    ) -> None:

        assert len(tasks) == 1

        if self.online:
            self.tune_online(
                agent=agent,
                task=tasks,
                agent_config=agent_config,
                replay_buffer=replay_buffer,
                episodes=episodes,
            )

        else:
            self.tune_offline(
                agent=agent,
                task=tasks,
                agent_config=agent_config,
                replay_buffer=replay_buffer,
            )

    def tune_offline(
        self,
        agent: Union[FB, CFB, CalFB],
        task: List[str],
        agent_config: Dict,
        replay_buffer: FBReplayBuffer,
    ) -> None:
        """
        Finetunes FB or CFB on one task offline, without online interaction.
        Args:
            agent: agent to finetune
            task: task to finetune on
            agent_config: agent config
            replay_buffer: replay buffer for z sampling
        """

        if self.wandb_logging:
            run = wandb.init(
                config=agent_config,
                tags=[agent.name, "finetuning"],
                reinit=True,
                entity='1155173723',
                project="zero-shot", 
            )

        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        # get observations and rewards for task inference
        if self.domain_name == "point_mass_maze":
            self.goal_states = {}

            goal_state = point_mass_maze_goals[task[0]]
            self.goal_states[task[0]] = torch.tensor(
                goal_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            (
                self.observations_z,
                self.rewards_z,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=self.z_inference_steps
            )

        best_mean_task_reward = -np.inf

        # get initial eval metrics
        logger.info("Getting init performance.")
        eval_metrics = self.eval(agent=agent, tasks=task)
        init_performance = eval_metrics["eval/task_reward_iqm"]

        logger.info(f"Finetuning {agent.name} on {self.domain_name}-{task[0]}.")

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(agent.batch_size)

            # infer z for task
            if self.domain_name == "point_mass_maze":
                z = agent.infer_z(self.goal_states[task[0]])
            else:
                z = agent.infer_z(self.observations_z, self.rewards_z[task[0]])

            z_batch = torch.tile(
                torch.as_tensor(z, dtype=torch.float32, device=self.device),
                (agent.batch_size, 1),
            )  # repeat z for batch size

            if self.critic_tuning:
                fb_metrics = agent.update_fb(
                    observations=batch.observations,
                    next_observations=batch.next_observations,
                    actions=batch.actions,
                    discounts=batch.discounts,
                    zs=z_batch,
                    step=i,
                )
                actor_metrics = agent.update_actor(
                    observation=batch.observations, z=z_batch, step=i
                )

                agent.soft_update_params(
                    network=agent.FB.forward_representation,
                    target_network=agent.FB.forward_representation_target,
                    tau=agent._tau,  # pylint: disable=protected-access
                )
                agent.soft_update_params(
                    network=agent.FB.backward_representation,
                    target_network=agent.FB.backward_representation_target,
                    tau=agent._tau,  # pylint: disable=protected-access
                )
                if agent.name in ("VCalFB", "MCalFB"):
                    agent.soft_update_params(
                        network=agent.FB.forward_mu,
                        target_network=agent.FB.forward_mu_target,
                        tau=agent._tau,  # pylint: disable=protected-access
                    )

                train_metrics = {**fb_metrics, **actor_metrics}

            else:
                train_metrics = agent.update_actor(
                    observation=batch.observations, z=z_batch, step=i
                )

            eval_metrics = {}

            if i % self.eval_frequency == 0:
                eval_metrics = self.eval(agent=agent, tasks=task)
                eval_metrics["eval/init_performance"] = init_performance

                if eval_metrics["eval/task_reward_iqm"] > best_mean_task_reward:
                    logger.info(
                        f"Finetuned performance:"
                        f"{eval_metrics['eval/task_reward_iqm']:.1f} |"
                        f" Init performance:"
                        f"{eval_metrics['eval/init_performance']:.1f}"
                    )

                    best_mean_task_reward = eval_metrics["eval/task_reward_iqm"]

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                run.log(metrics)

        if self.wandb_logging:
            # save to wandb_logging
            run.finish()

    def tune_online(
        self,
        agent: Union[FB, CFB, CalFB],
        task: List[str],
        agent_config: Dict,
        replay_buffer: OnlineFBReplayBuffer,
        episodes: int,
    ) -> None:
        """
        Finetunes FB or CFB on one task using online data.
        Args:
            agent: agent to finetune
            task: task to finetune on
            agent_config: agent config
            replay_buffer: replay buffer for z sampling
            episodes: number of episodes to finetune for
        """

        if self.wandb_logging:
            run = wandb.init(
                config=agent_config,
                tags=[agent.name, "finetuning"],
                reinit=True,
                entity='1155173723',
                project="zero-shot", 
            )

        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        # get observations and rewards for task inference
        if self.domain_name == "point_mass_maze":
            self.goal_states = {}

            goal_state = point_mass_maze_goals[task[0]]
            self.goal_states[task[0]] = torch.tensor(
                goal_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            (
                self.observations_z,
                self.rewards_z,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=self.z_inference_steps
            )

        # get initial eval metrics
        logger.info("Getting init performance.")
        eval_metrics = self.eval(agent=agent, tasks=task)
        init_performance = eval_metrics["eval/task_reward_iqm"]
        best_mean_task_reward = -np.inf

        logger.info(f"Online finetuning {agent.name} on {self.domain_name}-{task[0]}.")
        j = 0
        for i in tqdm(range(episodes)):

            # interact with env
            timestep = self.env.reset()
            while not timestep.last():

                # infer z for task
                if self.domain_name == "point_mass_maze":
                    z = agent.infer_z(self.goal_states[task[0]])
                else:
                    z = agent.infer_z(self.observations_z, self.rewards_z[task[0]])

                action, _ = agent.act(
                    timestep.observation["observations"],
                    task=z,
                    step=None,
                    sample=True,
                )

                observation = timestep.observation["observations"]
                timestep = self.env.step(action)
                reward = self.reward_functions[task[0]](self.env.physics)
                done = timestep.last()
                j += 1

                replay_buffer.add(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=timestep.observation["observations"],
                    done=done,
                )

                # start learning once batch size is reached
                if j >= agent.batch_size:
                    batch = replay_buffer.sample(agent.batch_size)

                    z_batch = torch.tile(
                        torch.as_tensor(z, dtype=torch.float32, device=self.device),
                        (agent.batch_size, 1),
                    )  # repeat z for batch size

                    if self.critic_tuning:
                        fb_metrics = agent.update_fb(
                            observations=batch.observations,
                            next_observations=batch.next_observations,
                            actions=batch.actions,
                            discounts=batch.discounts,
                            zs=z_batch,
                            step=i,
                        )
                        actor_metrics = agent.update_actor(
                            observation=batch.observations, z=z_batch, step=i
                        )

                        agent.soft_update_params(
                            network=agent.FB.forward_representation,
                            target_network=agent.FB.forward_representation_target,
                            tau=agent._tau,  # pylint: disable=protected-access
                        )
                        agent.soft_update_params(
                            network=agent.FB.backward_representation,
                            target_network=agent.FB.backward_representation_target,
                            tau=agent._tau,  # pylint: disable=protected-access
                        )

                        if agent.name in ("VCalFB", "MCalFB"):
                            agent.soft_update_params(
                                network=agent.FB.forward_mu,
                                target_network=agent.FB.forward_mu_target,
                                tau=agent._tau,  # pylint: disable=protected-access
                            )

                        train_metrics = {**fb_metrics, **actor_metrics}

                    else:
                        train_metrics = agent.update_actor(
                            observation=batch.observations, z=z_batch, step=i
                        )
                else:
                    train_metrics = {}

                if j % self.eval_frequency == 0:
                    eval_metrics = self.eval(agent=agent, tasks=task)
                    eval_metrics["eval/init_performance"] = init_performance

                    if eval_metrics["eval/task_reward_iqm"] > best_mean_task_reward:
                        logger.info(
                            f"Finetuned performance:"
                            f"{eval_metrics['eval/task_reward_iqm']:.1f} |"
                            f" Init performance:"
                            f"{eval_metrics['eval/init_performance']:.1f}"
                        )

                        best_mean_task_reward = eval_metrics["eval/task_reward_iqm"]

                    agent.train()
                else:
                    eval_metrics = {}

                metrics = {**train_metrics, **eval_metrics}

                if self.wandb_logging:
                    run.log(metrics)

        if self.wandb_logging:
            # save to wandb_logging
            run.finish()

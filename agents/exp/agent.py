import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

from agents.fb.models import ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule
from agents.exp.module import deepNetApproximator


class EXP(AbstractAgent):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_output_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_number_of_features: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        backward_activation: str,
        backward_preprocess: bool,
        backward_preprocess_hidden_dimension: int,
        backward_preprocess_hidden_layers: int,
        backward_preprocess_output_dim: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        critic_learning_rate: float,
        orthonormalisation_coefficient: float,
        q_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        total_action_samples: int,
        ood_action_weight: float,        
        alpha: float,
        target_conservative_penalty: float,
        device: torch.device,
        name: str,
        learning_rate_coefficient: float = 1.0,
        lagrange: bool = False,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        self.approximator = deepNetApproximator(
            z_dim=z_dimension,
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_activation=preprocessor_activation,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            backward_preprocess=backward_preprocess,
            backward_activation=backward_activation,
            backward_preporcess_activation=preprocessor_activation,
            backward_preporcess_hidden_dimension=backward_preprocess_hidden_dimension,
            backward_preporcess_hidden_layers=backward_preprocess_hidden_layers,
            backward_preporcess_output_dim=backward_preprocess_output_dim,
            discount=discount,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
        )
            
        self.approximator.apprximator_1_target.load_state_dict(
            self.approximator.apprximator_1.state_dict()
        )
        self.approximator.apprximator_2_target.load_state_dict(
            self.approximator.apprximator_2.state_dict()
        )
        self.approximator.bacewardNet_target.load_state_dict(
            self.approximator.bacewardNet.state_dict()
        )

        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            number_of_features=forward_number_of_features,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            gaussian_actor=gaussian_actor,
            std_dev_clip=std_dev_clip,
            device=device,
        )

        self.encoder = torch.nn.Identity()
        self.augmentation = torch.nn.Identity()

        # total_action_samples must be divisible by 4
        assert (ood_action_weight % 0.25 == 0) & (
            0 < ood_action_weight <= 1
        )  # ood_action_weight must be divisible by 0.25
        self.total_action_samples = total_action_samples
        self.ood_action_samples = int(self.total_action_samples * ood_action_weight)
        self.actor_action_samples = int(
            (self.total_action_samples - self.ood_action_samples) / 3
        )
        assert (
            self.ood_action_samples + (3 * self.actor_action_samples)
            == self.total_action_samples
        )

        self.alpha = alpha
        self.target_conservative_penalty = target_conservative_penalty

        
        # optimisers
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.approximator.apprximator_1.parameters()},
                {"params": self.approximator.apprximator_2.parameters()},
                {
                    "params": self.approximator.bacewardNet.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                }
            ],
            lr=critic_learning_rate,
        )
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )

        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._z_dimension = z_dimension
        self.std_dev_schedule = std_dev_schedule
        self.q_coefficient = q_coefficient

        # lagrange multiplier
        self.lagrange = lagrange
        self.critic_log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self.critic_alpha_optimizer = torch.optim.Adam(
            [self.critic_log_alpha], lr=critic_learning_rate
        )
        
    @torch.no_grad()
    def act(
        self,
        observation: Dict[str, np.ndarray],
        task: np.array,
        step: int,
        sample: bool = False,
    ) -> Tuple[np.array, float]:

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        h = self.encoder(observation)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        # get action from actor
        std_dev = schedule(self.std_dev_schedule, step)
        action, _ = self.actor(h, z, std_dev, sample=sample)

        return action.detach().cpu().numpy()[0], std_dev

    def update(self, batch: Batch, batch_rand: Batch, step: int) -> Dict[str, float]:
        zs = self.sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        backward_input = batch.observations[perm]
        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = self.approximator.bacewardNet(
                backward_input[mix_indices]
            ).detach()
            mix_zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                mix_zs, dim=1
            )

        zs[mix_indices] = mix_zs
        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)

        # update forward and backward models
        operate_metrics = self.update_operate(
            observations=batch.observations,
            observations_rand=batch_rand.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts,
            zs=zs,
            step=step,
        )

        # update actor
        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, step=step
        )

        # update target networks for forwards and backwards models
        self.soft_update_params(
            network=self.approximator.apprximator_1,
            target_network=self.approximator.apprximator_1_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.approximator.apprximator_2,
            target_network=self.approximator.apprximator_2_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.approximator.bacewardNet,
            target_network=self.approximator.bacewardNet_target,
            tau=self._tau,
        )

        metrics = {
            **operate_metrics,
            **actor_metrics,
        }

        return metrics

    def update_operate(
        self,
        observations: torch.Tensor,
        observations_rand: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:

        total_loss, metrics, _, _, _, _, _, actor_std_dev = self._update_operate_inner(
            observations, observations_rand ,actions, next_observations, discounts, zs, step
        )
        (conservative_penalty, conservative_metrics,) = self._value_conservative_penalty(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            zs=zs,
            actor_std_dev=actor_std_dev,
        )

        # tune alpha from conservative penalty
        alpha, alpha_metrics = self._tune_alpha(
            conservative_penalty=conservative_penalty
        )
        conservative_loss = alpha * conservative_penalty
        total_loss = total_loss + conservative_loss
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.approximator.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()
        
        metrics = {
            **metrics,
            **conservative_metrics,
            **alpha_metrics,
            "train/forward_backward_total_loss": total_loss,
        }

        return metrics

    def _update_operate_inner(
        self,
        observations: torch.Tensor,
        observations_rand: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ):

        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            next_actions, _ = self.actor(
                next_observations, zs, actor_std_dev, sample=True
            )

            target_B = self.approximator.bacewardNet_target(observation=observations_rand)
            target_M1 = self.approximator.apprximator_1_target(observation=next_observations, z=zs, action=next_actions, B=target_B)
            target_M2 = self.approximator.apprximator_2_target(observation=next_observations, z=zs, action=next_actions, B=target_B)
            target_M = torch.min(target_M1, target_M2)

        # --- Forward-backward representation loss ---
        B_next = self.approximator.bacewardNet(next_observations)
        B_rand = self.approximator.bacewardNet(observations_rand)
        M1_next = self.approximator.apprximator_1(observation=observations, z=zs, action=actions, B=B_next)
        M2_next = self.approximator.apprximator_2(observation=observations, z=zs, action=actions, B=B_next)
        M1_rand = self.approximator.apprximator_1(observation=observations, z=zs, action=actions, B=B_rand)
        M2_rand = self.approximator.apprximator_2(observation=observations, z=zs, action=actions, B=B_rand)

        M_off_diag_loss = 0.5 * sum(
            (M - discounts * target_M).pow(2).mean()
            for M in [M1_rand, M2_rand]
        )

        M_diag_loss = -sum(M.mean() for M in [M1_next, M2_next])

        M_loss = M_diag_loss + M_off_diag_loss

        with torch.no_grad():
            next_Q1 = self.approximator.apprximator_1_target(observation=next_observations, z=zs, action=next_actions).squeeze() 
            next_Q2 = self.approximator.apprximator_2_target(observation=next_observations, z=zs, action=next_actions).squeeze()
            next_Q = torch.min(next_Q1, next_Q2)
            cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
            inv_cov = torch.inverse(cov)
            implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1)  # batch_size
            target_Q = implicit_reward.detach() + discounts.squeeze() * next_Q  # batch_size
        Q1 = self.approximator.apprximator_1(observation=observations, z=zs, action=actions).squeeze()
        Q2 = self.approximator.apprximator_2(observation=observations, z=zs, action=actions).squeeze()
        Q = torch.min(Q1, Q2)
        q_loss = F.mse_loss(Q, target_Q) * self.q_coefficient

        # --- orthonormalisation loss ---
        covariance = torch.matmul(B_next, B_next.T)
        I = torch.eye(*covariance.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.approximator.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        total_loss = M_loss + ortho_loss + q_loss

        metrics = {
            "train/forward_backward_total_loss": total_loss,
            "train/M_loss": M_loss,
            "train/M_diag_loss": M_diag_loss,
            "train/M_off_diag_loss": M_off_diag_loss,
            "train/q_loss": q_loss,
            "train/ortho_diag_loss": ortho_loss_diag,
            "train/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/target_M": target_M.mean().item(),
            "train/M": M1_next.mean().item(),
            "train/B": B_next.mean().item(),
        }

        return total_loss, metrics, B_next, M1_next, M2_next, target_B, off_diagonal, actor_std_dev

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int
    ) -> Dict[str, float]:
        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(observation, z, std, sample=True)

        Q1 = self.approximator.apprximator_1(observation=observation, action=action, z=z).squeeze()
        Q2 = self.approximator.apprximator_2(observation=observation, action=action, z=z).squeeze()
        Q = torch.min(Q1, Q2)

        # update actor towards action that maximise Q (minimise -Q)
        actor_loss = -Q

        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            # add an entropy regularisation term
            log_prob = action_dist.log_prob(action).sum(-1)
            actor_loss += 0.1 * log_prob  # NOTE: currently hand-coded weight!
            mean_log_prob = log_prob.mean().item()
        else:
            mean_log_prob = 0.0

        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        metrics = {
            "train/actor_loss": actor_loss.item(),
            "train/actor_Q": Q.mean().item(),
            "train/actor_log_prob": mean_log_prob,
        }

        return metrics

    def load(self, filepath: Path):
        """Loads model."""
        pass

    def sample_z(self, size: int) -> torch.Tensor:
        """Samples z in the sphere of radius sqrt(D)."""
        gaussian_random_variable = torch.randn(
            size, self._z_dimension, dtype=torch.float32, device=self._device
        )
        gaussian_random_variable = torch.nn.functional.normalize(
            gaussian_random_variable, dim=1
        )
        z = math.sqrt(self._z_dimension) * gaussian_random_variable

        return z

    def infer_z(
        self, observations: torch.Tensor, rewards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        with torch.no_grad():
            z = self.approximator.bacewardNet(observations)

        if rewards is not None:
            z = torch.matmul(rewards.T, z) / rewards.shape[0]  # reward-weighted average

        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        z = z.squeeze().cpu().numpy()

        return z

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ):

        Q1 = self.approximator.apprximator_1(observation, action, z)
        Q2 = self.approximator.apprximator_2(observation, action, z)
        Q = torch.min(Q1, Q2)

        return Q

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:

        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _value_conservative_penalty(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        with torch.no_grad():
            # repeat observations, next_observations, zs, and Bs
            # we fold the action sample dimension into the batch dimension
            # to allow the tensors to be passed through F and B; we then
            # reshape the output back to maintain the action sample dimension
            repeated_observations_ood = observations.repeat(
                self.ood_action_samples, 1, 1
            ).reshape(self.ood_action_samples * self.batch_size, -1)
            repeated_zs_ood = zs.repeat(self.ood_action_samples, 1, 1).reshape(
                self.ood_action_samples * self.batch_size, -1
            )
            ood_actions = torch.empty(
                size=(self.ood_action_samples * self.batch_size, self.action_length),
                device=self._device,
            ).uniform_(-1, 1)

            if self.actor_action_samples > 0:
                repeated_observations_actor = observations.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * self.batch_size, -1)
                repeated_next_observations_actor = next_observations.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * self.batch_size, -1)
                repeated_zs_actor = zs.repeat(self.actor_action_samples, 1, 1).reshape(
                    self.actor_action_samples * self.batch_size, -1
                )
                actor_current_actions, _ = self.actor(
                    repeated_observations_actor,
                    repeated_zs_actor,
                    std=actor_std_dev,
                    sample=True,
                )  # [actor_action_samples * batch_size, action_length]

                actor_next_actions, _ = self.actor(
                    repeated_next_observations_actor,
                    z=repeated_zs_actor,
                    std=actor_std_dev,
                    sample=True,
                )  # [actor_action_samples * batch_size, action_length]


        if self.actor_action_samples > 0:
            repeated_obs = observations.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1)
            repeated_zs = zs.repeat(self.actor_action_samples, 1, 1).reshape(
                self.actor_action_samples * self.batch_size, -1
            )
            repeated_actions = actions.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1)
            
            cat_obs = torch.cat(
                [
                    repeated_observations_ood,
                    repeated_observations_actor,
                    repeated_next_observations_actor,
                    repeated_obs,
                ],
                dim=0,
            )
            cat_zs = torch.cat(
                [
                    repeated_zs_ood,
                    repeated_zs_actor,
                    repeated_zs_actor,
                    repeated_zs,
                ],
                dim=0,
            )
            cat_actions = torch.cat(
                [
                    ood_actions,
                    actor_current_actions,
                    actor_next_actions,
                    repeated_actions,
                ],
                dim=0,
            )
        else:
            cat_obs = repeated_observations_ood
            cat_zs = repeated_zs_ood
            cat_actions = ood_actions

        # convert to Qs
        cql_cat_Q1 = self.approximator.apprximator_1(observation=cat_obs, action=cat_actions, z=cat_zs).reshape(
            self.total_action_samples, self.batch_size, -1
        )
        cql_cat_Q2 = self.approximator.apprximator_2(observation=cat_obs, action=cat_actions, z=cat_zs).reshape(
            self.total_action_samples, self.batch_size, -1
        )
        

        cql_logsumexp = (
            torch.logsumexp(cql_cat_Q1, dim=0).mean()
            + torch.logsumexp(cql_cat_Q2, dim=0).mean()
        )

        # get existing Qs
        Q1 = self.approximator.apprximator_1(observation=observations, z=zs, action=actions).squeeze()
        Q2 = self.approximator.apprximator_2(observation=observations, z=zs, action=actions).squeeze()
        conservative_penalty = cql_logsumexp - (Q1 + Q2).mean()

        metrics = {
            "train/cql_penalty": conservative_penalty.item(),
            "train/cql_cat_Q1": cql_cat_Q1.mean().item(),
            "train/cql_cat_Q2": cql_cat_Q2.mean().item(),
        }

        return conservative_penalty, metrics

    def _tune_alpha(
        self,
        conservative_penalty: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # alpha auto-tuning
        if self.lagrange:
            alpha = torch.clamp(self.critic_log_alpha.exp(), min=0.0, max=1e6)
            alpha_loss = (
                -0.5 * alpha * (conservative_penalty - self.target_conservative_penalty)
            )

            self.critic_alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.critic_alpha_optimizer.step()
            alpha = torch.clamp(self.critic_log_alpha.exp(), min=0.0, max=1e6).detach()
            alpha_loss = alpha_loss.detach().item()

        # fixed alpha
        else:
            alpha = self.alpha
            alpha_loss = 0.0

        metrics = {
            "train/alpha": alpha,
            "train/alpha_loss": alpha_loss,
        }

        return alpha, metrics
    
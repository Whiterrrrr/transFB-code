"""Module defining the Forward-Backward Agent."""

import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import numpy as np
import torch.nn.functional as F
from agents.fb.models import ForwardBackwardRepresentation, ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class IFB(AbstractAgent):

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
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        critic_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        iql_tau: float,
        device: torch.device,
        name: str = 'IFB',
        iql_beta: float=3
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )
        
        self.FB = ForwardBackwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=forward_number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
            use_iql=True
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
        self.beta=iql_beta

        self.FB.forward_representation_target.load_state_dict(
            self.FB.forward_representation.state_dict()
        )
        self.FB.backward_representation_target.load_state_dict(
            self.FB.backward_representation.state_dict()
        )
        self.FB.state_forward_representation_target.load_state_dict(
            self.FB.state_forward_representation.state_dict()
        )
            
        self.FB_optimizer = torch.optim.Adam(
            [
                {"params": self.FB.forward_representation.parameters()},
                {"params": self.FB.state_forward_representation.parameters()},
                {
                    "params": self.FB.backward_representation.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                },
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
        self.iql_tau = iql_tau

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

        std_dev = schedule(self.std_dev_schedule, step)
        action, _ = self.actor(h, z, std_dev, sample=sample)

        return action.detach().cpu().numpy()[0], std_dev

    def update(self, batch: Batch, step: int) -> Dict[str, float]:

        zs = self.sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        backward_input = batch.observations[perm]
        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = self.FB.backward_representation(
                backward_input[mix_indices]
            ).detach()
            mix_zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                mix_zs, dim=1
            )

        zs[mix_indices] = mix_zs
        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)

        fb_metrics = self.update_fb(
            observations=batch.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts,
            zs=zs,
            step=step,
        )

        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, step=step, actions=batch.actions
        )

        self.soft_update_params(
            network=self.FB.forward_representation,
            target_network=self.FB.forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.FB.state_forward_representation,
            target_network=self.FB.state_forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.FB.backward_representation,
            target_network=self.FB.backward_representation_target,
            tau=self._tau,
        )

        metrics = {
            **fb_metrics,
            **actor_metrics,
        }

        return metrics
    
    def update_fb(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:

        total_loss, metrics, _, _, _, _, _, _, _ = self._update_fb_inner(
            observations, actions, next_observations, discounts, zs, step
        )

        self.FB_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.FB.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.FB_optimizer.step()

        return metrics

    def _update_fb_inner(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ):
        with torch.no_grad():
            target_F1, target_F2 = self.FB.forward_representation_target(
                observation=observations, action=actions, z=zs
            )
            target_K = self.FB.state_forward_representation_target(
                observation=next_observations, z=zs
            )
            target_B = self.FB.backward_representation_target(
                observation=next_observations
            )
            target_O = torch.einsum("sd, td -> st", target_K, target_B)
            next_V = torch.einsum("sd, td -> s", target_K, zs)
            target_M1, target_M2 = torch.einsum("sd, td -> st", target_F1, target_B), torch.einsum("sd, td -> st", target_F2, target_B)
            target_M = torch.min(target_M1, target_M2)

        F1, F2 = self.FB.forward_representation(observations, actions, zs)
        K = self.FB.state_forward_representation(observations, zs)
        B_next = self.FB.backward_representation(next_observations)
        O = torch.einsum("sd, td -> st", K, B_next)
        Q1, Q2 = torch.einsum("sd, sd -> s", F1, zs), torch.einsum("sd, sd -> s", F2, zs)
        Q = torch.min(Q1, Q2)
        V = torch.einsum("sd, td -> s", K, zs)

        M1_next = torch.einsum("sd, td -> st", F1, B_next)
        M2_next = torch.einsum("sd, td -> st", F2, B_next)

        I = torch.eye(*M1_next.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}

        fb_off_diag_loss = 0.5 * sum(
            (M - discounts * target_O)[off_diagonal].pow(2).mean()
            for M in [M1_next, M2_next]
        )

        fb_diag_loss = -sum(M.diag().mean() for M in [M1_next, M2_next])
        fb_loss = fb_diag_loss + fb_off_diag_loss
         
        adv = Q.detach() - V
        adv_fb = (target_M - O)[off_diagonal]
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        adv_fb_loss = asymmetric_l2_loss(adv_fb, self.iql_tau)
        
        cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
        inv_cov = torch.inverse(cov)
        implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1)
        targets = implicit_reward + discounts.squeeze() * next_V.detach()
        q_loss = F.mse_loss(Q, targets)
        
        covariance = torch.matmul(B_next, B_next.T)
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.FB.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        total_loss = fb_loss + ortho_loss + v_loss + q_loss + adv_fb_loss
        
        metrics = {
            "train/total_loss": total_loss,
            "train/forward_backward_fb_loss": fb_loss,
            "train/forward_backward_fb_diag_loss": fb_diag_loss,
            "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
            "train/adv_fb_loss":adv_fb_loss,
            "train/v_loss": v_loss,
            "train/q_loss": q_loss,
            "train/ortho_diag_loss": ortho_loss_diag,
            "train/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/M1_next": M1_next.mean().item(),
            "train/M2_next": M2_next.mean().item(),
            "train/target_B": target_B.mean().item(),
            "train/target_O": target_O.mean().item(),
            "train/target_K": target_K.mean().item(),
            "train/next_V": next_V.mean().item(),
            "train/target_M": target_M.mean().item(),
            "train/target_F1": target_F1.mean().item(),
            "train/target_F2": target_F2.mean().item(),
            "train/F1": F1.mean().item(),
            "train/F2": F2.mean().item(),
            "train/K": K.mean().item(),
            "train/B": B_next.mean().item(),
            "train/V": V.mean().item(),
            "train/Q": Q.mean().item(),
            "train/O": O.mean().item(),
            "train/implicit_reward": implicit_reward.mean().item(),
        }

        return total_loss, metrics, \
               F1, F2, B_next, M1_next, M2_next, target_B, off_diagonal

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int, actions: torch.Tensor
    ) -> Dict[str, float]:

        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(observation, z, std, sample=True)

        F1, F2 = self.FB.forward_representation(
            observation=observation, z=z, action=actions
        )
        K = self.FB.state_forward_representation_target(
                observation=observation, z=z
        )

        Q1, Q2, V = torch.einsum("sd, sd -> s", F1, z), torch.einsum("sd, sd -> s", F2, z), torch.einsum("sd, td -> s", K, z)
        Q = torch.min(Q1, Q2)
        adv = (Q - V.detach())
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
        
        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            log_prob = action_dist.log_prob(action).sum(-1)
            actor_loss += 0.1 * log_prob  # NOTE: currently hand-coded weight!
            mean_log_prob = log_prob.mean().item()
            raise NotImplementedError
        else:
            bc_losses = torch.sum((action - actions)**2, dim=1)
            mean_log_prob = 0.0

        actor_loss =  torch.mean(exp_adv * bc_losses)

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
        pass

    def sample_z(self, size: int) -> torch.Tensor:
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
            z = self.FB.backward_representation(observations)

        if rewards is not None:
            z = torch.matmul(rewards.T, z) / rewards.shape[0]  # reward-weighted average

        z = (math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)).squeeze().cpu().numpy()
        return z

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ):

        F1, F2 = self.FB.forward_representation(
            observation=observation, z=z, action=action
        )
        Q1, Q2 = torch.einsum("sd, sd -> s", F1, z), torch.einsum("sd, sd -> s", F2, z)
        return torch.min(Q1, Q2)

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:
        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

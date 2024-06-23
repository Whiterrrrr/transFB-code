import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import time
import numpy as np

from agents.cexp.module import MixNetRepresentation
from agents.fb.models import ActorModel
from agents.fb.base import BCQ_actor, VAE
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule


class CEXP(AbstractAgent):

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
        use_trans: bool,
        use_cons: bool,
        use_m_cons: bool,
        use_res: bool,
        operator_hidden_dimension: int,
        operator_hidden_layers: int,
        trans_hidden_dimension: int,
        num_attention_heads: int,
        n_attention_layers: int,
        n_linear_layers: int,
        dropout_rate: float,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        backward_preprocess: bool,
        backward_preporcess_hidden_dimension,
        backward_preprocess_hidden_layers,
        backward_preporcess_activation,
        backward_preprocess_output_dimension,
        operator_activation: str,
        actor_activation: str, 
        actor_learning_rate: float,
        critic_learning_rate: float,
        b_learning_rate_coefficient: float,
        g_learning_rate_coefficient: float,
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
        use_q_loss,
        use_fed,
        use_VIB,
        use_dual: bool,
        use_cross_attention: bool,
        use_2branch: bool = False,
        lagrange: bool = False,
        use_icm: bool = False,
        M_pealty_coefficient: float = 1.0,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        self.Operate = MixNetRepresentation(
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
            operator_hidden_layers=operator_hidden_layers,
            operator_hidden_dimension=operator_hidden_dimension,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            operator_activation=operator_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
            use_trasnformer=use_trans,
            trans_dimension=trans_hidden_dimension,
            num_attention_heads=num_attention_heads,
            n_attention_layers=n_attention_layers,
            n_linear_layers=n_linear_layers,
            dropout_rate=dropout_rate,
            backward_preprocess=backward_preprocess,
            backward_preporcess_hidden_dimension=backward_preporcess_hidden_dimension,
            backward_preporcess_hidden_layers=backward_preprocess_hidden_layers,
            backward_preporcess_activation=backward_preporcess_activation,
            backward_preporcess_output_dim=backward_preprocess_output_dimension,
            use_res=use_res,
            use_fed=use_fed,
            use_VIB=use_VIB,
            use_2branch=use_2branch, 
            use_cross_attention=use_cross_attention,
            use_dual=use_dual,
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

        # load weights into target networks
        self.Operate.forward_representation_target.load_state_dict(
            self.Operate.forward_representation.state_dict()
        )
        self.Operate.backward_representation_target.load_state_dict(
            self.Operate.backward_representation.state_dict()
        )
        self.Operate.operator.load_state_dict(
            self.Operate.operator_target.state_dict()
        )
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

        self.use_cons = use_cons
        self.alpha = alpha
        self.target_conservative_penalty = target_conservative_penalty
        self.use_q_loss=use_q_loss
        self.use_icm = use_icm
        self.use_m_cons = use_m_cons
        self.M_pealty_coefficient = M_pealty_coefficient
        self.use_VIB = use_VIB
        self.use_2branch = use_2branch
        self.use_dual = use_dual
        self.use_cross_attention = use_cross_attention
        
        # optimisers
        self.Operate_optimizer = torch.optim.AdamW(
            [
                {"params": self.Operate.forward_representation.parameters()},
                {
                    "params": self.Operate.backward_representation.parameters(),
                    "lr": critic_learning_rate * b_learning_rate_coefficient,
                },
                {
                    "params": self.Operate.operator.parameters(),
                    "lr": critic_learning_rate * g_learning_rate_coefficient,
                },
            ],
            lr=critic_learning_rate,
            weight_decay=0.03,
            betas=(0.9, 0.999),
            amsgrad=False
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=actor_learning_rate,
            weight_decay=0.03,
            betas=(0.9, 0.999),
            amsgrad=False
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
        self.critic_alpha_optimizer = torch.optim.AdamW(
            [self.critic_log_alpha], 
            lr=critic_learning_rate,
            # weight_decay=0.03,
            # betas=(0.9, 0.9),
            # amsgrad=False
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

    def update(self, batch: Batch, step: int) -> Dict[str, float]:

        zs = self.sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        backward_input = batch.observations[perm]
        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = self.Operate.backward_representation(
                backward_input[mix_indices]
            ).detach()
            mix_zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                mix_zs, dim=1
            )

        zs[mix_indices] = mix_zs
        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)
        
        operate_metrics = self.update_operate(
            observations=batch.observations,
            observations_rand=batch.other_observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts,
            zs=zs,
            step=step,
        )
        
        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, step=step
        )

        self.soft_update_params(
            network=self.Operate.forward_representation,
            target_network=self.Operate.forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.Operate.backward_representation,
            target_network=self.Operate.backward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.Operate.operator,
            target_network=self.Operate.operator_target,
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

        total_loss, metrics, F1, F2, B_next, B_rand, _, _, _, _, actor_std_dev = self._update_operate_inner(
            observations, observations_rand ,actions, next_observations, discounts, zs, step
        )
           
        if self.use_cons:
            (conservative_penalty, conservative_metrics,) = self._value_conservative_penalty(
                observations=observations,
                next_observations=next_observations,
                zs=zs,
                actor_std_dev=actor_std_dev,
                F1=F1,
                F2=F2,
            )

            # tune alpha from conservative penalty
            alpha, alpha_metrics = self._tune_alpha(
                conservative_penalty=conservative_penalty
            )
            conservative_loss = alpha * conservative_penalty
            total_loss = total_loss + conservative_loss
        

        self.Operate_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.Operate.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.Operate_optimizer.step()
        
        
        if self.use_cons:
            metrics = {
                **metrics,
                **conservative_metrics,
                **alpha_metrics,
                "train/forward_backward_total_loss": total_loss,
            }
        else:
            metrics = {
                **metrics,
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
        #TODO: add the implementation for the dual network
        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            next_actions, _ = self.actor(
                next_observations, zs, actor_std_dev, sample=True
            )

            target_F1, target_F2 = self.Operate.forward_representation_target(observation=next_observations, z=zs, action=next_actions)
            target_B = self.Operate.backward_representation_target(observation=observations_rand)
            if not self.use_VIB:
                target_M = self.Operate.operator_target(
                    torch.cat((
                        target_F1.repeat(int(target_B.shape[0] // target_F1.shape[0]), 1), 
                        target_F2.repeat(int(target_B.shape[0] // target_F2.shape[0]), 1)), dim=0), 
                    torch.cat((target_B, target_B), dim=0)
                )
                target_M = torch.min(target_M[:target_B.size(0)], target_M[target_B.size(0):])

        # --- Forward-backward representation loss ---
        F1, F2 = self.Operate.forward_representation(observations, actions, zs)
        B = self.Operate.backward_representation(torch.cat((next_observations, observations_rand), dim=0))
        B_next, B_rand = B[:next_observations.size(0)], B[next_observations.size(0):]

        if not self.use_VIB:
            M_next = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((B_next, B_next), dim=0))
            M_rand = self.Operate.operator(
                torch.cat((
                    F1.repeat(int(B_rand.shape[0] // F1.shape[0]), 1), 
                    F2.repeat(int(B_rand.shape[0] // F2.shape[0]), 1)), dim=0), 
                torch.cat((B_rand, B_rand), dim=0)
            )
            M1_next, M2_next = M_next[:B_next.size(0)],  M_next[B_next.size(0):]
            M1_rand, M2_rand = M_rand[:B_rand.size(0)], M_rand[B_rand.size(0):]

        fb_off_diag_loss = 0.5 * sum(
            (M - discounts.repeat(int(B_rand.shape[0] // F1.shape[0]), 1) * target_M).pow(2).mean()
            for M in [M1_rand, M2_rand]
        )

        fb_diag_loss = -sum(M.mean() for M in [M1_next, M2_next])

        fb_loss = fb_diag_loss + fb_off_diag_loss
        total_loss = fb_loss

        if self.use_q_loss:
            with torch.no_grad():
                if not self.use_VIB:
                    next_Q = self.Operate.operator_target(
                        torch.cat((target_F1, target_F2), dim=0), 
                        torch.cat((zs, zs), dim=0)
                    ).squeeze() 
                    next_Q = torch.min(next_Q[:zs.size(0)], next_Q[zs.size(0):])
                if self.use_cross_attention:
                    B_pinv = torch.pinverse(B_next)
                    implicit_reward = torch.diag(torch.matmul(zs, B_pinv))
                else:
                    cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
                    inv_cov = torch.inverse(cov)
                    implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1)  # batch_size
                target_Q = implicit_reward.detach() + discounts.squeeze() * next_Q  # batch_size
            if not self.use_VIB:
                Q = self.Operate.operator_target(
                    torch.cat((F1, F2), dim=0), 
                    torch.cat((zs, zs), dim=0)
                ).squeeze() 
                Q = torch.min(Q[:zs.size(0)], Q[zs.size(0):])

            q_loss = F.mse_loss(Q, target_Q) * self.q_coefficient
            total_loss = total_loss + q_loss

        # --- orthonormalisation loss ---
        covariance = torch.matmul(B_next, B_next.T)
        I = torch.eye(*covariance.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.Operate.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        total_loss = total_loss + ortho_loss

        if self.use_q_loss:
            metrics = {
                "train/forward_backward_total_loss": total_loss,
                "train/forward_backward_fb_loss": fb_loss,
                "train/forward_backward_fb_diag_loss": fb_diag_loss,
                "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
                "train/q_loss": q_loss,
                "train/ortho_diag_loss": ortho_loss_diag,
                "train/ortho_off_diag_loss": ortho_loss_off_diag,
                "train/target_M": target_M.mean().item(),
                "train/M": M1_next.mean().item(),
                "train/F": F1.mean().item(),
                "train/F_norm1": torch.mean(torch.norm(F1, p=1, dim=1)).item(),
                "train/B": B_next.mean().item(),
                "train/B_norm1": torch.mean(torch.norm(B_next, p=1, dim=1)).item(),
                "train/B_var": B_next.var(dim=1).mean().item(),
            }
        else: 
            metrics = {
                "train/forward_backward_total_loss": total_loss,
                "train/forward_backward_fb_loss": fb_loss,
                "train/forward_backward_fb_diag_loss": fb_diag_loss,
                "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
                "train/ortho_diag_loss": ortho_loss_diag,
                "train/ortho_off_diag_loss": ortho_loss_off_diag,
                "train/target_M": target_M.mean().item(),
                "train/M": M1_next.mean().item(),
                "train/F": F1.mean().item(),
                "train/F_norm1": torch.mean(torch.norm(F1, p=1, dim=1)).item(),
                "train/B": B_next.mean().item(),
                "train/B_norm1": torch.mean(torch.norm(B_next, p=1, dim=1)).item(),
                "train/B_var": B_next.var(dim=1).mean().item(),
            }

        return total_loss, metrics, \
               F1, F2, B_next, B_rand, M1_next, M2_next, target_B, off_diagonal, actor_std_dev

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int
    ) -> Dict[str, float]:

        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(observation, z, std, sample=True)

        # with torch.no_grad():
        F1, F2 = self.Operate.forward_representation(
            observation=observation, z=z, action=action
        )

        # get Qs from F and z'
        if not self.use_VIB:
            Q = self.Operate.operator_target(
                torch.cat((F1, F2), dim=0), 
                torch.cat((z, z), dim=0)
            ).squeeze() 
            Q = torch.min(Q[:z.size(0)], Q[z.size(0):])
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
            z = self.Operate.backward_representation(observations)

        if rewards is not None:
            z = torch.matmul(rewards.T, z) / rewards.shape[0]  # reward-weighted average

        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)
        z = z.squeeze().cpu().numpy()
        return z

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ):

        F1, F2 = self.Operate.forward_representation(
            observation=observation, z=z, action=action
        )

        # get Qs from F and z
        if not self.use_VIB:
            Q = self.Operate.operator_target(
                torch.cat((F1, F2), dim=0), 
                torch.cat((z, z), dim=0)
            ).squeeze() 
        return torch.min(Q[:z.size(0)], Q[z.size(0):])

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
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
        F1: torch.Tensor,
        F2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        with torch.no_grad():

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

        # get cml Fs
        ood_F1, ood_F2 = self.Operate.forward_representation(
            repeated_observations_ood, ood_actions, repeated_zs_ood
        )  # [ood_action_samples * batch_size, latent_dim]
        
        if self.actor_action_samples > 0:
            repeated_F1, repeated_F2 = F1.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1), F2.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(
                self.actor_action_samples * self.batch_size, -1
            )
            actor_current_F1, actor_current_F2 = self.Operate.forward_representation(
                repeated_observations_actor, actor_current_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]   
            actor_next_F1, actor_next_F2 = self.Operate.forward_representation(
                repeated_next_observations_actor, actor_next_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            
            cat_F1 = torch.cat(
                [
                    ood_F1,
                    actor_current_F1,
                    actor_next_F1,
                    repeated_F1,
                ],
                dim=0,
            )
            cat_F2 = torch.cat(
                [
                    ood_F2,
                    actor_current_F2,
                    actor_next_F2,
                    repeated_F2,
                ],
                dim=0,
            )
        else:
            cat_F1 = ood_F1
            cat_F2 = ood_F2

        repeated_zs = zs.repeat(self.total_action_samples, 1, 1).reshape(
            self.total_action_samples * self.batch_size, -1
        )

        # convert to Qs
        if not self.use_VIB:
            cql_cat_Q = self.Operate.operator(
                torch.cat((cat_F1, cat_F2), dim=0),
                torch.cat((repeated_zs, repeated_zs), dim=0)
            )
            cql_cat_Q1, cql_cat_Q2 = cql_cat_Q[:repeated_zs.size(0)], cql_cat_Q[repeated_zs.size(0):]

        cql_logsumexp_Q = (
            torch.logsumexp(cql_cat_Q1, dim=0).mean()
            + torch.logsumexp(cql_cat_Q2, dim=0).mean()
        )

        # get existing Qs
        if not self.use_VIB:
            Q = self.Operate.operator_target(
                torch.cat((F1, F2), dim=0), 
                torch.cat((zs, zs), dim=0)
            ).squeeze() 
            Q1, Q2 = Q[:zs.size(0)], Q[zs.size(0):]
                
        conservative_penalty_Q = cql_logsumexp_Q - (Q1 + Q2).mean()
        conservative_penalty = conservative_penalty_Q
        
        metrics = {
            "train/cql_penalty": conservative_penalty.item(),
            "train/cql_penalty_Q": conservative_penalty_Q.item(),
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
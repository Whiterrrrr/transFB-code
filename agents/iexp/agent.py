import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

from agents.iexp.module import MixNetRepresentation
from agents.fb.models import ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

torch.autograd.set_detect_anomaly(True)

class IEXP(AbstractAgent):

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
        backward_preprocess_hidden_dimension,
        backward_preprocess_hidden_layers,
        backward_preprocess_activation,
        backward_preprocess_output_dimension,
        operator_activation: str,
        actor_activation: str, 
        actor_learning_rate: float,
        critic_learning_rate: float,
        f_loss_coefficient: float,
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
        asymmetric_l2_tau: float,
        alpha: float,
        device: torch.device,
        name: str,
        use_q_loss,
        use_fed,
        use_VIB,
        use_2branch,
        use_cross_attention = False,
        use_AWAR = False,
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
            trans_dimension=trans_hidden_dimension,
            num_attention_heads=num_attention_heads,
            n_attention_layers=n_attention_layers,
            n_linear_layers=n_linear_layers,
            dropout_rate=dropout_rate,
            backward_preprocess=backward_preprocess,
            backward_preporcess_hidden_dimension=backward_preprocess_hidden_dimension,
            backward_preporcess_hidden_layers=backward_preprocess_hidden_layers,
            backward_preporcess_activation=backward_preprocess_activation,
            backward_preporcess_output_dim=backward_preprocess_output_dimension,
            use_res=use_res,
            use_fed=use_fed,
            use_VIB=use_VIB,
            use_cross_attention=use_cross_attention,
            use_2branch = use_2branch,
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
        self.Operate.state_forward_representation_target.load_state_dict(
            self.Operate.state_forward_representation.state_dict()
        )
        self.Operate.backward_representation_target.load_state_dict(
            self.Operate.backward_representation.state_dict()
        )
        self.Operate.operator.load_state_dict(
            self.Operate.operator_target.state_dict()
        )
        self.alpha = alpha
        self.use_q_loss=use_q_loss
        self.use_VIB = use_VIB
        
        # optimisers
        self.Operate_optimizer = torch.optim.AdamW(
            [
                {"params": self.Operate.forward_representation.parameters()},
                {"params": self.Operate.state_forward_representation.parameters()},
                {"params": self.Operate.backward_representation.parameters(), "lr": critic_learning_rate * b_learning_rate_coefficient,},
                {"params": self.Operate.operator.parameters(), "lr": critic_learning_rate * g_learning_rate_coefficient,},
            ],
            lr=critic_learning_rate,
        )      
        for param in self.Operate.parameters():
            param.data = param.data.to(torch.float32)
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)
                
        for param in self.actor.parameters():
            param.data = param.data.to(torch.float32)
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)
        # self.Operate_optimizer_Fgroup = torch.optim.AdamW(
        #     [
        #         {"params": self.Operate.forward_representation.parameters()},
        #         {"params": self.Operate.backward_representation.parameters(), "lr": critic_learning_rate * b_learning_rate_coefficient,},
        #         {"params": self.Operate.operator.parameters(), "lr": critic_learning_rate * g_learning_rate_coefficient,},
        #     ],
        #     lr=critic_learning_rate,
        # )
        # self.Operate_optimizer_SFgroup = torch.optim.AdamW(
        #     [
        #         {"params": self.Operate.state_forward_representation.parameters()},
        #         {"params": self.Operate.backward_representation.parameters(), "lr": critic_learning_rate * b_learning_rate_coefficient,},
        #         {"params": self.Operate.operator.parameters(), "lr": critic_learning_rate * g_learning_rate_coefficient,},
        #     ],
        #     lr=critic_learning_rate,
        # )
        # self.F_optimizer = torch.optim.AdamW(params=self.Operate.forward_representation.parameters(), lr=critic_learning_rate)
        # self.SF_optimizer = torch.optim.AdamW(params=self.Operate.state_forward_representation.parameters(), lr=critic_learning_rate)
        # self.B_optimizer = torch.optim.AdamW(params=self.Operate.backward_representation.parameters(), lr=critic_learning_rate * b_learning_rate_coefficient)
        # self.Operate_optimizer = torch.optim.AdamW(params=self.Operate.operator.parameters(), lr=critic_learning_rate * g_learning_rate_coefficient)
        
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=actor_learning_rate,
            # weight_decay=0.03,
            # betas=(0.9, 0.9),
            # amsgrad=False
        )

        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self.asymmetric_l2_tau = asymmetric_l2_tau
        self._z_dimension = z_dimension
        self.std_dev_schedule = std_dev_schedule
        self.q_coefficient = q_coefficient
        self.f_loss_coefficient = f_loss_coefficient
        self.use_AWAR_loss = use_AWAR
        
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
        # sample zs and mix
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
        actor_actions = batch.actions.clone().requires_grad_(True)

        # update forward and backward models
        operate_metrics, V, adv = self.update_operate(
            observations=batch.observations,
            observations_rand=batch.other_observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts,
            rewards=batch.rewards,
            not_dones=batch.not_dones,
            zs=zs,
            step=step,
        )

        # update target networks for forwards and backwards models
        self.soft_update_params(
            network=self.Operate.backward_representation,
            target_network=self.Operate.backward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.Operate.forward_representation,
            target_network=self.Operate.forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.Operate.state_forward_representation,
            target_network=self.Operate.state_forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.Operate.operator,
            target_network=self.Operate.operator_target,
            tau=self._tau,
        )

        # update actor
        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, step=step, actions=actor_actions, adv=adv
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
        rewards: torch.tensor,
        not_dones: torch.tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:

        total_loss, metrics, _, _, _, _, _, _, _, _, _, V, adv = self._update_operate_inner(
            observations, observations_rand ,actions, next_observations, discounts, rewards, not_dones, zs, step
        )

        return metrics, V, adv

    def _update_operate_inner(
        self,
        observations: torch.Tensor,
        observations_rand: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        rewards: torch.Tensor,
        not_dones: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ):
        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            cur_F1_tar, cur_F2_tar = self.Operate.forward_representation_target(observation=observations, action=actions, z=zs)
            next_K_tar = self.Operate.state_forward_representation_target(observation=next_observations, z=zs)
            target_B = self.Operate.backward_representation_target(observation=observations_rand)
            target_O = self.Operate.operator_target(next_K_tar.repeat(int(target_B.shape[0] // next_K_tar.shape[0]), 1), target_B)
            target_next_V = self.Operate.operator_target(next_K_tar, zs).squeeze()
            target_Q = torch.min(
                self.Operate.operator_target(cur_F1_tar, zs).squeeze(),
                self.Operate.operator_target(cur_F2_tar, zs).squeeze(),
            )
            target_M = torch.min(
                self.Operate.operator_target(cur_F1_tar, target_B),
                self.Operate.operator_target(cur_F2_tar, target_B),
            )
            
        F1, F2 = self.Operate.forward_representation(observations, actions, zs)
        K = self.Operate.state_forward_representation(observations, zs)
        V = self.Operate.operator(K, zs).squeeze()
        Q1, Q2 = [self.Operate.operator(Fi, zs).squeeze() for Fi in [F1, F2]]
        B_next = self.Operate.backward_representation(next_observations)
        B_rand = self.Operate.backward_representation(observations_rand)
        cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
        inv_cov = torch.inverse(cov)
        implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1) 
        if not self.use_VIB:
            M1_next = self.Operate.operator(F1, B_next)
            M2_next = self.Operate.operator(F2, B_next)
            M1_rand = self.Operate.operator(F1.repeat(int(B_rand.shape[0] // F1.shape[0]), 1), B_rand)
            M2_rand = self.Operate.operator(F2.repeat(int(B_rand.shape[0] // F2.shape[0]), 1), B_rand)
            O_rand = self.Operate.operator(K.repeat(int(B_rand.shape[0] // K.shape[0]), 1), B_rand)
            
        # fb loss
        fb_off_diag_loss = 0.5 * sum(
            (M - discounts.repeat(int(B_rand.shape[0] // K.shape[0]), 1) * target_O).pow(2).mean()
            for M in [M1_rand, M2_rand]
        )

        fb_diag_loss = -sum(M.mean() for M in [M1_next, M2_next])
        fb_loss = fb_diag_loss + fb_off_diag_loss
        total_loss = fb_loss * self.f_loss_coefficient
        
        # kb loss
        ado = target_M - O_rand
        kb_loss = asymmetric_l2_loss(ado, self.asymmetric_l2_tau)
        total_loss += kb_loss * self.f_loss_coefficient
        
        adv = target_Q - V
        if self.use_q_loss:
            # q loss
            targets = implicit_reward.detach() + discounts.repeat(int(B_rand.shape[0] // K.shape[0]), 1).squeeze() * target_next_V.detach()
            q_loss = sum(F.mse_loss(q, targets) for q in [Q1, Q2]) / 2
            total_loss += q_loss * self.q_coefficient
            
            # v loss
            v_loss = asymmetric_l2_loss(adv, self.asymmetric_l2_tau)
            total_loss += v_loss * self.q_coefficient

        # --- orthonormalisation loss ---
        covariance = torch.matmul(B_next, B_next.T)
        I = torch.eye(*covariance.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.Operate.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )
        total_loss += ortho_loss

        self.Operate_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.Operate.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.Operate_optimizer.step()

        if self.use_q_loss:
            metrics = {
                "train/forward_backward_fb_diag_loss": fb_diag_loss,
                "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
                "train/fb_loss": fb_loss,
                "train/kb_loss": kb_loss,
                "train/q_loss": q_loss,
                "train/v_loss": v_loss,
                "train/total_loss": total_loss,
                "train/ortho_diag_loss": ortho_loss_diag,
                "train/ortho_off_diag_loss": ortho_loss_off_diag,
                "train/M": M1_rand.mean().item(),
                "train/O": O_rand.mean().item(),
                "train/F": F1.mean().item(),
                "train/F_norm1": torch.mean(torch.norm(F1, p=1, dim=1)).item(),
                "train/B": B_next.mean().item(),
                "train/B_norm1": torch.mean(torch.norm(B_next, p=1, dim=1)).item(),
                "train/B_var": B_next.var(dim=1).mean().item(),
                "train/Q1": Q1.mean().item(),
                "train/Q2": Q2.mean().item(),
                "train/V": V.mean().item(),
                "train/next_V": target_next_V.mean().item(),
                "train/implicit_reward": implicit_reward.mean().item(),
            }
        else:
            metrics = {
                "train/forward_backward_fb_diag_loss": fb_diag_loss,
                "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
                "train/fb_loss": fb_loss,
                "train/kb_loss": kb_loss,
                "train/total_loss": total_loss,
                "train/ortho_diag_loss": ortho_loss_diag,
                "train/ortho_off_diag_loss": ortho_loss_off_diag,
                "train/M": M1_rand.mean().item(),
                "train/O": O_rand.mean().item(),
                "train/F": F1.mean().item(),
                "train/F_norm1": torch.mean(torch.norm(F1, p=1, dim=1)).item(),
                "train/B": B_next.mean().item(),
                "train/B_norm1": torch.mean(torch.norm(B_next, p=1, dim=1)).item(),
                "train/B_var": B_next.var(dim=1).mean().item(),
                "train/Q1": Q1.mean().item(),
                "train/Q2": Q2.mean().item(),
                "train/V": V.mean().item(),
                "train/next_V": target_next_V.mean().item(),
                "train/implicit_reward": implicit_reward.mean().item(),
            }

        return total_loss, metrics, \
               F1, F2, B_next, B_rand, M1_next, M2_next, target_B, off_diagonal, actor_std_dev, V, adv

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int, actions: torch.Tensor, adv: torch.Tensor
    ) -> Dict[str, float]:

        std = schedule(self.std_dev_schedule, step)
        policy_out, action_dist = self.actor(observation, z, std, sample=True)
        
        # with torch.no_grad():
        
        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            # add an entropy regularisation term
            log_prob = action_dist.log_prob(policy_out).sum(-1)
            actor_loss += 0.1 * log_prob  # NOTE: currently hand-coded weight!
            mean_log_prob = log_prob.mean().item()
        else:
            mean_log_prob = 0.0
            
        if self.use_AWAR_loss:
            with torch.no_grad():
                F1, F2 = self.Operate.forward_representation(
                    observation=observation, z=z, action=policy_out
                )
                actor_Q = torch.min(
                    self.Operate.operator(F1, z).squeeze(),
                    self.Operate.operator(F2, z).squeeze(),
                )
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
            exp_adv = torch.exp(self.alpha * adv.detach()).clamp(max=1e2)
            actor_loss = torch.mean(exp_adv * bc_losses)
        else:
            F1, F2 = self.Operate.forward_representation(
                observation=observation, z=z, action=policy_out
            )
            actor_Q = torch.min(
                self.Operate.operator(F1, z).squeeze(),
                self.Operate.operator(F2, z).squeeze(),
            )
            actor_loss = -actor_Q.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        metrics = {
            "train/adv": adv.mean().item(),
            "train/actor_loss": actor_loss.item(),
            "train/actor_log_prob": mean_log_prob,
            "train/actor_Q": actor_Q.mean().item(),
        }

        return metrics

    def load(self, filepath: Path):
        """Loads model."""
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
            Q1 = self.Operate.operator(F1, z)
            Q2 = self.Operate.operator(F2, z)

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

    
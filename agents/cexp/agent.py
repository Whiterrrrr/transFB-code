import math
from pathlib import Path
from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
from agents.cexp.module import MixNetRepresentation, V_net
from agents.fb.models import ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

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
        use_res: bool,
        trans_hidden_dimension: int,
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
        actor_activation: str, 
        actor_learning_rate: float,
        critic_learning_rate: float,
        b_learning_rate_coefficient: float,
        g_learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        name: str,
        use_fed:bool,
        use_cross_attention: bool = False,
        use_distribution: bool = False,
        use_OFE: bool = True,
        q_loss_coeff: float = 0.01,
        ensemble_size: int = 1,
        num_atoms: int = 51,
        minVal: int = 0,
        maxVal: int = 500,
        use_film_cond: bool = False,
        use_linear_res: bool = False,
        iql_tau: float = 0.7,
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
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
            trans_dimension=trans_hidden_dimension,
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
            use_OFE=use_OFE,
            use_cross_attention=use_cross_attention,
            use_distribution=use_distribution,
            ensemble_size=ensemble_size,
            num_atoms=num_atoms,
            use_film_cond=use_film_cond,
            use_linear_res=use_linear_res,
        )

        self.V_net = V_net(
            input_dimension=observation_length + z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            activation=forward_activation,
            device=device,
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

        self.Operate.forward_representation_target.load_state_dict(
            self.Operate.forward_representation.state_dict()
        )
        self.Operate.backward_representation_target.load_state_dict(
            self.Operate.backward_representation.state_dict()
        )
        self.Operate.operator_target.load_state_dict(
            self.Operate.operator.state_dict()
        )
        self.use_distribution = use_distribution
        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._z_dimension = z_dimension
        self.std_dev_schedule = std_dev_schedule
        self.minVal=minVal
        self.maxVal=maxVal
        self.num_atoms=num_atoms
        self.support = torch.linspace(minVal, maxVal, num_atoms, device=device)
        self.q_loss_coeff = q_loss_coeff
        self.iql_tau = iql_tau
        
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
            betas=(0.9, 0.99),
            amsgrad=False
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=actor_learning_rate,
        )
        self.v_optimizer = torch.optim.AdamW(
            self.V_net.parameters(),
            lr=1e-4,
            weight_decay=0.03,
            betas=(0.9, 0.99),
            amsgrad=False
        )  
        self.scheduler = CosineAnnealingWarmRestarts(self.Operate_optimizer, T_0=5, T_mult=2, eta_min=1e-5)
        
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

    def update(self, batch: Batch, batch_rand: Batch, step: int) -> Dict[str, float]:
        mask = torch.rand(batch.observations.shape[0]) < 0.1
        batch_rand.observations[mask] = batch.next_observations[mask]
        zs = self.sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        backward_input = batch.observations[perm]
        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = self.Operate.backward_representation(backward_input[mix_indices]).detach()
            mix_zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(mix_zs, dim=1)

        zs[mix_indices] = mix_zs
        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)
        
        operate_metrics = self.update_operate(
            observations=batch.observations,
            observations_rand=batch_rand.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts.squeeze(),
            zs=zs,
            step=step,
        )
        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, discounts=batch.discounts, step=step
        )
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]

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
        metrics['current_lr']=current_lr 
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

        total_loss, metrics = self._update_operate_inner(
            observations, observations_rand ,actions, next_observations, discounts, zs, step
        )

        self.Operate_optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        for param in self.Operate.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.Operate_optimizer.step()
    
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
        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            next_actions, _ = self.actor(next_observations, zs, actor_std_dev, sample=True)
            target_F1, target_F2 = self.Operate.forward_representation_target(observation=next_observations, z=zs, action=next_actions)
            target_B = self.Operate.backward_representation_target(observation=observations_rand)
            if self.use_distribution:
                target_M_dist = self.Operate.operator_target(
                    torch.cat((
                        target_F1.repeat(int(target_B.shape[0] // target_F1.shape[0]), 1), 
                        target_F2.repeat(int(target_B.shape[0] // target_F2.shape[0]), 1)), dim=0), 
                    torch.cat((target_B, target_B), dim=0)
                ).squeeze()
                target_M1_dist, target_M2_dist = target_M_dist[:target_B.size(0)], target_M_dist[target_B.size(0):]
                target_M_dist = torch.where(torch.sum(target_M1_dist * self.support, dim=1).unsqueeze(-1) <= torch.sum(target_M2_dist * self.support, dim=1).unsqueeze(-1), target_M1_dist, target_M2_dist)
                target_M = torch.sum(target_M_dist * self.support, dim=1)
                Tz = torch.all(observations_rand == next_observations, dim=1).unsqueeze(1) + discounts.unsqueeze(1) * self.support
                Tz = Tz.clamp(min=self.minVal, max=self.maxVal)
                b = (Tz - self.minVal) / (self.maxVal - self.minVal) * (self.num_atoms - 1)
                l, u = b.floor().long(), b.ceil().long()
                offset = torch.linspace(0, (b.size(0) - 1) * self.num_atoms, b.size(0), device=self._device).long().unsqueeze(1).expand(b.size(0), self.num_atoms)
                proj_dist = torch.zeros(target_M_dist.size(), device=self._device)
                proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_M_dist * (u.float() - b)).view(-1))
                proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_M_dist * (b - l.float())).view(-1))
            else:
                target_M = self.Operate.operator_target(
                    torch.cat((
                        target_F1.repeat(int(target_B.shape[0] // target_F1.shape[0]), 1), 
                        target_F2.repeat(int(target_B.shape[0] // target_F2.shape[0]), 1)), dim=0), 
                    torch.cat((target_B, target_B), dim=0)
                ).squeeze()
                target_M = torch.min(target_M[:target_B.size(0)], target_M[target_B.size(0):])
            
        F1, F2 = self.Operate.forward_representation(observations, actions, zs)
        B = self.Operate.backward_representation(torch.cat((next_observations, observations_rand), dim=0))
        B_next, B_rand = B[:next_observations.size(0)], B[next_observations.size(0):]

        if self.use_distribution:
            M_rand_dist = self.Operate.operator(
                torch.cat((
                    F1.repeat(int(B_rand.shape[0] // F1.shape[0]), 1), 
                    F2.repeat(int(B_rand.shape[0] // F2.shape[0]), 1)), dim=0), 
                torch.cat((B_rand, B_rand), dim=0)
            ).squeeze()
            M1_rand_dist, M2_rand_dist = M_rand_dist[:B_rand.size(0)], M_rand_dist[B_rand.size(0):]
            M_rand_dist = torch.where(torch.sum(M1_rand_dist * self.support, dim=1).unsqueeze(-1) <= torch.sum(M2_rand_dist * self.support, dim=1).unsqueeze(-1), M1_rand_dist, M2_rand_dist)
            M_rand = torch.sum(M_rand_dist * self.support, dim=1)
            fb_loss = -torch.sum(proj_dist * torch.log(M_rand_dist + 1e-8), dim=1).mean()
        else:
            M_next = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((B_next, B_next), dim=0)).squeeze()
            M_rand = self.Operate.operator(
                torch.cat((
                    F1.repeat(int(B_rand.shape[0] // F1.shape[0]), 1), 
                    F2.repeat(int(B_rand.shape[0] // F2.shape[0]), 1)), dim=0), 
                torch.cat((B_rand, B_rand), dim=0)
            ).squeeze()
            M1_next, M2_next = M_next[:B_next.size(0)],  M_next[B_next.size(0):]
            M1_rand, M2_rand = M_rand[:B_rand.size(0)], M_rand[B_rand.size(0):]
            fb_off_diag_loss = 0.5 * sum(
                (M - discounts.repeat(int(B_rand.shape[0] // F1.shape[0]), 1) * target_M).pow(2).mean()
                for M in [M1_rand, M2_rand]
            )
            fb_diag_loss = -sum(M.mean() for M in [M1_next, M2_next])
            fb_loss = fb_diag_loss + fb_off_diag_loss
        total_loss = fb_loss

        covariance = torch.matmul(B_next, B_next.T)
        I = torch.eye(*covariance.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.Operate.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        V = self.V_net(observations, zs).squeeze()
        Q = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((zs, zs), dim=0)).squeeze()
        Q = torch.min(Q[:zs.size(0)], Q[zs.size(0):])
        
        target_V = self.V_net(next_observations, zs).squeeze()
        with torch.no_grad():
            norm_B_next = torch.nn.functional.normalize(B_next, dim=1)
            cov = torch.matmul(norm_B_next.T, norm_B_next)
            inv_cov = torch.inverse(cov)
            implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1)
            target_V = implicit_reward.detach() + discounts.squeeze() * target_V
            
        adv = Q.detach() - V
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        q_loss = F.mse_loss(Q, target_V.detach()) * self.q_loss_coeff
        total_loss = total_loss + ortho_loss + q_loss
        metrics = {
            "train/forward_backward_total_loss": total_loss,
            "train/forward_backward_fb_loss": fb_loss,
            "train/forward_backward_fb_diag_loss": fb_diag_loss,
            "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
            "train/ortho_diag_loss": ortho_loss_diag,
            "train/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/target_M": target_M.mean().item(),
            "train/M_next": M_next.mean().item(),
            "train/M_rand": M_rand.mean().item(),
            "train/F1": F1.mean().item(),
            "train/F2": F2.mean().item(),
            "train/target_F1": target_F1.mean().item(),
            "train/target_F2": target_F2.mean().item(),
            "train/F1_norm1": torch.mean(torch.norm(F1, p=1, dim=1)).item(),
            "train/F2_norm1": torch.mean(torch.norm(F2, p=1, dim=1)).item(),
            "train/target_F1_norm1": torch.mean(torch.norm(target_F1, p=1, dim=1)).item(),
            "train/target_F2_norm1": torch.mean(torch.norm(target_F2, p=1, dim=1)).item(),
            "train/B_rand": B_rand.mean().item(),
            "train/B_next": B_next.mean().item(),
            "train/target_B": target_B.mean().item(),
            "train/B_rand_norm1": torch.mean(torch.norm(B_rand, p=1, dim=1)).item(),
            "train/B_next_norm1": torch.mean(torch.norm(B_next, p=1, dim=1)).item(),
            "train/target_B_norm1": torch.mean(torch.norm(target_B, p=1, dim=1)).item(),
            "train/B_rand_var": B_rand.var(dim=1).mean().item(),
            "train/B_next_var": B_next.var(dim=1).mean().item(),
            "train/target_B_var": target_B.var(dim=1).mean().item(),
            "train/Q": Q.mean().item(),
            "train/V": V.mean().item(),
            "train/target_V": target_V.mean().item(),
            "train/forward_backward_v_loss": v_loss.mean().item(),
            "train/forward_backward_q_loss": q_loss.mean().item(),
        }
        return total_loss, metrics

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, discounts: torch.Tensor, step: int
    ) -> Dict[str, float]:

        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(observation, z, std, sample=True)

        F1, F2 = self.Operate.forward_representation(observation=observation, z=z, action=action)
        if self.use_distribution:
            Q_dist = self.Operate.operator(torch.cat((F1, F2), dim=0),torch.cat((z, z), dim=0)).squeeze()
            Q1_dist, Q2_dist = Q_dist[:z.size(0)], Q_dist[z.size(0):]
            Q = torch.min(torch.sum(Q1_dist * self.support, dim=1), torch.sum(Q2_dist * self.support, dim=1))
        else:
            Q = self.Operate.operator_target(torch.cat((F1, F2), dim=0), torch.cat((z, z), dim=0)).squeeze() 
            Q = torch.min(Q[:z.size(0)], Q[z.size(0):])
        actor_loss = -Q

        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            log_prob = action_dist.log_prob(action).sum(-1)
            actor_loss += 0.1 * log_prob
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
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor, discounts: torch.Tensor
    ):
        F1, F2 = self.Operate.forward_representation(observation=observation, z=z, action=action)
        if self.use_distribution:
            Q_dist = self.Operate.operator(
                torch.cat((F1, F2), dim=0),
                torch.cat((z, z), dim=0)
            ).squeeze()
            Q1_dist, Q2_dist = Q_dist[:z.size(0)], Q_dist[z.size(0):]
            Q = torch.min(torch.sum(Q1_dist * self.support, dim=1), torch.sum(Q2_dist * self.support, dim=1))
        else:
            Q = self.Operate.operator_target(torch.cat((F1, F2), dim=0), torch.cat((z, z), dim=0)).squeeze() 
            Q = torch.min(Q[:z.size(0)], Q[z.size(0):])
        return Q

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:

        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
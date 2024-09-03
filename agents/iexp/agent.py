import math
from pathlib import Path
from typing import Tuple, Dict, Optional
import torch
import numpy as np
from agents.iexp.module import MixNetRepresentation
from agents.fb.models import ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from agents.diffusion_agent import OP_Agent
from agents.diffusion_model import *

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

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
        use_cross_attention = False,
        use_2branch: bool = False,
        use_feature_norm=False,
        use_linear_res=False,
        use_forward_backward_cross=False,
        iql_tau = 0.9,
        use_fed = False,
        use_diffusion=False,
        beta=3,
        ts=5,
        use_eql=False,
        use_sql=False,
        alpha=2,
        V_learning_rate=1e-3,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )
        self.use_diffusion = use_diffusion

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
            backward_preporcess_hidden_dimension=backward_preprocess_hidden_dimension,
            backward_preporcess_hidden_layers=backward_preprocess_hidden_layers,
            backward_preporcess_activation=backward_preprocess_activation,
            backward_preporcess_output_dim=backward_preprocess_output_dimension,
            use_res=use_res,
            use_2branch=use_2branch,
            use_feature_norm=use_feature_norm,
            use_cross_attention=use_cross_attention,
            use_linear_res=use_linear_res,
            use_forward_backward_cross=use_forward_backward_cross,
            use_fed=use_fed,
        )
        self.V_net = AbstractMLP(
            input_dim=observation_length + z_dimension,
            output_dim=1,
            hidden_dim=forward_hidden_dimension,
            num_blocks=forward_hidden_layers,
            ac_fn=forward_activation,
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
        if self.use_diffusion:
            self.policy = IDQLDiffusion(
                input_dim=action_length,  # a dim
                output_dim=action_length,  # a dim
                observation_length=observation_length,
                z_dimension=z_dimension,
                preprocessor_hidden_dimension=preprocessor_hidden_dimension,
                preprocessor_feature_space_dimension=preprocessor_output_dimension,
                preprocessor_hidden_layers=preprocessor_hidden_layers,
                preprocessor_activation=preprocessor_activation,
                hidden_dim=actor_hidden_dimension,
                num_blocks=actor_hidden_layers,
                time_dim=64,
                ac_fn='mish',
                time_embeding='fixed',
                device=device,
            )
            self.agent = OP_Agent(
                policy_model=self.policy,
                schedule='vp',
                num_timesteps=ts,
                num_sample=16,
                reward_temp=beta,
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

        self.use_2branch = use_2branch
        self.use_cross_attention = use_cross_attention
        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._z_dimension = z_dimension
        self.std_dev_schedule = std_dev_schedule
        self.iql_tau = iql_tau
        self.use_sql = use_sql
        self.use_eql = use_eql
        self.beta = beta
        self.alpha=alpha
        
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
        self.v_optimizer = torch.optim.AdamW(
            self.V_net.parameters(),
            lr=V_learning_rate,
            weight_decay=0.03,
            betas=(0.9, 0.99),
            amsgrad=False
        )   
        self.scheduler = CosineAnnealingWarmRestarts(self.Operate_optimizer, T_0=5, T_mult=2, eta_min=1e-5)
        self.mixnet_optimizer = torch.optim.AdamW(
            self.Operate.operator.parameters(),
            lr=critic_learning_rate,
            # weight_decay=0.03,
            # betas=(0.9, 0.9),
            # amsgrad=False
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=actor_learning_rate,
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
        if not self.use_diffusion:
            action, _ = self.actor(h, z, std_dev, sample=sample)
            action = action[0]
        else:
            with torch.no_grad():
                actions = self.agent.get_actions(observation, z)
                observation, z = observation.repeat(actions.shape[0], 1), z.repeat(actions.shape[0], 1)
                F1, F2 = self.Operate.forward_representation_target(observation=observation, z=z, action=actions)
                Q = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((z, z), dim=0)).squeeze()
                Q = torch.min(Q[:z.size(0)], Q[z.size(0):])
                idx = torch.argmax(Q)
                action = actions[idx]
        return action.detach().cpu().numpy(), std_dev

    def update(self, batch: Batch, batch_rand: Batch, step: int) -> Dict[str, float]:
        mask = torch.rand(batch.observations.shape[0]) < 0.1
        batch_rand.observations[mask] = batch.next_observations[mask]
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
            observation=actor_observations, z=actor_zs, discounts=batch.discounts, step=step, actions=actor_actions
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

        total_loss, metrics, _, _, _, _, _, _, _ = self._update_operate_inner(
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
            target_B = self.Operate.backward_representation_target(observations_rand)
            target_F1, target_F2 = self.Operate.forward_representation_target(next_observations, next_actions, zs)
            target_V = self.V_net(torch.cat((next_observations, zs), dim=1)).squeeze()
            target_M = self.Operate.operator_target(torch.cat((target_F1, target_F2), dim=0), torch.cat((target_B, target_B), dim=0)).squeeze()
            target_M = torch.min(target_M[:target_B.size(0)], target_M[target_B.size(0):])
            
        B = self.Operate.backward_representation(torch.cat((next_observations, observations_rand), dim=0))
        B_next, B_rand = B[:next_observations.size(0)], B[next_observations.size(0):]
        F1, F2 = self.Operate.forward_representation(observations, actions, zs)
        V = self.V_net(torch.cat((observations, zs), dim=1)).squeeze()
        Q = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((zs, zs), dim=0)).squeeze()
        Q = torch.min(Q[:zs.size(0)], Q[zs.size(0):])
        M_next = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((B_next, B_next), dim=0)).squeeze()
        M_rand = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((B_rand, B_rand), dim=0)).squeeze()
        M1_next, M2_next, M1_rand, M2_rand = M_next[:B_next.size(0)], M_next[B_next.size(0):], M_rand[:B_rand.size(0)], M_rand[B_rand.size(0):]
        M_rand = torch.min(M1_rand, M2_rand)
        
        fb_off_diag_loss = 0.5 * sum((M - discounts * target_M).pow(2).mean()for M in [M1_rand, M2_rand])
        fb_diag_loss = -sum(M.mean() for M in [M1_next, M2_next])
        fb_loss = fb_diag_loss + fb_off_diag_loss
        
        with torch.no_grad():
            cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
            inv_cov = torch.inverse(cov)
            implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1)
            target_V = implicit_reward.detach() + discounts.squeeze() * target_V
            
        q_loss = F.mse_loss(Q, target_V)
        
        if self.use_sql:
            sp_term = (Q.detach() - V) / (2 * self.alpha) + 1.0
            sp_weight = torch.where(sp_term > 0, torch.tensor(1.0), torch.tensor(0.0))
            v_loss = (sp_weight * (sp_term**2) + V / self.alpha).mean()
        elif self.use_eql:
            sp_term = (Q.detach() - V) / self.alpha
            sp_term = torch.minimum(sp_term, torch.tensor(5.0))
            max_sp_term = torch.max(sp_term, dim=0).values
            max_sp_term = torch.where(max_sp_term < -1.0, torch.tensor(-1.0), max_sp_term)
            max_sp_term = max_sp_term.detach()
            v_loss = (torch.exp(sp_term - max_sp_term) + torch.exp(-max_sp_term) * V / self.alpha).mean()
        else:
            adv = Q.detach() - V
            v_loss = asymmetric_l2_loss(adv, self.iql_tau)

        covariance = torch.matmul(B_next, B_next.T)
        I = torch.eye(*covariance.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.Operate.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        total_loss = fb_loss + ortho_loss + q_loss
        metrics = {
            "train/forward_backward_total_loss": total_loss,
            "train/forward_backward_v_loss": v_loss,
            "train/forward_backward_q_loss": q_loss,
            "train/forward_backward_fb_loss": fb_loss,
            "train/forward_backward_fb_diag_loss": fb_diag_loss,
            "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
            "train/ortho_diag_loss": ortho_loss_diag,
            "train/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/M_next": M_next.mean().item(),
            "train/M_rand": M_rand.mean().item(),
            "train/F1": F1.mean().item(),
            "train/F2": F2.mean().item(),
            "train/Q": Q.mean().item(),
            "train/V": V.mean().item(),
            "train/target_V": target_V.mean().item(),
            "train/B_rand": B_rand.mean().item(),
            "train/B_next": B_next.mean().item(),
            "train/target_B": target_B.mean().item(),
            "train/implicit_reward": implicit_reward.mean().item(),
        }
        return total_loss, metrics, \
            F1, F2, B_next, B_rand, target_B, off_diagonal, actor_std_dev

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, discounts: torch.Tensor, step: int, actions: torch.Tensor
    ) -> Dict[str, float]:
        F1, F2 = self.Operate.forward_representation_target(observation=observation, z=z, action=actions)
        V = self.V_net(torch.cat((observation, z), dim=1)).squeeze()
        Q = self.Operate.operator_target(torch.cat((F1, F2), dim=0), torch.cat((z, z), dim=0)).squeeze() 
        Q = torch.min(Q[:z.size(0)], Q[z.size(0):])
        adv = (Q - V.detach())
        
        std = schedule(self.std_dev_schedule, step)
        actor_output, action_dist = self.actor(observation, z, std, sample=True)
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
        
        if (type(self.actor.actor) == AbstractGaussianActor):
            log_prob = action_dist[1].log_prob(actions).sum(-1)
            mean_log_prob = log_prob        
            if self.use_eql:    
                weight = torch.exp(10 * adv.detach()/self.alpha).clamp(max=100)
                actor_loss = -(weight * log_prob).mean()
            elif self.use_sql:
                weight = torch.clamp(adv, min=0)
                actor_loss = -(weight * log_prob).mean()
            else:
                actor_loss = -(exp_adv * log_prob).mean()
        else:
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
            mean_log_prob = 0.0
            bc_losses = torch.sum((actor_output - actions)**2, dim=1)
            actor_loss =  torch.mean(exp_adv * bc_losses)
            
        if self.use_diffusion:
            diffusion_loss = self.agent.policy_loss(actions, observation, z, q=Q.detach(), v=V.detach()).mean()
            actor_loss += diffusion_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor.parameters():
            if param.grad is not None:
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
        gaussian_random_variable = torch.randn(size, self._z_dimension, dtype=torch.float32, device=self._device)
        gaussian_random_variable = torch.nn.functional.normalize(gaussian_random_variable, dim=1)
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
        Q = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((z, z), dim=0)).squeeze() 
        return torch.min(Q[:z.size(0)], Q[z.size(0):])

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:

        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

import math
from pathlib import Path
from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
from agents.calexp.module import MixNetRepresentation
from agents.fb.models import ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.fb.base import FF_pred_model
from agents.utils import schedule
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from agents.calexp.utils import perturb, cal_dormant_grad, perturb_factor, cal_dormant_ratio, dormant_perturb

class Calexp(AbstractAgent):
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
        use_dr3,
        dr3_coefficient = 1,
        use_auxiliary: bool = False,
        auxiliary_coefficient: float = 1, 
        use_cross_attention = False,
        use_distribution: bool = False,
        use_2branch: bool = False,
        lagrange: bool = False,
        use_icm: bool = False,
        M_pealty_coefficient: float = 1.0,
        update_freq=1,
        reset_interval=1000,
        use_feature_norm=False,
        use_dormant=True,
        use_OFE=True,
        FF_pred_hidden_dimension: int = 256,
        FF_pred_hidden_layers: int = 2,
        FF_pred_activation: str = 'ReLU',
        ensemble_size: int = 1,
        num_atoms: int = 51,
        minVal: int = 0,
        maxVal: int = 500,
        use_gamma_loss: bool = False,
        use_film_cond: bool = False,
        use_linear_res=False,
        use_forward_backward_cross=False
    ):
        assert not (use_gamma_loss and use_distribution), 'use_gamma_loss and use_distribution cannot be True at the same time'
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
            use_2branch=use_2branch,
            use_dr3=use_dr3,
            use_feature_norm=use_feature_norm,
            use_OFE=use_OFE,
            use_cross_attention=use_cross_attention,
            use_distribution=use_distribution,
            ensemble_size=ensemble_size,
            num_atoms=num_atoms,
            use_film_cond=use_film_cond,
            use_linear_res=use_linear_res,
            use_forward_backward_cross=use_forward_backward_cross
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

        self.update_freq=update_freq
        self.use_cons = use_cons
        self.alpha = alpha
        self.target_conservative_penalty = target_conservative_penalty
        self.use_q_loss=use_q_loss
        self.use_icm = use_icm
        self.use_m_cons = use_m_cons
        self.M_pealty_coefficient = M_pealty_coefficient
        self.use_2branch = use_2branch
        self.use_dr3=use_dr3
        self.dr3_coefficient=dr3_coefficient
        self.reset_interval = reset_interval
        self.use_dormant = use_dormant
        self.use_auxiliary = use_auxiliary
        self.auxiliary_coefficient = auxiliary_coefficient
        self.use_distribution = use_distribution
        self.use_cross_attention = use_cross_attention
        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._z_dimension = z_dimension
        self.std_dev_schedule = std_dev_schedule
        self.q_coefficient = q_coefficient
        self.lagrange = lagrange
        self.minVal=minVal
        self.maxVal=maxVal
        self.num_atoms=num_atoms
        self.support = torch.linspace(minVal, maxVal, num_atoms, device=device)
        self.use_gamma_loss = use_gamma_loss
        
        if self.use_auxiliary:
            self.pred_model = FF_pred_model(
                observation_length=observation_length,
                z_dim=z_dimension,
                hidden_dimension=FF_pred_hidden_dimension,
                hidden_layers=FF_pred_hidden_layers,
                activation=FF_pred_activation,
                device=device,
            )
            self.auxiliary_optimizer = torch.optim.AdamW(
                [
                    {"params": self.pred_model.parameters()},
                    {"params": self.Operate.forward_representation.parameters()},
                ],
                lr=critic_learning_rate,
            )
        # optimisers
        self.Operate_optimizer = torch.optim.AdamW(
            [
                {"params": self.Operate.forward_representation.parameters()},
                {"params": self.Operate.forward_mu.parameters()},
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
        self.scheduler = CosineAnnealingWarmRestarts(self.Operate_optimizer, T_0=5, T_mult=2, eta_min=1e-5)
        self.FB_optimizer = torch.optim.AdamW(
            [
                {"params": self.Operate.forward_representation.parameters()},
                {
                    "params": self.Operate.backward_representation.parameters(),
                    "lr": critic_learning_rate * b_learning_rate_coefficient,
                },
            ],
            lr=critic_learning_rate,
            # weight_decay=0.03,
            # betas=(0.9, 0.9),
            # amsgrad=False
        )
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
        
        operate_metrics, dormant_indices = self.update_operate(
            observations=batch.observations,
            observations_rand=batch_rand.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts.squeeze(),
            zs=zs,
            step=step,
        )
        actor_metrics, actor_dormant_indices = self.update_actor(
            observation=actor_observations, z=actor_zs, discounts=batch.discounts, step=step
        )
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        if self.use_dormant and step % self.reset_interval == 0 and step > 5000:
            operater_dormant_metrics = cal_dormant_grad(self.Operate, type='critic', percentage=0.05)
            actor_dormant_metrics = cal_dormant_grad(self.actor, type='actor', percentage=0.025)

            # dormant_metrics.update(grad_metrics)
            operater_dormant_metrics["factor"] = perturb_factor(operater_dormant_metrics['critic_grad_dormant_ratio']) if operater_dormant_metrics else 1
            actor_dormant_metrics["factor"] = perturb_factor(actor_dormant_metrics['actor_grad_dormant_ratio']) if actor_dormant_metrics else 1

            if operater_dormant_metrics["factor"] < 1: 
                self.Operate.forward_representation, self.Operate_optimizer = dormant_perturb(self.Operate.forward_representation, self.Operate_optimizer, dormant_indices['forward_representation'], operater_dormant_metrics["factor"])
                self.Operate.backward_representation, self.Operate_optimizer = dormant_perturb(self.Operate.backward_representation, self.Operate_optimizer, dormant_indices['backward_representation'], operater_dormant_metrics["factor"])
                self.Operate.operator, self.Operate_optimizer = dormant_perturb(self.Operate.operator, self.Operate_optimizer, dormant_indices['operator'], operater_dormant_metrics["factor"])
                self.Operate.forward_representation_target, self.Operate_optimizer = dormant_perturb(self.Operate.forward_representation_target, self.Operate_optimizer, dormant_indices['forward_target'], operater_dormant_metrics["factor"])
                self.Operate.backward_representation_target, self.Operate_optimizer = dormant_perturb(self.Operate.backward_representation_target, self.Operate_optimizer, dormant_indices['backward_target'], operater_dormant_metrics["factor"])
                self.Operate.operator_target, self.Operate_optimizer = dormant_perturb(self.Operate.operator_target, self.Operate_optimizer, dormant_indices['operator_target'], operater_dormant_metrics["factor"])
                self.Operate, self.Operate_optimizer = perturb(self.Operate, self.Operate_optimizer, operater_dormant_metrics["factor"])
            if actor_dormant_metrics["factor"] < 1:
                self.actor, self.actor_optimizer = dormant_perturb(self.actor, self.actor_optimizer, actor_dormant_indices, actor_dormant_metrics["factor"])
                self.actor, self.actor_optimizer = perturb(self.actor, self.actor_optimizer, actor_dormant_metrics["factor"])

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
        self.soft_update_params(
            network=self.Operate.forward_mu,
            target_network=self.Operate.forward_mu_target,
            tau=self._tau,
        )

        if self.use_dormant and step % self.reset_interval == 0 and step > 5000:
            metrics = {
                **operate_metrics,
                **actor_metrics,
                **operater_dormant_metrics,
                **actor_dormant_metrics
            }
        else:
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

        total_loss, metrics, F1, F2, B_next, B_rand, target_B, _, actor_std_dev, dormant_indices, auxiliary_loss, M1_rand, M2_rand = self._update_operate_inner(
            observations, observations_rand ,actions, next_observations, discounts, zs, step
        )
        with torch.no_grad():
            target_Fmu = self.Operate.forward_mu_target(next_observations)
            target_Mmu = self.Operate.operator_target(target_Fmu, target_B).squeeze()
        Fmu = self.Operate.forward_mu(observations)
        Mmu = self.Operate.operator(Fmu, B_rand).squeeze()
        fmub_off_diag_loss = (Mmu - discounts * target_Mmu).pow(2).mean()
        Mmu_next = self.Operate.operator(Fmu, B_next).squeeze()
        fmub_diag_loss = -2.0 * Mmu_next.mean()
        fmub_loss = fmub_off_diag_loss + fmub_diag_loss
        total_loss = total_loss + fmub_loss
        
        if self.use_cons:
            (conservative_penalty, conservative_metrics,) = self._value_conservative_penalty(
                observations=observations,
                next_observations=next_observations,
                rand_observations=observations_rand[:observations.shape[0]],
                zs=zs,
                actor_std_dev=actor_std_dev,
                F1=F1,
                F2=F2,
                discount=discounts,
                B_rand=B_rand,
                M_pealty_coefficient=self.M_pealty_coefficient,
                Fmu=Fmu,
                M1_next=M1_rand,
                M2_next=M2_rand,
            )

            alpha, alpha_metrics = self._tune_alpha(
                conservative_penalty=conservative_penalty
            )
            conservative_loss = alpha * conservative_penalty
            total_loss = total_loss + conservative_loss

        self.Operate_optimizer.zero_grad(set_to_none=True)
        if self.use_auxiliary:
            self.auxiliary_optimizer.zero_grad(set_to_none=True)
            
        total_loss.backward(retain_graph=True)
        for param in self.Operate.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        if self.use_auxiliary:
            auxiliary_loss.backward()
            for param in self.pred_model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
                
        self.Operate_optimizer.step()
        if self.use_auxiliary:
            self.auxiliary_optimizer.step()
        
        if self.use_cons:
            metrics = {
                **metrics,
                **conservative_metrics,
                **alpha_metrics,
                "train/fmub_loss": fmub_loss,
                "train/forward_backward_total_loss": total_loss,
            }
        else:
            metrics = {
                **metrics,
                "train/forward_backward_total_loss": total_loss,
            }
        return metrics, dormant_indices

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
                if not self.use_dr3:
                    target_M = self.Operate.operator_target(
                        torch.cat((
                            target_F1.repeat(int(target_B.shape[0] // target_F1.shape[0]), 1), 
                            target_F2.repeat(int(target_B.shape[0] // target_F2.shape[0]), 1)), dim=0), 
                        torch.cat((target_B, target_B), dim=0)
                    ).squeeze()
                else:
                    target_M, target_M_feature = self.Operate.operator_target(
                        torch.cat((
                            target_F1.repeat(int(target_B.shape[0] // target_F1.shape[0]), 1), 
                            target_F2.repeat(int(target_B.shape[0] // target_F2.shape[0]), 1)), dim=0), 
                        torch.cat((target_B, target_B), dim=0)
                    )
                    target_M, target_M_feature = target_M.squeeze(), target_M_feature.squeeze()
                target_M = torch.min(target_M[:target_B.size(0)], target_M[target_B.size(0):])
            
        F1, F2 = self.Operate.forward_representation(observations, actions, zs)
        B = self.Operate.backward_representation(torch.cat((next_observations, observations_rand), dim=0))
        B_next, B_rand = B[:next_observations.size(0)], B[next_observations.size(0):]

        M_next = dr3_implict_reg = auxiliary_loss = fb_diag_loss = fb_off_diag_loss = torch.zeros(1, device=self._device)
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
            total_loss = fb_loss
        else:
            if not self.use_dr3:
                M_next = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((B_next, B_next), dim=0)).squeeze()
                M_rand = self.Operate.operator(
                    torch.cat((
                        F1.repeat(int(B_rand.shape[0] // F1.shape[0]), 1), 
                        F2.repeat(int(B_rand.shape[0] // F2.shape[0]), 1)), dim=0), 
                    torch.cat((B_rand, B_rand), dim=0)
                ).squeeze()
            else:
                M_next, M_next_feature = self.Operate.operator(torch.cat((F1, F2), dim=0), torch.cat((B_next, B_next), dim=0))
                M_rand, M_rand_feature = self.Operate.operator(
                    torch.cat((
                        F1.repeat(int(B_rand.shape[0] // F1.shape[0]), 1), 
                        F2.repeat(int(B_rand.shape[0] // F2.shape[0]), 1)), dim=0), 
                    torch.cat((B_rand, B_rand), dim=0)
                )
                dr3_implict_reg = torch.mean(torch.sum(M_rand_feature*target_M_feature, dim=1))
                M_next, M_rand, M_next_feature, M_rand_feature = M_next.squeeze(), M_rand.squeeze(), M_next_feature.squeeze(), M_rand_feature.squeeze()
            M1_next, M2_next = M_next[:B_next.size(0)], M_next[B_next.size(0):]
            M1_rand, M2_rand = M_rand[:B_rand.size(0)], M_rand[B_rand.size(0):]
            fb_off_diag_loss = 0.5 * sum(
                (M - discounts.repeat(int(B_rand.shape[0] // F1.shape[0]), 1) * target_M).pow(2).mean()
                for M in [M1_rand, M2_rand]
            )

            fb_diag_loss = -sum(M.mean() for M in [M1_next, M2_next])

            fb_loss = fb_diag_loss + fb_off_diag_loss
            total_loss = fb_loss + dr3_implict_reg * self.dr3_coefficient
                
        if self.use_auxiliary:
            F1_clone, F2_clone = F1.clone(), F2.clone()
            F1_clone.retain_grad()
            F2_clone.retain_grad()
            pred_next_observations1, pred_next_observations2 = self.pred_model(F1_clone, F2_clone, zs)
            auxiliary_loss = (F.mse_loss(pred_next_observations1, next_observations) + F.mse_loss(pred_next_observations2, next_observations)) * self.auxiliary_coefficient
            
        dormant_indices = None
        if self.use_dormant and step % self.reset_interval == 0 and step > 5000:
            _, forward_representation_dormant_indices, _ = cal_dormant_ratio(self.Operate.forward_representation, observations, actions, zs, type='forward_representation', percentage=0.1)
            _, backward_representation_dormant_indices, _ = cal_dormant_ratio(self.Operate.backward_representation, observations_rand, type='backward_representation', percentage=0.1)
            _, operator_dormant_indices, _ = cal_dormant_ratio(self.Operate.operator, F1, B_next, type='operator', percentage=0.1)
            _, forward_target_dormant_indices, _ = cal_dormant_ratio(self.Operate.forward_representation_target, next_observations, next_actions, zs, type='forward_target', percentage=0.1)
            _, backward_target_dormant_indices, _ = cal_dormant_ratio(self.Operate.backward_representation_target, observations_rand, type='backward_target', percentage=0.1)
            _, operator_target_dormant_indices, _ = cal_dormant_ratio(self.Operate.operator_target, target_F1, target_B, type='operator_target', percentage=0.1)
            dormant_indices = {
                "forward_representation": forward_representation_dormant_indices,
                "backward_representation": backward_representation_dormant_indices,
                "operator": operator_dormant_indices,
                "forward_target": forward_target_dormant_indices,
                "backward_target": backward_target_dormant_indices,
                "operator_target": operator_target_dormant_indices
            }
        
        if self.use_q_loss:
            with torch.no_grad():
                cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1) 
                    
                if self.use_distribution:
                    next_Q_dist = self.Operate.operator_target(
                        torch.cat((target_F1, target_F2), dim=0), 
                        torch.cat((zs, zs), dim=0)
                    ).squeeze()
                    next_Q1_dist, next_Q2_dist = next_Q_dist[:zs.size(0)], next_Q_dist[zs.size(0):]
                    next_Q_dist = torch.where(torch.sum(next_Q1_dist * self.support, dim=1).unsqueeze(-1) <= torch.sum(next_Q2_dist * self.support, dim=1).unsqueeze(-1), next_Q1_dist, next_Q2_dist)
                    target_Q = torch.sum(next_Q_dist * self.support, dim=1)
                    Tz = implicit_reward.unsqueeze(1) + discounts.unsqueeze(1) * self.support
                    Tz = Tz.clamp(min=self.minVal, max=self.maxVal)
                    b = (Tz - self.minVal) / (self.maxVal - self.minVal) * (self.num_atoms - 1)
                    l = b.floor().long()
                    u = b.ceil().long()
                    offset = torch.linspace(0, (b.size(0) - 1) * self.num_atoms, b.size(0), device=self._device).long().unsqueeze(1).expand(b.size(0), self.num_atoms)
                    proj_dist = torch.zeros(next_Q_dist.size(), device=self._device)
                    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_Q_dist * (u.float() - b)).view(-1))
                    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_Q_dist * (b - l.float())).view(-1))
                else:
                    if not self.use_dr3:
                        next_Q = self.Operate.operator_target(
                            torch.cat((target_F1, target_F2), dim=0), 
                            torch.cat((zs, zs), dim=0)
                        ).squeeze() 
                        next_Q = next_Q.squeeze()
                    else:
                        next_Q, next_Q_feature = self.Operate.operator_target(
                            torch.cat((target_F1, target_F2), dim=0), 
                            torch.cat((zs, zs), dim=0)
                        )
                        next_Q, next_Q_feature = next_Q.squeeze(), next_Q_feature.squeeze()
                    next_Q = torch.min(next_Q[:zs.size(0)], next_Q[zs.size(0):])
                    target_Q = implicit_reward.detach() + discounts.squeeze() * next_Q  # batch_size
            if self.use_distribution:
                Q_dist = self.Operate.operator(
                    torch.cat((F1, F2), dim=0), 
                    torch.cat((zs, zs), dim=0)
                ).squeeze()
                Q1_dist, Q2_dist = Q_dist[:zs.size(0)], Q_dist[zs.size(0):]
                Q_dist = torch.where(torch.sum(Q1_dist * self.support, dim=1).unsqueeze(-1) <= torch.sum(Q2_dist * self.support, dim=1).unsqueeze(-1), Q1_dist, Q2_dist)
                Q = torch.sum(Q_dist * self.support, dim=1)
                q_loss = -torch.sum(proj_dist * torch.log(Q_dist + 1e-8), dim=1).mean()
            else:
                if not self.use_dr3:
                    Q = self.Operate.operator_target(
                        torch.cat((F1, F2), dim=0), 
                        torch.cat((zs, zs), dim=0)
                    ).squeeze() 
                else:
                    Q, Q_feature = self.Operate.operator_target(
                        torch.cat((F1, F2), dim=0), 
                        torch.cat((zs, zs), dim=0)
                    )
                    Q, Q_feature = Q.squeeze(), Q_feature.squeeze()
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
                "train/auxiliary_loss": auxiliary_loss,
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
                "train/implicit_reward": implicit_reward.mean().item(),
                "train/target_Q": target_Q.mean().item(),
                "train/Q": Q.mean().item(),
            }
        else: 
            metrics = {
                "train/forward_backward_total_loss": total_loss,
                "train/forward_backward_fb_loss": fb_loss,
                "train/forward_backward_fb_diag_loss": fb_diag_loss,
                "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
                "train/auxiliary_loss": auxiliary_loss,
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
            }
        if self.use_dr3:
            metrics["train/dr3_reg"] = dr3_implict_reg.item()
        return total_loss, metrics, \
            F1, F2, B_next, B_rand, target_B, off_diagonal, actor_std_dev, dormant_indices, auxiliary_loss, M1_rand, M2_rand

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, discounts: torch.Tensor, step: int
    ) -> Dict[str, float]:

        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(observation, z, std, sample=True)
        actor_dormant_indices = None
        if self.use_dormant and step % self.reset_interval == 0 and step > 5000:
            _, actor_dormant_indices, _ = cal_dormant_ratio(self.actor, observation, z, std, type='actor', percentage=0.1)

        F1, F2 = self.Operate.forward_representation(
            observation=observation, z=z, action=action
        )
        if self.use_distribution:
            # u, std = self.Operate.operator(
            #     torch.cat((F1, F2), dim=0),
            #     torch.cat((z, z), dim=0)
            # )
            # u, std = u.squeeze(), std.squeeze()
            # current_distribution = torch.distributions.normal.Normal(u, std)
            # Q = current_distribution.sample()
            # actor_loss = -u
            Q_dist = self.Operate.operator(
                torch.cat((F1, F2), dim=0),
                torch.cat((z, z), dim=0)
            ).squeeze()
            Q1_dist, Q2_dist = Q_dist[:z.size(0)], Q_dist[z.size(0):]
            Q = torch.min(torch.sum(Q1_dist * self.support, dim=1), torch.sum(Q2_dist * self.support, dim=1))
            actor_loss = -Q
        else:
            if not self.use_dr3:
                Q = self.Operate.operator_target(
                    torch.cat((F1, F2), dim=0), 
                    torch.cat((z, z), dim=0)
                ).squeeze() 
            else:
                Q, _ = self.Operate.operator_target(
                    torch.cat((F1, F2), dim=0), 
                    torch.cat((z, z), dim=0)
                )
                Q = Q.squeeze()
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

        return metrics, actor_dormant_indices

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

        F1, F2 = self.Operate.forward_representation(
            observation=observation, z=z, action=action
        )
        if self.use_distribution:
            Q_dist = self.Operate.operator(
                torch.cat((F1, F2), dim=0),
                torch.cat((z, z), dim=0)
            ).squeeze()
            Q1_dist, Q2_dist = Q_dist[:z.size(0)], Q_dist[z.size(0):]
            Q = torch.min(torch.sum(Q1_dist * self.support, dim=1), torch.sum(Q2_dist * self.support, dim=1))
        else:
            if not self.use_dr3:
                Q = self.Operate.operator_target(
                    torch.cat((F1, F2), dim=0), 
                    torch.cat((z, z), dim=0)
                ).squeeze() 
            else:
                Q, _ = self.Operate.operator_target(
                    torch.cat((F1, F2), dim=0), 
                    torch.cat((z, z), dim=0)
                )
                Q = Q.squeeze()
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
        rand_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
        F1: torch.Tensor,
        F2: torch.Tensor,
        discount: torch.Tensor,
        B_rand: torch.Tensor,
        M_pealty_coefficient: float = 1.0,
        Fmu = None,
        M1_next = None,
        M2_next = None
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
        repeated_B_rand = B_rand.repeat(self.total_action_samples, 1, 1).reshape(
            self.total_action_samples * self.batch_size, -1
        )
        cml_cat_M = self.Operate.operator(
            torch.cat((cat_F1, cat_F2), dim=0),
            torch.cat((repeated_B_rand, repeated_B_rand), dim=0)
        ).squeeze()
        cml_cat_M1, cml_cat_M2 = cml_cat_M[:repeated_zs.size(0)], cml_cat_M[repeated_zs.size(0):]
        cml_logsumexp = torch.logsumexp(cml_cat_M1, dim=0) + torch.logsumexp(
            cml_cat_M2, dim=0
        )
        # if self.use_distribution:
        #     cql_cat_Q_dist = self.Operate.operator(
        #         torch.cat((cat_F1, cat_F2), dim=0),
        #         torch.cat((repeated_zs, repeated_zs), dim=0)
        #     ).squeeze()
        #     cql_cat_Q1_dist, cql_cat_Q2_dist = cql_cat_Q_dist[:repeated_zs.size(0)], cql_cat_Q_dist[repeated_zs.size(0):]
        #     cql_cat_Q1, cql_cat_Q2 = torch.sum(cql_cat_Q1_dist * self.support, dim=1), torch.sum(cql_cat_Q2_dist * self.support, dim=1)
        # else:
        #     if not self.use_dr3:
        #         cql_cat_Q = self.Operate.operator(
        #             torch.cat((cat_F1, cat_F2), dim=0),
        #             torch.cat((repeated_zs, repeated_zs), dim=0)
        #         ).squeeze()
        #     else:
        #         cql_cat_Q, _ = self.Operate.operator(
        #             torch.cat((cat_F1, cat_F2), dim=0),
        #             torch.cat((repeated_zs, repeated_zs), dim=0)
        #         )
        #         cql_cat_Q = cql_cat_Q.squeeze()
        #     cql_cat_Q1, cql_cat_Q2 = cql_cat_Q[:repeated_zs.size(0)], cql_cat_Q[repeated_zs.size(0):]
            
        # cql_logsumexp_Q = (
        #     torch.logsumexp(cql_cat_Q1, dim=0).mean()
        #     + torch.logsumexp(cql_cat_Q2, dim=0).mean()
        # )
        
        Mmu = 2 * self.Operate.operator(Fmu, B_rand).squeeze().detach()
        
        # if self.use_distribution:
        #     Q_dist = self.Operate.operator(
        #         torch.cat((F1, F2), dim=0),
        #         torch.cat((zs, zs), dim=0)
        #     ).squeeze()
        #     Q1_dist, Q2_dist = Q_dist[:zs.size(0)], Q_dist[zs.size(0):]
        #     Q1, Q2 = torch.sum(Q1_dist * self.support, dim=1), torch.sum(Q2_dist * self.support, dim=1)
        # else:
        #     if not self.use_dr3:
        #         Q = self.Operate.operator_target(
        #             torch.cat((F1, F2), dim=0), 
        #             torch.cat((zs, zs), dim=0)
        #         ).squeeze() 
        #     else:
        #         Q, _ = self.Operate.operator_target(
        #             torch.cat((F1, F2), dim=0), 
        #             torch.cat((zs, zs), dim=0)
        #         )
        #         Q = Q.squeeze()
        #     Q1, Q2 = Q[:zs.size(0)], Q[zs.size(0):]
        
        # conservative_penalty = (
        #     torch.maximum(cql_logsumexp_Q, Vmu).mean() - (Q1 + Q2).mean()
        # )
        conservative_penalty = (
            torch.maximum(cml_logsumexp, Mmu).mean() - (M1_next + M2_next).mean()
        )
        
        # metrics = {
        #     "train/cql_penalty": conservative_penalty.item(),
        #     "train/cql_cat_Q1": cql_cat_Q1.mean().item(),
        #     "train/cql_cat_Q2": cql_cat_Q2.mean().item(),
        # }       ]
        metrics = {
            "train/cql_penalty_M": conservative_penalty.item(),
            "train/cql_cat_M1": cml_cat_M1.mean().item(),
            "train/cql_cat_M2": cml_cat_M2.mean().item(),
        }
        return conservative_penalty, metrics
        
    def _tune_alpha(
        self,
        conservative_penalty: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

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

        else:
            alpha = self.alpha
            alpha_loss = 0.0

        metrics = {
            "train/alpha": alpha,
            "train/alpha_loss": alpha_loss,
        }

        return alpha, metrics
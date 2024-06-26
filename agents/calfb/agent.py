"""Module for specifiying the Calibrated Conservative Forward-Backward Agent."""

import torch
from typing import Dict, Tuple

from agents.fb.agent import FB
from agents.calfb.models import ForwardBackwardRepresentation
from agents.base import Batch


class CalFB(FB):

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
        critic_learning_rate: float,
        actor_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        total_action_samples: int,
        ood_action_weight: float,
        alpha: float,
        target_conservative_penalty: float,
        vcfb: bool,
        mcfb: bool,
        lagrange: bool = False,
    ):
        assert vcfb != mcfb
        self.vcfb = vcfb
        self.mcfb = mcfb
        if self.vcfb:
            name = "VCalFB"
        elif self.mcfb:
            name = "MCalFB"
        else:
            raise ValueError("Either vcfb or mcfb must be True")

        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_output_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_number_of_features=forward_number_of_features,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            actor_activation=actor_activation,
            critic_learning_rate=critic_learning_rate,
            actor_learning_rate=actor_learning_rate,
            learning_rate_coefficient=learning_rate_coefficient,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            batch_size=batch_size,
            z_mix_ratio=z_mix_ratio,
            gaussian_actor=gaussian_actor,
            std_dev_clip=std_dev_clip,
            std_dev_schedule=std_dev_schedule,
            tau=tau,
            device=device,
            name=name,
        )

        # NOTE: overwrites attribute created on super().__init__
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
        )
        # also need to recreate optimiser to include forward_mu
        self.FB_optimizer = torch.optim.Adam(
            [
                {"params": self.FB.forward_representation.parameters()},
                {"params": self.FB.forward_mu.parameters()},
                {
                    "params": self.FB.backward_representation.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                },
            ],
            lr=critic_learning_rate,
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

        self.alpha = alpha
        self.target_conservative_penalty = target_conservative_penalty

        # lagrange multiplier
        self.lagrange = lagrange
        self.critic_log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

        # optimizer
        self.critic_alpha_optimizer = torch.optim.Adam(
            [self.critic_log_alpha], lr=critic_learning_rate
        )

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        
        metrics = super().update(batch, step)

        self.soft_update_params(
            network=self.FB.forward_mu,
            target_network=self.FB.forward_mu_target,
            tau=self._tau,
        )

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

        (
            core_loss,
            core_metrics,
            F1,
            F2,
            B_next,
            M1_next,
            M2_next,
            target_B,
            off_diagonal,
            actor_std_dev,
        ) = self._update_fb_inner(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            discounts=discounts,
            zs=zs,
            step=step,
        )

        # calculate FB loss for exploration policy
        with torch.no_grad():
            target_Fmu = self.FB.forward_mu_target(next_observations)
        target_Mmu = torch.einsum("sd, td -> st", target_Fmu, target_B)

        Fmu = self.FB.forward_mu(observations)
        Mmu_next = torch.einsum("sd, td -> st", Fmu, B_next)

        fmub_off_diag_loss = (
            (Mmu_next - discounts * target_Mmu)[off_diagonal].pow(2).mean()
        )

        fmub_diag_loss = -2.0 * Mmu_next.diag().mean()

        fmub_loss = fmub_off_diag_loss + fmub_diag_loss

        # calculate MC or VC penalty
        if self.mcfb:
            (
                conservative_penalty,
                conservative_metrics,
            ) = self._measure_conservative_penalty(
                observations=observations,
                next_observations=next_observations,
                zs=zs,
                actor_std_dev=actor_std_dev,
                F1=F1,
                F2=F2,
                Fmu=Fmu,
                B_next=B_next,
                M1_next=M1_next,
                M2_next=M2_next,
            )
        # VCFB
        else:
            (
                conservative_penalty,
                conservative_metrics,
            ) = self._value_conservative_penalty(
                observations=observations,
                next_observations=next_observations,
                zs=zs,
                actor_std_dev=actor_std_dev,
                F1=F1,
                F2=F2,
                Fmu=Fmu,
            )

        # get alpha from conservative penalty
        alpha, alpha_metrics = self._tune_alpha(
            conservative_penalty=conservative_penalty
        )
        conservative_loss = alpha * conservative_penalty

        total_loss = core_loss + fmub_loss + conservative_loss

        self.FB_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.FB.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.FB_optimizer.step()

        metrics = {
            **core_metrics,
            **conservative_metrics,
            **alpha_metrics,
            "train/fmub_loss": fmub_loss,
            "train/forward_backward_total_loss": total_loss,
        }

        return metrics

    def _measure_conservative_penalty(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
        F1: torch.Tensor,
        F2: torch.Tensor,
        Fmu: torch.Tensor,
        B_next: torch.Tensor,
        M1_next: torch.Tensor,
        M2_next: torch.Tensor,
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
        ood_F1, ood_F2 = self.FB.forward_representation(
            repeated_observations_ood, ood_actions, repeated_zs_ood
        )  # [ood_action_samples * batch_size, latent_dim]

        if self.actor_action_samples > 0:
            actor_current_F1, actor_current_F2 = self.FB.forward_representation(
                repeated_observations_actor, actor_current_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            actor_next_F1, actor_next_F2 = self.FB.forward_representation(
                repeated_next_observations_actor, actor_next_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            repeated_F1, repeated_F2 = F1.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1), F2.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(
                self.actor_action_samples * self.batch_size, -1
            )
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

        cml_cat_M1 = torch.einsum("sd, td -> st", cat_F1, B_next).reshape(
            self.total_action_samples, self.batch_size, -1
        )
        cml_cat_M2 = torch.einsum("sd, td -> st", cat_F2, B_next).reshape(
            self.total_action_samples, self.batch_size, -1
        )

        cml_logsumexp = torch.logsumexp(cml_cat_M1, dim=0) + torch.logsumexp(
            cml_cat_M2, dim=0
        )

        # get Mmu prediction using forward network for mu
        # NOTE: multiplying by 2 because all other terms are summed across F1 and F2
        Mmu = 2 * torch.einsum("sd, td -> st", Fmu, B_next).detach()

        conservative_penalty = (
            torch.maximum(cml_logsumexp, Mmu).mean() - (M1_next + M2_next).mean()
        )

        metrics = {
            "train/cml_penalty": conservative_penalty.item(),
            "train/cml_cat_M1": cml_cat_M1.mean().item(),
            "train/cml_cat_M2": cml_cat_M2.mean().item(),
            "train/Mmu": Mmu.mean().item(),
        }

        return conservative_penalty, metrics

    def _value_conservative_penalty(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
        F1: torch.Tensor,
        F2: torch.Tensor,
        Fmu: torch.Tensor,
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
        ood_F1, ood_F2 = self.FB.forward_representation(
            repeated_observations_ood, ood_actions, repeated_zs_ood
        )  # [ood_action_samples * batch_size, latent_dim]

        if self.actor_action_samples > 0:
            actor_current_F1, actor_current_F2 = self.FB.forward_representation(
                repeated_observations_actor, actor_current_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            actor_next_F1, actor_next_F2 = self.FB.forward_representation(
                repeated_next_observations_actor, actor_next_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            repeated_F1, repeated_F2 = F1.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1), F2.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(
                self.actor_action_samples * self.batch_size, -1
            )
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
        cql_cat_Q1 = torch.einsum("sd, sd -> s", cat_F1, repeated_zs).reshape(
            self.total_action_samples, self.batch_size, -1
        )
        cql_cat_Q2 = torch.einsum("sd, sd -> s", cat_F2, repeated_zs).reshape(
            self.total_action_samples, self.batch_size, -1
        )

        cql_logsumexp = torch.logsumexp(cql_cat_Q1, dim=0) + torch.logsumexp(
            cql_cat_Q2, dim=0
        )

        # get Vmu prediction using forward network for mu
        # NOTE: multiplying by 2 because all other terms are summed across F1 and F2
        Vmu = 2 * torch.einsum("sd, sd -> s", Fmu, zs).detach()

        # get existing Qs
        Q1, Q2 = [torch.einsum("sd, sd -> s", F, zs) for F in [F1, F2]]

        conservative_penalty = (
            torch.maximum(cql_logsumexp, Vmu).mean() - (Q1 + Q2).mean()
        )

        metrics = {
            "train/cql_penalty": conservative_penalty.item(),
            "train/cql_cat_Q1": cql_cat_Q1.mean().item(),
            "train/cql_cat_Q2": cql_cat_Q2.mean().item(),
            "train/Vmu": Vmu.mean().item(),
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

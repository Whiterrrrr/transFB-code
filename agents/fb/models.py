"""Module for fully defined models used by the forward-backward agent. """

from agents.fb.base import (
    ForwardModel,
    BackwardModel,
    ActorModel,
    AbstractPreprocessor,
)
from typing import Tuple

import torch


class ForwardRepresentation(torch.nn.Module):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        number_of_features: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        device: torch.device,
        forward_activation: str,
        use_2branch = False
    ):
        super().__init__()
        self.use_2branch = use_2branch
        self.z_dimension = z_dimension
        self.device = device
        # pre-processors
        self.obs_action_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=action_length,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        self.F1 = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=number_of_features,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
        )

        self.F2 = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=number_of_features,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
        )
        
        if use_2branch:
            self.scale_branch = ForwardModel(
                preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
                number_of_preprocessed_features=number_of_features,
                z_dimension=1,
                hidden_dimension=forward_hidden_dimension,
                hidden_layers=1,
                device=device,
                activation=forward_activation,
            )

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action_embedding = self.obs_action_preprocessor(
            torch.cat([observation, action], dim=-1)
        )
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))

        h = torch.cat([obs_action_embedding, obs_z_embedding], dim=-1)
        if self.use_2branch:
            scale = self.scale_branch(h)
            f1 = torch.sqrt(
                torch.tensor(self.z_dimension, dtype=torch.int, device=self.device)
            ) * torch.nn.functional.normalize(self.F1(h), dim=1)
            
            f2 = torch.sqrt(
                torch.tensor(self.z_dimension, dtype=torch.int, device=self.device)
            ) * torch.nn.functional.normalize(self.F2(h), dim=1)
            return f1 * scale, f2 * scale

        return self.F1(h), self.F2(h)


class BackwardRepresentation(torch.nn.Module):

    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        device: torch.device,
        backward_activation: torch.nn,
        preprocess: bool=False,
        backward_preporcess_hidden_dimension: int=None,
        backward_preporcess_output_dim: int=None,
        backward_preporcess_hidden_layers: int=None,
        backward_preporcess_activation: torch.nn=None,
    ):
        super().__init__()
        self.obs_preprocessor = preprocess
        if preprocess:
            self.obs_preprocessor = AbstractPreprocessor(
                observation_length=observation_length,
                concatenated_variable_length=0,
                hidden_dimension=backward_preporcess_hidden_dimension,
                feature_space_dimension=backward_preporcess_output_dim,
                hidden_layers=backward_preporcess_hidden_layers,
                activation=backward_preporcess_activation,
                device=device,
            )
            observation_length = backward_preporcess_output_dim
                
        self.B = BackwardModel(
            observation_length=observation_length,
            z_dimension=z_dimension,
            hidden_dimension=backward_hidden_dimension,
            hidden_layers=backward_hidden_layers,
            device=device,
            activation=backward_activation,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        if self.obs_preprocessor:
            observation = self.obs_preprocessor(observation)
        return self.B(observation)


class ForwardBackwardRepresentation(torch.nn.Module):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        number_of_features: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
        use_iql = False
    ):
        super().__init__()
        self.forward_representation = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
        )

        self.backward_representation = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
        )

        self.forward_representation_target = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
        )

        self.backward_representation_target = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
        )
        
        if use_iql:
            self.state_forward_representation = StateForwardRepresentation(
                observation_length=observation_length,
                preprocessor_hidden_dimension=preprocessor_hidden_dimension,
                preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
                preprocessor_hidden_layers=preprocessor_hidden_layers,
                preprocessor_activation=preprocessor_activation,
                number_of_features=number_of_features,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                device=device,
                forward_activation=forward_activation,
            )
            
            self.state_forward_representation_target = StateForwardRepresentation(
                observation_length=observation_length,
                preprocessor_hidden_dimension=preprocessor_hidden_dimension,
                preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
                preprocessor_hidden_layers=preprocessor_hidden_layers,
                preprocessor_activation=preprocessor_activation,
                number_of_features=number_of_features,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                device=device,
                forward_activation=forward_activation,
            )

        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
        
        
class StateForwardRepresentation(torch.nn.Module):
    
    def __init__(
        self,
        observation_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        number_of_features: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        device: torch.device,
        forward_activation: str,
        forward_output_dimension = None,
        use_2branch = False
    ):
        super().__init__()
        if not forward_output_dimension:
            forward_output_dimension = z_dimension
            
        # pre-processors
        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )
        
        self.K = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=number_of_features-1,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            # forward_output_dimension=forward_output_dimension,
            # use_2branch=use_2branch
        )
        
    def forward(
        self, observation: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))
        K = self.K(obs_z_embedding)
        return K


class Actor(torch.nn.Module):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        z_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_activation: torch.nn,
        std_dev_schedule: str,
        std_dev_clip: float,
        device: torch.device,
        gaussian_actor: bool,
    ):
        super().__init__()

        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=2,
            z_dimension=z_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            std_dev_clip=std_dev_clip,
            device=device,
            gaussian_actor=gaussian_actor,
        )

        self._std_dev_schedule = std_dev_schedule

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        std: float,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        action, action_dist = self.actor(observation, z, std, sample)

        return action, action_dist

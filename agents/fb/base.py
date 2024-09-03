"""Module defining base classed for forward-backward agent."""

import abc
import torch
import torch.nn as nn
from agents.base import AbstractMLP, AbstractActor, AbstractGaussianActor
from typing import Tuple
import torch.nn.functional as F

class AbstractPreprocessor(AbstractMLP, metaclass=abc.ABCMeta):

    def __init__(
        self,
        observation_length: int,
        concatenated_variable_length: int,
        hidden_dimension: int,
        feature_space_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
    ):
        super().__init__(
            input_dimension=observation_length + concatenated_variable_length,
            output_dimension=feature_space_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            preprocessor=True,
        )

    def forward(self, concatenation: torch.tensor) -> torch.tensor:

        features = self.trunk(concatenation)  # pylint: disable=E1102

        return features


class ForwardModel(AbstractMLP):

    def __init__(
        self,
        preprocessor_feature_space_dimension: int,
        number_of_preprocessed_features: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
    ):
        super().__init__(
            input_dimension=preprocessor_feature_space_dimension
            * number_of_preprocessed_features,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        
        z_ = self.trunk(h)  # pylint: disable=E1102
        return z_


class BackwardModel(AbstractMLP):

    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )
        self._z_dimension = z_dimension

    def forward(self, observation: torch.Tensor) -> torch.Tensor:

        z = self.trunk(observation)  # pylint: disable=E1102

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.int, device=self.device)
        ) * torch.nn.functional.normalize(z, dim=1)

        return z


class FF_pred_model(AbstractMLP):
    def __init__(
        self, 
        observation_length: int,
        z_dim: int, 
        hidden_dimension: int, 
        hidden_layers: int, 
        activation: str, 
        device: torch.device, 
    ):
        super().__init__(
            input_dimension=z_dim*2,
            output_dimension=observation_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )
    
    def forward(self, F1: torch.Tensor, F2: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        input1, input2 = torch.cat([F1, z], dim=-1), torch.cat([F2, z], dim=-1)
        z1_, z2_ = self.trunk(input1), self.trunk(input2)
        return z1_, z2_
    

class ActorModel(nn.Module):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        number_of_features: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        z_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        gaussian_actor: bool,
        actor_activation: torch.nn,
        std_dev_clip: float,
        device: torch.device,
    ):
        super().__init__()

        self.actor = (AbstractGaussianActor if gaussian_actor else AbstractActor)(
            observation_length=preprocessor_feature_space_dimension
            * number_of_features,
            action_length=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            activation=actor_activation,
            device=device,
        )

        # pre-procossors
        self.obs_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=0,  # preprocess observation alone
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

        self._std_dev_clip = std_dev_clip

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        std: float,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:

        obs_embedding = self.obs_preprocessor(observation)
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1)) # TODO understand
        h = torch.cat([obs_embedding, obs_z_embedding], dim=-1)

        action_dist = (
            self.actor(h)
            if type(self.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
            else self.actor(h, std)
        )

        if sample:
            action = (
                action_dist[0]
                if type(self.actor)  # pylint: disable=unidiomatic-typecheck
                == AbstractGaussianActor
                else action_dist.sample(clip=self._std_dev_clip)
            )

        else:
            if type(self.actor) == AbstractGaussianActor:
                action = action_dist[0]
            else:
                action = action_dist.mean

        return action.clip(-1, 1), action_dist
    
    
class BCQ_actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, max_action=1, phi=0.05):
        super(BCQ_actor, self).__init__()
        self.device = device
        self.l1 = nn.Linear(state_dim + action_dim, 400).to(self.device)
        self.l2 = nn.Linear(400, 300).to(self.device)
        self.l3 = nn.Linear(300, action_dim).to(self.device)
        
        self.max_action = max_action
        self.phi = phi


    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1))).to(self.device)
        a = F.relu(self.l2(a)).to(self.device)
        a = self.phi * self.max_action * torch.tanh(self.l3(a)).to(self.device)
        return (a + action).clamp(-self.max_action, self.max_action)


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, device, latent_dim=25, max_action=1):
        super(VAE, self).__init__()
        self.device = device
        self.e1 = nn.Linear(state_dim + action_dim, 750).to(self.device)
        self.e2 = nn.Linear(750, 750).to(self.device)

        self.mean = nn.Linear(750, latent_dim).to(self.device)
        self.log_std = nn.Linear(750, latent_dim).to(self.device)

        self.d1 = nn.Linear(state_dim + latent_dim, 750).to(self.device)
        self.d2 = nn.Linear(750, 750).to(self.device)
        self.d3 = nn.Linear(750, action_dim).to(self.device)

        self.max_action = max_action
        self.latent_dim = latent_dim


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std).to(self.device)
        
        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
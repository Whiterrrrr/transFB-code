from agents.fb.models import (
    BackwardRepresentation,
    AbstractPreprocessor
)
from agents.base import AbstractMLP
import torch
from torch import nn
from torch.nn import functional as F
import math

def weight_init(m) -> None:
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
            
            
class Approximator(AbstractMLP):
    def __init__(
        self, 
        input_dimension: int, 
        output_dimension: int, 
        hidden_dimension: int, 
        hidden_layers: int, 
        activation: str, 
        device: torch.device, 
        layernorm: bool = True
        ):
        
        super().__init__(
            input_dimension, 
            output_dimension, 
            hidden_dimension, 
            hidden_layers, 
            activation, 
            device, 
            layernorm
        )
        
    def forward(self, x):
        return self.trunk(x)
    
class MQapproximator(nn.Module):
    def __init__(
        self, 
        z_dim,         
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_activation: str,
        device: torch.device,      
        ):
        super(MQapproximator, self).__init__()
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
            concatenated_variable_length=z_dim,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )
        
        self.approximator = Approximator(
            input_dimension=2*preprocessor_feature_space_dimension+z_dim,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            activation=forward_activation,
            device=device,
            output_dimension=1,
        )
        self.apply(weight_init)
        
    def forward(self, observation: torch.Tensor, action: torch.Tensor, z: torch.Tensor, B=None):

        obs_action_embedding = self.obs_action_preprocessor(
            torch.cat([observation, action], dim=-1)
        )
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))
        if B is not None:
            h = torch.cat([obs_action_embedding, obs_z_embedding, B], dim=-1)
        else:
            h = torch.cat([obs_action_embedding, obs_z_embedding, z], dim=-1)
        return self.approximator(h)     
    
class deepNetApproximator(torch.nn.Module):
    
    def __init__(
        self,
        z_dim,
        observation_length,
        action_length,
        preprocessor_hidden_dimension,
        preprocessor_feature_space_dimension,
        preprocessor_hidden_layers,
        preprocessor_activation,
        forward_hidden_dimension,
        forward_hidden_layers,
        forward_activation,
        device,
        backward_hidden_dimension,
        backward_hidden_layers,
        backward_activation,
        discount,
        orthonormalisation_coefficient,
        backward_preporcess_hidden_dimension=None,
        backward_preporcess_output_dim=None,
        backward_preporcess_hidden_layers=None,
        backward_preporcess_activation=None,
        backward_preprocess=None,
        ):
        super().__init__()                  
        
        self.apprximator_1 = MQapproximator(
            z_dim=z_dim,         
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,  
        )
        self.apprximator_1_target = MQapproximator(
            z_dim=z_dim,         
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,  
        )
        
        self.apprximator_2 = MQapproximator(
            z_dim=z_dim,         
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,  
        )
        
        self.apprximator_2_target = MQapproximator(
            z_dim=z_dim,         
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,  
        )
        
        self.bacewardNet = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dim,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
            backward_preporcess_hidden_dimension=backward_preporcess_hidden_dimension,
            backward_preporcess_output_dim=backward_preporcess_output_dim,
            backward_preporcess_hidden_layers=backward_preporcess_hidden_layers,
            backward_preporcess_activation=backward_preporcess_activation,
            preprocess=backward_preprocess,
        )
        self.bacewardNet_target = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dim,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
            backward_preporcess_hidden_dimension=backward_preporcess_hidden_dimension,
            backward_preporcess_output_dim=backward_preporcess_output_dim,
            backward_preporcess_hidden_layers=backward_preporcess_hidden_layers,
            backward_preporcess_activation=backward_preporcess_activation,
            preprocess=backward_preprocess,
        )
        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
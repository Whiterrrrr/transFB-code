from agents.fb.models import (
    ForwardRepresentation,
    BackwardRepresentation,
    StateForwardRepresentation
)
import torch.distributions as dist

import torch
from torch import nn
from torch.nn import functional as F

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
            
class SelfAttention(nn.Module):
    def __init__(self, z_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(z_dim, z_dim)
        self.key = nn.Linear(z_dim, z_dim)
        self.value = nn.Linear(z_dim, z_dim)
        self.z_dim = z_dim
        self.apply(weight_init)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)   
        V = self.value(x) 
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.bmm(attention_weights, V)
        return output

    
class BidirectionalAttentionMixnet(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_attention_heads, n_attention_layers, n_linear_layers, dropout_rate, device, use_res, use_fed, use_VIB=False, use_cross_attention=True):
        self.use_res = use_res
        self.use_VIB = use_VIB
        super(BidirectionalAttentionMixnet, self).__init__()
        self.attention_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.use_cross_attention = use_cross_attention
        
        for _ in range(n_attention_layers):
            self.attention_layers.append(SelfAttention(z_dim).to(device))
            self.attention_layers.append(nn.LayerNorm(z_dim).to(device))
            self.attention_layers.append(nn.Dropout(dropout_rate).to(device))
            
            if use_fed:
                self.attention_layers.append(nn.Linear(z_dim, z_dim * 4).to(device))
                self.attention_layers.append(nn.Linear(z_dim * 4, z_dim).to(device))
                self.attention_layers.append(nn.LayerNorm(z_dim).to(device)) 
            
        self.linear_layers.append(nn.Linear(z_dim * 2, hidden_dim).to(device))
        
        for _ in range(n_linear_layers-1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device))

        if self.use_VIB:
            self.fc_mu  = nn.Linear(hidden_dim, hidden_dim).to(device)
            self.fc_std = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.output = nn.Linear(hidden_dim, 1).to(device)
        self.apply(weight_init)
    
    
    def forward(self, forward_rep, backward_rep):
        combined = torch.cat((forward_rep.unsqueeze(1), backward_rep.unsqueeze(1)), dim=1)
        for layer in self.attention_layers:
            if isinstance(layer, SelfAttention) and self.use_res:
                combined = combined + layer(combined)
            else:
                combined = layer(combined)
        combined_output = torch.cat((combined[:, 0, :], combined[:, 1, :]), dim=-1)
        for layer in self.linear_layers:
            combined_output = layer(combined_output)
        if self.use_VIB:
            mu = self.fc_mu(combined_output)
            log_std = self.fc_std(combined_output)
            std = torch.exp(log_std)
            normal_dist = dist.Normal(mu, std)
            combined_output = normal_dist.rsample()
            return self.output(combined_output), mu, std
        return self.output(combined_output)
    
    
class MixNetRepresentation(torch.nn.Module):
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
        backward_preprocess: bool,
        backward_preporcess_hidden_dimension: int,
        backward_preporcess_output_dim: int,
        backward_preporcess_hidden_layers: int,
        backward_preporcess_activation: str,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        operator_hidden_dimension: int,
        operator_hidden_layers: int,
        operator_activation: str,
        trans_dimension: int,
        num_attention_heads: int,
        n_attention_layers: int,
        n_linear_layers: int,
        dropout_rate: float,
        forward_activation: str,
        backward_activation: str,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
        use_res: bool,
        use_fed: bool,
        use_VIB: bool,
        use_cross_attention,
        use_2branch: bool
        
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
            use_2branch=use_2branch
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
            use_2branch=use_2branch
        )
        
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
            use_2branch=use_2branch
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
            use_2branch=use_2branch
        )
        
        self.backward_representation = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
            preprocess=backward_preprocess,
            backward_preporcess_hidden_dimension=backward_preporcess_hidden_dimension,
            backward_preporcess_output_dim=backward_preporcess_output_dim,
            backward_preporcess_hidden_layers=backward_preporcess_hidden_layers,
            backward_preporcess_activation=backward_preporcess_activation,
        )
        
        self.backward_representation_target = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
            preprocess=backward_preprocess,
            backward_preporcess_hidden_dimension=backward_preporcess_hidden_dimension,
            backward_preporcess_output_dim=backward_preporcess_output_dim,
            backward_preporcess_hidden_layers=backward_preporcess_hidden_layers,
            backward_preporcess_activation=backward_preporcess_activation,
        )
        
        self.operator = BidirectionalAttentionMixnet(
            z_dim=z_dimension,
            hidden_dim=trans_dimension,
            num_attention_heads=num_attention_heads,
            n_attention_layers=n_attention_layers,
            n_linear_layers=n_linear_layers,
            dropout_rate=dropout_rate,
            device=device,
            use_res=use_res,
            use_fed=use_fed,
            use_VIB=use_VIB,
            use_cross_attention=use_cross_attention
        )
        
        self.operator_target = BidirectionalAttentionMixnet(
            z_dim=z_dimension,
            hidden_dim=trans_dimension,
            num_attention_heads=num_attention_heads,
            n_attention_layers=n_attention_layers,
            n_linear_layers=n_linear_layers,
            dropout_rate=dropout_rate,
            device=device,
            use_res=use_res,
            use_fed=use_fed,
            use_VIB=use_VIB,
            use_cross_attention=use_cross_attention
        )
            
        self.apply(weight_init)
        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
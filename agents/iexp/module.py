from agents.fb.models import (
    ForwardRepresentation,
    BackwardRepresentation,
    StateForwardRepresentation
)
import torch.distributions as dist
from agents.base import AbstractMLP
import torch
from torch import nn
from torch.nn import functional as F
import math

def weight_init(m) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            
            
class Linear_block(nn.Module):
    def __init__(self, hidden_dim, n_linear_layers, device):
        super(Linear_block, self).__init__()
        self.mean = nn.ModuleList()
        self.std = nn.ModuleList()
        for _ in range(n_linear_layers):
            self.mean.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            self.mean.append(nn.LeakyReLU())
            self.std.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            self.std.append(nn.LeakyReLU())
        self.mean.append(nn.Linear(hidden_dim, 1).to(device))
        self.std.append(nn.Linear(hidden_dim, 1).to(device))
        self.std.append(nn.Softplus())
        
    def forward(self, x):
        u, std = x, x
        for layer in self.mean:
            u = layer(u)
        for layer in self.std:
            std = layer(std)
        std = std + 1e-6
        return u, std
    
    
class MixNet(AbstractMLP):
    def __init__(
        self,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
        layernorm = True
    ):
        super().__init__(
            input_dimension=2*z_dimension,
            output_dimension=1,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm
        )
        self._z_dimension = z_dimension
        self.apply(weight_init)

    def forward(self, observation: torch.Tensor, B) -> torch.Tensor:
        z = self.trunk(torch.cat([observation, B], dim=-1))  # pylint: disable=E1102
        return z   
    
    
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


class CrossAttention(nn.Module):
    def __init__(self, z_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(z_dim, z_dim)
        self.key = nn.Linear(z_dim, z_dim)
        self.value = nn.Linear(z_dim, z_dim)
        self.z_dim = z_dim
        self.apply(weight_init)

    def forward(self, x, y):
        Q = self.query(x)
        K = self.key(y)   
        V = self.value(y) 
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.bmm(attention_weights, V)
        return output
    
    
class MulHeadSelfAttention(nn.Module):
    def __init__(self, z_dim, num_attention_heads, dropout_prob):   
        super(MulHeadSelfAttention, self).__init__()
        if z_dim % num_attention_heads != 0: 
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (z_dim, num_attention_heads))
        self.num_attention_heads = num_attention_heads  
        self.attention_head_size = int(z_dim / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        
        self.query = nn.Linear(z_dim, self.all_head_size)
        self.key = nn.Linear(z_dim, self.all_head_size)
        self.value = nn.Linear(z_dim, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) 
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  
        mixed_key_layer = self.key(hidden_states)    
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) 
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) 

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  

        attention_probs = nn.Softmax(dim=-1)(attention_scores) 
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer) 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    
    
class BidirectionalAttentionMixnet(nn.Module):
    def __init__(
        self, 
        z_dim, 
        hidden_dim, 
        num_attention_heads, 
        n_attention_layers, 
        n_linear_layers, 
        dropout_rate, 
        device, 
        use_res, 
        use_fed, 
        use_cross_attention=False, 
        use_dr3=False,
        use_feature_norm=False,
        use_linear_res=False,
        use_forward_backward_cross=False,
    ):
        super(BidirectionalAttentionMixnet, self).__init__()
        self.use_res = use_res
        self.attention_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.use_cross_attention = use_cross_attention
        self.use_dr3 = use_dr3
        self.use_feature_norm=use_feature_norm
        self.use_linear_res=use_linear_res
        self.use_forward_backward_cross=use_forward_backward_cross
        
        for _ in range(n_attention_layers):
            self.attention_layers.append(CrossAttention(z_dim).to(device)) if self.use_cross_attention else self.attention_layers.append(SelfAttention(z_dim).to(device))
            self.attention_layers.append(nn.LayerNorm(z_dim).to(device))
            self.attention_layers.append(nn.Dropout(dropout_rate).to(device))
            if use_fed:
                self.attention_layers.append(nn.Linear(z_dim, z_dim * 4).to(device))
                self.attention_layers.append(nn.Linear(z_dim * 4, z_dim).to(device))
                self.attention_layers.append(nn.LayerNorm(z_dim).to(device))
        self.linear_layers.append(nn.Linear(z_dim, hidden_dim).to(device)) if self.use_cross_attention else self.linear_layers.append(nn.Linear(z_dim * 2, hidden_dim).to(device))
        # self.linear_layers.append(nn.LeakyReLU().to(device))
        n_linear_layers -= 1

        for _ in range(n_linear_layers-1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            # self.linear_layers.append(nn.LeakyReLU().to(device))
                # self.linear_layers.append(nn.LayerNorm(hidden_dim).to(device))
        self.output = nn.Linear(hidden_dim, 1).to(device)
        self.apply(weight_init)
    
    def forward(self, forward_rep, backward_rep):
        f_len, b_len = torch.norm(forward_rep, p=2, dim=1), torch.norm(backward_rep, p=2, dim=1)
        if self.use_forward_backward_cross:
            combined_output = None
            for layer in self.attention_layers:
                if isinstance(layer, CrossAttention) and combined_output == None:
                    combined1 = layer(backward_rep.unsqueeze(1), forward_rep.unsqueeze(1))
                    combined2 = layer(forward_rep.unsqueeze(1), backward_rep.unsqueeze(1))
                    combined_output = combined1.squeeze(1) + combined2.squeeze(1)
                else:
                    combined_output = layer(combined_output)
                    combined_output = combined_output.squeeze(1)
        elif not self.use_cross_attention:
            combined = torch.cat((forward_rep.unsqueeze(1), backward_rep.unsqueeze(1)), dim=1)
            for layer in self.attention_layers:
                combined = combined + layer(combined) if (isinstance(layer, SelfAttention) and self.use_res) else layer(combined)
            combined_output = torch.cat((combined[:, 0, :], combined[:, 1, :]), dim=-1)
        else:
            forward_rep = backward_rep = F.normalize(forward_rep, p=2, dim=1) + F.normalize(backward_rep, p=2, dim=1)
            # forward_rep = backward_rep = forward_rep + backward_rep
            combined = None
            for layer in self.attention_layers:
                if isinstance(layer, CrossAttention):
                    combined = layer(backward_rep.unsqueeze(1), forward_rep.unsqueeze(1)) if combined is None else layer(combined, forward_rep.unsqueeze(1))
                    combined = combined + backward_rep.unsqueeze(1) if self.use_res else combined
                else:
                    combined = layer(combined)
            combined_output = combined.squeeze(1)
        linear_cnt = 0
        for layer in self.linear_layers:
            combined_output = layer(combined_output) if ((not self.use_linear_res) or linear_cnt == 0) else combined_output + layer(combined_output)
            linear_cnt += 1
        if self.use_dr3:
            return self.output(combined_output), combined_output
        if self.use_feature_norm:
            combined_output = combined_output / combined_output.norm(p=2, dim=1, keepdim=True)
        return self.output(combined_output).squeeze() * f_len * b_len
        # return self.output(combined_output).squeeze()
    
    
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
        use_trasnformer: bool,
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
        trans_dimension: int,
        num_attention_heads: int,
        n_attention_layers: int,
        n_linear_layers: int,
        dropout_rate: float,
        forward_activation: str,
        backward_activation: str,
        operator_activation: str,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
        use_res: bool,
        use_fed: bool,
        use_2branch: bool,
        use_feature_norm,
        use_cross_attention: bool,
        use_linear_res=False,
        use_forward_backward_cross=False
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
            use_cross_attention=use_cross_attention,
            use_feature_norm=use_feature_norm,
            use_linear_res=use_linear_res,
            use_forward_backward_cross=use_forward_backward_cross
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
            use_cross_attention=use_cross_attention,
            use_feature_norm=use_feature_norm,
            use_linear_res=use_linear_res,
            use_forward_backward_cross=use_forward_backward_cross
        )
            
        self.apply(weight_init)
        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
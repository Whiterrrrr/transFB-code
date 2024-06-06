from agents.fb.models import (
    ForwardRepresentation,
    BackwardRepresentation,
)
import torch.distributions as dist
from agents.base import AbstractMLP, AbstractActor, AbstractGaussianActor

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
            
class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, device) -> None:
        super().__init__()
        # self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        # self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim, 'tanh')
        self.inverse_dynamic_net = AbstractMLP(2 * z_dim, action_dim, hidden_dim, 2, 'relu', device=device)
        self.apply(weight_init)

    def forward(self, backeard_rep: torch.Tensor, action: torch.Tensor, next_backeard_rep: torch.Tensor):
        # predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        # forward_error = (next_phi.detach() - predicted_next_obs).pow(2).mean()
        predicted_action = self.inverse_dynamic_net.trunk(torch.cat([backeard_rep, next_backeard_rep], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        # icm_loss = forward_error + backward_error
        return icm_loss
            
            
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

class CrossAttentionModel(nn.Module):
    def __init__(self, d_model, nhead, dropout):   
        super(CrossAttentionModel, self).__init__()
        self.self_attn_f = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.self_attn_b = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm1_f = nn.LayerNorm(d_model)
        self.norm1_b = nn.LayerNorm(d_model)
        self.norm2_cross = nn.LayerNorm(d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, F, B):
        F = F.unsqueeze(0)
        B = B.unsqueeze(0)
        F, _ = self.self_attn_f(F, F, F)
        F = self.norm1_f(F)
        B, _ = self.self_attn_b(B, B, B)
        B = self.norm1_b(B)
        attn_output, _ = self.cross_attention(F, B, B)
        attn_output = self.norm2_cross(attn_output)
        attn_output = attn_output.squeeze(0)
        final_output = self.final_linear(attn_output)
        return final_output
    

class CrossAttentionModel_with_fed(nn.Module):
    def __init__(self, d_model, nhead, dropout):   
        super(CrossAttentionModel_with_fed, self).__init__()
        self.self_attn_f = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.self_attn_b = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        
        # Layer normalization modules
        self.norm1_f = nn.LayerNorm(d_model)
        self.norm1_b = nn.LayerNorm(d_model)
        self.norm2_cross = nn.LayerNorm(d_model)

        # Feed-forward networks for each attention
        self.ffn_f = self._make_feedforward(d_model, dropout)
        self.ffn_b = self._make_feedforward(d_model, dropout)
        self.ffn_cross = self._make_feedforward(d_model, dropout)

        # Final linear and normalization
        self.final_linear = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def _make_feedforward(self, d_model, dropout):
        return nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, F, B):
        # Add batch dimension
        F = F.unsqueeze(0)
        B = B.unsqueeze(0)
        # Self-attention and feed-forward for F
        F_attn, _ = self.self_attn_f(F, F, F)
        F = self.norm1_f(F_attn)  # Apply layer normalization
        F_ffn = self.ffn_f(F)

        # Self-attention and feed-forward for B
        B_attn, _ = self.self_attn_b(B, B, B)
        B = self.norm1_b(B_attn)  # Apply layer normalization
        B_ffn = self.ffn_b(B)

        # Cross-attention and feed-forward
        attn_output, _ = self.cross_attention(F_ffn, B_ffn, B_ffn)
        attn_output = attn_output + self.dropout(attn_output)  # Apply residual connection
        attn_output = self.norm2_cross(attn_output)  # Apply layer normalization
        cross_ffn = self.ffn_cross(attn_output)

        # Remove batch dimension
        attn_output = cross_ffn.squeeze(0)

        # Final linear layer and normalization
        final_output = self.final_linear(attn_output)
        final_output = self.final_norm(final_output)  # Apply layer normalization

        return final_output


    
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
    
# class BidirectionalAttention(nn.Module):
#     def __init__(self, z_dim, hidden_dim):
#         super(BidirectionalAttention, self).__init__()
#         self.attention = SelfAttention(z_dim)
#         self.linear = nn.Linear(z_dim * 2, hidden_dim)

#     def forward(self, forward_rep, backward_rep):
#         combined = torch.cat((forward_rep.unsqueeze(1), backward_rep.unsqueeze(1)), dim=1)
#         attention_output = self.attention(combined)
#         combined_output = torch.cat((attention_output[:, 0, :], attention_output[:, 1, :]), dim=-1)
#         output = self.linear(combined_output)
#         return output
    
class BidirectionalAttentionMixnet(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_attention_heads, n_attention_layers, n_linear_layers, dropout_rate, device, use_res, use_fed, use_VIB=False, use_cross_attention=False):
        self.use_res = use_res
        self.use_VIB = use_VIB
        super(BidirectionalAttentionMixnet, self).__init__()
        self.attention_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.use_cross_attention = use_cross_attention
        
        # if use_cross_attention:
        #     self.attention_layers.append(CrossAttentionModel(z_dim, num_attention_heads, dropout_rate).to(device))
        # else:
        for _ in range(n_attention_layers):
            self.attention_layers.append(SelfAttention(z_dim).to(device))
            self.attention_layers.append(nn.LayerNorm(z_dim).to(device))
            self.attention_layers.append(nn.Dropout(dropout_rate).to(device))
            
            if use_fed:
                self.attention_layers.append(nn.Linear(z_dim, z_dim * 4).to(device))
                # self.attention_layers.append(nn.ELU())
                self.attention_layers.append(nn.Linear(z_dim * 4, z_dim).to(device))
                self.attention_layers.append(nn.LayerNorm(z_dim).to(device)) 
            
        if not self.use_cross_attention:
            self.linear_layers.append(nn.Linear(z_dim * 2, hidden_dim).to(device))
        else:
            self.linear_layers.append(nn.Linear(z_dim, hidden_dim).to(device))
            n_linear_layers -= 1
        # self.linear_layers.append(nn.ELU())
        
        for _ in range(n_linear_layers-1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device))
            self.linear_layers.append(nn.LeakyReLU())
                # self.linear_layers.append(nn.LayerNorm(hidden_dim).to(device))
        if self.use_VIB:
            self.fc_mu  = nn.Linear(hidden_dim, hidden_dim).to(device)
            self.fc_std = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.output = nn.Linear(hidden_dim, 1).to(device)
        self.apply(weight_init)
    
    
    def forward(self, forward_rep, backward_rep):
        if not self.use_cross_attention:
            combined = torch.cat((forward_rep.unsqueeze(1), backward_rep.unsqueeze(1)), dim=1)
            for layer in self.attention_layers:
                if isinstance(layer, SelfAttention) and self.use_res:
                    combined = combined + layer(combined)
                else:
                    combined = layer(combined)
            combined_output = torch.cat((combined[:, 0, :], combined[:, 1, :]), dim=-1)
        else:
            combined = forward_rep
            for layer in self.attention_layers:
                if isinstance(layer, CrossAttentionModel_with_fed):
                    combined = layer(combined, backward_rep)
            combined_output = combined
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
        use_VIB: bool,
        use_2branch: bool,
        use_cross_attention: bool = False,
        
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
            use_2branch = use_2branch
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
        
        if not use_trasnformer:
            self.operator = MixNet(
                z_dimension=z_dimension,
                hidden_dimension=operator_hidden_dimension,
                hidden_layers=operator_hidden_layers,
                device=device,
                activation=operator_activation,
            )
            
            self.operator_target = MixNet(
                z_dimension=z_dimension,
                hidden_dimension=operator_hidden_dimension,
                hidden_layers=operator_hidden_layers,
                device=device,
                activation=operator_activation,
            )
        else:
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
            
            # self.icm = ICM(
            #     obs_dim=observation_length,
            #     action_dim=action_length,
            #     z_dim=z_dimension,
            #     hidden_dim=operator_hidden_dimension,
            #     device=device
            # )
        self.apply(weight_init)
        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
        
        
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
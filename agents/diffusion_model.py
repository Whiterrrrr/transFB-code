import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base import AbstractMLP
AC_FN ={'relu': F.relu, 'mish': F.mish, 'gelu': F.gelu}


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
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class MLP(nn.Module):
    def __init__(
            self, 
            input_size:int, 
            hidden_sizes:list, 
            output_size:int, 
            ac_fn: str='relu', 
            use_layernorm: bool=False, 
            dropout_rate: float=0.
    ):
        super().__init__()             
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout_rate
        
        # initialize layers
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList() if use_layernorm else None
        self.ac_fn = AC_FN[ac_fn]
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                
        if self.use_layernorm:
            self.layernorms.append(nn.LayerNorm(input_size))
                
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

            
    def forward(self, x):
        if self.use_layernorm:
            x = self.layernorms[-1](x)
        
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x = self.ac_fn(x)

        x = self.layers[-1](x)
        return x
       

class MLPResNetBlock(nn.Module):
    """
    the MLPResnet Blocks used in IDQL: arXiv:2304.10573, Appendix G
    """
    def __init__(self, hidden_dim:int, ac_fn='relu', use_layernorm=False, dropout_rate=0.1):
        super(MLPResNetBlock, self).__init__()
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ac_fn = AC_FN[ac_fn]
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)
            
    def forward(self, x):
        identity = x
        
        out = self.dropout(x)
        out = self.norm1(out)
        out = self.dense1(out)
        out = self.ac_fn(out)
        out = self.dense2(out)
        out = identity + out
        
        return out


class MLPResNet(nn.Module):
    """
    the LN_Resnet used in IDQL: arXiv:2304.10573
    """
    def __init__(self, num_blocks:int, input_dim:int, hidden_dim:int, output_size:int, ac_fn='relu', use_layernorm=True, dropout_rate=0.1):
        super(MLPResNet, self).__init__()
        
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.ac_fn = AC_FN[ac_fn]
        self.dense2 = nn.Linear(hidden_dim, output_size)
        self.mlp_res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_res_blocks.append(MLPResNetBlock(hidden_dim, ac_fn, use_layernorm, dropout_rate))
            
    def forward(self, x):
        out = self.dense1(x)
        for mlp_res_block in self.mlp_res_blocks:
            out = mlp_res_block(out)
        out = self.ac_fn(out)
        out = self.dense2(out)
        return out
              


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
            
    def forward(self, x):
        device = x.device
        half_dim = self.output_size // 2
        f = math.log(10000) / (half_dim - 1)
        f = torch.exp(torch.arange(half_dim, device=device) * -f)
        f = x * f[None, :]
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f

# learned positional embeds
class LearnedPosEmb(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)
            
    def forward(self, x):
        f = 2 * torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f       

TIMEEMBED = {"fixed": SinusoidalPosEmb, "learned": LearnedPosEmb}

# the simplest mlp model that takes
class MLPDiffusion(nn.Module):
    def __init__(
            self, 
            input_dim,
            output_dim,
            cond_dim=0,
            time_embeding='fixed',
            device='cpu',
    ):
        super(MLPDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.cond_dim = cond_dim
        
        # base encoder
        hidden_size = 256
        self.base_model = MLP(input_dim+cond_dim, [hidden_size, hidden_size], hidden_size)
        
        # time embedding
        if time_embeding not in TIMEEMBED.keys():
            raise ValueError(f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}")
        self.time_process = TIMEEMBED[time_embeding](1, hidden_size)
        
        # decoder
        self.decoder = MLP(hidden_size+hidden_size, [hidden_size, hidden_size], output_dim)
        
        self.device = device
            
    def forward(self, xt, t, cond=None):
        # encode
        time_embedding = self.time_process(t.view(-1, 1))
        if cond is not None:
            xt = torch.concat([xt, cond], dim=-1)
        base_embedding = self.base_model(xt)
        embedding = torch.cat([time_embedding, base_embedding], dim=-1)
        
        # decode
        noise_pred = self.decoder(embedding)
        return noise_pred
          
          
# the diffusion model used in IDQL
class IDQLDiffusion(nn.Module):
    """
    the diffusion model used in IDQL: arXiv:2304.10573
    """
    def __init__(
        self, 
        input_dim,  # a dim
        output_dim,  # a dim
        observation_length,  # s dim, if condition on s
        z_dimension,  # s dim, if condition on s
        preprocessor_hidden_dimension,
        preprocessor_feature_space_dimension,
        preprocessor_hidden_layers,
        preprocessor_activation,
        hidden_dim=256,
        num_blocks=3,
        time_dim=64,
        ac_fn='mish',
        time_embeding='fixed',
        device='cpu',
    ):
        super(IDQLDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
              
        # time embedding
        if time_embeding not in TIMEEMBED.keys():
            raise ValueError(f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}")
        self.time_process = TIMEEMBED[time_embeding](1, time_dim)
        self.time_encoder = MLP(time_dim, [128], 128, ac_fn='mish').to(device)
        
        # decoder
        self.decoder = MLPResNet(num_blocks, input_dim + 128 + preprocessor_feature_space_dimension*2, hidden_dim, output_dim, ac_fn, True, 0.1).to(device)
        
        self.device = device
              
    def forward(self, xt, t, cond_s=None, cond_z=None):
        # encode
        obs_embedding = self.obs_preprocessor(cond_s)
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([cond_s, cond_z], dim=-1))
        time_embedding = self.time_process(t.view(-1, 1))
        time_embedding = self.time_encoder(time_embedding)
        xt = torch.concat([xt, obs_embedding, obs_z_embedding], dim=-1)
        embedding = torch.cat([time_embedding, xt], dim=-1)
            
        # decode
        noise_pred = self.decoder(embedding)
        return noise_pred       
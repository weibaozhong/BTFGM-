import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_plus import Mamba
# class Add_Norm(nn.Module):
#     def __init__(self, d_model, dropout, residual, drop_flag=1):
#         super(Add_Norm, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(d_model)
#         self.residual = residual
#         self.drop_flag = drop_flag
    
#     def forward(self, new, old):
#         new = self.dropout(new) if self.drop_flag else new
#         return self.norm(old + new) if self.residual else self.norm(new)


# class EncoderLayer(nn.Module):
#     def __init__(self, Mambablock, Mambablocked, d_model, d_ff=None,
#                  dropout=0.1, activation="relu", residual=1):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.Mambablock = Mambablock
#         self.Mambablocked = Mambablocked
#         self.residual = residual
        
#         self.addnorm_for = Add_Norm(d_model, dropout, residual, drop_flag=0)
#         self.addnorm_back = Add_Norm(d_model, dropout, residual, drop_flag=0)
        
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.ffn = nn.Sequential(
#             nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
#             nn.ReLU() if activation == "relu" else nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         )
#         self.addnorm_ffn = Add_Norm(d_model, dropout, residual, drop_flag=1)

#     def forward(self, x, x_mask=None, tau=None, delta=None):
        
#         output_forward = self.Mambablock(x)
#         output_forward = self.addnorm_for(output_forward, x)

#         output_backward = self.Mambablocked(x.flip(dims=[1])).flip(dims=[1])
#         output_backward = self.addnorm_back(output_backward, x)
#         output = output_forward + output_backward

#         temp = output
#         output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
#         output = self.addnorm_ffn(output, temp)

#         return output

# class Encoder(nn.Module):
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Encoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection

#     def forward(self, x, x_mask=None, tau=None, delta=None):
#         for layer in self.layers:
#             x = layer(x, x_mask=x_mask, tau=tau, delta=delta)

#         if self.norm is not None:
#             x = self.norm(x)

#         if self.projection is not None:
#             x = self.projection(x)
#         return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.man = Mamba(
        #     d_model=11,  # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=2,  # Local convolution width
        #     expand=1,  # Block expansion factor)
        # )
        # self.man2 = Mamba(
        #     d_model=11,  # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=2,  # Local convolution width
        #     expand=1,  # Block expansion factor)
        # )
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        attn = 1

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


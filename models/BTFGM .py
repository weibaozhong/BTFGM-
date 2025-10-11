import math
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from layers.RevIN import RevIN
from mamba_plus import Mamba  # 确保已安装 mamba-ssm
from layers.BMambaXer_Enc import EncoderLayer, Encoder
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer

wavelet_filter_lengths = {
    "haar": 2,
    "db1": 2,
    "db2": 4,
    "db3": 6,
    "db4": 8,
    "db5": 10,
    "db6": 12,
    "db7": 14,
    "db8": 16,
    "db9": 18,
    "db10": 20,
    "sym2": 4,
    "sym3": 6,
    "sym4": 8,
    "sym5": 10,
    "sym6": 12,
    "sym7": 14,
    "sym8": 16,
    "coif1": 6,
    "coif2": 12,
    "coif3": 18,
    "coif4": 24,
    "coif5": 30,
    "bior1.1": 2,
    "bior2.2": 6,
    "bior3.3": 10,
    "bior4.4": 14,
    "rbio1.1": 2,
    "rbio2.2": 6,
    "rbio3.3": 10,
    "rbio4.4": 14,
    "dmey": 102,
    "gaus1": 2,
    "gaus2": 4,
    "gaus3": 6,
    "mexh": "N/A",
    "morl": "N/A"
}


def compute_dwt_dimensions(T, J, wav):
    filter_length = wavelet_filter_lengths[wav]
    P = filter_length - 1
    yh_lengths = []
    for j in range(1, J + 1):
        T = math.floor((T + P) / 2)
        yh_lengths.append(T)
    yl_length = T
    return yl_length, yh_lengths


class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual, drop_flag=1):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.residual = residual
        self.drop_flag = drop_flag
    
    def forward(self, new, old):
        new = self.dropout(new) if self.drop_flag else new
        return self.norm(old + new) if self.residual else self.norm(new)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_ff, residual, dropout, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        residual = 1
        dropout = 0.1
        self.mamba_f = Mamba(
                d_model=d_model,
                d_state=8,
                d_conv=2,
                expand=1
            )
        self.mamba_b = Mamba(
                d_model=d_model,
                d_state=8,
                d_conv=2,
                expand=1
            )
        self.addnorm_for = Add_Norm(d_model, dropout, residual, drop_flag=0)
        self.addnorm_back = Add_Norm(d_model, dropout, residual, drop_flag=0)
        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        )
        self.addnorm_ffn = Add_Norm(d_model, dropout, residual, drop_flag=1)
    def forward(self, x):
        # x: [B, D, T] → [B, T, D]
        x = x.permute(0, 2, 1)
        output_forward = self.mamba_f(x)
        output_forward = self.addnorm_for(output_forward, x)

        output_backward = self.mamba_b(x.flip(dims=[1])).flip(dims=[1])
        output_backward = self.addnorm_back(output_backward, x)
        output = output_forward + output_backward

        temp = output
        output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
        output = self.addnorm_ffn(output, temp)

        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_norm = configs.use_norm
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.use_convdropout = configs.use_convdropout
        self.conv_dropout = configs.conv_dropout
        self.wav = 'haar'
        self.J = 1
        self.not_independent = configs.not_independent
        self.no_revin = configs.no_revin
        self.dropout = nn.Dropout(configs.dropout)
        

        self.dwt = DWT1DForward(wave=self.wav, J=self.J)
        self.idwt = DWT1DInverse(wave=self.wav)

        yl, _ = compute_dwt_dimensions(self.seq_len, self.J, self.wav)
        yl_, _ = compute_dwt_dimensions(self.pred_len, self.J, self.wav)

        if self.not_independent:
            self.yl_upsampler = nn.ModuleList([
                nn.Linear(yl, yl_) for _ in range(self.enc_in)
            ])
        else:
            self.yl_upsampler = nn.Linear(yl, yl_)

        self.mamba_block = MambaBlock(d_model=2, d_ff=8, residual=1, dropout = configs.dropout, activation="relu")

        if self.use_convdropout:
            self.dropout1 = nn.Dropout(self.conv_dropout)

        # self.en_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.en_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                    # residual=configs.residual==1
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.fusion_proj = nn.Linear(configs.enc_in * 2, configs.enc_in)
        
        self.c_linear = nn.Linear(configs.pred_len, configs.d_model, bias=True)
        self.cat_linear = nn.Linear(configs.d_model * 2, configs.d_model, bias=True)
    

        self.crossattention1 = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),
                configs.d_model, configs.n_heads)
        self.crossattention2 = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),
                configs.d_model, configs.n_heads)
        

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
# -------------------小波变换mamba-------------------------
        x_enc = x
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        # print(x_enc.shape)
        
        if self.no_revin:
            seq_mean = torch.mean(x, dim=1).unsqueeze(1)
            x = (x - seq_mean).permute(0, 2, 1)
        else:
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # DWT
        yl, yh = self.dwt(x)  # [B, C, T] → yl: [B, C, T'], yh: list[T']
        # print("yl",yl.shape)
        yh = yh[0]  # Level-1 only
        y = torch.stack([yl, yh], dim=-2)  # [B, C, 2, T']

        B, C, D, T_ = y.shape
        y_reshaped = y.reshape(B * C, D, T_)

        # Mamba
        # y_mamba = self.mamba_block(y_reshaped) + y_reshaped  # 残差连接
        y_mamba = self.mamba_block(y_reshaped) 
        y = y_mamba.reshape(B, C, D, T_)

        if self.use_convdropout:
            y = self.dropout1(y)

        # Upsample
        if self.not_independent:
            y_out = torch.zeros([B, C, D, self.pred_len // 2], dtype=y.dtype).to(y.device)
            for i in range(self.enc_in):
                y_out[:, i, :, :] = self.yl_upsampler[i](y[:, i, :, :])
            y = y_out
        else:
            y = self.yl_upsampler(y)
        # y torch.Size([8, 7, 2, 48])
        # print("y",y.shape)


        yl_, yh_ = y[:, :, 0, :], [y[:, :, 1, :]]

        # IDWT
        y = self.idwt((yl_, yh_))
        

        if self.no_revin:
            y = y.permute(0, 2, 1) + seq_mean
        else:
            y = y.permute(0, 2, 1)
            y = self.revin_layer(y, 'denorm')
# --------------------------------------------
        # y torch.Size([8, 96, 7])
        # print("y",y.shape)
        # print(x_enc.shape)
        en_embed = self.en_embedding(x_enc, None)
        

        enc_out, attn = self.encoder(en_embed)
        enc_out = enc_out + en_embed

        y_cross = y.permute(0, 2, 1)
        # print("y_cross",y_cross.shape)
        # print("enc_out",enc_out.shape)
        # print(y_cross.shape)
        y_cross = self.c_linear(y_cross)
        ycross, attn1 = self.crossattention1(y_cross, enc_out, enc_out, attn_mask=None)
        y_cross = self.dropout(ycross)
        y_cross = y_cross + ycross


        # # 提取选定批次和头的注意力权重
        # attn_weights = attn1[0, 0, :, :]

        # # 绘制热力图
        # plt.figure(figsize=(10, 6))
        # plt.imshow(attn_weights.detach().cpu().numpy(), cmap='viridis', aspect='auto')
        # plt.colorbar(label="Attention Weight")
        # plt.xlabel("Key")
        # plt.ylabel("Query")
        # plt.title("FQ-Cross-Attention Heatmap")

        # # 保存为图片
        # plt.savefig('cross_attention_heatmap.png', dpi=500)  # 指定文件名和分辨率
        # plt.close()  # 关闭图形以释放内存



        encross, attn2 = self.crossattention1(enc_out, y_cross, y_cross, attn_mask=None)
        encross = self.dropout(encross)
        enc_out = encross + enc_out     

        dec_out = torch.cat([y_cross, enc_out], -1)
        dec_out = self.cat_linear(dec_out)
        dec_out = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]
        
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

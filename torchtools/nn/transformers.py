import torch.nn as nn


# Based on the GPT2 implementatyion from MinGPT https://github.com/karpathy/minGPT by Andrej Karpathy
class GPTTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.ln1(x)
        x = x + self.attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        x = x + self.mlp(self.ln2(x))
        return x
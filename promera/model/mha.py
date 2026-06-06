import math
import torch.nn as nn
import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except:
    print("Could not import flash attn")


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, layers=2, act=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LayerNorm(dim))
        if act == SwiGLU:
            out_mul = 2
        else:
            out_mul = 1
        self.layers.append(nn.Linear(dim, ff_dim * out_mul))
        for i in range(layers - 2):
            self.layers.append(act())
            self.layers.append(nn.Linear(ff_dim, ff_dim * out_mul))
        self.layers.append(act())
        self.layers.append(nn.Linear(ff_dim, dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_expand=4,
        ff_layers=2,
        dropout=0.0,
        token_dropout=0.0,
        qk_norm=False,
        act="relu",
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            dim=dim,
            heads=heads,
            dropout=token_dropout,
            qk_norm=qk_norm,
        )
        self.ff = FeedForward(
            dim,
            ff_expand * dim,
            layers=ff_layers,
            act={
                "relu": nn.ReLU,
                "swiglu": SwiGLU,
                "gelu": nn.GELU,
                "silu": nn.SiLU,
            }[act],
        )

        self.mha_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.mha_dropout = nn.Dropout(p=dropout)
        self.ff_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x,
        mask,
        idx=None,
    ):

        x = x + self.mha_dropout(
            self.mha(
                x=self.mha_norm(x),
                mask=mask.bool(),
                idx=idx,
            )
        )

        x = x + self.ff_dropout(self.ff(self.ff_norm(x)))

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout=0.0,
        qk_norm=False,
    ):

        super().__init__()
        self.dim = dim
        self.heads = heads

        ## basic stuff we always need
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(dim) if qk_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x,
        mask,
        idx=None,
        flash_attn=False,
    ):

        B, L, D = x.shape
        dev = x.device
        if idx is None:
            idx = torch.arange(L, device=dev)
        if mask is None:
            mask = torch.ones(B, L, D, dtype=bool, device=dev)

        query = (
            self.q_norm(self.w_q(x)).view(B, L, self.heads, -1).transpose(1, 2)
        )  # B H L D
        key = self.k_norm(self.w_k(x)).view(B, L, self.heads, -1).transpose(1, 2)

        ## scalar values
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)

        if flash_attn:
            mask = mask.to(query).view(B, 1, L, 1)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = torch.nn.functional.scaled_dot_product_attention(
                    query * mask, key * mask, value * mask
                )

        else:
            attn = query @ key.mT / math.sqrt(D / self.heads)  # B H L L

            mask = mask.view(B, 1, 1, -1)
            attn = torch.where(mask, attn, -float("inf"))
            attn = torch.softmax(attn, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attn = self.dropout(attn)

            out = attn @ value

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.w_o(out)

        return out

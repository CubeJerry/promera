# Adapted from https://github.com/jwohlwend/boltz
from typing import Dict, Tuple

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
import torch
from torch import Tensor, nn


from .layers.attention import AttentionPairBias
from .layers.dropout import get_dropout_mask
from .layers.outer_product_mean import OuterProductMean
from .layers.pair_averaging import PairWeightedAveraging
from .layers.transition import Transition
from .layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from .layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .encoders import AtomAttentionEncoder

_chunk_size_threshold = 384

from tinyprot.feature import _ntoks


class InputEmbedder(nn.Module):
    """Input embedder."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.atom_attention_encoder = AtomAttentionEncoder(
            **cfg.dims,
            atom_encoder_depth=cfg.atom_encoder_depth,
            atom_encoder_heads=cfg.atom_encoder_heads,
            structure_prediction=False,
        )

    def forward(self, feats: Dict[str, Tensor], mask_ligand=False) -> Tensor:
        # Load relevant features
        res_type = feats["restype"]
        profile = feats["profile"]
        deletion_mean = feats["deletion_mean"].unsqueeze(-1)

        a, _, _, _, _ = self.atom_attention_encoder(feats)

        if self.cfg.feature.mask_std_feats:
            a = torch.where(feats["is_std"][..., None], 0.0, a)

        is_epitope = feats["is_epitope"].float().unsqueeze(-1)
        res_type = torch.nn.functional.one_hot(res_type, _ntoks)
        s = torch.cat([a, res_type, profile, deletion_mean, is_epitope], dim=-1)
        return s


class MSAModule(nn.Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)

        self.msa_proj = nn.Linear(
            _ntoks + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = nn.ModuleList()
        for i in range(msa_blocks):
            if activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        MSALayer(
                            msa_s,
                            token_z,
                            msa_dropout,
                            z_dropout,
                            pairwise_head_width,
                            pairwise_num_heads,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    MSALayer(
                        msa_s,
                        token_z,
                        msa_dropout,
                        z_dropout,
                        pairwise_head_width,
                        pairwise_num_heads,
                    )
                )

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: Dict[str, Tensor],
    ) -> Tensor:

        # Set chunk sizes
        if not self.training:
            if z.shape[1] > _chunk_size_threshold:
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # Load relevant features
        msa = feats["msa"]

        msa = torch.nn.functional.one_hot(msa, _ntoks)

        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)

        is_paired = feats["msa_paired"].unsqueeze(-1)
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"].float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        # Perform MSA blocks
        for i in range(self.msa_blocks):
            z, m = self.layers[i](
                z,
                m,
                token_mask,
                msa_mask,
                chunk_heads_pwa,
                chunk_size_transition_z,
                chunk_size_transition_msa,
                chunk_size_outer_product,
                chunk_size_tri_attn,
            )
        s = (m * msa_mask[..., None]).sum(1) / (msa_mask.sum(1)[..., None] + 1e-5)
        return s, z


class MSALayer(nn.Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:

        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
        )
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int = None,
        chunk_size_transition_msa: int = None,
        chunk_size_outer_product: int = None,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:

        # Communication to MSA stack
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        m = m + msa_dropout * self.pair_weighted_averaging(
            m, z, token_mask, chunk_heads_pwa
        )
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        # Communication to pairwise stack
        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # Compute pairwise stack
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
        )

        dropout = get_dropout_mask(self.z_dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
        )

        z = z + self.z_transition(z, chunk_size_transition_z)

        return z, m


class PairformerModule(nn.Module):
    """Pairformer module."""

    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for i in range(cfg.num_blocks):
            block = PairformerLayer(
                cfg.dims.token_s,
                cfg.dims.token_z,
                cfg.num_heads,
                cfg.dropout,
                cfg.pairwise_head_width,
                cfg.pairwise_num_heads,
                getattr(cfg, "no_update_s", False),
                i == cfg.num_blocks - 1,
            )
            if cfg.activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(block, offload_to_cpu=cfg.offload_to_cpu)
                )
            else:
                self.layers.append(block)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:

        if not self.training:
            if z.shape[1] > _chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, chunk_size_tri_attn)
        return s, z


class PairformerLayer(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
    ) -> None:

        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        if not self.no_update_s:
            self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        if not self.no_update_s:
            self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass."""
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
        )

        z = z + self.transition_z(z)

        # Compute sequence stack
        if not self.no_update_s:
            s = s + self.attention(s, z, mask)
            s = s + self.transition_s(s)

        return s, z


class DistogramModule(nn.Module):
    """Distogram Module."""

    def __init__(self, token_z: int, num_bins: int) -> None:
        super().__init__()
        self.distogram = nn.Linear(token_z, num_bins)

    def forward(self, z: Tensor) -> Tensor:
        z = z + z.transpose(1, 2)
        return self.distogram(z)

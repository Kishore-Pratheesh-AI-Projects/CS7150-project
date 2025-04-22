"""
Fusion modules for multimodal learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossModalFusion(nn.Module):
    """
    Bidirectional cross-modal fusion using multi-head attention.
    """
    def __init__(self, latent_dim=256, n_heads=4):
        super().__init__()
        self.attn_img_to_txt = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)
        self.attn_txt_to_img = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)

    def forward(self, img_feat, txt_feat):
        img_feat = img_feat.unsqueeze(1)  # [B, 1, D]
        txt_feat = txt_feat.unsqueeze(1)  # [B, 1, D]

        # Image attends to text
        img_attended, _ = self.attn_img_to_txt(img_feat, txt_feat, txt_feat)  # Query=img, Key/Value=text

        # Text attends to image
        txt_attended, _ = self.attn_txt_to_img(txt_feat, img_feat, img_feat)

        # Aggregate
        fused = torch.cat([img_attended, txt_attended], dim=1)  # [B, 2, D]
        return fused.mean(dim=1)  # Final Z: [B, D]


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion for sequences.
    """
    def __init__(self, latent_dim=256, n_heads=4):
        super().__init__()
        self.img_proj = nn.Linear(latent_dim, latent_dim)
        self.txt_proj = nn.Linear(latent_dim, latent_dim)

        self.attn_img_to_txt = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)
        self.attn_txt_to_img = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)

    def forward(self, z_img_seq, z_txt_seq):
        # z_img_seq: (B, 1, D) or (B, seq_len_img, D)
        # z_txt_seq: (B, seq_len_txt, D)

        q_img = self.img_proj(z_img_seq)  # queries from image
        k_txt = self.txt_proj(z_txt_seq)  # keys from text

        # Image attends to text
        img_attended, _ = self.attn_img_to_txt(q_img, k_txt, k_txt)  # (B, 1, D)

        # Text attends to image
        txt_attended, _ = self.attn_txt_to_img(k_txt, q_img, q_img)  # (B, seq_len_txt, D)

        # Fuse via mean pooling
        fused = torch.cat([
            img_attended.mean(dim=1),        # (B, D)
            txt_attended.mean(dim=1)         # (B, D)
        ], dim=1)                             # → (B, 2D)

        return fused


class SimpleConcatenationFusion(nn.Module):
    """
    Simple concatenation-based fusion of modalities.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        # After concatenation, reduce dimensions back to latent_dim
        self.projection = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, img_feat, txt_feat):
        # Simple concatenation of image and text features
        concat_feat = torch.cat([img_feat, txt_feat], dim=1)  # [B, 2*D]

        # Project back to original latent dimension
        fused = self.projection(concat_feat)  # [B, D]

        return fused
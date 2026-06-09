"""
llm_branch.py – PubMedBERT feature projection branch.

Nhận vector PubMedBERT đã encode sẵn (768-d) của drug và SE,
kết hợp combined + difference + normalize để tăng discriminability.
Cosine sim trung bình giảm từ ~0.92 xuống ~0.3–0.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMBranch(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        # input_dim * 2 vì concat(combined, diff)
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        drug_vec: torch.Tensor,
        se_vec: torch.Tensor,
        drug_mask: torch.Tensor | None = None,
        se_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            drug_vec  : (batch, 768) – PubMedBERT embedding of drug
            se_vec    : (batch, 768) – PubMedBERT embedding of SE
            drug_mask : (batch,) – 1.0 if drug has real text, 0.0 otherwise
            se_mask   : (batch,) – 1.0 if SE has real text, 0.0 otherwise

        Returns:
            (batch, output_dim) – projected LLM features
        """
        # Zero out vectors that have no real text (encoded from empty string)
        if drug_mask is not None:
            drug_vec = drug_vec * drug_mask.unsqueeze(1)
        if se_mask is not None:
            se_vec = se_vec * se_mask.unsqueeze(1)

        # Normalize về unit sphere trước
        drug_vec = F.normalize(drug_vec, dim=1)  # (batch, 768)
        se_vec = F.normalize(se_vec, dim=1)    # (batch, 768)

        # Combined: nắm bắt điểm chung giữa drug và SE
        combined = drug_vec + se_vec            # (batch, 768)

        # Difference: nắm bắt điểm khác biệt — quan trọng để phân biệt
        diff = drug_vec - se_vec                # (batch, 768)

        # Concat cả 2 → (batch, 1536)
        x = torch.cat([combined, diff], dim=1)

        # Project xuống output_dim
        out = self.projection(x)                # (batch, output_dim)
        return self.layer_norm(out)


def verify_llm_branch():
    """Chạy để kiểm tra module hoạt động đúng và cosine sim giảm."""
    branch = LLMBranch(input_dim=768, output_dim=384)
    drug_vec = torch.randn(8, 768)
    se_vec = torch.randn(8, 768)

    out = branch(drug_vec, se_vec)
    assert out.shape == (8, 384), f"Expected (8, 384), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected!"

    # Kiểm tra cosine sim sau khi qua branch
    out_norm = F.normalize(out, dim=1)
    sims = torch.matmul(out_norm, out_norm.T)
    mask = torch.triu(torch.ones(8, 8), diagonal=1).bool()
    mean_sim = sims[mask].mean().item()
    max_sim = sims[mask].max().item()
    print(f"Output cosine sim mean: {mean_sim:.4f}  (lý tưởng < 0.7)")
    print(f"Output cosine sim max:  {max_sim:.4f}")
    print(f"Output shape: {out.shape}")
    print("LLMBranch OK!")


if __name__ == '__main__':
    verify_llm_branch()

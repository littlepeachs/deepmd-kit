import torch
import torch.nn as nn

class RMSLayerNorm(nn.Module):
    """Root Mean Square Layer Normalization over a configurable number of trailing dimensions.

    Parameters
    ----------
    dim : int
        Size of the last (feature) dimension.
    rank : int
        Number of preceding dimensions (e.g., spatial dims) to include in normalization.
    eps : float, optional
        Small constant to avoid division by zero.
    """
    def __init__(self, dim: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.eps = eps
        # Learnable scale parameter for the feature dimension
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine which axes to normalize over: the last rank+1 dims
        axes = tuple(range(-self.rank - 1, 0))  # e.g., rank=0 -> (-1,), rank=2 -> (-3,-2,-1)
        # Compute mean of squares
        mean_sq = x.pow(2).mean(dim=axes, keepdim=True)

        # Root mean square
        rms = torch.sqrt(mean_sq + self.eps)
        # Normalize
        x_norm = x / rms
        # Reshape gamma for broadcasting over all but the last dim
        gamma = self.gamma.view(*([1] * (x.dim() - 1)), -1).to(x.device)
        return x_norm * gamma


def test_rms_layer_norm():
    batch_size, d1, d2, d3, dim = 2, 3, 3, 3, 8
    x = torch.randn(batch_size, d1, d2, d3, dim)
    # Test for ranks 0 through 3
    for rank in range(4):
        norm = RMSLayerNorm(dim, rank)
        out = norm(x)
        print(f"rank={rank}, input shape={x.shape}, output shape={out.shape}")
        # Compute RMS over the normalization axes to verify ~1
        axes = tuple(range(-rank - 1, 0))
        rms_out = torch.sqrt(out.pow(2).mean(dim=axes))
        print(f"Sample RMS values (should be ~1): {rms_out.flatten()[:5]}")

if __name__ == "__main__":
    test_rms_layer_norm()

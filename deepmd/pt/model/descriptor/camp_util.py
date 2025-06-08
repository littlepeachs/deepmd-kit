import math

import torch
from torch import Tensor, nn

import warnings
from typing import Optional, Any
import string
from collections import Counter
def letter_index(n: int, start: int = 0) -> str:
    """
    Get a list of letters 'abc...' of length n.

    Args:
        n: the length of the letters
    """
    return string.ascii_lowercase[start : start + n]

def get_unique(values: list[list[Any]]) -> list[list[Any]]:
    """Get unique inner lists of an outer list.

    The order of the elements of the inner list does not matter.

    Example:
        >>> get_unique([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
        [[0, 1, 1], [0, 0, 1]]
    """
    seen = set()
    unique = []
    for x in values:
        # using Counter to distinguish the case of x being, e.g. [0, 0, 1] and [0, 1, 1]
        # if we use frozenset(x), then both will be [0, 1], which is not distinguishable
        x_fs = frozenset(Counter(x).items())
        if x_fs not in seen:
            unique.append(x)
            seen.add(x_fs)

    return unique

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int] = None,
        activation: nn.Module = nn.SiLU(),
        out_activation: bool = False,
    ):
        """
        MLP with SiLU activation.

        Total number of layers is len(hidden_features) + 1.

        Args:
            in_features:
            out_features:
            hidden_features: list of hidden layer sizes. If None, no hidden layers.
            activation:
            out_activation: whether to apply activation to the output layer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.out_activation = out_activation

        sizes = [in_features]
        if hidden_features is not None:
            sizes += list(hidden_features)
        sizes += [out_features]

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation)

        if not out_activation:
            layers.pop()

        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


class LinearCombination(nn.Module):
    """
    Linear combination of tensors.

    Given a tensor of shape (d0, d1, d2 ...), this module computes the linear
    combination along the first dimension d0, but separately for each d1 dimension,
    resulting in a tensor of shape (d1, d2, ...).

    Args:
        in_features: d0
        const_features: d1
    """

    def __init__(self, in_features: int, const_features: int):
        super().__init__()
        self.in_features = in_features
        self.const_features = const_features

        self.weight = nn.Parameter(torch.empty(in_features, const_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#L109
        """
        k = 1 / self.in_features**0.5
        nn.init.uniform_(self.weight, -k, k)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            x: tensor of shape (d0, d1, d2, ...)

        Returns:
            tensor of shape (d1, d2, ...)
        """

        out = torch.einsum("ij,ij...->j...", self.weight, input)

        return out


class LinearMap(nn.Module):
    """
    Linear map of tensors.

    Given a tensor of shape (d0, d1, d2 ...), this module computes the linear
    combination of the tensor along the first dimension d0, and returns a tensor of
    shape (d0', d1, d2, ...).

    Args:
        in_features: d0
        out_features: d0'
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#l109
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            x: tensor of shape (d0, d1,...)

        Returns:
            tensor of shape (d0', d1,...)
        """

        out = torch.tensordot(self.weight, input, dims=1)

        return out

def get_dyadic_tensor(r: Tensor, rank: int = 2, normalize: bool = True) -> Tensor:
    r"""
    Create a generalized dyadic tensor.

    For rank = 0, the dyadic tensor is a scalar, simply equal to 1.
    For rank = 1, the dyadic tensor is a vector, simply equal to r.
    For rank >= 2, the generalized dyadic tensor is the tensor product of the vector r
    with itself, i.e. :math:`r \otimes r \otimes \cdots \otimes r`. The rank is the
    number of vectors in the tensor product.

    Args:
        r: shape (..., 3) the vector to construct the generalized dyadic tensor. Only
            the last dimension is used to construct the tensor. The ellipsis represents
            any number of dimensions that allows batching.
        rank: rank of the generalized dyadic tensor, i.e. the number of times to tensor
            product the vector r with itself. Rank must be greater than or equal to 1.
        normalize: whether to normalize the vector r as a unit vector before
            constructing the generalized dyadic tensor.

    Returns:
        A tensor of shape (..., 3, 3, 3), where the ... represents the batching
        dimensions, and the number of 3's is equal to the rank.
    """
    if rank < 0:
        raise ValueError("Rank must be greater than or equal to 0.")
    elif rank == 0:
        shape = r.shape[:-1]
        return torch.ones(shape).to(r.device)
    else:
        if normalize:
            norm = torch.norm(r, dim=-1, keepdim=True)
            if torch.any(norm < 1e-3):
                warnings.warn("The norm of the vector(s) is smaller than 1e-3.")
            r = r / norm

        indices = letter_index(rank)
        data = [r] * rank
        t = torch.einsum(f"{','.join(['...'+i for i in indices])}->...{indices}", data)

        return t



class RadialPart(nn.Module):
    """Radial part of the MTP.

    f_nu_i_j(r) = \sum_\beta c_nu_i_j * radial_basis_\beta(r)

    Eq. 3 of Shapeev.
    """

    def __init__(
        self,
        n_u: int,
        n_z: int,
        max_chebyshev_degree: int = 9,
        r_cut: float = 5,
        envelope: Optional[int] = None,
    ):
        """
        Args:
            n_u: number of radial basis functions.
            n_z: number of atom types.
            max_chebyshev_degree: max degree of the Chebyshev polynomial. The total
                number of chebyshev polynomials is `max_chebyshev_degree + 1`; +1 for
                the zeroth degree.
            r_cut: cutoff distance.
            envelope: envelope function to make the radial basis function smooth at
                r_cut. if None, using the MTP 2nd order polynomial envelope. Otherwise,
                p is a positive integer, and the envelope function in dimenet is used.
        """
        super().__init__()

        self.n_u = n_u
        self.n_z = n_z
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.envelope = envelope

        self.c = nn.Parameter(torch.empty(n_z, n_z, n_u, max_chebyshev_degree + 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights to:

            uniform(-1/sqrt(in_features), 1/sqrt(in_features)).

        Note, self.c can be regarded as a collection of multiple linear layers, each for
        a specific combination of zi and zj.

        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#L109
        """
        k = 1 / (self.max_chebyshev_degree + 1) ** 0.5
        nn.init.uniform_(self.c, -k, k)

    def forward(self, r: Tensor, zi: Tensor, zj: Tensor):
        """
        Args:
            r: 1D tensor of distances between atoms i and j.
            zi: 1D tensor of integers. type of atom i. The choice are 0, 1, 2, ...
                the number of atom types.
            zj: 1D tensor of integers. type of atom j. The choice are 0, 1, 2, ...
                the number of atom types.

        Note:
            The shape of r, zi, and zj should be the same.

        Returns:
            A tensor of shape (len(r), n_nu). The first dimension corresponds to `nu`
            in Eq. 3 of Shapeev, and the second dimension denotes the size of the
            distances.
        """
        # shape (n_nu, len(r))
        radial = radial_basis(
            self.max_chebyshev_degree, r, r_cut=self.r_cut, envelope=self.envelope
        )

        # select c for r according to zi and zj
        c = self.c[zi, zj, :, :]  # shape(len(r), n_nu, len(degrees))

        # linear combination of radial basis functions of different degrees
        out = torch.einsum("rub, br -> ru", c, radial)

        return out


def radial_basis(
    degree: int,
    r: Tensor,
    r_min: float = 0,
    r_cut: float = 5,
    envelope: Optional[int] = None,
) -> Tensor:
    """
    Radial basis function, using Chebyshev polynomials.

    I.e. Q in Eq. 4 of Shapeev.

    Args:
        degree: max degree of the Chebyshev polynomial to use.
        r: distance, 1D tensor.
        r_min: minimum distance.
        r_cut: cutoff distance.
        envelope: envelope function to make the radial basis function smooth at r_cut.
            if None, using the MTP 2nd order polynomial envelope. Otherwise, p is a
            positive integer, and the envelope function in dimenet is used.

    Returns:
        A tensor X of shape (degree+1, *r.shape); +1 to include the zeroth degree.
        The first dimension denotes the degree of the polynomial. X[i] is the result
        for the i-th degree polynomial.
    """
    # select r < r_cut ones for computation
    mask = r < r_cut
    selected_r = r[mask]

    # normalize r to [0, 1]
    normalized_r = (selected_r - r_min) / (r_cut - r_min)

    che = chebyshev_first(degree, normalized_r)

    if envelope is None:
        env = mtp_envelope(normalized_r)
    else:
        env = dimenet_envelope(normalized_r, p=envelope)

    Q = che * env

    # prepare output
    shape = torch.Size([degree + 1]) + r.shape
    out = torch.zeros(shape, dtype=r.dtype, device=r.device)
    out[:, mask] = Q

    return out


def chebyshev_first(n: int, x: Tensor) -> Tensor:
    """Chebyshev polynomials of the first kind.

    Args:
        n: highest degree of the polynomial to compute.
        x: input tensor.

    Returns:
        A tensor of shape (n + 1, *x.shape). The first dimension denotes the degree
        of the polynomial, e.g. T[1] is the result of the first degree polynomial.
    """
    T = [torch.ones_like(x), x]  # T0 and T1
    for i in range(2, n + 1):
        T.append(2.0 * x * T[i - 1] - T[i - 2])

    T = torch.stack(T, dim=0)

    return T


def mtp_envelope(r: Tensor):
    """The envelope function used in the MTP."""
    return (1 - r) ** 2


def dimenet_envelope(r: Tensor, p: int = 6):
    """The envelope function used in DimNet.

    1 - (p+1)(p+2)/2*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)

    This is also the envelope function used hybrid NN of Mingjian Wen when p = 3.
    """
    if p == 6:
        return 1 - 28 * r**6 + 48 * r**7 - 21 * r**8
    elif p == 3:
        return 1 - 10 * r**3 + 15 * r**4 - 6 * r**5
    else:
        return (
            1
            - (p + 1) * (p + 2) / 2 * r**p
            + p * (p + 2) * r ** (p + 1)
            - p * (p + 1) / 2 * r ** (p + 2)
        )
    
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    reduce: str,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    A wrapper around torch.scatter_reduce_ that creates output tensor if not provided.

    The only difference is that this wrapper creates the output tensor if not provided.

    See: https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html

    Argues:
        reduce: "sum", "prod", "mean", "amax", "amin".
            Note, "amax"="max" and "amin"="min".
    """

    # index should have the same shape as src as that backward pass works,
    # per the PyTorch docs.
    index = broadcast(index, src, dim)

    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # setting include_self=False so that
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)

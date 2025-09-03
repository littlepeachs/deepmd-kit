from typing import Dict
import torch
from torch import nn
from typing import List, Optional
from typing import Callable, Union, Optional, Sequence,Any

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["build_mlp", "Dense"]

def grad(y: torch.Tensor, x: torch.Tensor, training: bool = True) -> torch.Tensor:
    """
    a wrapper for the gradient calculation
    alow multiple dimensional and/or complex y
    y: [n_graphs, ] or [n_graphs, dim_y]
    x: [n_nodes, :]
    """
    if y.is_complex():
        get_imag = True
    else:
        get_imag = False

    if len(y.shape) == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
        gradient_real = torch.autograd.grad(
            outputs=[y],  # [n_graphs, ]
            inputs=[x],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=(training or get_imag),  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]  # [n_nodes, 3]
        assert gradient_real is not None, "Gradient real is None"
        if get_imag:
            gradient_imag = torch.autograd.grad(
                outputs=[y/1j],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=training,  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
            )[0]  # [n_nodes, 3]
            assert gradient_imag is not None, "Gradient imag is None"
        else:
            gradient_imag = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    else:
        dim_y = y.shape[1] 
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y[:,0])]
        grad_list_real = []
        for i in range(dim_y):
            g = torch.autograd.grad(
                outputs=[y[:, i]],  # [n_graphs, ]
                inputs=[x],         # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < dim_y - 1) or get_imag),
                create_graph=training, # Create graph for second derivative
                allow_unused=True, # For complete dissociation turn to true
            )[0]
            assert g is not None, f"Gradient real for channel {i} is None"
            grad_list_real.append(g)
        gradient_real = torch.stack(grad_list_real, dim=2)  # [n_nodes, 3, dim_y]
        # if y is complex, we need to calculate the imaginary part
        if get_imag:
            grad_list_imag = []
            for i in range(dim_y):
                g = torch.autograd.grad(
                    outputs=[y[:, i]/1j], # [n_graphs, ]
                    inputs=[x], # [n_nodes, 3]
                    grad_outputs=grad_outputs,
                    retain_graph=(training or (i < dim_y - 1)), # Make sure the graph is not destroyed during training
                    create_graph=training, # Create graph for second derivative
                    allow_unused=True, # For complete dissociation turn to true
                )[0]
                assert g is not None, f"Gradient imag for channel {i} is None"
                grad_list_imag.append(g)
            gradient_imag = torch.stack(grad_list_imag, dim=2)  # [n_nodes, 3, dim_y]
        else:
            gradient_imag = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    if get_imag:
        return gradient_real + 1j * gradient_imag
    else:
        return gradient_real

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.script
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    bias: bool = True,
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation, bias=bias)
        for i in range(n_layers - 1)
    ]

    # assign a Dense layer (without activation function) to the output layer
    layers.append(
        Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias)
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net

class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = nn.Identity(),
    ):
        """
        Fully connected linear layer with an optional activation function and batch normalization.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If False, the layer will not have a bias term.
            activation (Callable or nn.Module): Activation function. Defaults to Identity.
        """
        super().__init__()
        # Dense layer
        self.linear = nn.Linear(in_features, out_features, bias)

        # Activation function
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def forward(self, input: torch.Tensor):
        
        y = self.linear(input)
        y = self.activation(y)
        return y



__all__ = ["Atomwise"]

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
        self,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Callable = F.silu,
        add_linear_nn: bool = False,
        output_scaling_factor: float = 1.0,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            add_linear_nn: whether to add a linear NN to the output of the MLP 
        """
        super().__init__()

        self.n_in = None
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.add_linear_nn = add_linear_nn
        self.bias = bias
        self.output_scaling_factor = output_scaling_factor
        self.outnet = None

    def forward(self, 
                desc: torch.Tensor, # [n_atoms, n_features]
                batch: torch.Tensor, # [n_atoms]
                training: bool = None,
               ) -> torch.Tensor:

        if self.n_in is None:
            self.n_in = desc.shape[1]
        else:
            assert self.n_in == desc.shape[1]

        if self.outnet is None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                bias=self.bias,
                )
            self.outnet = self.outnet.to(desc.device).to(desc.dtype)
            
            if self.add_linear_nn:
                self.linear_nn = Dense(
                   self.n_in,
                   self.n_out,
                   bias=self.bias,
                   activation=None,
                   )
                self.linear_nn = self.linear_nn.to(desc.device).to(desc.dtype)
            else:
                self.linear_nn = None

            les_state = torch.load('/aisi/mnt/data_nas/liwentao/auto_model_test/results/0824-DPA3les-Omol25/les_state.pt')
            new_les_state = {}
            for old_key, value in les_state.items():
                # 去掉前缀，如 'model.Default.atomic_model.les.atomwise.outnet.0.linear.weight'
                # 变为 'outnet.0.linear.weight'
                new_key = old_key.replace('model.Default.atomic_model.les.atomwise.', '')
                new_les_state[new_key] = value
            self.load_state_dict(new_les_state)
            
        # predict atomwise contributions
        y = self.outnet(desc)
        if self.add_linear_nn:
            y += self.linear_nn(desc)

        return y * self.output_scaling_factor

    def __repr__(self):
        return f"Atomwise(n_in={self.n_in}, n_out={self.n_out}, n_hidden={self.n_hidden}, n_layers={self.n_layers}, bias={self.bias}, activation={self.activation.__name__})"

class BEC(nn.Module):
    def __init__(self,
                 remove_mean: bool = True,
                 epsilon_factor: float = 1., # \epsilon_infty
                 ):
        super().__init__()
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                output_index: Optional[int] = None, # 0, 1, 2 to select only one component
                ) -> torch.Tensor:

        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)
        unique_batches = torch.unique(batch)  # Get unique batch indices

        # compute the polarization for each batch
        all_P = []
        all_phases = [] 
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            r_now, q_now = r[mask], q[mask]
            if self.remove_mean:
                q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)
    
            if cell is not None:
                box_now = cell[i]  # Get the box for the i-th configuration

            # check if the box is periodic or not
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                polarization = torch.sum(q_now * r_now, dim=0)
                phase = torch.ones_like(r_now, dtype=torch.complex64)
            else:
                polarization, phase = self.compute_pol_pbc(r_now, q_now, box_now)
            if output_index is not None:
                polarization = polarization[output_index]
                phase = phase[:, output_index]

            all_P.append(polarization * self.normalization_factor)
            all_phases.append(phase)
        P = torch.stack(all_P, dim=0)
        phases = torch.cat(all_phases, dim=0)

        # take the gradient of the polarization w.r.t. the positions to get the complex BEC
        bec_complex = grad(y=P, x=r)
   
        # dephase
        result = bec_complex * phases.unsqueeze(1).conj()
        return result.real
 
    def compute_pol_pbc(self, r_now, q_now, box_now):
        r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
        phase = torch.exp(1j * 2.* torch.pi * r_frac)
        S = torch.sum(q_now * phase, dim=0)
        polarization = torch.matmul(box_now.to(S.dtype), 
                                    S.unsqueeze(1)) / (1j * 2.* torch.pi)
        return polarization.reshape(-1), phase

    def __repr__(self):
        return f'BEC(remove_mean={self.remove_mean}, epsilon_factor={self.epsilon_factor})'

class Ewald(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 remove_self_interaction=True,
                 norm_factor=90.0474,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        unique_batches = torch.unique(batch)  # Get unique batch indices
        
        results = []
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now = r[mask], q[mask]
            if cell is not None:
                box_now = cell[i]  # Get the box for the i-th configuration
            
            # check if the box is periodic or not
            
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                pot = self.compute_potential_realspace(r_raw_now, q_now)
            else:
                # the box is periodic, we use the reciprocal sum
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now)
            results.append(pot)
        
        return torch.stack(results, dim=0).sum(dim=1)

    def compute_potential_realspace(self, r_raw, q):
        # Compute pairwise distances (norm of vector differences)
        # Add epsilon for safe Hessian compute
        epsilon = 1e-6
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        torch.diagonal(r_ij).add_(epsilon)
        r_ij_norm = torch.norm(r_ij, dim=-1)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
   
        # Compute inverse distance
        r_p_ij = 1.0 / (r_ij_norm)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)
        
        #Exclude diagonal terms from energy
        mask = ~torch.eye(pot.shape[0], device=pot.device).to(torch.bool).unsqueeze(-1)
        mask = torch.vstack([mask.transpose(0,-1)]*pot.shape[-1]).transpose(0,-1)
        pot = pot[mask].sum().view(-1) / self.twopi / 2.0

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
    
        return pot * self.norm_factor
 
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T
        #print('G', G.type())

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3).to(G.dtype)
        kvec = nvec @ G  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]

        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        if q.dim() == 1:  
            q = q.unsqueeze(1)

         #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot -= torch.sum(q**2) / (self.sigma * (2*torch.pi)**1.5)

        return pot * self.norm_factor

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"

class Les(nn.Module):

    def __init__(self, les_arguments: Union[Dict[str, Any], str] = {}):
        """
        LES model for long-range interations
        """
        super().__init__()

        if isinstance(les_arguments, str):
            import yaml
            with open(les_arguments, 'r') as file:
                les_arguments = yaml.safe_load(file)
                if les_arguments is None:
                    les_arguments = {}

        self._parse_arguments(les_arguments)
 
        self.atomwise: nn.Module = (
            Atomwise(
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor, 
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl
            )

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )

    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.n_layers = les_arguments.get('n_layers', 3)
        self.n_hidden = les_arguments.get('n_hidden', [32, 16])
        self.add_linear_nn = les_arguments.get('add_linear_nn', True)
        self.output_scaling_factor = les_arguments.get('output_scaling_factor', 0.1)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        self.use_atomwise = les_arguments.get('use_atomwise', True)

    def forward(self, 
               positions: torch.Tensor, # [n_atoms, 3]
               cell: Optional[torch.Tensor] = None, # [batch_size, 3, 3]
               desc: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               latent_charges: Optional[torch.Tensor] = None, # [n_atoms, ]
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               ) -> Dict[str, Optional[torch.Tensor]]:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        latent_charges: torch.Tensor
        One can also directly input the latent charges. Shape: (n_atoms, )
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
        # check the input shapes
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)


        if latent_charges is not None:
            # check the shape of latent charges
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None and latent_charges is None:
            if not self.use_atomwise:
                raise ValueError("desc must be provided and use_atomwise must be True if latent_charges is not provided")
            # compute the latent charges
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        # compute the long-range interactions
        if compute_energy:
            E_lr = self.ewald(q=latent_charges,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              )
        else:
            E_lr = None

        # compute the BEC
        if compute_bec:
            bec = self.bec(q=latent_charges,
                           r=positions,
                           cell=cell,
                           batch=batch,
                           output_index=bec_output_index,
		           )
        else:
            bec = None

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'BEC': bec,
            }
        return output 

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")

if __name__ == "__main__":
    les = Les(les_arguments={}).to("cuda")

    # set the same random seed for reproducibility
    torch.manual_seed(0)
    r = torch.rand(10, 3) * 10  # Random positions in a 10x10x10 box
    r.requires_grad_(requires_grad=True)
    q = torch.rand(10) * 2 - 1 # Random charges

    box = torch.tensor([10.0, 10.0, 10.0])  # Box dimensions
    box_full = torch.tensor([
        [10.0, 0,0],
        [0,10.0, 0], 
        [0,0,10.0]])  # Box dimensions
    
    result = les(desc=r,
        positions=r,
        cell=box_full.unsqueeze(0),
        batch=None,
        compute_bec=True,
        bec_output_index=1)
    print(result)
import torch
import sys
import os

# Ensure we pick up local graph_parallel
sys.path.append(os.getcwd())

import graph_parallel
from deepmd.pt.model.descriptor.repflows import DescrptBlockRepflows
from deepmd.pt.utils import env

def test_gp_simulation():
    # Setup
    torch.manual_seed(42)
    
    nframes = 2
    nloc = 100
    nall = 100
    nnei = 10
    ntypes = 1
    n_dim = 128
    
    # Enable dynamic selection for GP path
    # Note: smooth_edge_update must be True for dynamic sel
    model = DescrptBlockRepflows(
        e_rcut=6.0, e_rcut_smth=5.0, e_sel=[nnei], 
        a_rcut=4.0, a_rcut_smth=3.0, a_sel=5,
        ntypes=ntypes, nlayers=1, n_dim=n_dim,
        use_dynamic_sel=True, smooth_edge_update=True,
        trainable=False
    )
    
    # Create valid inputs
    # Need to be somewhat realistic for prod_env_mat to not crash or produce nans
    extended_coord = torch.randn(nframes, nall * 3)
    extended_atype = torch.zeros(nframes, nall, dtype=torch.int32)
    nlist = torch.randint(0, nloc, (nframes, nloc, nnei), dtype=torch.int32)
    extended_atype_embd = torch.randn(nframes, nloc, n_dim)
    
    # 1. Baseline (Non-Parallel)
    print("Running Baseline...")
    graph_parallel.set_graph_parallel_enabled(False)
    node_ebd_base, _, _, _, _ = model(
        nlist, extended_coord, extended_atype, extended_atype_embd=extended_atype_embd
    )
    
    # 2. GP Simulation (4 splits)
    print("Running Graph Parallel Simulation...")
    graph_parallel.set_graph_parallel_enabled(True)
    
    # We assume uniform 200 nodes.
    # graph_parallel._balanced_partition_sizes(200, 4) -> [50, 50, 50, 50]
    
    results = []
    ranks = 4
    for rank in range(ranks):
        print(f"  Rank {rank}/{ranks}...")
        graph_parallel.set_gp_rank(rank)
        
        # In a real scenario, inputs might be different (distributed), 
        # but here we pass full inputs and rely on the internal slicing logic we verified.
        node_ebd_part, _, _, _, _ = model(
            nlist, extended_coord, extended_atype, extended_atype_embd=extended_atype_embd
        )
        # Expected output: (50, 128)
        results.append(node_ebd_part)
        
    node_ebd_sim = torch.cat(results, dim=0)
    
    print(f"Base Shape: {node_ebd_base.shape}")
    print(f"Sim Shape:  {node_ebd_sim.shape}")
    
    # Compare
    diff = (node_ebd_base - node_ebd_sim).abs().max()
    print(f"Maximum Difference: {diff.item()}")
    
    if diff < 1e-4:
        print("Test PASSED: Outputs match.")
    else:
        print("Test FAILED: Outputs mismatch.")
        print("Base sample:", node_ebd_base[0, :5])
        print("Sim sample: ", node_ebd_sim[0, :5])

if __name__ == "__main__":
    test_gp_simulation()

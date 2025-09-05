import numpy as np

for id in range(1, 112):
    data = np.load(f'/mnt/data_nas/mptraj_clean_0301/mptraj_energy_0.05/split_clean-mix/first/{id}/set.000/real_atom_types.npy')
    
    if data.max() < 88:
        print(id)

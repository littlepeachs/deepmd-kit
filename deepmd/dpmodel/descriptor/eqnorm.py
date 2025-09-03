# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)


class EqnormArgs:
    def __init__(
        self,
        DIPOLE: bool = False,  # calculate dipole or not
        POLAR: bool = False,  # calculate polar or not
        STRESS: bool = True,  # calculate stress or not
        energy_per_atom: bool = True,  # calculate metric per atom or not
        relative_energy: bool = False,  # calculate conformer relative energy or not
        shift: str = "per_species",  # per_species, per_atom
        scale: str = "force_rms",  # per_species, per_atom, force_rms
        shift_trainable: bool = False,  # shift_scale trainable or not
        scale_trainable: bool = False,  # shift_scale trainable or not
        grad_mode: str = "edge",  # "node", "edge"
        irreps_hidden: str = "128x0e+64x1o+32x2e+32x3o",  # Irreps for hidden layers
        irreps_sh: str = "1x0e+1x1o+1x2e+1x3o",  # Irreps for shell layers
        num_convs: int = 4,  # number of MPNN layers
        num_features: int = 128,  # number of features per node
        r_cutoff: float = 6.0,  # unit A
        num_basis: int = 8,  # number of basis functions
        invariant_layers: int = 2,  # number of invariant layers
        invariant_neurons: int = 64,  # neurons per invariant layer
        poly_p: int = 6,  # polynomial degree
        use_ema: bool = True,  # whether to use EMA (Exponential Moving Average)
        avg_nbr: float = 61.84,  # average number of neighbors
        avg_atoms: float = 31.19  # average number of atoms
    ) -> None:
        # Initialize all parameters
        self.DIPOLE = DIPOLE
        self.POLAR = POLAR
        self.STRESS = STRESS
        self.energy_per_atom = energy_per_atom
        self.relative_energy = relative_energy
        self.shift = shift
        self.scale = scale
        self.shift_trainable = shift_trainable
        self.scale_trainable = scale_trainable
        self.grad_mode = grad_mode
        self.irreps_hidden = irreps_hidden
        self.irreps_sh = irreps_sh
        self.num_convs = num_convs
        self.num_features = num_features
        self.r_cutoff = r_cutoff
        self.num_basis = num_basis
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.poly_p = poly_p
        self.use_ema = use_ema
        self.avg_nbr = avg_nbr
        self.avg_atoms = avg_atoms

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "DIPOLE": self.DIPOLE,
            "POLAR": self.POLAR,
            "STRESS": self.STRESS,
            "energy_per_atom": self.energy_per_atom,
            "relative_energy": self.relative_energy,
            "shift": self.shift,
            "scale": self.scale,
            "shift_trainable": self.shift_trainable,
            "scale_trainable": self.scale_trainable,
            "grad_mode": self.grad_mode,
            "irreps_hidden": self.irreps_hidden,
            "irreps_sh": self.irreps_sh,
            "num_convs": self.num_convs,
            "num_features": self.num_features,
            "r_cutoff": self.r_cutoff,
            "num_basis": self.num_basis,
            "invariant_layers": self.invariant_layers,
            "invariant_neurons": self.invariant_neurons,
            "poly_p": self.poly_p,
            "use_ema": self.use_ema,
            "avg_nbr": self.avg_nbr,
            "avg_atoms": self.avg_atoms,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EqnormArgs":
        return cls(**data)
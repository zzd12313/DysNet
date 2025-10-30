from typing import Iterable, Union, Tuple

import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Gate, resolve_actfn
from ..utils.config import NetConfig
from ..utils.qc import ATOM_MASS

class ScalarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        node_bias: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.layernorm = nn.LayerNorm(self.node_dim) if use_layernorm else nn.Identity()
        
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.constant_(self.out_mlp[2].bias, node_bias)
        
        nn.init.xavier_normal_(self.out_mlp[0].weight, gain=0.1)
        nn.init.xavier_normal_(self.out_mlp[2].weight, gain=0.1)
    
    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        batch = data.batch
        
        x_scalar = self.layernorm(x_scalar)
        
        atom_out = self.out_mlp(x_scalar)
        
        res = scatter(atom_out, batch, dim=0)
        
        return res

class NegGradOut(ScalarOut):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
    ) -> None:
        """
        super().__init__(node_dim, hidden_dim, 1, actfn, node_bias)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch = data.batch; coord = data.pos
        atom_out = self.out_mlp(x_scalar)
        res =  scatter(atom_out, batch, dim=0)
        grad = torch.autograd.grad(
            [atom_out.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        return res, -grad

class VectorOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1o",
        output_dim: int = 3,
        actfn: str = "silu",
        center_of_mass_correction: bool = False,
    ) -> None:
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.center_of_mass_correction = center_of_mass_correction
        
        if self.center_of_mass_correction:
            self.register_buffer("masses", ATOM_MASS)
        
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps),
            o3.Linear(self.hidden_irreps, "1x1o"),
        )
        if output_dim != 3 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 3, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        batch = data.batch
        
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        
        if self.center_of_mass_correction:
            coord = data.pos
            at_no = data.at_no
            masses = self.masses[at_no]
            
            total_masses = scatter(masses, batch, dim=0)
            centroids = scatter(masses.unsqueeze(-1) * coord, batch, dim=0) / total_masses.unsqueeze(-1)
            
            coord_centered = coord - centroids[batch]
            
            atom_out = spherical_out * scalar_out
            res = scatter(atom_out, batch, dim=0)
        else:
             atom_out = spherical_out * scalar_out
             res = scatter(atom_out, batch, dim=0)
        
        if self.output_dim == 1:
            res = torch.linalg.norm(res, dim=-1, keepdim=True)
        
        return res

class PolarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "64x0e + 16x2e",
        output_dim: int = 9,
        actfn: str = "silu",
    ) -> None:
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps),
            o3.Linear(self.hidden_irreps, "1x0e + 1x2e", biases=True),
        )
        nn.init.zeros_(self.spherical_out_mlp[0].bias)
        nn.init.zeros_(self.spherical_out_mlp[2].bias)
        
        self.tensor_product = o3.ElementwiseTensorProduct("1x0e + 1x2e", "1x0e")
        
        irreps = o3.Irreps("1x0e + 1x2e")
        self.register_buffer("sh_to_tensor_transform", irreps.to_matrix())
        
        if output_dim != 9 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 9, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        batch = data.batch
        
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        
        atom_out = self.tensor_product(spherical_out, scalar_out)
        
        mol_out_coeffs = scatter(atom_out, batch, dim=0)
        
        polarizability = torch.einsum('bi,ijk->bjk', mol_out_coeffs, self.sh_to_tensor_transform)

        if self.output_dim == 1:
            res = torch.diagonal(polarizability, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
        elif self.output_dim == 9:
            res = polarizability.view(-1, 9)
            
        return res

class SpatialOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        use_layernorm: bool = True,
        use_physical_definition: bool = False,
    ) -> None:
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.use_physical_definition = use_physical_definition
        self.register_buffer("masses", ATOM_MASS)
        
        self.layernorm = nn.LayerNorm(self.node_dim) if use_layernorm else nn.Identity()
        
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        batch = data.batch
        coord = data.pos
        at_no = data.at_no
        
        x_scalar = self.layernorm(x_scalar)
        
        masses = self.masses[at_no]
        
        total_masses = scatter(masses, batch, dim=0)
        centroids = scatter(masses.unsqueeze(-1) * coord, batch, dim=0) / total_masses.unsqueeze(-1)
        
        coord_centered = coord - centroids[batch]
        
        scalar_out = self.scalar_out_mlp(x_scalar)
        
        squared_distances = torch.sum(coord_centered ** 2, dim=1, keepdim=True)
        
        if self.use_physical_definition:
            numerator = scatter(masses.unsqueeze(-1) * squared_distances, batch, dim=0)
            res = numerator / total_masses.unsqueeze(-1)
        else:
            weighted_distances = scalar_out * squared_distances
            res = scatter(weighted_distances, batch, dim=0)
        
        return res

def resolve_output(config: NetConfig):
    if config.output_mode == "scalar":
        return ScalarOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.output_dim,
            actfn=config.activation,
            node_bias=config.node_average,
            use_layernorm=config.use_layernorm,
        )
    elif config.output_mode == "grad":
        return NegGradOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
            node_bias=config.node_average,
        )
    elif config.output_mode == "vector":
        return VectorOut(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            output_dim=config.output_dim,
            actfn=config.activation,
            center_of_mass_correction=getattr(config, 'center_of_mass_correction', False),
        )
    elif config.output_mode == "polar":
        return PolarOut(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            output_dim=config.output_dim,
            actfn=config.activation,
        )
    elif config.output_mode == "spatial":
        return SpatialOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
            use_layernorm=config.use_layernorm,
            use_physical_definition=getattr(config, 'use_physical_definition', False),
        )
    else:
        raise NotImplementedError(f"output mode {config.output_mode} is not implemented")

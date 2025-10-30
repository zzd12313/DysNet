from typing import Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter, scatter_add
from e3nn import o3

from .dysnet import Embedding
from .output import resolve_output
from ..utils import NetConfig

class SphericalProcessor(nn.Module):
    def __init__(self, node_spherical_irreps, L_max=3):
        super().__init__()
        self.node_spherical_irreps = o3.Irreps(node_spherical_irreps)
        self.L_max = L_max
        
        self.irrep_processors = nn.ModuleDict()
        
        current_offset = 0
        for i, (mul, ir) in enumerate(self.node_spherical_irreps):
            irrep_dim = mul * ir.dim
            if irrep_dim > 0:
                irrep_str = f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"
                self.irrep_processors[f"irrep_{i}"] = o3.Linear(
                    irreps_in=irrep_str,
                    irreps_out=irrep_str,
                    biases=True
                )
            current_offset += irrep_dim
    
    def forward(self, spherical_features):
        """处理球谐特征，保持等变性"""
        if spherical_features.numel() == 0:
            return spherical_features
            
        processed_parts = []
        current_offset = 0
        
        for i, (mul, ir) in enumerate(self.node_spherical_irreps):
            irrep_dim = mul * ir.dim
            if irrep_dim > 0:
                irrep_part = spherical_features.narrow(-1, current_offset, irrep_dim)
                
                processor_key = f"irrep_{i}"
                if processor_key in self.irrep_processors:
                    processed_part = self.irrep_processors[processor_key](irrep_part)
                    processed_parts.append(processed_part)
                else:
                    processed_parts.append(irrep_part)
                    
            current_offset += irrep_dim
        
        if processed_parts:
            return torch.cat(processed_parts, dim=-1)
        else:
            return spherical_features

class Message(nn.Module):
    
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps] = "128x0e + 64x1o + 32x2e",
        num_basis: int = 20,
        body_order: int = 2,
        actfn: str = "silu",
        norm_type: str = "layer",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.body_order = body_order
        self.num_basis = num_basis
        
        self.filter_dim = self.node_dim + self.edge_num_irreps * 2
        
        self.scalar_mlp = self._build_multibody_mlp(actfn)
        
        self.rbf_lin = nn.Linear(self.num_basis, self.filter_dim, bias=True)
        nn.init.zeros_(self.rbf_lin.bias)
        
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.edge_irreps, 
            f"{self.edge_num_irreps}x0e"
        )
        
        from .o3layer import resolve_norm, resolve_o3norm
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps)
        
        self._init_weights()
    
    def _build_multibody_mlp(self, actfn):
        """根据体序构建不同复杂度的MLP，保持合理的参数量"""
        from .o3layer import resolve_actfn
        
        if self.body_order == 2:
            mlp = nn.Sequential(
                nn.Linear(self.node_dim, self.node_dim),
                resolve_actfn(actfn),
                nn.Linear(self.node_dim, self.filter_dim),
            )
        elif self.body_order == 3:
            hidden_dim = int(self.node_dim * 1.5)
            mlp = nn.Sequential(
                nn.Linear(self.node_dim, hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, self.filter_dim),
            )
        else:
            hidden_dim = self.node_dim * 2
            mlp = nn.Sequential(
                nn.Linear(self.node_dim, hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, self.filter_dim),
            )
        
        return mlp
    
    def _init_weights(self):
        """初始化权重，确保训练稳定性和多体差异"""
        for i, layer in enumerate(self.scalar_mlp):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=0.8)
                
                nn.init.uniform_(layer.bias, -0.1, 0.1)
        
        nn.init.xavier_uniform_(self.rbf_lin.weight, gain=0.5)
        nn.init.uniform_(self.rbf_lin.bias, -0.05, 0.05)
    
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
        fcut: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scalar_in = self.norm(x_scalar)
        spherical_in = self.o3norm(x_spherical)
        
        scalar_out = self.scalar_mlp(scalar_in)
        
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        gate_state_spherical, gate_edge_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.edge_num_irreps, self.node_dim],
            dim=-1,
        )
        
        message_spherical = self.rsh_conv(
            spherical_in[edge_index[1]], 
            gate_state_spherical
        )
        
        edge_spherical = self.rsh_conv(rsh, gate_edge_spherical)
        
        message_spherical = message_spherical + edge_spherical
        
        num_nodes = x_scalar.shape[0]
        
        aggregated_scalar = scatter_add(
            message_scalar, edge_index[0], 
            dim=0, dim_size=num_nodes
        )
        
        aggregated_spherical = scatter_add(
            message_spherical, edge_index[0], 
            dim=0, dim_size=num_nodes
        )
        
        new_scalar = x_scalar + aggregated_scalar
        new_spherical = x_spherical + aggregated_spherical
        
        return new_scalar, new_spherical

class Update(nn.Module):
    
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps] = "128x0e + 64x1o + 32x2e",
        actfn: str = "silu",
        norm_type: str = "layer",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.edge_num_irreps
        
        self.update_U = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.update_V = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        
        from .o3layer import Invariant, EquivariantDot
        self.invariant = Invariant(self.edge_irreps)
        self.equidot = EquivariantDot(self.edge_irreps)
        self.dot_lin = nn.Linear(self.edge_num_irreps, self.node_dim, bias=False)
        
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.edge_irreps, 
            f"{self.edge_num_irreps}x0e"
        )
        
        from .o3layer import resolve_actfn
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.edge_num_irreps, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        
        from .o3layer import resolve_norm, resolve_o3norm
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.update_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scalar_in = self.norm(x_scalar)
        spherical_in = self.o3norm(x_spherical)
        
        U_spherical = self.update_U(spherical_in)
        V_spherical = self.update_V(spherical_in)
        
        V_invariant = self.invariant(V_spherical)
        
        mlp_in = torch.cat([scalar_in, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_out,
            [self.edge_num_irreps, self.node_dim, self.node_dim],
            dim=-1
        )
        
        d_spherical = self.rsh_conv(U_spherical, a_vv)
        
        inner_prod = self.equidot(U_spherical, V_spherical)
        inner_prod = self.dot_lin(inner_prod)
        d_scalar = a_sv * inner_prod + a_ss
        
        return x_scalar + d_scalar, x_spherical + d_spherical

class DysNet(nn.Module):
    
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        
        self.embed = Embedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        
        body_orders = getattr(config, "body_orders", [2, 3, 4])
        if isinstance(body_orders, int):
            body_orders = [body_orders] * config.action_blocks
        
        self.message_layers = nn.ModuleList([
            Message(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                num_basis=config.num_basis,
                body_order=body_orders[i % len(body_orders)],
                    actfn=config.activation,
                norm_type=config.norm_type,
            )
            for i in range(config.action_blocks)
        ])
        
        self.update_layers = nn.ModuleList([
            Update(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                    actfn=config.activation,
                    norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        
        self.out = resolve_output(config)
        
        self.spherical_init_proj = o3.Linear(
            irreps_in=config.edge_irreps,
            irreps_out=config.edge_irreps,
            biases=True
        )
    
    def _initialize_spherical_features(
        self, 
        x_scalar: torch.Tensor, 
        rsh: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        num_nodes = x_scalar.shape[0]
        
        initial_spherical = scatter(
            rsh, edge_index[0], 
            dim=0, dim_size=num_nodes, 
            reduce="mean"
        )
        
        initial_spherical = self.spherical_init_proj(initial_spherical)
        
        return initial_spherical

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        at_no, pos, edge_index = data.at_no, data.pos, data.edge_index
        shifts = getattr(data, 'shifts', torch.zeros((edge_index.shape[1], 3), device=pos.device))
        
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        
        x_spherical = self._initialize_spherical_features(x_scalar, rsh, edge_index)
        
        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            x_scalar, x_spherical = msg_layer(
                x_scalar, x_spherical, rbf, fcut, rsh, edge_index
            )
            x_scalar, x_spherical = upd_layer(x_scalar, x_spherical)
                
        return self.out(data, x_scalar, x_spherical)

def resolve_model(config: NetConfig) -> nn.Module:
    if config.version == "dysnet" or config.version == "DysNet":
        return DysNet(config)
    else:
        print(f"Warning: Unknown model version '{config.version}', using DysNet")
        return DysNet(config)

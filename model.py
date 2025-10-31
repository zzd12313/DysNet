from typing import Union, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter, scatter_add
from e3nn import o3
from .xpainn import Embedding
from .output import resolve_output
from ..utils import NetConfig
class SphericalProcessor(nn.Module):
    def __init__(self, node_spherical_irreps, L_max=3, config=None):
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
        from .o3layer import GatedNonlinearity
        if config is not None:
            gate_mlp_hidden_dim = getattr(config, "gate_mlp_hidden_dim", 64)
            use_smooth_activation = getattr(config, "use_smooth_activation", True)
            smooth_factor = getattr(config, "smooth_factor", 0.1)
            residual_weight = getattr(config, "residual_weight", 0.1)
        else:
            gate_mlp_hidden_dim = 64
            use_smooth_activation = True
            smooth_factor = 0.1
            residual_weight = 0.1
        self.spherical_gate = GatedNonlinearity(
            irreps_in=self.node_spherical_irreps,
            gate_mlp_hidden_dim=gate_mlp_hidden_dim,
            use_smooth_activation=use_smooth_activation,
            smooth_factor=smooth_factor,
            residual_weight=residual_weight
        )
    def forward(self, spherical_features):
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
            combined_features = torch.cat(processed_parts, dim=-1)
            if self.spherical_gate is not None:
                gated_features = self.spherical_gate(combined_features)
            else:
                gated_features = combined_features
            return gated_features
        else:
            return spherical_features
class ImprovedMessage(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps] = "128x0e + 64x1o + 32x2e",
        num_basis: int = 20,
        body_order: int = 2,
        actfn: str = "silu",
        norm_type: str = "layer",
        std_balance_degrees: bool = True,
        config: NetConfig = None,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.body_order = body_order
        self.num_basis = num_basis
        self.config = config
        self.filter_dim = self.node_dim + self.edge_num_irreps * 2
        self.scalar_mlp = self._build_multibody_mlp(actfn, config)
        self.rbf_lin = nn.Linear(self.num_basis, self.filter_dim, bias=True)
        nn.init.zeros_(self.rbf_lin.bias)
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.edge_irreps,
            f"{self.edge_num_irreps}x0e"
        )
        from .o3layer import resolve_norm, resolve_o3norm
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps, std_balance_degrees=std_balance_degrees)
        from .o3layer import GatedNonlinearity
        if config is not None:
            gate_mlp_hidden_dim = getattr(config, "gate_mlp_hidden_dim", 64)
            use_smooth_activation = getattr(config, "use_smooth_activation", True)
            smooth_factor = getattr(config, "smooth_factor", 0.1)
            residual_weight = getattr(config, "residual_weight", 0.1)
        else:
            gate_mlp_hidden_dim = 64
            use_smooth_activation = True
            smooth_factor = 0.1
            residual_weight = 0.1
        self.spherical_gate = GatedNonlinearity(
            irreps_in=self.edge_irreps,
            gate_mlp_hidden_dim=gate_mlp_hidden_dim,
            use_smooth_activation=use_smooth_activation,
            smooth_factor=smooth_factor,
            residual_weight=residual_weight
        )
        self._init_weights()
    def _build_multibody_mlp(self, actfn, config=None):
        from .o3layer import resolve_actfn
        if config is not None:
            scale_2 = getattr(config, "body_order_2_hidden_scale", 1.0)
            scale_3 = getattr(config, "body_order_3_hidden_scale", 1.5)
            scale_4 = getattr(config, "body_order_4_hidden_scale", 2.0)
        else:
            scale_2 = 1.0
            scale_3 = 1.5
            scale_4 = 2.0
        if self.body_order == 2:
            hidden_dim = int(self.node_dim * scale_2)
            mlp = nn.Sequential(
                nn.Linear(self.node_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, self.filter_dim),
            )
        elif self.body_order == 3:
            hidden_dim = int(self.node_dim * scale_3)
            mlp = nn.Sequential(
                nn.Linear(self.node_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, self.filter_dim),
            )
        else:
            hidden_dim = int(self.node_dim * scale_4)
            mlp = nn.Sequential(
                nn.Linear(self.node_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_actfn(actfn),
                nn.Linear(hidden_dim, self.filter_dim),
            )
        return mlp
    def _init_weights(self):
        weight_init_gain = getattr(self.config, "weight_init_gain", 0.5) if hasattr(self, 'config') and self.config is not None else 0.5
        bias_init_range = getattr(self.config, "bias_init_range", 0.1) if hasattr(self, 'config') and self.config is not None else 0.1
        rbf_weight_init_gain = getattr(self.config, "rbf_weight_init_gain", 0.5) if hasattr(self, 'config') and self.config is not None else 0.5
        rbf_bias_init_range = getattr(self.config, "rbf_bias_init_range", 0.05) if hasattr(self, 'config') and self.config is not None else 0.05
        for i, layer in enumerate(self.scalar_mlp):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=weight_init_gain)
                nn.init.uniform_(layer.bias, -bias_init_range, bias_init_range)
        nn.init.xavier_uniform_(self.rbf_lin.weight, gain=rbf_weight_init_gain)
        nn.init.uniform_(self.rbf_lin.bias, -rbf_bias_init_range, rbf_bias_init_range)
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
        fcut: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar_in = self.norm(x_scalar)
        spherical_in = self.o3norm(x_spherical)
        if torch.isnan(scalar_in).any():
            print(f"Warning: ImprovedMessage - scalar_in contains NaN after normalization")
        if torch.isnan(spherical_in).any():
            print(f"Warning: ImprovedMessage - spherical_in contains NaN after normalization")
        scalar_out = self.scalar_mlp(scalar_in)
        if torch.isnan(scalar_out).any():
            print(f"Warning: ImprovedMessage - scalar_out contains NaN after MLP")
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
        if self.spherical_gate is not None:
            message_spherical = self.spherical_gate(message_spherical)
        num_nodes = x_scalar.shape[0]
        aggregated_scalar = scatter_add(
            message_scalar, edge_index[0],
            dim=0, dim_size=num_nodes
        )
        aggregated_spherical = scatter_add(
            message_spherical, edge_index[0],
            dim=0, dim_size=num_nodes
        )
        residual_weight = getattr(self.config, 'residual_weight', 0.1) if hasattr(self, 'config') and self.config is not None else 0.1
        new_scalar = x_scalar + residual_weight * aggregated_scalar
        new_spherical = x_spherical + residual_weight * aggregated_spherical
        if self.training:
            with torch.no_grad():
                scalar_grad_norm = torch.norm(new_scalar - x_scalar, dim=-1)
                if scalar_grad_norm.max() > 1000.0:
                    delta_scalar = torch.clamp(aggregated_scalar, -100.0, 100.0)
                    new_scalar = x_scalar + residual_weight * delta_scalar
                    print(f"Warning: Scalar gradient too large, clamped to [-100, 100]")
                spherical_grad_norm = torch.norm(new_spherical - x_spherical, dim=-1)
                if spherical_grad_norm.max() > 1000.0:
                    delta_spherical = torch.clamp(aggregated_spherical, -100.0, 100.0)
                    new_spherical = x_spherical + residual_weight * delta_spherical
                    print(f"Warning: Spherical gradient too large, clamped to [-100, 100]")
        if torch.isnan(new_scalar).any():
            print(f"Warning: ImprovedMessage - new_scalar contains NaN after residual connection")
        if torch.isnan(new_spherical).any():
            print(f"Warning: ImprovedMessage - new_spherical contains NaN after residual connection")
        return new_scalar, new_spherical
class ImprovedUpdate(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps] = "128x0e + 64x1o + 32x2e",
        actfn: str = "silu",
        norm_type: str = "layer",
        std_balance_degrees: bool = True,
        config: NetConfig = None,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.edge_num_irreps
        self.config = config
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
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps, std_balance_degrees=std_balance_degrees)
        from .o3layer import GatedNonlinearity
        if config is not None:
            gate_mlp_hidden_dim = getattr(config, "gate_mlp_hidden_dim", 64)
            use_smooth_activation = getattr(config, "use_smooth_activation", True)
            smooth_factor = getattr(config, "smooth_factor", 0.1)
            residual_weight = getattr(config, "residual_weight", 0.1)
        else:
            gate_mlp_hidden_dim = 64
            use_smooth_activation = True
            smooth_factor = 0.1
            residual_weight = 0.1
        self.update_gate = GatedNonlinearity(
            irreps_in=self.edge_irreps,
            gate_mlp_hidden_dim=gate_mlp_hidden_dim,
            use_smooth_activation=use_smooth_activation,
            smooth_factor=smooth_factor,
            residual_weight=residual_weight
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.update_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar_in = self.norm(x_scalar)
        spherical_in = self.o3norm(x_spherical)
        if torch.isnan(scalar_in).any():
            print(f"Warning: ImprovedUpdate - scalar_in contains NaN after normalization")
        if torch.isnan(spherical_in).any():
            print(f"Warning: ImprovedUpdate - spherical_in contains NaN after normalization")
        U_spherical = self.update_U(spherical_in)
        V_spherical = self.update_V(spherical_in)
        if torch.isnan(U_spherical).any():
            print(f"Warning: ImprovedUpdate - U_spherical contains NaN after linear transform")
        if torch.isnan(V_spherical).any():
            print(f"Warning: ImprovedUpdate - V_spherical contains NaN after linear transform")
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
        if self.update_gate is not None:
            d_spherical = self.update_gate(d_spherical)
        new_scalar = x_scalar + d_scalar
        new_spherical = x_spherical + d_spherical
        if torch.isnan(new_scalar).any():
            print(f"Warning: ImprovedUpdate - new_scalar contains NaN after residual connection")
        if torch.isnan(new_spherical).any():
            print(f"Warning: ImprovedUpdate - new_spherical contains NaN after residual connection")
        return new_scalar, new_spherical
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
            ImprovedMessage(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                num_basis=config.num_basis,
                body_order=body_orders[i % len(body_orders)],
                actfn=config.activation,
                norm_type=config.norm_type,
                std_balance_degrees=getattr(config, "std_balance_degrees", True),
                config=config,
            )
            for i in range(config.action_blocks)
        ])
        self.update_layers = nn.ModuleList([
            ImprovedUpdate(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                actfn=config.activation,
                norm_type=config.norm_type,
                std_balance_degrees=getattr(config, "std_balance_degrees", True),
                config=config,
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
        num_nodes = x_scalar.shape[0]
        initial_spherical = scatter(
            rsh, edge_index[0],
            dim=0, dim_size=num_nodes,
            reduce="mean"
        )
        if initial_spherical.numel() == 0:
            print(f"Error: Found {num_nodes} nodes with no edges! This should not happen in chemical systems.")
            print(f"Edge index shape: {edge_index.shape}")
            print(f"RSH shape: {rsh.shape}")
            print(f"Node positions range: {x_scalar.min():.3f} ~ {x_scalar.max():.3f}")
            print(f"Number of unique nodes in edge_index: {edge_index[0].unique().numel()}")
            raise ValueError("No edges found in molecular graph. Check data preprocessing and cutoff distance.")
        initial_spherical = self.spherical_init_proj(initial_spherical)
        return initial_spherical
    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        at_no, pos, edge_index = data.at_no, data.pos, data.edge_index
        shifts = getattr(data, 'shifts', torch.zeros((edge_index.shape[1], 3), device=pos.device))
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        if torch.isnan(x_scalar).any():
            print(f"Warning: DysNet - x_scalar contains NaN after embedding")
            print(f"Embedding input at_no range: {at_no.min()} to {at_no.max()}")
            print(f"Position range: {pos.min():.3f} to {pos.max():.3f}")
        if torch.isnan(rsh).any():
            print(f"Warning: DysNet - rsh contains NaN after embedding")
        x_spherical = self._initialize_spherical_features(x_scalar, rsh, edge_index)
        if torch.isnan(x_spherical).any():
            print(f"Warning: DysNet - x_spherical contains NaN after initialization")
        for i, (msg_layer, upd_layer) in enumerate(zip(self.message_layers, self.update_layers)):
            x_scalar, x_spherical = msg_layer(
                x_scalar, x_spherical, rbf, fcut, rsh, edge_index
            )
            x_scalar, x_spherical = upd_layer(x_scalar, x_spherical)
            if torch.isnan(x_scalar).any():
                print(f"Warning: DysNet - x_scalar contains NaN after layer {i}")
            if torch.isnan(x_spherical).any():
                print(f"Warning: DysNet - x_spherical contains NaN after layer {i}")
        return self.out(data, x_scalar, x_spherical)
def resolve_model(config: NetConfig) -> nn.Module:
    if config.version in ["v2", "xpainn-multibody-v2", "xpainn-multibody", "dysnet"]:
        return DysNet(config)
    else:
        print(f"Warning: Unknown model version '{config.version}', using DysNet")
        return DysNet(config)

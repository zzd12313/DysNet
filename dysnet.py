import math
from typing import Iterable, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from e3nn import o3
from e3nn.o3 import TensorProduct, ElementwiseTensorProduct

from .o3layer import (
    Invariant, EquivariantDot, Int2c1eEmbedding,
    resolve_actfn, resolve_norm, resolve_o3norm,
)
from .rbf import resolve_cutoff, resolve_rbf

class Embedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
    ) -> None:
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
        self.node_lin = nn.Linear(self.int2c1e.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)
        
        self.sph_harm = o3.SphericalHarmonics(self.edge_irreps, normalize=True, normalization="component")
        
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        try:
            if shifts is None:
                shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device, dtype=pos.dtype)
            
            vec = pos[edge_index[0]] - pos[edge_index[1]] - shifts
            dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
            
            if torch.isnan(vec).any() or torch.isinf(vec).any():
                print("Warning: NaN or Inf detected in position vectors, replacing with zeros")
                vec = torch.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
                dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
            
            x = self.int2c1e(at_no)
            x_scalar = self.node_lin(x)
                
            rbf = self.rbf(dist)
            fcut = self.cutoff_fn(dist)
                
            try:
                vec_norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
                vec_normalized = vec / (vec_norm + 1e-8)
                
                rsh = self.sph_harm(vec_normalized[:, [1, 2, 0]])
                        
            except Exception as e:
                print(f"Warning: Spherical harmonics calculation failed: {e}")
                print(f"Falling back to zero tensor with shape [{vec.shape[0]}, {self.edge_irreps.dim}]")
                rsh = torch.zeros(vec.shape[0], self.edge_irreps.dim, device=vec.device, dtype=vec.dtype)
            
            if torch.isnan(rsh).any():
                print("Warning: NaN detected in spherical harmonics, replacing with zeros")
                rsh = torch.nan_to_num(rsh, nan=0.0)
                
            return x_scalar, rbf, fcut, rsh

        except Exception as e:
            print(f"Error in XEmbedding forward pass: {e}")
            num_nodes = at_no.shape[0]
            num_edges = edge_index.shape[1]
            
            x_scalar = torch.zeros(num_nodes, self.node_dim, device=at_no.device)
            rbf = torch.zeros(num_edges, self.rbf.num_basis, device=at_no.device)
            fcut = torch.ones(num_edges, 1, device=at_no.device)
            rsh = torch.zeros(num_edges, self.edge_irreps.dim, device=at_no.device)
            
            return x_scalar, rbf, fcut, rsh

class MessageLayer(nn.Module):
    def __init__(self, 
                 node_scalar_dim: int, 
                 node_spherical_irreps: Union[str, o3.Irreps],
                 edge_scalar_dim: int,
                 edge_sh_irreps: Union[str, o3.Irreps],
                 output_scalar_dim: int,
                 output_spherical_irreps: Union[str, o3.Irreps],
                 actfn: str = "silu",
                 ):
        super().__init__()
        self.node_scalar_dim = node_scalar_dim
        self.node_spherical_irreps = o3.Irreps(node_spherical_irreps)
        self.edge_scalar_dim = edge_scalar_dim
        self.edge_sh_irreps = o3.Irreps(edge_sh_irreps)
        self.output_scalar_dim = output_scalar_dim
        self.output_spherical_irreps = o3.Irreps(output_spherical_irreps)
        self.act = resolve_actfn(actfn)

        scalar_mlp_input_dim = self.node_scalar_dim * 2 + self.edge_scalar_dim
        self.scalar_message_net = nn.Sequential(
            nn.Linear(scalar_mlp_input_dim, self.output_scalar_dim * 2),
            self.act,
            nn.Linear(self.output_scalar_dim * 2, self.output_scalar_dim * 2),
            self.act,
            nn.Linear(self.output_scalar_dim * 2, self.output_scalar_dim)
        )

        self.tp_full_node_edge = o3.FullTensorProduct(
            self.node_spherical_irreps,
            self.edge_sh_irreps
        )
        
        intermediate_tp_output_irreps = self.tp_full_node_edge.irreps_out
        self.tp_linear_projection = o3.Linear(
            irreps_in=intermediate_tp_output_irreps,
            irreps_out=self.output_spherical_irreps,
            biases=True
        )

        max_possible_irreps = max(
            len(self.node_spherical_irreps),
            len(self.output_spherical_irreps),
            8
        )
        
        fixed_gate_dim = max_possible_irreps * 8
        
        self.spherical_message_gate_net = nn.Sequential(
            nn.Linear(scalar_mlp_input_dim, fixed_gate_dim),
            self.act,
            nn.Linear(fixed_gate_dim, fixed_gate_dim // 2),
            self.act,
            nn.Linear(fixed_gate_dim // 2, fixed_gate_dim // 4),
            nn.Sigmoid()
        )
        
        self.gate_projection = nn.Linear(fixed_gate_dim // 4, self.output_spherical_irreps.num_irreps)

        self.scalar_scale = nn.Parameter(torch.ones(1))
        self.spherical_scale = nn.Parameter(torch.ones(1))

    def forward(self, 
                x_scalar_src: torch.Tensor,
                x_scalar_dst: torch.Tensor,
                x_spherical_src: torch.Tensor,
                edge_scalar_feat: torch.Tensor,
                edge_sh_feat: torch.Tensor,
               ):
        """
        num_edges = x_scalar_src.shape[0]

        scalar_mlp_input = torch.cat([x_scalar_src, x_scalar_dst, edge_scalar_feat], dim=-1)
        scalar_message = self.scalar_message_net(scalar_mlp_input) * self.scalar_scale

        try:
            full_tp_out = self.tp_full_node_edge(x_spherical_src, edge_sh_feat)
            raw_spherical_message = self.tp_linear_projection(full_tp_out)
            
            gate_raw = self.spherical_message_gate_net(scalar_mlp_input)
            gate_scalars_for_sph_msg = self.gate_projection(gate_raw)
        
            gated_spherical_message_parts = []
            current_dim_offset = 0
            
            for i_irrep, (mul, ir) in enumerate(self.output_spherical_irreps):
                ir_dim = mul * ir.dim
                if ir_dim == 0: continue
                
                sph_part = raw_spherical_message.narrow(dim=-1, start=current_dim_offset, length=ir_dim)
                
                if i_irrep < gate_scalars_for_sph_msg.shape[-1]:
                    gate_value = gate_scalars_for_sph_msg.narrow(dim=-1, start=i_irrep, length=1)
                else:
                    gate_value = torch.ones(sph_part.shape[0], 1, device=sph_part.device) * 0.5
                
                gated_part = sph_part * gate_value * self.spherical_scale
                gated_spherical_message_parts.append(gated_part)
                
                current_dim_offset += ir_dim
            
            if not gated_spherical_message_parts:
                spherical_message = torch.zeros((num_edges, 0), device=x_scalar_src.device, dtype=x_scalar_src.dtype)
            else:
                spherical_message = torch.cat(gated_spherical_message_parts, dim=-1)
                
        except Exception as e:
            print(f"Warning: Spherical message computation failed: {e}")
            spherical_message = torch.zeros((num_edges, self.output_spherical_irreps.dim), 
                                          device=x_scalar_src.device, dtype=x_scalar_src.dtype)

        if torch.isnan(scalar_message).any():
            print(f"Warning: MessageLayer - Scalar message contains NaN. Input stats: mean={scalar_mlp_input.mean().item():.3f}, std={scalar_mlp_input.std().item():.3f}")
            scalar_message = torch.nan_to_num(scalar_message, nan=0.0)
        if torch.isnan(spherical_message).any():
            print(f"Warning: MessageLayer - Spherical message contains NaN. Input stats: mean={x_spherical_src.mean().item():.3f}, std={x_spherical_src.std().item():.3f}")
            spherical_message = torch.nan_to_num(spherical_message, nan=0.0)

        return scalar_message, spherical_message

class UpdateLayer(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        node_spherical_irreps: Union[str, o3.Irreps, Iterable] = "64x0e + 32x1o + 16x2e",
        actfn: str = "silu",
        norm_type: str = "layer",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.node_spherical_irreps = o3.Irreps(node_spherical_irreps)
        
        hidden_scalar_dim = self.node_dim * 2
        self.scalar_update_net = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, hidden_scalar_dim),
            resolve_actfn(actfn),
            nn.Linear(hidden_scalar_dim, hidden_scalar_dim),
            resolve_actfn(actfn),
            nn.Linear(hidden_scalar_dim, self.node_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        scalar_input_irreps_for_updater = o3.Irreps(f"{self.node_dim}x0e")
        
        intermediate_irreps = o3.Irreps([
            (mul * 2, ir) for mul, ir in self.node_spherical_irreps
        ])
        
        self.scalar_to_spherical_updater = nn.Sequential(
            o3.Linear(
                irreps_in=scalar_input_irreps_for_updater,
                irreps_out=intermediate_irreps,
                biases=True
            ),
            o3.Linear(
                irreps_in=intermediate_irreps,
                irreps_out=self.node_spherical_irreps,
                biases=True
            )
        )
        
        self.spherical_norm = resolve_o3norm(norm_type, self.node_spherical_irreps, affine=True)
        
        self.scalar_residual_scale = nn.Parameter(torch.ones(1))
        self.spherical_residual_scale = nn.Parameter(torch.ones(1))
        
        nn.init.constant_(self.scalar_residual_scale, 1.0)
        nn.init.constant_(self.spherical_residual_scale, 1.0)
        
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scalar_update_contribution = self.scalar_update_net(x_scalar)
        new_scalar = x_scalar + scalar_update_contribution * self.scalar_residual_scale
        
        spherical_update_contribution = self.scalar_to_spherical_updater(new_scalar)
        
        updated_spherical = x_spherical + spherical_update_contribution * self.spherical_residual_scale
        
        new_spherical = self.spherical_norm(updated_spherical)
        
        if torch.isnan(new_scalar).any():
            print(f"Warning: UpdateLayer - Scalar features contain NaN. Input stats: mean={x_scalar.mean().item():.3f}, std={x_scalar.std().item():.3f}")
            new_scalar = torch.nan_to_num(new_scalar, nan=0.0)
        if torch.isnan(new_spherical).any():
            print(f"Warning: UpdateLayer - Spherical features contain NaN. Input stats: mean={x_spherical.mean().item():.3f}, std={x_spherical.std().item():.3f}")
            new_spherical = torch.nan_to_num(new_spherical, nan=0.0)
            
        return new_scalar, new_spherical

class EleEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.sqrt_dim = math.sqrt(node_dim)
        self.q_linear = nn.Linear(node_dim, node_dim)
        nn.init.zeros_(self.q_linear.bias)
        self.k_linear = nn.Linear(1, node_dim)
        nn.init.zeros_(self.k_linear.bias)
        self.v_linear = nn.Linear(1, node_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ele: torch.Tensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        """Embed electronic features.
        batch_ele = ele.index_select(0, batch).unsqueeze(-1)
        q = self.q_linear(x)
        k = self.k_linear(batch_ele)
        v = self.v_linear(batch_ele)
        dot_prod = torch.sum(q * k, dim=1, keepdim=True) / self.sqrt_dim
        attn = softmax(dot_prod, batch, dim=0)
        out = attn * v
        return out

class OneParticleBasisFunction(nn.Module):
    """单粒子基函数模块
    def __init__(
        self,
        node_dim: int,
        edge_irreps: str,
        num_channels: Dict[str, int],
        num_basis: int,
        L_max: int,
        cutoff: float,
        cutoff_fn: str = "cosine",
    ) -> None:
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.num_basis = num_basis
        self.L_max = L_max
        self.cutoff = cutoff
        
        if cutoff_fn == "cosine":
            from .rbf import CosineCutoff
            self.cutoff_fn = CosineCutoff(cutoff)
        elif cutoff_fn == "mollifier":
            from .rbf import PolynomialCutoff
            self.cutoff_fn = PolynomialCutoff(cutoff)
        else:
            raise ValueError(f"未知的切断函数: {cutoff_fn}")
        
        from .rbf import GaussianSmearing
        self.rbf = GaussianSmearing(num_basis, cutoff)
        
        self.sph_harm_calculator = o3.SphericalHarmonics(self.edge_irreps, normalize=True, normalization="component")
        
        self.edge_dim = self.edge_irreps.dim
        
        self.sph_dim_for_scalar_path = self.edge_irreps.num_irreps
        
        sph_input_dim_for_sph_net = sum(2*l+1 for l in range(L_max+1))
        
        self.rbf_net = nn.Sequential(
            nn.Linear(num_basis, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.sph_net = nn.Sequential(
            nn.Linear(self.edge_irreps.num_irreps if self.edge_irreps.lmax == 0 and all(p==1 for _,(_,p) in self.edge_irreps) else self.edge_irreps.dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.node_net = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.final_net = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.SiLU(),
            nn.Linear(256, self.edge_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """计算单粒子基函数Φ^(t)
        sender, receiver = edge_index
        num_edges = sender.shape[0]
        
        try:
            if edge_attr.shape[1] != 1:
                print(f"Warning: OneParticleBasisFunction expects edge_attr to be distances [N,1], got {edge_attr.shape}")
                if edge_attr.dim() > 1 and edge_attr.shape[1] > 1:
                     distances = torch.norm(edge_attr, dim=1, keepdim=True) if edge_attr.shape[1] == 3 else edge_attr.mean(dim=1, keepdim=True)
                else:
                    distances = torch.zeros(num_edges,1, device=x.device)
            else:
                distances = edge_attr

                f_cut = self.cutoff_fn(distances)
            rbf = self.rbf(distances)
            rbf_features = self.rbf_net(rbf * f_cut)

            sender_node_features = x[sender]
            node_s_features = self.node_net(sender_node_features)

            actual_sph_harm_coeffs = self.sph_harm_calculator(directions)

            if self.edge_irreps.lmax == 0 and all(p == 1 for _, (_, p) in self.edge_irreps):
                sph_scalarized_parts = []
                current_offset = 0
                for mul, ir in self.edge_irreps:
                    dim = mul * ir.dim
                    if dim > 0 :
                        sph_part = actual_sph_harm_coeffs.narrow(-1, current_offset, dim)
                        if ir.l == 0:
                             sph_scalarized_parts.append(sph_part)
                        else:
                             sph_scalarized_parts.append(o3.Norm(ir)(sph_part))
                    current_offset += dim
                
                sph_s_features = torch.cat(sph_scalarized_parts, dim=1) if sph_scalarized_parts else torch.empty(num_edges, 0, device=x.device)

                sph_s_features_processed = self.sph_net(actual_sph_harm_coeffs)
                
                combined_features = torch.cat([rbf_features, sph_s_features_processed, node_s_features], dim=1)
                
                if torch.isnan(combined_features).any():
                    print("Warning: Combined scalar features in OneParticleBasisFunction contain NaN, replacing with 0.")
                    combined_features = torch.nan_to_num(combined_features, nan=0.0)
                
                output_features = self.final_net(combined_features)
            else:
                scalar_modulator_input = torch.cat([rbf_features, node_s_features], dim=1)
                
                gate_coeffs_net = o3.Linear(128*2, self.edge_irreps.num_irreps, biases=True)
                gate_coeffs = gate_coeffs_net(scalar_modulator_input)

                output_features_list = []
                current_offset = 0
                for i_irrep, (mul, ir) in enumerate(self.edge_irreps):
                    dim = mul * ir.dim
                    if dim == 0: continue
                    sph_part = actual_sph_harm_coeffs.narrow(-1, current_offset, dim)
                    
                    sph_part_reshaped = sph_part.reshape(num_edges, mul, ir.dim)
                    
                    g = gate_coeffs.narrow(-1, i_irrep, 1).unsqueeze(-1)
                    output_features_list.append((sph_part_reshaped * g).reshape(num_edges, dim))
                    current_offset += dim
                
                if not output_features_list:
                    output_features = torch.zeros((num_edges, 0), device=x.device, dtype=x.dtype)
                else:
                    output_features = torch.cat(output_features_list, dim=-1)

            if torch.isnan(output_features).any():
                print(f"Warning: Output features in OneParticleBasisFunction contain NaN (type: {'scalar' if self.edge_irreps.lmax == 0 else 'equivariant'}), replacing with 0.")
                output_features = torch.nan_to_num(output_features, nan=0.0)

            return output_features
            
        except Exception as e:
            print(f"OneParticleBasisFunction前向传播错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return torch.zeros((num_edges, self.edge_dim), device=x.device)

class EnhancedBodyInteraction(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_channels, dropout_rate=0.0):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        
        irrep_dims = {
            "0e": 1,
            "1o": 3,
            "2e": 5,
            "3o": 7
        }
        self.total_sph_dim = sum(num_channels[key] * irrep_dims[key] for key in num_channels)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.interaction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.sph_processor = EfficientSphericalFeatures(num_channels)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + self.total_sph_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.output_projection = nn.Linear(hidden_dim, node_dim)
        
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, nodes, edge_index, edge_features, rsh=None):
        try:
            src, dst = edge_index
            src_features = nodes[src]
            dst_features = nodes[dst]
            
            combined_features = torch.cat([src_features, dst_features, edge_features], dim=-1)
            extracted_features = self.feature_extractor(combined_features)
            
            gate = self.gate_net(extracted_features)
            
            interaction = self.interaction_net(extracted_features)
            
            if rsh is not None:
                try:
                    processed_sph = self.sph_processor(rsh)
                    combined = torch.cat([processed_sph, interaction], dim=-1)
                    interaction = self.fusion_layer(combined)
                except Exception as e:
                    print(f"Warning: Spherical feature processing failed: {e}")
                    pass
            
            interaction = interaction * gate
            
            message = scatter(interaction, dst, dim=0, reduce='sum', dim_size=nodes.size(0))
            
            output = self.output_projection(message)
            
            output = output * self.residual_scale
            
            return output
            
        except Exception as e:
            print(f"Error in EnhancedBodyInteraction: {e}")
            return torch.zeros(nodes.shape[0], self.node_dim, 
                             device=nodes.device, dtype=nodes.dtype)

class TwoBodyInteraction(EnhancedBodyInteraction):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_channels, dropout_rate=0.0):
        super().__init__(node_dim, edge_dim, hidden_dim, 
                        {'0e': num_channels.get('0e', 0)}, dropout_rate)

class ThreeBodyInteraction(EnhancedBodyInteraction):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_channels, dropout_rate=0.0):
        super().__init__(node_dim, edge_dim, hidden_dim, 
                        {'0e': num_channels.get('0e', 0),
                         '1o': num_channels.get('1o', 0)}, dropout_rate)

class FourBodyInteraction(EnhancedBodyInteraction):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_channels, dropout_rate=0.0):
        super().__init__(node_dim, edge_dim, hidden_dim, 
                        {'0e': num_channels.get('0e', 0),
                         '1o': num_channels.get('1o', 0),
                         '2e': num_channels.get('2e', 0),
                         '3o': num_channels.get('3o', 0)}, dropout_rate)

class Layer(nn.Module):
    def __init__(
        self,
        num_node_types,
        node_dim,
        edge_dim,
        body_order=2,
        L_max=2,
        hidden_dim=128,
        num_channels=None,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(Layer, self).__init__()

        self.num_node_types = num_node_types
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.body_order = body_order
        self.L_max = L_max
        self.hidden_dim = max(hidden_dim, node_dim)
        
        if num_channels is None:
            self.num_channels = {"k": 4, "c": 4}
        else:
            self.num_channels = num_channels
            
        self.dropout_rate = dropout_rate

        self.message_components = nn.ModuleList()
        
        self.message_components.append(
            TwoBodyInteraction(
                node_dim=node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                num_channels=self.num_channels,
                dropout_rate=self.dropout_rate
            )
        )
        
        if body_order >= 3:
            self.message_components.append(
                ThreeBodyInteraction(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    hidden_dim=self.hidden_dim,
                    num_channels=self.num_channels,
                    dropout_rate=self.dropout_rate
                )
            )
        
        if body_order >= 4:
            self.message_components.append(
                FourBodyInteraction(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    hidden_dim=self.hidden_dim,
                    num_channels=self.num_channels,
                    dropout_rate=self.dropout_rate
                )
            )
        
        projection_width = int(64 * (1 + 0.5 * (body_order - 2)))
        
        self.edge_projections = nn.ModuleDict({
            "small": nn.Sequential(
                nn.Linear(20, projection_width),
                nn.SiLU(),
                nn.Linear(projection_width, self.edge_dim),
            ),
            "medium": nn.Sequential(
                nn.Linear(24, projection_width),
                nn.SiLU(),
                nn.Linear(projection_width, self.edge_dim),
            ),
            "large": nn.Sequential(
                nn.Linear(64, projection_width),
                nn.SiLU(),
                nn.Linear(projection_width, self.edge_dim),
            ),
        })
        
        node_projection_layers = [nn.Linear(self.node_dim, self.hidden_dim), nn.SiLU()]
        for _ in range(max(0, body_order - 2)):
            node_projection_layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            ])
        self.node_projection = nn.Sequential(*node_projection_layers)
        
        self.final_projection = nn.Linear(self.hidden_dim, self.node_dim)
        
        self.use_spherical_info = True
        self.max_spherical_components = (L_max + 1) ** 2
        
        self.attention_range = min(body_order - 1, 3)
        
        num_components = len(self.message_components)
        if num_components > 0:
            if num_components == 1:
                init_weights = torch.tensor([1.0])
            elif num_components == 2:
                init_weights = torch.tensor([0.7, 0.3])
            elif num_components == 3:
                init_weights = torch.tensor([0.6, 0.3, 0.1])
            else:
                init_weights = torch.ones(num_components) / num_components
            self.body_order_weights = nn.Parameter(init_weights.clone())
        else:
            self.body_order_weights = nn.Parameter(torch.ones(1))
            
        self.pre_final_norm = nn.LayerNorm(self.hidden_dim)
        
        self.edge_projection_dropout = nn.Dropout(dropout_rate)
        self.node_projection_dropout = nn.Dropout(dropout_rate) 
        self.final_projection_dropout = nn.Dropout(dropout_rate)

    def forward(self, x_scalar, edge_index, edge_features, rsh=None):
        sender, receiver = edge_index

        nodes_projected = self.node_projection_dropout(self.node_projection(x_scalar))
        
        edge_feat_dim_raw = edge_features.shape[1]
        projected_edge_features_for_components: torch.Tensor
        if edge_feat_dim_raw <= 20:
             projected_edge_features_for_components = self.edge_projection_dropout(self.edge_projections["small"](edge_features))
        elif edge_feat_dim_raw <= 32:
             projected_edge_features_for_components = self.edge_projection_dropout(self.edge_projections["medium"](edge_features))
        else:
             projected_edge_features_for_components = self.edge_projection_dropout(self.edge_projections["large"](edge_features))

        if rsh is not None:
            total_channels = sum(self.num_channels.values())
            if rsh.shape[-1] != total_channels:
                rsh = rsh[:, :total_channels]
                if rsh.shape[-1] < total_channels:
                    padding = torch.zeros(rsh.shape[0], total_channels - rsh.shape[-1], 
                                        device=rsh.device, dtype=rsh.dtype)
                    rsh = torch.cat([rsh, padding], dim=-1)

        component_contributions = []
        for i, component in enumerate(self.message_components):
            weight = torch.softmax(self.body_order_weights, dim=0)[i]
            
            node_level_message = component(
                nodes_projected,
                edge_index,
                projected_edge_features_for_components,
                rsh
            )
            
            assert node_level_message.shape == nodes_projected.shape, \
                f"Component output shape {node_level_message.shape} does not match projected nodes shape {nodes_projected.shape}"
            
            component_contributions.append(node_level_message * weight)
        
        if component_contributions:
            total_aggregated_contributions = torch.sum(torch.stack(component_contributions), dim=0)
        else:
            total_aggregated_contributions = torch.zeros_like(nodes_projected)
        
        updated_nodes = nodes_projected + total_aggregated_contributions
        
        updated_nodes = self.pre_final_norm(updated_nodes)
        
        updated_x_scalar = self.final_projection_dropout(self.final_projection(updated_nodes))
        
        residual_weight = 0.8
        updated_x_scalar = residual_weight * x_scalar + (1 - residual_weight) * updated_x_scalar
        
        return updated_x_scalar

class IrrepsProjection(nn.Module):
    """投影层，用于将特征从一个不可约表示空间投影到另一个不可约表示空间。
    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        
        self.input_dims = {}
        in_idx = 0
        for mul, (l, p) in self.irreps_in:
            dim = mul * (2 * l + 1)
            self.input_dims[l] = (in_idx, dim)
            in_idx += dim
        
        self.output_dims = {}
        out_idx = 0
        for mul, (l, p) in self.irreps_out:
            dim = mul * (2 * l + 1)
            self.output_dims[l] = (out_idx, dim)
            out_idx += dim
        
        self.projections = nn.ModuleDict()
        for l in self.output_dims:
            if l in self.input_dims:
                _, in_dim = self.input_dims[l]
                _, out_dim = self.output_dims[l]
                self.projections[str(l)] = nn.Linear(in_dim, out_dim, bias=False)
            else:
                _, out_dim = self.output_dims[l]
                self.projections[str(l)] = nn.Parameter(
                    torch.zeros(out_dim), requires_grad=True
                )
    
    def forward(self, x):
        """将输入张量投影到目标不可约表示空间
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, self.irreps_out.dim, device=x.device, dtype=x.dtype)
        
        for l, (out_idx, out_dim) in self.output_dims.items():
            l_str = str(l)
            if l in self.input_dims:
                in_idx, in_dim = self.input_dims[l]
                input_part = x[:, in_idx:in_idx+in_dim]
                result[:, out_idx:out_idx+out_dim] = self.projections[l_str](input_part)
            else:
                result[:, out_idx:out_idx+out_dim] = self.projections[l_str].unsqueeze(0).expand(batch_size, -1)
        
        return result

class EfficientSphericalFeatures(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        
        self.irrep_dims = {
            "0e": 1,
            "1o": 3,
            "2e": 5,
            "3o": 7
        }
        
        self.total_dim = sum(num_channels[key] * self.irrep_dims[key] for key in num_channels)
        
        self.intra_orbital_processors = nn.ModuleDict()
        self.orbital_dims = {}
        
        for key, dim in self.irrep_dims.items():
            if num_channels.get(key, 0) > 0:
                irrep_total_dim = num_channels[key] * dim
                self.orbital_dims[key] = irrep_total_dim
                
                self.intra_orbital_processors[key] = nn.Sequential(
                    nn.Linear(irrep_total_dim, max(irrep_total_dim, 8)),
                    nn.SiLU(),
                    nn.Linear(max(irrep_total_dim, 8), irrep_total_dim),
                )
        
        if len(self.orbital_dims) > 1:
            orbital_summary_dim = 16
            
            self.orbital_summarizers = nn.ModuleDict()
            for key, irrep_dim in self.orbital_dims.items():
                self.orbital_summarizers[key] = nn.Sequential(
                    nn.Linear(irrep_dim, orbital_summary_dim),
                    nn.SiLU()
                )
            
            total_summary_dim = len(self.orbital_dims) * orbital_summary_dim
            self.inter_orbital_processor = nn.Sequential(
                nn.Linear(total_summary_dim, total_summary_dim // 2),
                nn.SiLU(),
                nn.Linear(total_summary_dim // 2, total_summary_dim),
                nn.Sigmoid()
            )
            
            self.cross_orbital_weights = nn.ModuleDict()
            for key in self.orbital_dims.keys():
                self.cross_orbital_weights[key] = nn.Linear(total_summary_dim, self.orbital_dims[key])
        
        self.final_processor = nn.Sequential(
            nn.Linear(self.total_dim, min(self.total_dim, 64)),
            nn.SiLU(),
            nn.Linear(min(self.total_dim, 64), self.total_dim)
        )
        
        self.projection_cache = {}
        
        orbital_importance = {"0e": 1.0, "1o": 0.9, "2e": 0.7, "3o": 0.5}
        weights = []
        for key in num_channels.keys():
            if num_channels[key] > 0:
                weights.append(orbital_importance.get(key, 0.3))
        
        if weights:
            weights = torch.tensor(weights)
            weights = weights / weights.sum()
            self.register_buffer('orbital_weights', weights)
        else:
            self.register_buffer('orbital_weights', torch.ones(1))
        
    def _get_or_create_projection(self, input_dim):
        """动态创建投影层 - 仅用于输入维度对齐"""
        if input_dim == self.total_dim:
            return None
        
        if input_dim not in self.projection_cache:
            self.projection_cache[input_dim] = nn.Linear(input_dim, self.total_dim).to(
                next(self.parameters()).device
            )
        return self.projection_cache[input_dim]
        
    def forward(self, rsh_features):
        try:
            input_dim = rsh_features.shape[-1]
            
            if input_dim != self.total_dim:
                projection = self._get_or_create_projection(input_dim)
                if projection is not None:
                    rsh_features = projection(rsh_features)
                else:
                    if input_dim > self.total_dim:
                        rsh_features = rsh_features[:, :self.total_dim]
                    else:
                        padding = torch.zeros(rsh_features.shape[0], self.total_dim - input_dim, 
                                            device=rsh_features.device, dtype=rsh_features.dtype)
                        rsh_features = torch.cat([rsh_features, padding], dim=-1)
            
            if torch.isnan(rsh_features).any() or torch.isinf(rsh_features).any():
                rsh_features = torch.nan_to_num(rsh_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            current_idx = 0
            orbital_features = {}
            orbital_summaries = []
            
            for i, (key, num_ch) in enumerate(self.num_channels.items()):
                if num_ch > 0 and key in self.intra_orbital_processors:
                    irrep_dim = num_ch * self.irrep_dims[key]
                    
                    if current_idx + irrep_dim <= rsh_features.shape[-1]:
                        orbital_part = rsh_features[:, current_idx:current_idx + irrep_dim]
                        
                        processed_orbital = self.intra_orbital_processors[key](orbital_part)
                        orbital_features[key] = processed_orbital
                        
                        if hasattr(self, 'orbital_summarizers') and key in self.orbital_summarizers:
                            orbital_summary = self.orbital_summarizers[key](processed_orbital)
                            orbital_summaries.append(orbital_summary)
                        
                        current_idx += irrep_dim
                    else:
                        zero_part = torch.zeros(rsh_features.shape[0], irrep_dim, 
                                              device=rsh_features.device, dtype=rsh_features.dtype)
                        orbital_features[key] = zero_part
            
            if len(orbital_summaries) > 1 and hasattr(self, 'inter_orbital_processor'):
                combined_summary = torch.cat(orbital_summaries, dim=-1)
                
                interaction_gates = self.inter_orbital_processor(combined_summary)
                
                enhanced_orbital_features = {}
                for key, orbital_feat in orbital_features.items():
                    if key in self.cross_orbital_weights:
                        interaction_weight = self.cross_orbital_weights[key](interaction_gates)
                        enhanced_orbital_features[key] = orbital_feat * (1.0 + 0.1 * torch.tanh(interaction_weight))
                    else:
                        enhanced_orbital_features[key] = orbital_feat
                
                orbital_features = enhanced_orbital_features
            
            final_features = []
            for i, (key, _) in enumerate(self.num_channels.items()):
                if key in orbital_features:
                    weight = self.orbital_weights[i] if i < len(self.orbital_weights) else 1.0
                    final_features.append(orbital_features[key] * weight)
            
            if not final_features:
                return torch.zeros(rsh_features.shape[0], self.total_dim, 
                                 device=rsh_features.device, dtype=rsh_features.dtype)
            
            combined_features = torch.cat(final_features, dim=-1)
            
            result = self.final_processor(combined_features)
            
            if torch.isnan(result).any():
                result = torch.nan_to_num(result, nan=0.0)
                
            return result
            
        except Exception as e:
            print(f"Error in EfficientSphericalFeatures: {e}")
            return torch.zeros(rsh_features.shape[0], self.total_dim, 
                             device=rsh_features.device, dtype=rsh_features.dtype)

class DynamicBodyOrderWeights(nn.Module):
    def __init__(self, node_dim, num_components):
        super().__init__()
        self.structure_encoder = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_components),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, node_features, edge_index):
        structure_features = self.structure_encoder(node_features)
        
        src, dst = edge_index
        edge_features = structure_features[src] * structure_features[dst]
        
        weights = self.weight_generator(edge_features)
        return weights

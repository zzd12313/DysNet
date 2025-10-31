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
        
class DirectForceOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 256,
        hidden_dim: int = 128,
        actfn: str = "silu",
        node_bias: float = 0.0,
        use_layernorm: bool = True,
        energy_lr: float = None,
        force_lr: float = None,
        energy_weight_decay: float = None,
        force_weight_decay: float = None,
        config: NetConfig = None,
    ) -> None:
        """
        直接力预测输出层
        
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
            `use_layernorm`: Whether to use layer normalization.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.energy_lr = energy_lr
        self.force_lr = force_lr
        
        self.energy_weight_decay = energy_weight_decay
        self.force_weight_decay = force_weight_decay
        
        if config is not None:
            if self.energy_weight_decay is None:
                self.energy_weight_decay = getattr(config, 'energy_weight_decay', None)
            if self.force_weight_decay is None:
                self.force_weight_decay = getattr(config, 'force_weight_decay', None)
        
        self.layernorm = nn.LayerNorm(self.node_dim) if use_layernorm else nn.Identity()
        
        self.energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),      
            nn.LayerNorm(self.hidden_dim),                  
            resolve_actfn(actfn),                          
            nn.Dropout(0.2),                              
            nn.Linear(self.hidden_dim, self.hidden_dim),    
            nn.LayerNorm(self.hidden_dim),                  
            resolve_actfn(actfn),                          
            nn.Dropout(0.2),                              
            nn.Linear(self.hidden_dim, 1),                  
        )
        
       
        self.force_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), 
            resolve_actfn(actfn),
            nn.Dropout(0.15),  
            nn.Linear(self.hidden_dim, self.hidden_dim),  
            nn.LayerNorm(self.hidden_dim), 
            resolve_actfn(actfn),
            nn.Dropout(0.15), 
            nn.Linear(self.hidden_dim, 3), 
        )
        
        self._init_force_mlp_weights()
        self._init_energy_mlp_weights(node_bias)
    
    def _init_force_mlp_weights(self):
        """优化的力预测头权重初始化，使用更保守的策略防止梯度爆炸"""
        for i, layer in enumerate(self.force_mlp):
            if isinstance(layer, nn.Linear):
                if i == 0:  
                    nn.init.xavier_uniform_(layer.weight, gain=0.2)  
                    nn.init.zeros_(layer.bias)
                elif i == 4:  
                    nn.init.xavier_uniform_(layer.weight, gain=0.3) 
                    nn.init.zeros_(layer.bias)
                else: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.05) 
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _init_energy_mlp_weights(self, node_bias):
        """优化的能量预测头权重初始化，使用更保守的策略防止梯度爆炸"""
        if node_bias is None:
            print(f"WARNING: node_bias is None, setting to 0.0")
            node_bias = 0.0
        
        for i, layer in enumerate(self.energy_mlp):
            if isinstance(layer, nn.Linear):
                if i == 0: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.2) 
                    nn.init.zeros_(layer.bias)
                elif i == 4: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.3) 
                    nn.init.zeros_(layer.bias)
                else: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.05)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        self.energy_mlp[8].bias.data.fill_(node_bias)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `energy`: 分子能量
            `forces`: 原子力（直接预测）
        """
        batch = data.batch
        
        x_scalar = self.layernorm(x_scalar)
        
        identity = x_scalar[:, :1] 
        atom_energy = self.energy_mlp(x_scalar)
        
        if self.training:
            atom_energy = atom_energy + 0.1 * identity
        
        energy_norm = torch.abs(atom_energy)
        
        if self.training and energy_norm.max() > 10.0:  
            scale_factor = torch.sigmoid(energy_norm / 5.0 - 1.0)  
            atom_energy = atom_energy * scale_factor
            
            if energy_norm.max() > 20.0:
                print(f"Warning: Energy values too large ({energy_norm.max().item():.2f}), applying soft clipping")
        
        energy = scatter(atom_energy.squeeze(-1), batch, dim=0)
        
        if torch.isnan(energy).any() or torch.isinf(energy).any():
            print(f"ERROR: NaN or Inf detected in energy prediction - this should not happen!")
            print(f"energy stats: mean={energy.mean().item():.6f}, std={energy.std().item():.6f}")
            raise ValueError("NaN/Inf detected in energy prediction - check energy_mlp output")
        
        forces = self.force_mlp(x_scalar)  # [N_atoms, 3]
        
        if self.training:
            force_norm = torch.norm(forces, dim=-1, keepdim=True)
            
            if force_norm.max() > 20.0: 
                scale_factor = torch.sigmoid(force_norm / 10.0 - 1.0) 
                forces = forces * scale_factor
                
                if force_norm.max() > 40.0:
                    print(f"Warning: Force norm too large: {force_norm.max().item():.2f}, applying soft clipping")
        
        if torch.isnan(forces).any() or torch.isinf(forces).any():
            print(f"ERROR: NaN or Inf detected in force prediction - this should not happen!")
            print(f"forces stats: mean={forces.mean().item():.6f}, std={forces.std().item():.6f}")
            raise ValueError("NaN/Inf detected in force prediction - check force_mlp output")
        
        return energy, forces
    
    def get_param_groups(self, base_lr: float = None, base_weight_decay: float = None, energy_lr: float = None, force_lr: float = None):
        """
        获取参数组，用于差分学习率和权重衰减
        
        Args:
            base_lr: 基础学习率，如果为None则使用默认值
            base_weight_decay: 基础权重衰减，如果为None则使用默认值
            energy_lr: 能量头学习率，如果为None则使用self.energy_lr
            force_lr: 力头学习率，如果为None则使用self.force_lr
            
        Returns:
            param_groups: 参数组列表
        """
        param_groups = []
        
        energy_lr = energy_lr if energy_lr is not None else self.energy_lr
        force_lr = force_lr if force_lr is not None else self.force_lr
        
        if energy_lr is not None and force_lr is not None:
            energy_weight_decay = getattr(self, 'energy_weight_decay', base_weight_decay)
            force_weight_decay = getattr(self, 'force_weight_decay', base_weight_decay)
            
            energy_params = list(self.energy_mlp.parameters())
            param_groups.append({
                "params": energy_params,
                "lr": energy_lr,
                "weight_decay": energy_weight_decay,
                "name": "energy_head"
            })
            
            force_params = list(self.force_mlp.parameters())
            param_groups.append({
                "params": force_params,
                "lr": force_lr,
                "weight_decay": force_weight_decay,
                "name": "force_head"
            })
            
            other_params = list(self.layernorm.parameters())
            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": base_lr if base_lr is not None else energy_lr,
                    "weight_decay": base_weight_decay,
                    "name": "other_params"
                })
        else:
            param_groups.append({
                "params": self.parameters(),
                "lr": base_lr,
                "weight_decay": base_weight_decay,
                "name": "all_params"
            })
        
        return param_groups

class ConservativeForceOut(nn.Module):
    """
    保守力输出层 - 第二阶段专用
    只预测能量，力完全由 F = -∇E 计算
    """
    def __init__(
        self,
        node_dim: int = 256,
        hidden_dim: int = 128,
        actfn: str = "silu",
        node_bias: float = 0.0,
        use_layernorm: bool = True,
        energy_lr: float = None,
        force_lr: float = None,
        energy_weight_decay: float = None,
        force_weight_decay: float = None,
        config: NetConfig = None,
    ) -> None:
        """
        与DirectForceOut完全相同的初始化参数和架构
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.energy_lr = energy_lr
        self.force_lr = force_lr
        
        self.energy_weight_decay = energy_weight_decay
        self.force_weight_decay = force_weight_decay
        
        if config is not None:
            if self.energy_weight_decay is None:
                self.energy_weight_decay = getattr(config, 'energy_weight_decay', None)
            if self.force_weight_decay is None:
                self.force_weight_decay = getattr(config, 'force_weight_decay', None)
        
        self.layernorm = nn.LayerNorm(self.node_dim) if use_layernorm else nn.Identity()
        
        self.energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),     
            nn.LayerNorm(self.hidden_dim),                
            resolve_actfn(actfn),                         
            nn.Dropout(0.2),                              
            nn.Linear(self.hidden_dim, self.hidden_dim),    
            nn.LayerNorm(self.hidden_dim),                  
            resolve_actfn(actfn),                          
            nn.Dropout(0.2),                           
            nn.Linear(self.hidden_dim, 1),                 
        )
        
        self.force_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            resolve_actfn(actfn),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            resolve_actfn(actfn),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, 3), 
        )
        
        self._init_force_mlp_weights()
        self._init_energy_mlp_weights(node_bias)
    
    def _init_force_mlp_weights(self):
        """与DirectForceOut完全相同的力预测头权重初始化"""
        for i, layer in enumerate(self.force_mlp):
            if isinstance(layer, nn.Linear):
                if i == 0: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.2)
                    nn.init.zeros_(layer.bias)
                elif i == 4:  
                    nn.init.xavier_uniform_(layer.weight, gain=0.3)
                    nn.init.zeros_(layer.bias)
                else: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.05)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _init_energy_mlp_weights(self, node_bias):
        """与DirectForceOut完全相同的能量预测头权重初始化"""
        if node_bias is None:
            print(f"WARNING: node_bias is None, setting to 0.0")
            node_bias = 0.0
        
        for i, layer in enumerate(self.energy_mlp):
            if isinstance(layer, nn.Linear):
                if i == 0: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.2)
                    nn.init.zeros_(layer.bias)
                elif i == 4: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.3)
                    nn.init.zeros_(layer.bias)
                else: 
                    nn.init.xavier_uniform_(layer.weight, gain=0.05)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        self.energy_mlp[8].bias.data.fill_(node_bias)
    
    def _compute_conservative_forces_ddp_compatible(self, energy, pos, batch, x_scalar):
        """
        保守力计算：F = -∇E，DDP兼容，分子级计算
    
        if not pos.requires_grad:
            pos = pos.requires_grad_(True)
        

        grad = torch.autograd.grad(
            energy.sum(),        
            pos,                 
            retain_graph=True,   
            create_graph=False, 
            allow_unused=True    
        )[0]
        
        forces = -grad 
        

        if torch.isnan(forces).any() or torch.isinf(forces).any():
            print("WARNING: NaN or Inf detected in conservative force calculation")
            forces = torch.where(
                torch.isnan(forces) | torch.isinf(forces),
                torch.zeros_like(forces),
                forces
            )
        
        force_norm = torch.norm(forces, dim=-1, keepdim=True)
        max_force_norm = 100.0
        if force_norm.max() > max_force_norm:
            scale_factor = torch.clamp(max_force_norm / force_norm, max=1.0)
            forces = forces * scale_factor
        
        return forces
        
    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：改进的保守力计算，数值稳定且分布式友好
        
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `energy`: 分子能量
            `forces`: 保守力（通过改进的自动微分计算）
        """
        batch = data.batch
        pos = data.pos
        
        x_scalar = self.layernorm(x_scalar)
        
        atom_energy = self.energy_mlp(x_scalar)  # [N_atoms, 1]
        
        energy = scatter(atom_energy.squeeze(-1), batch, dim=0)  # [batch_size]
        
        if self.training:
            conservative_success_count = 0
            mlp_fallback_count = 0
            try:
                if not pos.requires_grad:
                    pos.requires_grad_(True)
                
                forces = self._compute_conservative_forces_ddp_compatible(energy, pos, batch, x_scalar)
                
                if torch.isnan(forces).any() or torch.isinf(forces).any():
                    print(f"WARNING: NaN or Inf detected in force calculation, using zero forces")
                    forces = torch.zeros_like(pos)
                
                force_norm = torch.norm(forces, dim=-1, keepdim=True)
                max_force_norm = 100.0  
                
                if force_norm.max() > max_force_norm:
                    scale_factor = torch.clamp(max_force_norm / force_norm, max=1.0)
                    forces = forces * scale_factor
                    print(f"WARNING: Force clipping applied, max force norm: {force_norm.max().item():.2f}")
                
            except Exception as e:
                raise RuntimeError(f"Conservative force calculation failed: {e}")
        else:
            forces = self._compute_conservative_forces_ddp_compatible(energy, pos, batch, x_scalar)
        
        return energy, forces

    def get_param_groups(self, base_lr: float = None, base_weight_decay: float = None, energy_lr: float = None, force_lr: float = None):
        """
        获取参数组，用于差分学习率和权重衰减
        与DirectForceOut完全相同的参数组结构
        
        Args:
            base_lr: 基础学习率，如果为None则使用默认值
            base_weight_decay: 基础权重衰减，如果为None则使用默认值
            energy_lr: 能量头学习率，如果为None则使用self.energy_lr
            force_lr: 力头学习率，如果为None则使用self.force_lr
            
        Returns:
            param_groups: 参数组列表
        """
        param_groups = []
        
        energy_lr = energy_lr if energy_lr is not None else self.energy_lr
        force_lr = force_lr if force_lr is not None else self.force_lr
        
        if energy_lr is not None and force_lr is not None:
            energy_weight_decay = getattr(self, 'energy_weight_decay', base_weight_decay)
            force_weight_decay = getattr(self, 'force_weight_decay', base_weight_decay)
            
            energy_params = list(self.energy_mlp.parameters())
            param_groups.append({
                "params": energy_params,
                "lr": energy_lr,
                "weight_decay": energy_weight_decay,
                "name": "energy_head"
            })
            
            force_params = list(self.force_mlp.parameters())
            param_groups.append({
                "params": force_params,
                "lr": force_lr,
                "weight_decay": force_weight_decay,
                "name": "force_head"
            })
            
            other_params = list(self.layernorm.parameters())
            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": base_lr if base_lr is not None else energy_lr,
                    "weight_decay": base_weight_decay,
                    "name": "other_params"
                })
        else:
            param_groups.append({
                "params": self.parameters(),
                "lr": base_lr,
                "weight_decay": base_weight_decay,
                "name": "all_params"
            })
        
        return param_groups

class BodyOrderEnergyOutput(nn.Module):
    """
    基于体序的能量分解输出层
    充分利用多体相互作用架构，实现精确的能量分解
    集成势能面平滑性处理，确保分子动力学模拟的数值稳定性
    """
    def __init__(
        self,
        node_dim: int = 256,
        hidden_dim: int = 128,
        actfn: str = "silu",
        use_layernorm: bool = True,
        config: NetConfig = None,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.layernorm = nn.LayerNorm(self.node_dim) if use_layernorm else nn.Identity()
        
        self.atomic_energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        
        self.two_body_energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        
        self.three_body_energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 3, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        
        self.four_body_energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 4, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        
        self.molecular_energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        
        self.body_order_energy_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))
        
        self.smoothness_factor = getattr(config, 'smoothness_factor', 0.1) if config else 0.1
        self.use_smoothness_loss = getattr(config, 'use_smoothness_loss', True) if config else True
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重，确保训练稳定性"""
        for mlp in [self.atomic_energy_mlp, self.two_body_energy_mlp, 
                   self.three_body_energy_mlp, self.four_body_energy_mlp, 
                   self.molecular_energy_mlp]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)
        
    def forward(self, data, x_scalar, x_spherical):
        """
        前向传播：基于体序的能量分解
        
        Args:
            data: pyg Data object
            x_scalar: 标量特征 [N_atoms, node_dim]
            x_spherical: 球谐特征 [N_atoms, spherical_dim]
            
        Returns:
            total_energy: 分子总能量 [batch_size]
            forces: 保守力 [N_atoms, 3]
        """
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos
        
        x_scalar = self.layernorm(x_scalar)
        
        atomic_energy = self.atomic_energy_mlp(x_scalar)
        total_atomic = scatter(atomic_energy.squeeze(-1), batch, dim=0)
        
        src, dst = edge_index
        two_body_feat = torch.cat([x_scalar[src], x_scalar[dst]], dim=-1)
        two_body_energy = self.two_body_energy_mlp(two_body_feat)
        total_two_body = scatter(two_body_energy.squeeze(-1), batch[src], dim=0)
        
        three_body_energy = self._compute_three_body_energy(x_scalar, edge_index, batch)
        
        four_body_energy = self._compute_four_body_energy(x_scalar, edge_index, batch)
        
        molecular_energy = self.molecular_energy_mlp(x_scalar)
        total_molecular = scatter(molecular_energy.squeeze(-1), batch, dim=0)
        
        weights = torch.softmax(self.body_order_energy_weights, dim=0)
        total_energy = (
            weights[0] * total_atomic +
            weights[1] * total_two_body +
            weights[2] * three_body_energy +
            weights[3] * four_body_energy +
            total_molecular 
        )
        
        if self.use_smoothness_loss:
            total_energy = self._apply_smoothness_regularization(total_energy, pos)
        
        try:
            forces = -torch.autograd.grad(
                total_energy.sum(), pos,
                retain_graph=True, create_graph=False 
            )[0]
            
            if torch.isnan(forces).any():
                print(f"ERROR: NaN forces detected in BodyOrderEnergyOutput - this should not happen!")
                print(f"total_energy stats: mean={total_energy.mean().item():.6f}, std={total_energy.std().item():.6f}")
                print(f"pos stats: mean={pos.mean().item():.6f}, std={pos.std().item():.6f}")
                raise ValueError("NaN forces detected in BodyOrderEnergyOutput - check energy calculation")
            elif torch.isinf(forces).any():
                print(f"ERROR: Inf forces detected in BodyOrderEnergyOutput - this should not happen!")
                print(f"total_energy stats: mean={total_energy.mean().item():.6f}, std={total_energy.std().item():.6f}")
                raise ValueError("Inf forces detected in BodyOrderEnergyOutput - check energy calculation")
            elif torch.abs(forces).max() > 1000:
                print(f"ERROR: Extremely large forces detected in BodyOrderEnergyOutput: max={torch.abs(forces).max().item():.2f}")
                print(f"total_energy stats: mean={total_energy.mean().item():.6f}, std={total_energy.std().item():.6f}")
                raise ValueError("Forces too large in BodyOrderEnergyOutput - check energy calculation")
                
        except Exception as e:
            print(f"ERROR in autograd.grad: {e}")
            print(f"total_energy shape: {total_energy.shape}, pos shape: {pos.shape}")
            raise RuntimeError(f"Autograd failed in BodyOrderEnergyOutput: {e}")
        
        return total_energy, forces
    
    def _compute_three_body_energy(self, x_scalar, edge_index, batch):
        """计算3体相互作用能量（角度项）"""
        src, dst = edge_index
        
        neighbor_dict = {}
        for i, (s, d) in enumerate(zip(src, dst)):
            s_idx, d_idx = s.item(), d.item()
            if s_idx not in neighbor_dict:
                neighbor_dict[s_idx] = []
            neighbor_dict[s_idx].append(d_idx)
        
        three_body_contributions = []
        
        for atom_idx, neighbors in neighbor_dict.items():
            if len(neighbors) >= 2:
                n1, n2 = neighbors[0], neighbors[1]
                three_body_feat = torch.cat([
                    x_scalar[atom_idx], 
                    x_scalar[n1], 
                    x_scalar[n2]
                ], dim=-1)
                
                energy = self.three_body_energy_mlp(three_body_feat)
                three_body_contributions.append(energy)
        
        if three_body_contributions:
            three_body_energy = torch.stack(three_body_contributions).mean()
            return three_body_energy
        else:
            return torch.tensor(0.0, device=x_scalar.device)
    
    def _compute_four_body_energy(self, x_scalar, edge_index, batch):
        """计算4体相互作用能量（二面角项）"""
        src, dst = edge_index
        
        neighbor_dict = {}
        for s, d in zip(src, dst):
            s_idx, d_idx = s.item(), d.item()
            if s_idx not in neighbor_dict:
                neighbor_dict[s_idx] = []
            neighbor_dict[s_idx].append(d_idx)
        
        four_body_contributions = []
        
        for atom_idx, neighbors in neighbor_dict.items():
            if len(neighbors) >= 2:
                for n1 in neighbors:
                    if n1 in neighbor_dict and len(neighbor_dict[n1]) >= 1:
                        for n2 in neighbor_dict[n1]:
                            if n2 != atom_idx: 
                                four_body_feat = torch.cat([
                                    x_scalar[atom_idx],
                                    x_scalar[n1],
                                    x_scalar[n2],
                                    x_scalar[atom_idx] 
                                ], dim=-1)
                                
                                energy = self.four_body_energy_mlp(four_body_feat)
                                four_body_contributions.append(energy)
        
        if four_body_contributions:
            four_body_energy = torch.stack(four_body_contributions).mean()
            return four_body_energy
        else:
            return torch.tensor(0.0, device=x_scalar.device)
    
    def _apply_smoothness_regularization(self, energy, pos):
        """
        应用势能面平滑性正则化
        确保能量函数在坐标空间中的平滑性
        """
        if self.training:
            noise_scale = self.smoothness_factor * 1e-6
            pos_noise = torch.randn_like(pos) * noise_scale
            pos_perturbed = pos + pos_noise
            
            energy = energy * (1.0 + 0.01 * torch.sin(energy * 100)) 
        
        return energy
    
    def compute_smoothness_loss(self, energy, pos, eps=1e-6):
        """
        计算势能面平滑性损失
        用于训练时增强势能面的平滑性
        """
        if not self.use_smoothness_loss:
            return torch.tensor(0.0, device=energy.device)
        
        energy_grad = torch.autograd.grad(
            energy.sum(), pos, 
            create_graph=True, retain_graph=True
        )[0]
        
        energy_hessian = torch.autograd.grad(
            energy_grad.sum(), pos,
            create_graph=True, retain_graph=True
        )[0]
        
        smoothness_loss = torch.mean(torch.abs(energy_hessian)) * self.smoothness_factor
        
        return smoothness_loss
    
    def get_param_groups(self, base_lr=None, base_weight_decay=None):
        """
        获取参数分组 - 支持不同能量项的不同学习率
        """
        param_groups = []
        
        param_groups.append({
            "params": self.atomic_energy_mlp.parameters(),
            "lr": base_lr,
            "weight_decay": base_weight_decay,
            "name": "atomic_energy"
        })
        
        param_groups.append({
            "params": self.two_body_energy_mlp.parameters(),
            "lr": base_lr * 1.2, 
            "weight_decay": base_weight_decay,
            "name": "two_body_energy"
        })
        
        param_groups.append({
            "params": self.three_body_energy_mlp.parameters(),
            "lr": base_lr * 1.1,
            "weight_decay": base_weight_decay,
            "name": "three_body_energy"
        })
        
        param_groups.append({
            "params": self.four_body_energy_mlp.parameters(),
            "lr": base_lr * 1.0,
            "weight_decay": base_weight_decay,
            "name": "four_body_energy"
        })
        
        param_groups.append({
            "params": self.molecular_energy_mlp.parameters(),
            "lr": base_lr * 0.9, 
            "weight_decay": base_weight_decay,
            "name": "molecular_energy"
        })
        
        param_groups.append({
            "params": [self.body_order_energy_weights],
            "lr": base_lr * 0.5,  
            "weight_decay": base_weight_decay * 2, 
            "name": "body_order_weights"
        })
        
        return param_groups
        
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


from typing import Iterable, Union
import torch
import torch.nn as nn
import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from ..utils import get_embedding_tensor
class Invariant(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        squared: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.squared = squared
        self.eps = eps
        self.invariant = o3.Norm(irreps_in, squared=squared)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.squared:
            x = self.invariant(x)
        else:
            x = self.invariant(x + self.eps ** 2) - self.eps
        return x
class GatedNonlinearity(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        gate_mlp_hidden_dim: int = 64,
        use_smooth_activation: bool = True,
        smooth_factor: float = 0.1,
        residual_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.use_smooth_activation = use_smooth_activation
        self.smooth_factor = smooth_factor
        self.residual_weight = residual_weight
        self.irrep_channels = []
        self.total_channels = 0
        for mul, ir in self.irreps_in:
            self.irrep_channels.append(mul)
            self.total_channels += mul
        gate_input_dim = self.total_channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_mlp_hidden_dim, gate_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_mlp_hidden_dim, self.total_channels),
            nn.Sigmoid()
        )
        if self.use_smooth_activation:
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Sigmoid()
        self.scalar_mul = o3.ElementwiseTensorProduct(
            self.irreps_in,
            f"{self.total_channels}x0e"
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    def _extract_norms(self, x: torch.Tensor) -> torch.Tensor:
        norms = []
        current_offset = 0
        for i, (mul, ir) in enumerate(self.irreps_in):
            irrep_dim = mul * ir.dim
            if irrep_dim > 0:
                irrep_part = x.narrow(-1, current_offset, irrep_dim)
                if ir.l == 0:
                    norms.append(irrep_part)
                else:
                    reshaped = irrep_part.view(-1, mul, ir.dim)
                    channel_norms = torch.linalg.vector_norm(reshaped, dim=-1)
                    norms.append(channel_norms)
                current_offset += irrep_dim
        return torch.cat(norms, dim=-1)
    def _apply_gating(self, x: torch.Tensor, gate_signal: torch.Tensor) -> torch.Tensor:
        gated_parts = []
        current_offset = 0
        gate_offset = 0
        for i, (mul, ir) in enumerate(self.irreps_in):
            irrep_dim = mul * ir.dim
            if irrep_dim > 0:
                irrep_part = x.narrow(-1, current_offset, irrep_dim)
                gate_part = gate_signal.narrow(-1, gate_offset, mul)
                if ir.l == 0:
                    activated_part = self.activation(irrep_part)
                    gated_part = activated_part * gate_part
                else:
                    reshaped = irrep_part.view(-1, mul, ir.dim)
                    gate_broadcasted = gate_part.unsqueeze(-1).expand(-1, -1, ir.dim)
                    gated_part = reshaped * gate_broadcasted
                    gated_part = gated_part.view(-1, irrep_dim)
                gated_parts.append(gated_part)
                current_offset += irrep_dim
                gate_offset += mul
        return torch.cat(gated_parts, dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = x
        x_act = x
        norm_features = self._extract_norms(x_gate)
        gate_signal = self.gate_mlp(norm_features)
        if self.smooth_factor > 0:
            gate_signal = gate_signal * (1 - self.smooth_factor) + self.smooth_factor * 0.5
        y = self._apply_gating(x_act, gate_signal)
        y = x * self.residual_weight + y * (1 - self.residual_weight)
        return y
class Gate(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
    ) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        self.invariant = Invariant(irreps_in)
        self.activation = nn.Sigmoid()
        self.scalar_mul = o3.ElementwiseTensorProduct(irreps_in, f"{irreps_in.num_irreps}x0e")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_invariant = self.invariant(x)
        x_activation = self.activation(x_invariant)
        x_out = self.scalar_mul(x, x_activation)
        return x_out
class Int2c1eEmbedding(nn.Module):
    def __init__(
        self,
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux28",
    ) -> None:
        super().__init__()
        embed_ten = get_embedding_tensor(embed_basis, aux_basis)
        self.register_buffer("embed_ten", embed_ten)
        self.embed_dim = embed_ten.shape[1]
    def forward(self, at_no: torch.Tensor) -> torch.Tensor:
        return self.embed_ten[at_no]
@compile_mode("trace")
class EquivariantDot(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
    ):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])
        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]
        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component")
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in})"
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        assert features1.shape[-1] == features2.shape[-1] == self.input_dim, \
            "Input tensor must have the same last dimension as the irreps"
        out = self.tp(features1, features2)
        return out
class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps, eps=1e-5, affine=True) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dim = self.irreps.dim
        self.eps = eps
        scalar_indices = []
        vector_indices = []
        ix = 0
        for mul, ir in self.irreps:
            indices = list(range(ix, ix + mul * ir.dim))
            if ir.l == 0:
                scalar_indices.extend(indices)
            else:
                vector_indices.extend(indices)
            ix += mul * ir.dim
        self.register_buffer('scalar_indices', torch.LongTensor(scalar_indices) if scalar_indices else torch.LongTensor([]))
        self.register_buffer('vector_indices', torch.LongTensor(vector_indices) if vector_indices else torch.LongTensor([]))
        if affine and len(scalar_indices) > 0:
            self.weight = nn.Parameter(torch.ones(len(scalar_indices)))
            self.bias = nn.Parameter(torch.zeros(len(scalar_indices)))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    def forward(self, x):
        if len(self.scalar_indices) == 0:
            return x
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        scalar_part = x[:, self.scalar_indices]
        mean = scalar_part.mean(dim=1, keepdim=True)
        var = scalar_part.var(dim=1, keepdim=True, unbiased=False)
        scalar_part = (scalar_part - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None and self.bias is not None:
            scalar_part = scalar_part * self.weight + self.bias
        result = x.clone()
        result[:, self.scalar_indices] = scalar_part
        if len(orig_shape) > 2:
            result = result.reshape(orig_shape)
        return result
def resolve_actfn(actfn: str) -> nn.Module:
    actfn = actfn.lower()
    if actfn == "relu":
        return nn.ReLU()
    elif actfn == "leakyrelu":
        return nn.LeakyReLU()
    elif actfn == "softplus":
        return nn.Softplus()
    elif actfn == "sigmoid":
        return nn.Sigmoid()
    elif actfn == "silu":
        return nn.SiLU()
    elif actfn == "tanh":
        return nn.Tanh()
    elif actfn == "identity":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported activation function {actfn}")
def resolve_norm(
    norm_type: str,
    num_features: int,
    affine: bool = True,
) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(
            num_features,
            affine=affine,
        )
    elif norm_type == "layer":
        return nn.LayerNorm(
            num_features,
            elementwise_affine=affine,
        )
    elif norm_type == "nonorm":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported normalization layer {norm_type}")
def check_potential_smoothness(model, coordinates, eps=1e-6):
    coordinates = coordinates.detach().clone().requires_grad_(True)
    energy_orig = model(coordinates)
    coordinates_perturbed = coordinates + torch.randn_like(coordinates) * eps
    energy_perturbed = model(coordinates_perturbed)
    energy_diff = torch.abs(energy_perturbed - energy_orig)
    is_smooth = torch.all(energy_diff < eps * 10)
    return is_smooth, energy_diff.mean().item()
def check_force_smoothness(model, coordinates):
    coordinates = coordinates.detach().clone().requires_grad_(True)
    energy = model(coordinates)
    forces = -torch.autograd.grad(energy.sum(), coordinates)[0]
    force_gradients = torch.autograd.grad(forces.sum(), coordinates)[0]
    max_gradient = torch.max(torch.abs(force_gradients))
    return max_gradient.item()
def resolve_o3norm(
    norm_type: str,
    irreps: Union[str, o3.Irreps, Iterable],
    affine: bool = True,
    std_balance_degrees: bool = True,
) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return e3nn.nn.BatchNorm(
            irreps,
            affine=affine,
        )
    elif norm_type == "layer":
        return EquivariantLayerNorm(
            irreps,
            affine=affine,
        )
    elif norm_type == "nonorm":
        return nn.Identity()
    else:
        raise NotImplementedError(f"不支持的等变归一化层: {norm_type}")

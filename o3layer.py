from typing import Iterable, Union

import torch
import torch.nn as nn

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from ..utils import get_embedding_tensor

class Invariant(nn.Module):
    """
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        squared: bool = False,
        eps: float = 1e-6,
    ) -> None:
        """
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
        """
        super().__init__()
        embed_ten = get_embedding_tensor(embed_basis, aux_basis)
        self.register_buffer("embed_ten", embed_ten)
        self.embed_dim = embed_ten.shape[1]
    
    def forward(self, at_no: torch.Tensor) -> torch.Tensor:
        """
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
    """等变层归一化，只对标量特征进行归一化，保持高阶特征不变"""
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
        """前向传播，仅对标量部分归一化
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
    """Helper function to return activation function"""
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
    """Helper function to return normalization layer"""
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

def resolve_o3norm(
    norm_type: str,
    irreps: Union[str, o3.Irreps, Iterable],
    affine: bool = True,
) -> nn.Module:
    """Helper function to return equivariant normalization layer"""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        try:
            return e3nn.nn.BatchNorm(
                irreps,
                affine=affine,
            )
        except Exception as e:
            print(f"警告: e3nn BatchNorm初始化失败: {str(e)}，使用EquivariantLayerNorm代替")
            return EquivariantLayerNorm(
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

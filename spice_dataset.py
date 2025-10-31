
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_cluster import radius_graph
from typing import Optional, List, Dict, Any
import os
def unit_conversion(from_unit: str, to_unit: str) -> float:
    if from_unit.lower() == "bohr" and to_unit.lower() == "angstrom":
        return 0.529177249
    elif from_unit.lower() == "angstrom" and to_unit.lower() == "bohr":
        return 1.889726125
    else:
        return 1.0
class SPICEDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        split: str = 'train',
        cutoff: float = 6.0,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        grad_target_mean: Optional[float] = None,
        grad_target_std: Optional[float] = None,
        coord_unit: str = "bohr",
        cutoff_unit: str = "bohr",
        transform=None,
        pre_transform=None,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.split = split
        self.cutoff = cutoff
        self.coord_unit = coord_unit
        self.cutoff_unit = cutoff_unit
        self.transform = transform
        self.pre_transform = pre_transform
        self.target_mean = target_mean
        self.target_std = target_std
        self.grad_target_mean = grad_target_mean
        self.grad_target_std = grad_target_std
        self.data_list = []
        self._load_data()
        print(f"✅ SPICE {split}集加载完成:")
        print(f"  总样本数: {len(self.data_list)}")
        print(f"  坐标单位: {self.coord_unit}")
        print(f"  截断距离: {self.cutoff} {self.cutoff_unit}")
        print(f"  归一化参数:")
        print(f"    能量均值: {self.target_mean}")
        print(f"    能量标准差: {self.target_std}")
        print(f"    力均值: {self.grad_target_mean}")
        print(f"    力标准差: {self.grad_target_std}")
        if len(self.data_list) > 0:
            total_atoms = sum(data.num_atoms for data in self.data_list)
            avg_atoms = total_atoms / len(self.data_list)
            print(f"  统计信息:")
            print(f"    总原子数: {total_atoms}")
            print(f"    平均原子数: {avg_atoms:.1f}")
            print(f"    平均边数: {sum(data.edge_index.shape[1] for data in self.data_list) / len(self.data_list):.1f}")
    def _load_data(self):
        print(f"开始加载SPICE {self.split}集数据...")
        with h5py.File(self.h5_file_path, 'r') as f:
            split_data = f['splits'][self.split]
            molecule_ids = list(split_data.keys())
            print(f"找到 {len(molecule_ids)} 个分子")
            from tqdm import tqdm
            for mol_id in tqdm(molecule_ids, desc=f"处理{self.split}集分子", unit="分子"):
                molecule = split_data[mol_id]
                data_list = self._extract_molecule_data(molecule, mol_id)
                self.data_list.extend(data_list)
    def _extract_molecule_data(self, molecule: h5py.Group, mol_id: str) -> List[Data]:
        data_list = []
        atomic_numbers = molecule['atomic_numbers'][:]
        num_atoms = len(atomic_numbers)
        if 'dft_total_energy' in molecule:
            energies = molecule['dft_total_energy'][:]
            num_confs = len(energies)
        else:
            print(f"警告: 分子 {mol_id} 没有能量数据")
            return data_list
        if 'dft_total_gradient' in molecule:
            forces = molecule['dft_total_gradient'][:]
        else:
            print(f"警告: 分子 {mol_id} 没有力数据")
            return data_list
        if 'conformations' in molecule:
            conformations = molecule['conformations'][:]
        else:
            print(f"警告: 分子 {mol_id} 没有构象数据")
            return data_list
        if num_confs > 100:
            from tqdm import tqdm
            conf_range = tqdm(range(num_confs), desc=f"处理分子{mol_id}构象", leave=False)
        else:
            conf_range = range(num_confs)
        for conf_idx in conf_range:
            try:
                coords = conformations[conf_idx]
                energy_raw = energies[conf_idx]
                forces_raw = forces[conf_idx]
                if coords.shape != (num_atoms, 3):
                    print(f"警告: 分子 {mol_id} 构象 {conf_idx} 坐标形状错误: {coords.shape}, 期望: ({num_atoms}, 3)")
                    continue
                if forces_raw.shape != (num_atoms, 3):
                    print(f"警告: 分子 {mol_id} 构象 {conf_idx} 力形状错误: {forces_raw.shape}, 期望: ({num_atoms}, 3)")
                    continue
                if not np.isfinite(energy_raw):
                    print(f"警告: 分子 {mol_id} 构象 {conf_idx} 能量值无效: {energy_raw}")
                    continue
                if not np.all(np.isfinite(forces_raw)):
                    print(f"警告: 分子 {mol_id} 构象 {conf_idx} 力值包含无效值")
                    continue
                energy_normalized = self._normalize_energy(energy_raw)
                forces_normalized = self._normalize_forces(forces_raw)
                if self.coord_unit != self.cutoff_unit:
                    conversion_factor = unit_conversion(self.coord_unit, self.cutoff_unit)
                    coords_converted = coords * conversion_factor
                    if conf_idx == 0:
                        print(f"单位转换: {self.coord_unit} → {self.cutoff_unit}, 系数: {conversion_factor:.6f}")
                else:
                    coords_converted = coords
                edge_index = radius_graph(
                    torch.tensor(coords_converted, dtype=torch.float32),
                    r=self.cutoff
                )
                if edge_index.shape[1] > 0:
                    if torch.any(edge_index >= num_atoms):
                        print(f"错误: 分子 {mol_id} 构象 {conf_idx} 边索引超出原子范围")
                        continue
                data = Data(
                    at_no=torch.tensor(atomic_numbers, dtype=torch.long),
                    pos=torch.tensor(coords_converted, dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.tensor([energy_normalized], dtype=torch.float32),
                    force=torch.tensor(forces_normalized, dtype=torch.float32),
                    mol_id=mol_id,
                    conf_idx=conf_idx,
                    num_atoms=num_atoms,
                )
                data_list.append(data)
            except Exception as e:
                print(f"警告: 处理分子 {mol_id} 构象 {conf_idx} 时出错: {e}")
                print(f"错误类型: {type(e).__name__}")
                import traceback
                print(f"错误详情: {traceback.format_exc()}")
                continue
        return data_list
    def _normalize_energy(self, energy_raw: float) -> float:
        if self.target_mean is None or self.target_std is None:
            return energy_raw
        energy_normalized = (energy_raw - self.target_mean) / (self.target_std + 1e-8)
        return energy_normalized
    def _normalize_forces(self, forces_raw: np.ndarray) -> np.ndarray:
        if self.grad_target_mean is None or self.grad_target_std is None:
            return forces_raw
        forces_normalized = (forces_raw - self.grad_target_mean) / (self.grad_target_std + 1e-8)
        return forces_normalized
    def __len__(self) -> int:
        return len(self.data_list)
    def __getitem__(self, idx: int) -> Data:
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
def create_spice_datasets(
    h5_file_path: str,
    target_mean: float,
    target_std: float,
    grad_target_mean: float,
    grad_target_std: float,
    cutoff: float = 6.0,
    max_edges: int = 100,
    transform=None,
    pre_transform=None,
) -> Dict[str, SPICEDataset]:
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = SPICEDataset(
            h5_file_path=h5_file_path,
            split=split,
            cutoff=cutoff,
            target_mean=target_mean,
            target_std=target_std,
            grad_target_mean=grad_target_mean,
            grad_target_std=grad_target_std,
            transform=transform,
            pre_transform=pre_transform,
        )
    return datasets
def test_spice_dataset():
    print("=== 测试SPICE数据集 ===")
    target_mean = -1579.8712028723482
    target_std = 1291.860111079821
    grad_target_mean = -2.8962126448550086e-07
    grad_target_std = 0.5982427300960868
    h5_file_path = "/root/autodl-tmp/raw/SPICE_PUBCHEM_SUBSETS_SPLITS_CLEANED.hdf5"
    train_dataset = SPICEDataset(
        h5_file_path=h5_file_path,
        split='train',
        target_mean=target_mean,
        target_std=target_std,
        grad_target_mean=grad_target_mean,
        grad_target_std=grad_target_std,
        cutoff=6.0,
    )
    print(f"训练集大小: {len(train_dataset)}")
    for i in range(min(3, len(train_dataset))):
        data = train_dataset[i]
        print(f"\n样本 {i}:")
        print(f"  分子ID: {data.mol_id}")
        print(f"  构象索引: {data.conf_idx}")
        print(f"  原子数: {data.num_atoms}")
        print(f"  归一化能量: {data.y.item():.6f}")
        print(f"  归一化力形状: {data.force.shape}")
        print(f"  边数: {data.edge_index.shape[1]}")
        energy_original = data.y.item() * target_std + target_mean
        print(f"  原始能量: {energy_original:.6f} hartree")
        force_mean = data.force.mean().item()
        force_std = data.force.std().item()
        print(f"  归一化力均值: {force_mean:.6f}")
        print(f"  归一化力标准差: {force_std:.6f}")
    print("\n测试完成！")
if __name__ == "__main__":
    test_spice_dataset()

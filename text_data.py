from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..utils import get_default_unit, unit_conversion
from ase.io import read as ase_read
class TextDataset(Dataset):
    def __init__(
        self,
        file: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self._file = file
        self._transform = transform
        self.data_list = []
        _, self.len_unit = get_default_unit()
        self.process()
    def process(self) -> None:
        atoms_list = ase_read(self._file, index=":")
        for atoms in atoms_list:
            at_no = torch.from_numpy(atoms.get_atomic_numbers()).to(torch.long)
            coord = torch.from_numpy(atoms.get_positions()).to(torch.get_default_dtype())
            coord *= unit_conversion("Angstrom", self.len_unit)
            info = atoms.info
            charge = torch.Tensor([info.get("charge", 0.0)]).to(torch.get_default_dtype())
            if "multiplicity" in info:
                spin = torch.Tensor([info["multiplicity"] - 1]).to(torch.get_default_dtype())
            else:
                spin = torch.Tensor([info.get("spin", 0.0)]).to(torch.get_default_dtype())
            data = Data(at_no=at_no, pos=coord, charge=charge, spin=spin)
            pbc = atoms.get_pbc()
            if pbc.any():
                data.pbc = torch.from_numpy(pbc).to(torch.bool)
                data.lattice = torch.from_numpy(atoms.get_cell()).to(torch.get_default_dtype())
            self.data_list.append(data)
    def __len__(self) -> int:
        return len(self.data_list)
    def __getitem__(self, idx) -> Data:
        data = self.data_list[idx]
        if self._transform is not None:
            data = self._transform(data)
        return data

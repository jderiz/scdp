import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from scdp.common.pyg import DataLoader
from scdp.data.data import AtomicData
from scdp.model.module import ChgLightningModule
from scdp.scripts.preprocess import get_atomic_number_table_from_zs

z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())

metadata = "caffeine"
mol_file = "caffeine_1.sdf"

# metadata='test'
# mol_file='/home/iwe20/Projects/scdp/experiments/1.001_100.sdf'

device = "cuda"
dimensional_padding = 2.0
resolution = 0.25

bohr = 1.88973

mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
mol_atom_types = []
mol_pos = []
origin = [9999.9, 9999.9, 9999.9]
max_edge = [-9999.9, -9999.9, -9999.9]
for i in range(mol.GetNumAtoms()):
    mol_atom_types.append(mol.GetAtomWithIdx(i).GetAtomicNum())
    pos = mol.GetConformer().GetAtomPosition(i)
    pos = [pos.x, pos.y, pos.z]
    mol_pos.append(pos)
    for i in range(3):
        origin[i] = min(origin[i], pos[i])
    for i in range(3):
        max_edge[i] = max(max_edge[i], pos[i])

for i in range(3):
    origin[i] -= dimensional_padding
    max_edge[i] += dimensional_padding


# atomic numbers in sequence
atom_types = torch.tensor(mol_atom_types)
# positions in angstrom
atom_coords = torch.tensor(mol_pos)

# this is a fantastic 90 degree rotation counterclockwise around the x-axis
rotation = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float)

atom_coords_rot = atom_coords @ rotation

# base vector of coordinate system
cube_length = [max_edge[i] - origin[i] for i in range(3)]
cell = torch.tensor(
    [
        [cube_length[0], 0.0000, 0.0000],
        [0.0000, cube_length[1], 0.0000],
        [0.0000, 0.0000, cube_length[2]],
    ]
)
# the whole block of densities, but I'll try to skip it
chg_dimension = [int(cube_length[i] / resolution) + 1 for i in range(3)]

print("Parsed molecule with RDKit.")
print("\tDensity cube size:", ", ".join([f"{size:.4f}" for size in cube_length]))
print("\tProbe dimension:", chg_dimension)
print("\tResolution:", resolution)
print("\tPadding:", dimensional_padding)

chg_density = torch.zeros(chg_dimension)
# is omitted during parsing as well
origin = torch.tensor(origin)

# No virtual nodes added!
data_object = AtomicData.build_graph_with_vnodes(
    atom_coords=atom_coords,
    atom_types=atom_types,
    cell=cell,
    chg_density=chg_density,
    origin=origin,
    metadata=metadata,
    z_table=z_table,
    atom_cutoff=6.0,
    vnode_method="none",
    vnode_factor=3,
    vnode_res=0.8,
    disable_pbc=True,
    max_neighbors=None,
    device=device,
)
data_object_rot = AtomicData.build_graph_with_vnodes(
    atom_coords=atom_coords_rot,
    atom_types=atom_types,
    cell=cell,
    chg_density=chg_density,
    origin=origin,
    metadata=metadata + "_rot",
    z_table=z_table,
    atom_cutoff=6.0,
    vnode_method="none",
    vnode_factor=3,
    vnode_res=0.8,
    disable_pbc=True,
    max_neighbors=None,
    device=device,
)
print("Created AtomicData object")


class mol_data(Dataset):

    def __init__(self, atom_data):
        self.atom_data = atom_data

    def __len__(self):
        return len(self.atom_data)

    def __getitem__(self, index):
        return self.atom_data[index]


md = mol_data([data_object, data_object_rot])

model = ChgLightningModule.load_from_checkpoint(
    checkpoint_path="qm9_none_K4L3_beta_2.0/epoch=59-step=464400.ckpt"
).to(device)
model.eval()
model.ema.copy_to(model.parameters())

pytorch_total_params = sum(p.numel() for p in model.parameters())

print("Loaded model to", device)
print("\tParameters:", pytorch_total_params)

loader = DataLoader(md, batch_size=1)

coeffs_arr = []

print(model.gto_dict)
print(model.gto_dict["1"])

with torch.no_grad():
    for batch in loader:
        print("Start processing", batch.metadata[0])
        batch = batch.to(device)
        coeffs, expo_scaling = model.predict_coeffs(batch)
        coeffs_arr.append(coeffs)
        print(
            "Calculated",
            coeffs.shape[1],
            "coefficients each for",
            coeffs.shape[0],
            "atoms",
        )
        print(
            "\tExponent scaling:",
            expo_scaling.shape if expo_scaling != None else "None",
        )
        np.savetxt(batch.metadata[0] + ".coeffs", coeffs.cpu().numpy())

        pred = model.orbital_inference(
            batch, coeffs, expo_scaling, batch.n_probe, batch.probe_coords
        )
        print("Calculated densities for", batch.n_probe.item(), "probes")

        print("Start writing cube file...")
        with open(batch.metadata[0] + ".cube", "w") as file:
            string = batch.metadata[0] + "\n" + "Created by Paul and scdp\n"
            string += f"{len(atom_types):4}{origin[0]*bohr:12.6f}{origin[1]*bohr:12.6f}{origin[2]*bohr:12.6f}\n"
            for axis in range(3):
                string += f"{chg_dimension[axis]:4}"
                for xyz in range(3):
                    string += f"{cell[axis][xyz].item()*bohr/chg_dimension[axis]:12.6f}"
                string += "\n"
            for atom in range(len(atom_types)):
                string += f"{atom_types[atom].item():4}{0.0:12.6f}"
                for coord in range(3):
                    string += f"{atom_coords[atom][coord].item()*bohr:12.6f}"
                string += "\n"
            for i in range(batch.n_probe):
                string += f"{pred[i].item():14.6e}"
                if (i + 1) % 6 == 0:
                    string += "\n"
            file.write(string)
        print("Done.")

print(torch.isclose(coeffs_arr[0], coeffs_arr[1], atol=1e-2, rtol=1e-2)[0])

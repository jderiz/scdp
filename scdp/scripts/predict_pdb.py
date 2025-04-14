
from rdkit import Chem

from scdp.data.data import AtomicData
from scdp.scripts.preprocess import get_atomic_number_table_from_zs
import torch
import numpy as np
from scdp.model.module import ChgLightningModule
from torch.utils.data import Dataset
from scdp.common.pyg import DataLoader

PERIODIC_TABLE = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
    "CS": 55,
    "BA": 56,
    "LA": 57,
    "CE": 58,
    "PR": 59,
    "ND": 60,
    "PM": 61,
    "SM": 62,
    "EU": 63,
    "GD": 64,
    "TB": 65,
    "DY": 66,
    "HO": 67,
    "ER": 68,
    "TM": 69,
    "YB": 70,
    "LU": 71,
    "HF": 72,
    "TA": 73,
    "W": 74,
    "RE": 75,
    "OS": 76,
    "IR": 77,
    "PT": 78,
    "AU": 79,
    "HG": 80,
    "TL": 81,
    "PB": 82,
    "BI": 83,
    "PO": 84,
    "AT": 85,
    "RN": 86,
    "FR": 87,
    "RA": 88,
    "AC": 89,
    "TH": 90,
    "PA": 91,
    "U": 92,
    "NP": 93,
    "PU": 94,
    "AM": 95,
    "CM": 96,
    "BK": 97,
    "CF": 98,
    "ES": 99,
    "FM": 100,
    "MD": 101,
    "NO": 102,
    "LR": 103,
    "RF": 104,
    "DB": 105,
    "SG": 106,
    "BH": 107,
    "HS": 108,
    "MT": 109,
    "DS": 110,
    "RG": 111,
    "CN": 112,
    "NH": 113,
    "FL": 114,
    "MC": 115,
    "LV": 116,
    "TS": 117,
    "OG": 118
}

table_is_correct = True
# 1) Verify the total number of elements
if len(set(PERIODIC_TABLE.keys())) != 118:
    table_is_correct = False
    print(f"ERROR: Expected 118 elements, but found {len(PERIODIC_TABLE)}.")

# 2) Verify atomic numbers cover the full range 1..118
all_atomic_numbers = set(range(1, 119))
table_atomic_numbers = set(PERIODIC_TABLE.values())
if table_atomic_numbers != all_atomic_numbers:
    missing = all_atomic_numbers - table_atomic_numbers
    extra = table_atomic_numbers - all_atomic_numbers
    if missing:
        table_is_correct = False
        print(f"ERROR: Missing atomic numbers: {missing}")
    if extra:
        table_is_correct = False
        print(f"ERROR: Extra atomic numbers: {extra}")


if table_is_correct: print("SUCCESS: The periodic table is correct!")

z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())

#metadata='caffeine'
#mol_file='caffeine.sdf'


metadata='test'
pdb_file='/home/iwe20/Projects/scdp/raw_data/1a0q/1a0q_relax.pdb'

device = 'cuda'
dimensional_padding = 2.0
resolution = 0.1

bohr = 1.88973

mol_atom_types = []
mol_pos = []

origin = [9999.9,9999.9,9999.9]
max_edge = [-9999.9,-9999.9,-9999.9]

with open( pdb_file ) as file:
    for line in file:
        if line[:4] == 'ATOM' or line[:6] == 'HETATM':
            element = line[76:78].strip().upper()
            mol_atom_types.append( PERIODIC_TABLE[element] )
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            pos = [ x, y, z ]
            mol_pos.append(pos)

            for i in range(3):
                origin[i] = min(origin[i], pos[i])
            for i in range(3):
                max_edge[i] = max(max_edge[i], pos[i])

for i in range(3):
    origin[i] -= dimensional_padding
    max_edge[i] += dimensional_padding


# atomic numbers in sequence
atom_types = torch.tensor( mol_atom_types )
# positions in angstrom
atom_coords = torch.tensor(mol_pos)
# base vector of coordinate system
cube_length = [ max_edge[i]-origin[i] for i in range(3) ]
cell = torch.tensor([
    [cube_length[0], 0.0000, 0.0000],
    [0.0000, cube_length[1], 0.0000],
    [0.0000, 0.0000, cube_length[2]]
])
# the whole block of densities, but I'll try to skip it
chg_dimension = [ int(cube_length[i] / resolution) + 1 for i in range(3) ]

print("Parsed molecule with RDKit.")
print('\tDensity cube size:', ", ".join([f'{size:.4f}' for size in cube_length]))
print('\tProbe dimension:', chg_dimension)
print('\tResolution:', resolution)
print('\tPadding:', dimensional_padding)

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
                vnode_method='none',
                vnode_factor=3,
                vnode_res=0.8,
                disable_pbc=True,
                max_neighbors=None,
                device=device,
            )
print("Created AtomicData object")

class mol_data( Dataset ):

    def __init__(self, atom_data):
        self.atom_data = atom_data

    def __len__( self ):
        return len(self.atom_data)
    
    def __getitem__(self, index):
        return self.atom_data[index]

md = mol_data( [data_object] )

model = ChgLightningModule.load_from_checkpoint(checkpoint_path= 'qm9_none_K4L3_beta_2.0/epoch=59-step=464400.ckpt').to(device)
model.eval()
model.ema.copy_to(model.parameters())

pytorch_total_params = sum(p.numel() for p in model.parameters())

print("Loaded model to", device)
print("\tParameters:", pytorch_total_params)

loader = DataLoader( md )

with( torch.no_grad() ):
    for batch in loader:
        batch = batch.to(device)
        coeffs, expo_scaling = model.predict_coeffs(batch)
        print('Calculated', coeffs.shape[1], 'coefficients each for', coeffs.shape[0], 'atoms')
        print('\tExponent scaling:', expo_scaling.shape if expo_scaling != None else 'None')

        pred = model.orbital_inference( batch, coeffs, expo_scaling, batch.n_probe, batch.probe_coords )
        print( 'Calculated densities for', batch.n_probe.item(), 'probes' )

        print('Start writing cube file...')
        with open( metadata + '.cube', 'w') as file:
            string = metadata + '\n' + 'Created by Paul and scdp\n'
            string += f'{len(atom_types):4}{origin[0]*bohr:12.6f}{origin[1]*bohr:12.6f}{origin[2]*bohr:12.6f}\n'
            for axis in range(3):
                string += f'{chg_dimension[axis]:4}'
                for xyz in range(3):
                    string += f'{cell[axis][xyz].item()*bohr/chg_dimension[axis]:12.6f}'
                string += '\n'
            for atom in range( len(atom_types) ):
                string += f'{atom_types[atom].item():4}{0.0:12.6f}'
                for coord in range(3):
                    string += f'{atom_coords[atom][coord].item()*bohr:12.6f}'
                string += '\n'
            for i in range( batch.n_probe ):
                string += f'{pred[i].item():14.6e}'
                if (i + 1) % 6 == 0:
                    string += '\n'
            file.write(string)
        print('Done.')
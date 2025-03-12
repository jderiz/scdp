
from scdp.data.data import AtomicData
from scdp.scripts.preprocess import get_atomic_number_table_from_zs
import torch
import numpy as np
from scdp.model.module import ChgLightningModule
from torch.utils.data import Dataset
from scdp.common.pyg import DataLoader


z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())

device = 'cuda'

# atomic numbers in sequence
atom_types = torch.tensor( [6, 1, 1, 1, 1] )
# positions in angstrom
atom_coords = torch.tensor([
    [2.5281, 3.0918, 2.8846],
    [2.5430, 2.0000, 2.8786],
    [3.5525, 3.4698, 2.8769],
    [2.0000, 3.4536, 2.0000],
    [2.0170, 3.4440, 3.7830]
])
# base vector of coordinate system
cell = torch.tensor([
    [5.5525, 0.0000, 0.0000],
    [0.0000, 5.4698, 0.0000],
    [0.0000, 0.0000, 5.7830]
])
# the whole block of densities, but I'll try to skip it
chg_density = torch.zeros([56,56,56])
# is omitted during parsing as well
origin = None
metadata='caffeine'

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
print(data_object)

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

loader = DataLoader( md )

with( torch.no_grad() ):
    for batch in loader:
        batch = batch.to(device)
        coeffs, expo_scaling = model.predict_coeffs(batch)
        print(coeffs)
        print(expo_scaling)

        print(batch.n_probe)

        pred = model.orbital_inference( batch, coeffs, expo_scaling, batch.n_probe, batch.probe_coords )
        #with open('densities.txt', 'w') as file:
        #    string = ""
        #    for i in range( batch.n_probe ):
        #        string += " " + str( pred[i].item() )
        #        if (i + 1) % 5 == 0:
        #            string += '\n'
        #    file.write(string)
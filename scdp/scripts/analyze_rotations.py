import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from scdp.common.pyg import DataLoader
from scdp.data.data import AtomicData
from scdp.model.module import ChgLightningModule
from scdp.scripts.coeff_transform_utils import (
    compare_transformations,
    extract_l_mapping,
    predict_transformed_coeffs_rotation,
)
from scdp.scripts.plotting_helpers import (
    analyze_by_l_value,
    analyze_coefficient_stability,
    compare_coefficient_pairs,
    plot_coefficient_stability,
    plot_isclose_analysis,
    plot_l_value_analysis,
    plot_transformation_comparison,
)
from scdp.scripts.preprocess import get_atomic_number_table_from_zs

plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18


def random_rotation_matrix():
    """Generate a random rotation matrix using scipy's Rotation."""
    return torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)


def main():
    """Main function to analyze orbital coefficient stability under rotation."""
    # Configuration
    z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())
    metadata = "caffeine"
    mol_file = "caffeine.sdf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensional_padding = 2.0
    resolution = 0.25
    num_rotations = 5  # Number of random rotations to generate
    torch.manual_seed(42)  # For reproducibility

    # Constants
    bohr = 1.88973

    # Load the molecule
    mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
    if mol is None:
        print(f"Error: Could not load molecule from {mol_file}")
        return

    # Extract atom types and positions from the molecule
    mol_atom_types = []  # List to store atomic numbers
    mol_pos = []  # List to store 3D coordinates
    origin = [9999.9, 9999.9, 9999.9]  # Initialize origin with large values
    max_edge = [-9999.9, -9999.9, -9999.9]  # Initialize max edge with small values

    # Iterate through all atoms in the molecule
    for i in range(mol.GetNumAtoms()):
        # Get atomic number and 3D position for current atom
        mol_atom_types.append(mol.GetAtomWithIdx(i).GetAtomicNum())
        pos = mol.GetConformer().GetAtomPosition(i)
        pos = [pos.x, pos.y, pos.z]
        mol_pos.append(pos)

        # Update origin (minimum coordinates) and max_edge (maximum coordinates)
        for j in range(3):
            origin[j] = min(origin[j], pos[j])
            max_edge[j] = max(max_edge[j], pos[j])

    # Add padding to ensure we capture all electron density
    for i in range(3):
        origin[i] -= dimensional_padding
        max_edge[i] += dimensional_padding

    # Convert to tensors
    atom_types = torch.tensor(mol_atom_types)
    atom_coords = torch.tensor(mol_pos)

    # Generate rotation matrices:
    # First is a standard 90-degree rotation around the x-axis
    # Second is identity (no rotation)
    # Followed by random rotations
    rotations = [
        # 90-degree rotation counterclockwise around the x-axis
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float
        ),
        # Identity matrix (no rotation)
        torch.eye(3, dtype=torch.float),
    ]

    # Add random rotations
    for _ in range(num_rotations):
        rotations.append(random_rotation_matrix())

    # Calculate cube dimensions for electron density representation
    cube_length = [max_edge[i] - origin[i] for i in range(3)]
    cell = torch.tensor(
        [
            [cube_length[0], 0.0000, 0.0000],
            [0.0000, cube_length[1], 0.0000],
            [0.0000, 0.0000, cube_length[2]],
        ]
    )
    chg_dimension = [int(cube_length[i] / resolution) + 1 for i in range(3)]
    chg_density = torch.zeros(chg_dimension)
    origin_tensor = torch.tensor(origin)

    print("Parsed molecule with RDKit.")
    print(
        f"\tDensity cube size: {cube_length[0]:.4f}, {cube_length[1]:.4f}, {cube_length[2]:.4f}"
    )
    print(f"\tProbe dimension: {chg_dimension}")
    print(f"\tResolution: {resolution}")
    print(f"\tPadding: {dimensional_padding}")
    print(f"\tNumber of rotations: {len(rotations)}")
    print(f"\tRotation 0: 90-degree rotation around x-axis")
    print(f"\tRotation 1: Identity matrix (no rotation)")
    print(f"\tRotations 2-{len(rotations)-1}: Random rotations")

    # Create AtomicData objects for each rotation
    data_objects = []

    # Get number of atoms and set max_neighbors
    # This prevents errors from having too large max_neighbors value
    num_atoms = len(atom_types)
    max_neighbors = num_atoms - 1  # Each atom can connect to all others at most

    # Create data objects for each rotation
    for i, rotation in enumerate(rotations):
        # Apply rotation to atomic coordinates
        rotated_coords = atom_coords @ rotation
        data_obj = AtomicData.build_graph_with_vnodes(
            atom_coords=rotated_coords,
            atom_types=atom_types,
            cell=cell,
            chg_density=chg_density,
            origin=origin_tensor,
            metadata=f"{metadata}_rot{i}",
            z_table=z_table,
            atom_cutoff=6.0,  # Maximum distance for atom-atom interactions
            vnode_method="none",  # No virtual nodes used
            vnode_factor=3,
            vnode_res=0.8,
            disable_pbc=True,  # No periodic boundary conditions
            max_neighbors=max_neighbors,  # Maximum number of neighbors per atom
            device=device,
        )
        data_objects.append(data_obj)

    print(f"Created {len(data_objects)} AtomicData objects")

    # Create dataset and loader
    class MolData(Dataset):
        def __init__(self, atom_data):
            self.atom_data = atom_data

        def __len__(self):
            return len(self.atom_data)

        def __getitem__(self, index):
            return self.atom_data[index]

    dataset = MolData(data_objects)

    # Load trained model
    checkpoint_path = "qm9_none_K4L3_beta_2.0/epoch=59-step=464400.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found")
        return

    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=checkpoint_path).to(
        device
    )
    model.eval()
    model.ema.copy_to(model.parameters())

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model to {device}")
    print(f"\tParameters: {pytorch_total_params}")

    # Set up dataloader
    loader = DataLoader(dataset, batch_size=1)

    # Collect coefficients for all rotations
    all_coeffs = []
    all_expo_scaling = []

    # Process each rotation and get the predicted coefficients
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Processing rotation {i}")
            batch = batch.to(device)
            # Get the model's predicted coefficients for this rotation
            coeffs, expo_scaling = model.predict_coeffs(batch)
            all_coeffs.append(coeffs.cpu())
            if expo_scaling is not None:
                all_expo_scaling.append(expo_scaling.cpu())

    # Convert to tensor for easier analysis
    all_coeffs_tensor = torch.stack(all_coeffs)
    print(f"Collected coefficients shape: {all_coeffs_tensor.shape}")

    # Extract L-value mapping from the model to analyze by angular momentum
    coeff_to_l_mapping = extract_l_mapping(model, atom_types)

    # Compare the first two rotations (90-degree vs identity)
    if len(all_coeffs) >= 2:
        rot_90_coeffs = all_coeffs[0]  # 90-degree rotation
        identity_coeffs = all_coeffs[1]  # Identity (no rotation)

        # Use helper functions to compare coefficients
        close_mask, rel_diff = compare_coefficient_pairs(
            rot_90_coeffs,
            identity_coeffs,
            title_prefix="90-degree rotation vs Identity",
        )

        # Analyze by L value
        l_value_stats = analyze_by_l_value(close_mask, atom_types, coeff_to_l_mapping)

        # Visualize results
        plot_isclose_analysis(
            close_mask,
            atom_types,
            coeff_to_l_mapping,
            l_value_stats,
            title_prefix="90-degree vs Identity",
            filename="rotation_isclose_analysis.png",
        )

    # Compare actual rotated coefficients with theoretical predictions
    compare_with_theoretical_predictions(
        model, all_coeffs_tensor, rotations, atom_types, coeff_to_l_mapping, device
    )

    # Analyze overall coefficient stability across all rotations
    relative_std, invariant_mask, l_value_stats = analyze_coefficient_stability(
        all_coeffs_tensor, atom_types, coeff_to_l_mapping
    )

    # Visualize stability results
    plot_coefficient_stability(
        relative_std,
        title="Coefficient Stability Under Rotation",
        filename="rotation_coefficient_stability.png",
    )


def compare_with_theoretical_predictions(
    model, all_coeffs_tensor, rotations, atom_types, coeff_to_l_mapping, device
):
    """
    Compare actual rotated coefficients with theoretical predictions using the
    predict_transformed_coeffs_rotation function.

    Args:
        model: The trained ChgLightningModule instance
        all_coeffs_tensor: Tensor of shape [num_rotations, num_atoms, num_coefficients]
                          containing all predicted coefficients
        rotations: List of rotation matrices used to generate the data
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        device: Device to perform computations on
    """
    # Use the coefficients from the identity rotation (index 1) as the reference
    reference_coeffs = all_coeffs_tensor[1]  # Identity rotation

    # Initialize arrays to store comparison metrics
    num_rotations = len(rotations)
    agreement_percentages = []
    mean_rel_diffs = []
    max_rel_diffs = []
    rotation_labels = []

    # Only compare rotations, skip the identity rotation (index 1)
    rotation_indices = [i for i in range(num_rotations) if i != 1]

    for i in rotation_indices:
        rotation_matrix = rotations[i].cpu().numpy()
        actual_coeffs = all_coeffs_tensor[i]

        # Create label for this rotation
        rotation_name = "90-degree rotation" if i == 0 else f"Random rotation {i-1}"
        rotation_labels.append(rotation_name)

        # Predict the theoretical coefficients using the function from coeff_transform_utils
        theo_coeffs = predict_transformed_coeffs_rotation(
            model=model,
            coeffs_original=reference_coeffs,
            rotation_matrix=rotation_matrix,
            device=device,
        )

        # Compare theoretical predictions with actual rotated coefficients using coeff_transform_utils
        close_mask, agreement_percentage, mean_rel_diff, max_rel_diff, l_value_stats = (
            compare_transformations(
                theo_coeffs.cpu(),
                actual_coeffs,
                atom_types,
                coeff_to_l_mapping,
                transformation_name=f"Rotation {rotation_name}",
            )
        )

        # Store metrics
        agreement_percentages.append(agreement_percentage)
        mean_rel_diffs.append(mean_rel_diff)
        max_rel_diffs.append(max_rel_diff)

        # If this is the 90-degree rotation, create detailed visualizations
        if i == 0:
            # Create visualization for L value analysis
            plot_isclose_analysis(
                close_mask,
                atom_types,
                coeff_to_l_mapping,
                l_value_stats,
                title_prefix="Theoretical vs Predicted 90-degree Rotation",
                filename="theoretical_vs_predicted_90deg_analysis.png",
            )

            # Create a plot specifically for L value agreement
            l_values = sorted(l_value_stats.keys())
            accuracy_by_l = [
                (
                    100 * l_value_stats[l]["close"] / l_value_stats[l]["total"]
                    if l_value_stats[l]["total"] > 0
                    else 0
                )
                for l in l_values
            ]

            plot_l_value_analysis(
                l_values,
                accuracy_by_l,
                title="Theoretical vs. Actual Coefficient Agreement by L Value",
                filename="theoretical_prediction_by_l_value.png",
            )

    # Create comparison visualization for all rotations
    plot_transformation_comparison(
        agreement_percentages,
        mean_rel_diffs,
        max_rel_diffs,
        rotation_labels,
        title_prefix="Theoretical vs. Actual Rotations",
        filename="theoretical_vs_actual_rotation_comparison.png",
    )


if __name__ == "__main__":
    main()

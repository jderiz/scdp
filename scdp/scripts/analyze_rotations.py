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
    num_rotations = 20  # Number of random rotations to generate

    # Constants
    bohr = 1.88973

    # Load the molecule
    mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
    if mol is None:
        print(f"Error: Could not load molecule from {mol_file}")
        return

    # Extract atom types and positions
    mol_atom_types = []
    mol_pos = []
    origin = [9999.9, 9999.9, 9999.9]
    max_edge = [-9999.9, -9999.9, -9999.9]

    for i in range(mol.GetNumAtoms()):
        mol_atom_types.append(mol.GetAtomWithIdx(i).GetAtomicNum())
        pos = mol.GetConformer().GetAtomPosition(i)
        pos = [pos.x, pos.y, pos.z]
        mol_pos.append(pos)
        for j in range(3):
            origin[j] = min(origin[j], pos[j])
        for j in range(3):
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

    # Compare the first two rotations (90-degree vs identity)
    if len(all_coeffs) >= 2:
        # Calculate relative differences between 90-degree rotation and identity
        rot_90_coeffs = all_coeffs[0]
        identity_coeffs = all_coeffs[1]

        # Use torch.isclose() approach similar to predict_sdf.py
        # Check which coefficients are close between 90-degree rotation and identity
        # atol: absolute tolerance, rtol: relative tolerance
        close_mask = torch.isclose(rot_90_coeffs, identity_coeffs, atol=1e-2, rtol=1e-2)

        # Count how many coefficients are close for each atom
        close_counts = close_mask.sum(dim=1)
        total_counts = close_mask.numel()

        print("\nIsClose Analysis (comparing 90-degree rotation to identity):")
        print("===========================================================")

        # Per-atom analysis
        for i in range(len(close_counts)):
            atom_type = atom_types[i].item()
            count = close_counts[i].item()
            total = close_mask.shape[1]
            percentage = 100 * count / total
            print(
                f"Atom {i} (Z={atom_type}): {count}/{total} ({percentage:.2f}%) coefficients are close"
            )

        # Overall statistics
        total_close = close_mask.sum().item()
        total_coeffs = close_mask.numel()
        print(
            f"\nOverall: {total_close}/{total_coeffs} ({100*total_close/total_coeffs:.2f}%) coefficients are close"
        )

        # Extract L-value mapping from the model
        coeff_to_l_mapping = extract_l_mapping(model, atom_types)

        # Analysis by L value
        l_value_close_stats = {}

        for i in range(len(atom_types)):
            atom_type = atom_types[i].item()
            if atom_type not in coeff_to_l_mapping:
                continue

            l_values = coeff_to_l_mapping[atom_type]

            for j, l in enumerate(l_values):
                if j >= close_mask.shape[1]:
                    continue

                if l not in l_value_close_stats:
                    l_value_close_stats[l] = {"total": 0, "close": 0}

                l_value_close_stats[l]["total"] += 1
                if close_mask[i, j]:
                    l_value_close_stats[l]["close"] += 1

        # Print statistics for each L value
        print("\nIsClose Analysis by Angular Momentum (L):")
        print("========================================")
        for l, stats in sorted(l_value_close_stats.items()):
            percentage = (
                100 * stats["close"] / stats["total"] if stats["total"] > 0 else 0
            )
            print(
                f"L={l}: {stats['close']}/{stats['total']} ({percentage:.2f}%) coefficients are close"
            )

        # Calculate absolute difference
        abs_diff = torch.abs(rot_90_coeffs - identity_coeffs)

        # Calculate relative difference
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        rel_diff = abs_diff / (torch.abs(identity_coeffs) + epsilon)

        # Print summary statistics
        print("\nDifference Statistics between 90-degree rotation and identity:")
        print(f"Mean absolute difference: {abs_diff.mean().item():.4e}")
        print(f"Max absolute difference: {abs_diff.max().item():.4e}")
        print(f"Mean relative difference: {rel_diff.mean().item():.4f}")
        print(f"Max relative difference: {rel_diff.max().item():.4f}")

        # Create visualizations for isClose analysis
        visualize_isclose_analysis(close_mask, atom_types, coeff_to_l_mapping, l_value_close_stats)

    # Extract L-value mapping from the model to analyze by angular momentum
    coeff_to_l_mapping = extract_l_mapping(model, atom_types)

    # Analyze coefficient stability across rotations
    analyze_coefficient_stability(all_coeffs_tensor, atom_types, coeff_to_l_mapping)


def visualize_isclose_analysis(close_mask, atom_types, coeff_to_l_mapping, l_value_close_stats):
    """
    Create visualizations specifically for the isClose analysis comparing 90-degree rotation to identity.
    
    Args:
        close_mask: Boolean tensor of shape [num_atoms, num_coefficients] indicating which coefficients are close
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        l_value_close_stats: Dictionary containing statistics by L value
    """
    plt.figure(figsize=(20, 16))
    
    # 1. Heatmap of close coefficients (boolean mask)
    plt.subplot(2, 2, 1)
    plt.imshow(close_mask.numpy(), aspect='auto', cmap='Blues')
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Different', 'Close'], fontsize=18)
    plt.xlabel('Coefficient Index', fontsize=20)
    plt.ylabel('Atom Index', fontsize=20)
    plt.title('Coefficients that Remain Close Under 90° Rotation', fontsize=22)
    
    # 2. Bar chart showing percentage of close coefficients by atom
    plt.subplot(2, 2, 2)
    close_percentages = (close_mask.sum(dim=1).float() / close_mask.shape[1] * 100).numpy()
    atom_types_str = [f"{i} (Z={atom_types[i].item()})" for i in range(len(atom_types))]
    
    # Sort by percentage
    sorted_indices = np.argsort(close_percentages)
    sorted_percentages = close_percentages[sorted_indices]
    sorted_labels = [atom_types_str[i] for i in sorted_indices]
    
    plt.barh(range(len(sorted_percentages)), sorted_percentages, color='royalblue')
    plt.yticks(range(len(sorted_percentages)), sorted_labels)
    plt.xlabel('Percentage of Coefficients that Remain Close (%)', fontsize=20)
    plt.ylabel('Atom Index (Z=atom type)', fontsize=20)
    plt.title('Percentage of Close Coefficients by Atom', fontsize=22)
    plt.axvline(x=np.mean(close_percentages), color='red', linestyle='--', 
                label=f'Mean: {np.mean(close_percentages):.1f}%')
    plt.legend(fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 3. Bar chart showing percentage of close coefficients by L value
    plt.subplot(2, 2, 3)
    l_values = sorted(l_value_close_stats.keys())
    l_percentages = [100 * l_value_close_stats[l]["close"] / l_value_close_stats[l]["total"] 
                    if l_value_close_stats[l]["total"] > 0 else 0 
                    for l in l_values]
    
    plt.bar(l_values, l_percentages, color='royalblue')
    plt.xlabel('Angular Momentum (L)', fontsize=20)
    plt.ylabel('Percentage of Close Coefficients (%)', fontsize=20)
    plt.title('Percentage of Close Coefficients by L Value', fontsize=22)
    plt.xticks(l_values)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Heatmap by L value
    plt.subplot(2, 2, 4)
    
    # Reorganize data by L value for each atom
    # Create a matrix where rows are atoms and columns are L values
    l_max = max(l_value_close_stats.keys())
    data = np.zeros((len(atom_types), l_max + 1))
    
    for i in range(len(atom_types)):
        atom_type = atom_types[i].item()
        if atom_type not in coeff_to_l_mapping:
            continue
            
        l_values = coeff_to_l_mapping[atom_type]
        
        # Count close coefficients for each L value for this atom
        l_counts = {}
        l_totals = {}
        
        for j, l in enumerate(l_values):
            if j >= close_mask.shape[1]:
                continue
                
            if l not in l_counts:
                l_counts[l] = 0
                l_totals[l] = 0
                
            l_totals[l] += 1
            if close_mask[i, j]:
                l_counts[l] += 1
        
        # Calculate percentage for each L value
        for l in range(l_max + 1):
            if l in l_counts and l_totals[l] > 0:
                data[i, l] = 100 * l_counts[l] / l_totals[l]
    
    im = plt.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(im, label='Percentage of Close Coefficients (%)')
    plt.xlabel('Angular Momentum (L)', fontsize=20)
    plt.ylabel('Atom Index', fontsize=20)
    plt.title('Close Coefficient Percentage by Atom and L Value', fontsize=22)
    plt.xticks(range(l_max + 1))
    
    plt.tight_layout()
    plt.savefig('isclose_analysis.png')
    print("\nIsClose analysis visualization saved as 'isclose_analysis.png'")
    
    # Additional visualization: comparison with absolute differences
    create_coefficient_comparison_visualization(close_mask, atom_types, coeff_to_l_mapping)


def create_coefficient_comparison_visualization(close_mask, atom_types, coeff_to_l_mapping):
    """
    Create additional visualizations showing close coefficients distribution across L values.
    
    Args:
        close_mask: Boolean tensor indicating which coefficients are close
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
    """
    plt.figure(figsize=(16, 12))
    
    # Create a visualization showing the distribution of close vs. not close coefficients by L value
    l_values_by_idx = []
    
    # Collect all L values and their close/not close status
    for i in range(len(atom_types)):
        atom_type = atom_types[i].item()
        if atom_type not in coeff_to_l_mapping:
            continue
            
        l_values = coeff_to_l_mapping[atom_type]
        
        for j, l in enumerate(l_values):
            if j >= close_mask.shape[1]:
                continue
                
            l_values_by_idx.append((l, close_mask[i, j].item()))
    
    # Count coefficients for each L value
    l_max = max(x[0] for x in l_values_by_idx)
    close_counts = [0] * (l_max + 1)
    not_close_counts = [0] * (l_max + 1)
    
    for l, is_close in l_values_by_idx:
        if is_close:
            close_counts[l] += 1
        else:
            not_close_counts[l] += 1
    
    # Create stacked bar chart
    l_values = list(range(l_max + 1))
    
    # Plot stacked bar chart
    plt.figure(figsize=(15, 10))
    bar_width = 0.8
    
    # Calculate total counts and percentages
    total_counts = [close_counts[l] + not_close_counts[l] for l in range(l_max + 1)]
    close_percentages = [100 * close_counts[l] / total_counts[l] if total_counts[l] > 0 else 0 for l in range(l_max + 1)]
    
    # Create the stacked bar chart
    plt.bar(l_values, not_close_counts, bar_width, label='Different after rotation', color='tab:red')
    plt.bar(l_values, close_counts, bar_width, bottom=not_close_counts, label='Close after rotation', color='tab:blue')
    
    # Add counts as text
    for i, l in enumerate(l_values):
        # Add total count on top
        plt.text(l, total_counts[l] + 20, f'{total_counts[l]}', ha='center', fontsize=14)
        # Add percentage in the middle of the blue bar
        if close_counts[l] > 50:  # Only add text if the bar is big enough
            plt.text(l, not_close_counts[l] + close_counts[l]/2, 
                    f'{close_percentages[l]:.1f}%', ha='center', fontsize=14, color='white')
    
    plt.xlabel('Angular Momentum (L)', fontsize=20)
    plt.ylabel('Number of Coefficients', fontsize=20)
    plt.title('Distribution of Coefficients by L Value', fontsize=22)
    plt.xticks(l_values)
    plt.legend(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('coefficient_distribution_by_L.png')
    print("Coefficient distribution visualization saved as 'coefficient_distribution_by_L.png'")


def extract_l_mapping(model, atom_types):
    """
    Extract mapping from coefficient indices to L values from the model.
    
    This function maps each coefficient to its corresponding angular momentum (L) value,
    which is essential for understanding which types of orbitals (s, p, d, f, etc.)
    are more invariant under rotation.
    
    Args:
        model: ChgLightningModule with orbital information
        atom_types: Tensor of atom types in the molecule
    
    Returns:
        Dictionary mapping atom types to list of L values for each coefficient
    """
    # Initialize the result dictionary
    coeff_to_l = {}

    # Process each unique atom type
    unique_atom_types = torch.unique(atom_types)

    for atom_type in unique_atom_types:
        atom_type_int = atom_type.item()
        # Get the Gaussian Type Orbital (GTO) object for this atom type
        gto = model.gto_dict[str(atom_type_int)]

        # Extract L values for each coefficient
        l_values = []

        # The orbital indexing is based on angular momentum (L) (s, p, d, f, etc.)
        # For each L value, there are 2L+1 coefficients (corresponding to m = -L, -L+1, ..., L-1, L)
        for l in range(gto.Lmax + 1):
            # Get number of orbitals with this L value
            n_orbs = gto.n_orbitals_per_L[l].item()
            # Each orbital with angular momentum L has 2L+1 coefficients
            for _ in range(n_orbs):
                for _ in range(2 * l + 1):
                    l_values.append(l)

        coeff_to_l[atom_type_int] = l_values

        # Additional check: verify that the number of coefficients matches
        # with the number in the model's orbital index
        orb_index = model.orb_index[atom_type_int]
        num_coeffs = orb_index.sum().item()

        if len(l_values) != num_coeffs:
            print(
                f"Warning: Mismatch for atom type {atom_type_int}: {len(l_values)} L values but {num_coeffs} coefficients"
            )
            # Try to recover by creating a default mapping
            coeff_to_l[atom_type_int] = [0] * num_coeffs

    return coeff_to_l


def analyze_coefficient_stability(all_coeffs, atom_types, coeff_to_l_mapping):
    """
    Analyze how coefficients change under rotation and identify invariant coefficients.

    This function calculates the stability of each coefficient under rotation and
    performs detailed analysis based on atom types and angular momentum values.

    Args:
        all_coeffs: Tensor of shape [num_rotations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
    """
    num_rotations, num_atoms, num_coeffs = all_coeffs.shape

    # Calculate standard deviation for each coefficient across rotations
    # This measures how much each coefficient varies across different rotations
    coeff_std = torch.std(all_coeffs, dim=0)  # [num_atoms, num_coeffs]

    # Calculate mean absolute value of coefficients
    # This gives a measure of the coefficient's magnitude
    coeff_abs_mean = torch.mean(torch.abs(all_coeffs), dim=0)  # [num_atoms, num_coeffs]

    # Calculate relative standard deviation (coefficient of variation)
    # This normalizes the standard deviation by the mean magnitude
    # Small values indicate coefficients that are stable under rotation
    epsilon = 1e-10  # Small constant to prevent division by zero
    relative_std = coeff_std / (coeff_abs_mean + epsilon)

    # Consider a coefficient "invariant" if its relative standard deviation is below threshold
    invariance_threshold = 0.01  # 1% variation is the threshold for invariance
    invariant_mask = relative_std < invariance_threshold

    # For each atom, count how many coefficients are invariant
    invariant_counts = invariant_mask.sum(dim=1)

    print("\nCoefficient Stability Analysis:")
    print("================================")

    # Per-atom analysis
    for i in range(num_atoms):
        atom_type = atom_types[i].item()
        count = invariant_counts[i].item()
        percentage = 100 * count / num_coeffs
        print(
            f"Atom {i} (Z={atom_type}): {count}/{num_coeffs} ({percentage:.2f}%) coefficients are invariant to rotation"
        )

    # Overall statistics
    total_invariant = invariant_mask.sum().item()
    total_coeffs = num_atoms * num_coeffs
    print(
        f"\nOverall: {total_invariant}/{total_coeffs} ({100*total_invariant/total_coeffs:.2f}%) coefficients are invariant to rotation"
    )

    # Group coefficients by atom type
    atom_types_unique = torch.unique(atom_types)
    for atom_type in atom_types_unique:
        mask = atom_types == atom_type
        type_invariant = invariant_mask[mask].sum().item()
        type_total = mask.sum().item() * num_coeffs
        print(
            f"Atom type Z={atom_type.item()}: {type_invariant}/{type_total} ({100*type_invariant/type_total:.2f}%) coefficients are invariant"
        )

    # Analysis by angular momentum
    print("\nAnalysis by Angular Momentum (L):")
    print("================================")

    # Collect statistics per L value
    l_value_stats = {}

    for i in range(num_atoms):
        atom_type = atom_types[i].item()

        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        # Create or append to the statistics for each L value
        for j, l in enumerate(l_values):
            if j >= invariant_mask.shape[1]:
                continue

            if l not in l_value_stats:
                l_value_stats[l] = {"total": 0, "invariant": 0}

            l_value_stats[l]["total"] += 1
            if invariant_mask[i, j]:
                l_value_stats[l]["invariant"] += 1

    # Print statistics for each L value
    for l, stats in sorted(l_value_stats.items()):
        percentage = (
            100 * stats["invariant"] / stats["total"] if stats["total"] > 0 else 0
        )
        print(
            f"L={l}: {stats['invariant']}/{stats['total']} ({percentage:.2f}%) coefficients are invariant"
        )

    # Create a single visualization for the coefficient stability
    plt.figure(figsize=(10, 8))

    # Heatmap of relative standard deviations
    plt.imshow(relative_std.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Relative Standard Deviation")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Atom Index")
    plt.title("Coefficient Stability Under Rotation")

    plt.tight_layout()
    plt.savefig("coefficient_stability_analysis.png")
    print("\nCoefficient stability visualization saved as 'coefficient_stability_analysis.png'")

    # Create additional visualization: Heatmap of coefficients by L value and stability
    create_l_value_heatmap(all_coeffs, atom_types, coeff_to_l_mapping)

    # Create specific comparison between 90-degree rotation and identity
    if num_rotations >= 2:
        create_rotation_comparison(all_coeffs, atom_types, coeff_to_l_mapping)


def create_l_value_heatmap(all_coeffs, atom_types, coeff_to_l_mapping):
    """
    Create a heatmap showing the stability of coefficients grouped by their L values.

    This provides a visual representation of how coefficients with different
    angular momentum values respond to rotation.

    Args:
        all_coeffs: Tensor of shape [num_rotations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
    """
    num_rotations, num_atoms, num_coeffs = all_coeffs.shape

    # Calculate relative standard deviation
    coeff_std = torch.std(all_coeffs, dim=0)
    coeff_abs_mean = torch.mean(torch.abs(all_coeffs), dim=0)
    epsilon = 1e-10
    relative_std = coeff_std / (coeff_abs_mean + epsilon)

    # Create a figure for the heatmap
    plt.figure(figsize=(14, 10))

    # Process each atom separately - limit to first 4 atoms to keep plot manageable
    for i in range(min(num_atoms, 4)):
        atom_type = atom_types[i].item()

        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        # Skip if no L values available
        if not l_values:
            continue

        # Create data for heatmap
        max_l = max(l_values)

        # Reorganize data by L value - create a matrix where rows are L values
        # and columns are coefficient indices
        data = np.zeros((max_l + 1, num_coeffs))

        # Fill in the relative standard deviation values
        for j, l in enumerate(l_values):
            if j < relative_std.shape[1]:
                data[l, j] = relative_std[i, j].item()

        # Remove empty columns to focus on actual coefficients
        data = data[:, : len(l_values)]

        plt.subplot(2, 2, i + 1)
        # Use plt.imshow to create a heatmap
        im = plt.imshow(data, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Relative Standard Deviation")
        plt.xlabel("Coefficient Index")
        plt.ylabel("Angular Momentum (L)")
        plt.title(f"Coefficient Stability by L Value - Atom {i} (Z={atom_type})")

    plt.tight_layout()
    plt.savefig("l_value_stability_heatmap.png")
    print("L-value stability heatmap saved as 'l_value_stability_heatmap.png'")


def create_rotation_comparison(all_coeffs, atom_types, coeff_to_l_mapping):
    """
    Create visualizations specifically comparing the 90-degree rotation to the identity matrix.

    Args:
        all_coeffs: Tensor of shape [num_rotations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
    """
    # Extract the coefficients for the first two rotations
    rot_90_coeffs = all_coeffs[0]  # 90-degree rotation
    identity_coeffs = all_coeffs[1]  # Identity (no rotation)

    # Calculate absolute difference
    abs_diff = torch.abs(rot_90_coeffs - identity_coeffs)

    # Calculate relative difference
    epsilon = 1e-10
    rel_diff = abs_diff / (torch.abs(identity_coeffs) + epsilon)

    # Create a figure
    plt.figure(figsize=(14, 10))

    # 1. Heatmap of relative differences
    plt.subplot(2, 2, 1)
    plt.imshow(rel_diff.numpy(), aspect="auto", cmap="hot")
    plt.colorbar(label="Relative Difference")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Atom Index")
    plt.title("Coefficient Changes: 90-degree vs. Identity")

    # 2. Histogram of relative differences
    plt.subplot(2, 2, 2)
    plt.hist(rel_diff.numpy().flatten(), bins=50, log=True)
    plt.xlabel("Relative Difference")
    plt.ylabel("Count (log scale)")
    plt.title("Distribution of Coefficient Changes")

    # 3. Differences by L value
    plt.subplot(2, 2, 3)

    # Collect differences by L value
    l_value_diffs = {}

    for i in range(len(atom_types)):
        atom_type = atom_types[i].item()
        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        for j, l in enumerate(l_values):
            if j >= rel_diff.shape[1]:
                continue

            if l not in l_value_diffs:
                l_value_diffs[l] = []

            l_value_diffs[l].append(rel_diff[i, j].item())

    # Box plot of differences by L
    if l_value_diffs:
        data = [l_value_diffs[l] for l in sorted(l_value_diffs.keys())]
        plt.boxplot(data, labels=sorted(l_value_diffs.keys()))
        plt.yscale("log")
        plt.xlabel("Angular Momentum (L)")
        plt.ylabel("Relative Difference (log scale)")
        plt.title("Coefficient Changes by L Value")

    # 4. Comparison of specific coefficients for a selected atom
    plt.subplot(2, 2, 4)

    # Select the first atom
    atom_idx = 0

    # Select representative coefficients
    num_coeffs = rot_90_coeffs.shape[1]
    if num_coeffs > 5:
        selected_coeffs = [
            0,
            num_coeffs // 4,
            num_coeffs // 2,
            3 * num_coeffs // 4,
            num_coeffs - 1,
        ]
    else:
        selected_coeffs = list(range(num_coeffs))

    # Access L values safely
    atom_type_int = atom_types[atom_idx].item()
    l_values = coeff_to_l_mapping.get(atom_type_int, [])

    # Plot coefficient values for both rotations
    x = np.arange(len(selected_coeffs))
    width = 0.35

    rot_values = [rot_90_coeffs[atom_idx, idx].item() for idx in selected_coeffs]
    id_values = [identity_coeffs[atom_idx, idx].item() for idx in selected_coeffs]

    plt.bar(x - width / 2, rot_values, width, label="90-degree rotation")
    plt.bar(x + width / 2, id_values, width, label="Identity")

    # Label with coefficient indices and L values
    labels = []
    for idx in selected_coeffs:
        l_value = l_values[idx] if idx < len(l_values) else "?"
        labels.append(f"{idx}\n(L={l_value})")

    plt.xticks(x, labels)
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title(f"Coefficient Comparison for Atom {atom_idx}")
    plt.legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig("rotation_comparison.png")
    print("Rotation comparison visualization saved as 'rotation_comparison.png'")


if __name__ == "__main__":
    main()

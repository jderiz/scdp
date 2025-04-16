import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from scdp.common.pyg import DataLoader
from scdp.data.data import AtomicData
from scdp.model.module import ChgLightningModule
from scdp.scripts.preprocess import get_atomic_number_table_from_zs

plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18


def main():
    """Main function to analyze orbital coefficient stability under translation."""
    # Configuration
    z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())
    metadata = "caffeine"
    mol_file = "caffeine.sdf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensional_padding = 2.0
    resolution = 0.25
    translation_distance = 2.0  # Distance to translate in Angstroms

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

    # Define translations:
    # First 5 translations: No translation + 6 pure axis translations
    # Last 4 translations: Random translations in 3D space
    translations = [
        # Identity (no translation)
        torch.zeros(3),
        # +x direction
        torch.tensor([translation_distance, 0.0, 0.0]),
        # -x direction
        torch.tensor([-translation_distance, 0.0, 0.0]),
        # +y direction
        torch.tensor([0.0, translation_distance, 0.0]),
        # -y direction
        torch.tensor([0.0, -translation_distance, 0.0]),
        # +z direction
        torch.tensor([0.0, 0.0, translation_distance]),
        # -z direction
        torch.tensor([0.0, 0.0, -translation_distance]),
    ]

    # Add 4 random translations
    torch.manual_seed(42)  # For reproducibility
    for _ in range(4):
        # Generate random direction in 3D space
        random_dir = torch.randn(3)
        # Normalize to unit vector
        random_dir = random_dir / torch.norm(random_dir)
        # Scale by translation distance
        random_translation = random_dir * translation_distance
        translations.append(random_translation)

    # Calculate cube dimensions for electron density representation
    cube_length = [max_edge[i] - origin[i] for i in range(3)]

    # Ensure the cube is large enough to accommodate translations
    for i in range(3):
        cube_length[i] += 2 * translation_distance
        origin[i] -= translation_distance

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
    print(f"\tNumber of translations: {len(translations)}")
    print(f"\tTranslation 0: Identity (no translation)")
    for i in range(1, len(translations)):
        t = translations[i]
        print(f"\tTranslation {i}: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")

    # Create AtomicData objects for each translation
    data_objects = []

    # Get number of atoms and set max_neighbors
    # This prevents errors from having too large max_neighbors value
    num_atoms = len(atom_types)
    max_neighbors = num_atoms - 1  # Each atom can connect to all others at most

    # Create data objects for each translation
    for i, translation in enumerate(translations):
        # Apply translation to atomic coordinates
        translated_coords = atom_coords + translation
        data_obj = AtomicData.build_graph_with_vnodes(
            atom_coords=translated_coords,
            atom_types=atom_types,
            cell=cell,
            chg_density=chg_density,
            origin=origin_tensor,
            metadata=f"{metadata}_trans{i}",
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

    # Collect coefficients for all translations
    all_coeffs = []
    all_expo_scaling = []

    # Process each translation and get the predicted coefficients
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Processing translation {i}")
            batch = batch.to(device)
            # Get the model's predicted coefficients for this translation
            coeffs, expo_scaling = model.predict_coeffs(batch)
            all_coeffs.append(coeffs.cpu())
            if expo_scaling is not None:
                all_expo_scaling.append(expo_scaling.cpu())

    # Convert to tensor for easier analysis
    all_coeffs_tensor = torch.stack(all_coeffs)
    print(f"Collected coefficients shape: {all_coeffs_tensor.shape}")

    # Extract L-value mapping from the model to analyze by angular momentum
    coeff_to_l_mapping = extract_l_mapping(model, atom_types)

    # Analyze coefficient stability across translations
    analyze_coefficient_stability(all_coeffs_tensor, atom_types, coeff_to_l_mapping)

    # Create comprehensive visualization comparing all translations
    create_comprehensive_visualization(
        all_coeffs_tensor, atom_types, coeff_to_l_mapping, translations
    )
    
    # Perform isclose analysis comparing translation to reference (no translation)
    analyze_isclose(all_coeffs_tensor, atom_types, coeff_to_l_mapping)


def extract_l_mapping(model, atom_types):
    """
    Extract mapping from coefficient indices to L values from the model.

    This function maps each coefficient to its corresponding angular momentum (L) value,
    which is essential for understanding which types of orbitals (s, p, d, f, etc.)
    are more invariant under translation.

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
    Analyze how coefficients change under translation and identify invariant coefficients.

    This function calculates the stability of each coefficient under translation and
    performs detailed analysis based on atom types and angular momentum values.

    Args:
        all_coeffs: Tensor of shape [num_translations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
    """
    num_translations, num_atoms, num_coeffs = all_coeffs.shape

    # Calculate standard deviation for each coefficient across translations
    # This measures how much each coefficient varies across different translations
    coeff_std = torch.std(all_coeffs, dim=0)  # [num_atoms, num_coeffs]

    # Calculate mean absolute value of coefficients
    # This gives a measure of the coefficient's magnitude
    coeff_abs_mean = torch.mean(torch.abs(all_coeffs), dim=0)  # [num_atoms, num_coeffs]

    # Calculate relative standard deviation (coefficient of variation)
    # This normalizes the standard deviation by the mean magnitude
    # Small values indicate coefficients that are stable under translation
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
            f"Atom {i} (Z={atom_type}): {count}/{num_coeffs} ({percentage:.2f}%) coefficients are invariant to translation"
        )

    # Overall statistics
    total_invariant = invariant_mask.sum().item()
    total_coeffs = num_atoms * num_coeffs
    print(
        f"\nOverall: {total_invariant}/{total_coeffs} ({100*total_invariant/total_coeffs:.2f}%) coefficients are invariant to translation"
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


def create_comprehensive_visualization(
    all_coeffs, atom_types, coeff_to_l_mapping, translations
):
    """
    Create a comprehensive visualization comparing all translations in a single figure.

    Args:
        all_coeffs: Tensor of shape [num_translations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        translations: List of translation vectors
    """
    num_translations, num_atoms, num_coeffs = all_coeffs.shape

    # Define translation names
    translation_names = ["No translation"]
    translation_names.extend([f"Axis translation {i}" for i in range(1, 7)])
    translation_names.extend(
        [f"Random translation {i}" for i in range(7, num_translations)]
    )

    # Use the first translation (identity) as reference
    reference_coeffs = all_coeffs[0]

    # Calculate relative differences for all translations compared to reference
    rel_diffs = []
    for i in range(1, num_translations):
        current_coeffs = all_coeffs[i]
        abs_diff = torch.abs(current_coeffs - reference_coeffs)
        epsilon = 1e-12  # cant divide by 0
        rel_diff = abs_diff / (torch.abs(reference_coeffs) + epsilon)
        rel_diffs.append(rel_diff)

    # Convert to tensor for easier analysis
    rel_diffs_tensor = torch.stack(rel_diffs)

    # Create a figure with 2 subplots
    plt.figure(figsize=(20, 16))

    # 1. First subplot: Summary of coefficient stability across all translations
    plt.subplot(2, 1, 1)

    # Calculate standard deviation for each coefficient across all translations
    # This is different from relative difference - it directly measures variability
    coeff_std = torch.std(all_coeffs, dim=0)  # [num_atoms, num_coeffs]

    # Create heatmap showing standard deviation
    im = plt.imshow(coeff_std.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Average Standard Deviation")
    plt.xlabel("Coefficient Index", fontsize=16)
    plt.ylabel("Atom Index", fontsize=16)
    plt.title("Average Coefficient Changes Across All Translations", fontsize=18)

    # Add L value annotations for selected atoms if number of atoms is reasonable
    if num_atoms <= 5:
        # Select a few atoms to annotate
        for atom_idx in range(min(5, num_atoms)):
            atom_type = atom_types[atom_idx].item()
            if atom_type in coeff_to_l_mapping:
                l_values = coeff_to_l_mapping[atom_type]

                # Add text annotations for L values at regular intervals
                step = max(1, len(l_values) // 10)  # Show at most 10 labels
                for j in range(0, len(l_values), step):
                    if j < coeff_std.shape[1]:
                        plt.text(
                            j,
                            atom_idx,
                            f"L={l_values[j]}",
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="white" if coeff_std[atom_idx, j] > 0.5 else "black",
                        )

    # 2. Second subplot: Detailed comparison of each translation
    plt.subplot(2, 1, 2)

    # Create a grid to show each translation's effect
    # We'll show translation vs atom for the average coefficient change
    avg_by_atom_translation = torch.mean(
        rel_diffs_tensor, dim=2
    )  # [num_translations-1, num_atoms]

    # Create heatmap
    im = plt.imshow(avg_by_atom_translation.numpy(), aspect="auto", cmap="hot")
    plt.colorbar(im, label="Average Relative Difference")
    plt.xlabel("Atom Index", fontsize=16)
    plt.ylabel("Translation", fontsize=16)
    plt.title("Effect of Each Translation on Atoms", fontsize=18)

    # Label the y-axis with translation descriptions
    ytick_labels = translation_names[1:]  # Skip the reference (identity) translation
    plt.yticks(range(len(ytick_labels)), ytick_labels)

    # Add atom type annotations
    for atom_idx in range(num_atoms):
        atom_type = atom_types[atom_idx].item()
        plt.text(
            atom_idx,
            -0.5,
            f"Z={atom_type}",
            rotation=90,
            fontsize=10,
            ha="center",
            va="top",
        )

    # Add a text box summarizing translation vectors
    trans_text = "Translation vectors:\n"
    for i, t in enumerate(translations[1:], 1):  # Skip identity
        trans_text += f"{translation_names[i]}: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]\n"

    # Add text box in the bottom right corner
    plt.annotate(
        trans_text,
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("translation_analysis.png")
    print("\nComprehensive translation analysis saved as 'translation_analysis.png'")


# Add a new function for isclose analysis
def analyze_isclose(all_coeffs, atom_types, coeff_to_l_mapping):
    """
    Perform isclose analysis comparing all translations to the reference (no translation).
    
    Args:
        all_coeffs: Tensor of shape [num_translations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
    """
    # Use the first coefficient set (identity/no translation) as reference
    reference_coeffs = all_coeffs[0]
    
    # Define translation names for labeling
    translation_names = ["No translation (reference)"]
    translation_names.extend([f"Axis {i}" for i in range(1, 7)])
    translation_names.extend([f"Random {i}" for i in range(7, all_coeffs.shape[0])])
    
    # Create a grid of isclose results for all translations vs. reference
    # We'll focus on the first 6 translations for better visualization
    max_trans_to_show = min(7, all_coeffs.shape[0])
    close_masks = []
    
    for i in range(1, max_trans_to_show):
        current_coeffs = all_coeffs[i]
        # Use torch.isclose with relative and absolute tolerances
        close_mask = torch.isclose(reference_coeffs, current_coeffs, atol=1e-2, rtol=1e-2)
        close_masks.append(close_mask)
    
    # Stack close masks to create a 3D tensor: [num_translations-1, num_atoms, num_coeffs]
    close_masks_tensor = torch.stack(close_masks)
    
    # Create a figure
    plt.figure(figsize=(15, 12))
    
    # 1. Heatmap showing which coefficients remain close under each translation
    # Combine all close masks to show average stability across translations
    avg_close_mask = torch.mean(close_masks_tensor.float(), dim=0)
    
    plt.subplot(2, 1, 1)
    plt.imshow(avg_close_mask.numpy(), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('Fraction of Translations where Coefficient Remains Close')
    plt.xlabel('Coefficient Index', fontsize=16)
    plt.ylabel('Atom Index', fontsize=16)
    plt.title('Coefficient Stability Across All Translations (IsClose Analysis)', fontsize=18)
    
    # Add L value annotations for selected atoms if number of atoms is reasonable
    if len(atom_types) <= 5:
        for atom_idx in range(min(5, len(atom_types))):
            atom_type = atom_types[atom_idx].item()
            if atom_type in coeff_to_l_mapping:
                l_values = coeff_to_l_mapping[atom_type]
                
                # Add text annotations for L values at regular intervals
                step = max(1, len(l_values) // 10)  # Show at most 10 labels
                for j in range(0, len(l_values), step):
                    if j < avg_close_mask.shape[1]:
                        plt.text(
                            j,
                            atom_idx,
                            f"L={l_values[j]}",
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black" if avg_close_mask[atom_idx, j] > 0.5 else "white",
                        )
    
    # 2. Translation-specific heatmap 
    plt.subplot(2, 1, 2)
    
    # Reshape close_masks_tensor for visualization
    # We want to show translations (y-axis) vs atoms and coefficients (flattened on x-axis)
    num_trans, num_atoms, num_coeffs = close_masks_tensor.shape
    
    # For better visualization, we'll sample a subset of coefficients
    # rather than showing all of them
    sample_step = max(1, num_coeffs // 40)  # Show at most 40 coefficients per atom
    sampled_coeffs = []
    coeff_labels = []
    
    for atom_idx in range(num_atoms):
        for coeff_idx in range(0, num_coeffs, sample_step):
            sampled_coeffs.append(close_masks_tensor[:, atom_idx, coeff_idx])
            atom_type = atom_types[atom_idx].item()
            
            # Add L value if available
            l_value = "?"
            if atom_type in coeff_to_l_mapping:
                l_values = coeff_to_l_mapping[atom_type]
                if coeff_idx < len(l_values):
                    l_value = l_values[coeff_idx]
                    
            coeff_labels.append(f"A{atom_idx}:C{coeff_idx}\n(L={l_value})")
    
    # Stack sampled coefficients side by side
    sampled_data = torch.stack(sampled_coeffs, dim=1)
    
    # Plot the heatmap
    plt.imshow(sampled_data.numpy(), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_ticklabels(['Different', 'Close'])
    
    # Label axes
    plt.xlabel('Atom:Coefficient (L value)', fontsize=16)
    plt.ylabel('Translation', fontsize=16)
    plt.title('Coefficients Remaining Close Under Different Translations', fontsize=18)
    
    # Set x-axis ticks
    if len(coeff_labels) <= 40:
        plt.xticks(range(len(coeff_labels)), coeff_labels, rotation=90, fontsize=8)
    else:
        # If we have too many labels, show a subset
        tick_indices = np.linspace(0, len(coeff_labels)-1, 40, dtype=int)
        plt.xticks(tick_indices, [coeff_labels[i] for i in tick_indices], rotation=90, fontsize=8)
    
    # Set y-axis ticks
    plt.yticks(range(num_trans), translation_names[1:max_trans_to_show])
    
    # Add summary statistics above the plot
    total_close = close_masks_tensor.sum().item()
    total_elements = close_masks_tensor.numel()
    percentage = 100 * total_close / total_elements
    
    plt.suptitle(
        f"IsClose Analysis: {total_close}/{total_elements} ({percentage:.2f}%) "
        f"coefficients remain close across all translations",
        fontsize=20,
        y=0.98
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("isclose_translation_analysis.png")
    print("\nIsClose translation analysis saved as 'isclose_translation_analysis.png'")


if __name__ == "__main__":
    main()

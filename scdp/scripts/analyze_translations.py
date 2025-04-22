import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from scdp.common.pyg import DataLoader
from scdp.data.data import AtomicData
from scdp.model.module import ChgLightningModule
from scdp.scripts.coeff_transform_utils import (
    compare_transformations,
    extract_l_mapping,
    predict_transformed_coeffs_translation,
)
from scdp.scripts.plotting_helpers import (
    analyze_by_l_value,
    analyze_coefficient_stability,
    compare_coefficient_pairs,
    plot_coefficient_stability,
    plot_isclose_analysis,
    plot_transformation_comparison,
)
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
    translation_distance = 0.5  # Distance to translate in Angstroms

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
    relative_std, invariant_mask, l_value_stats = analyze_coefficient_stability(
        all_coeffs_tensor, atom_types, coeff_to_l_mapping
    )

    # Visualize coefficient stability
    plot_coefficient_stability(
        relative_std,
        title="Coefficient Stability Under Translation",
        filename="translation_coefficient_stability.png",
    )

    # Create comprehensive visualization comparing all translations
    create_comprehensive_visualization(
        all_coeffs_tensor, atom_types, coeff_to_l_mapping, translations
    )

    # Perform isclose analysis comparing translation to reference (no translation)
    analyze_isclose_translations(all_coeffs_tensor, atom_types, coeff_to_l_mapping)

    # Compare with theoretical predictions (coefficients should be invariant to translation)
    compare_with_theoretical_predictions(
        model, all_coeffs_tensor, translations, atom_types, coeff_to_l_mapping, device
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


def analyze_isclose_translations(all_coeffs, atom_types, coeff_to_l_mapping):
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

    # Analyze the first axis translation as an example
    if all_coeffs.shape[0] > 1:
        close_mask, rel_diff = compare_coefficient_pairs(
            reference_coeffs,
            all_coeffs[1],
            title_prefix="No translation vs +X translation",
        )

        l_value_stats = analyze_by_l_value(close_mask, atom_types, coeff_to_l_mapping)

        # Create isclose visualization
        plot_isclose_analysis(
            close_mask,
            atom_types,
            coeff_to_l_mapping,
            l_value_stats,
            title_prefix="Translation Analysis",
            filename="isclose_translation_analysis.png",
        )


def compare_with_theoretical_predictions(
    model, all_coeffs_tensor, translations, atom_types, coeff_to_l_mapping, device
):
    """
    Compare actual translated coefficients with theoretical predictions.
    For true GTO orbitals, coefficients should be invariant under translation.

    Args:
        model: The trained ChgLightningModule instance
        all_coeffs_tensor: Tensor of shape [num_translations, num_atoms, num_coefficients]
                          containing all predicted coefficients
        translations: List of translation vectors
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        device: Device to perform computations on
    """
    # Use the coefficients from the first translation (index 0) as the reference
    reference_coeffs = all_coeffs_tensor[0]  # No translation

    # Initialize arrays to store comparison metrics
    num_translations = len(translations)
    agreement_percentages = []
    mean_rel_diffs = []
    max_rel_diffs = []
    translation_labels = []

    # Skip the first translation (identity/no translation)
    translation_indices = range(1, min(7, num_translations))

    for i in translation_indices:
        translation_vector = translations[i].cpu().numpy()
        actual_coeffs = all_coeffs_tensor[i]

        # Create label for this translation
        translation_name = f"Translation {i} ({translations[i][0]:.2f}, {translations[i][1]:.2f}, {translations[i][2]:.2f})"
        translation_labels.append(translation_name)

        # Predict the theoretical coefficients - should be invariant to translation
        theo_coeffs = predict_transformed_coeffs_translation(
            model=model,
            coeffs_original=reference_coeffs,
            translation_vector=translation_vector,
            device=device,
        )

        # Compare theoretical predictions with actual translated coefficients
        close_mask, agreement_percentage, mean_rel_diff, max_rel_diff, l_value_stats = (
            compare_transformations(
                theo_coeffs.cpu(),
                actual_coeffs,
                atom_types,
                coeff_to_l_mapping,
                transformation_name=f"Translation {translation_name}",
            )
        )

        # Store metrics
        agreement_percentages.append(agreement_percentage)
        mean_rel_diffs.append(mean_rel_diff)
        max_rel_diffs.append(max_rel_diff)

    # Create comparison visualization for all translations
    plot_transformation_comparison(
        agreement_percentages,
        mean_rel_diffs,
        max_rel_diffs,
        translation_labels,
        title_prefix="Theoretical vs. Actual Translations",
        filename="theoretical_vs_actual_translation_comparison.png",
    )


if __name__ == "__main__":
    main()

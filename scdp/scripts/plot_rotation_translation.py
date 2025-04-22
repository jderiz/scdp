import os

import numpy as np
import plotly.graph_objects as go
import torch
from ase.data import chemical_symbols
from rdkit import Chem
from scipy.spatial.transform import Rotation

from scdp.common.visualization import draw_volume, lattice_object
from scdp.scripts.preprocess import get_atomic_number_table_from_zs


def random_rotation_matrix():
    """Generate a random rotation matrix using scipy's Rotation."""
    return torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)


def plot_molecule_transformations(
    mol_file,
    num_rotations=3,
    num_translations=3,
    translation_distance=2.0,
    padding=2.0,
    output_dir="visualizations",
    save_format="both",  # Options: 'html', 'png', 'both'
    dtype="volume",
    
):
    """
    Generate visualizations for molecule rotations and translations.

    Args:
        mol_file (str): Path to the molecule file (.sdf format)
        num_rotations (int): Number of random rotations to visualize
        num_translations (int): Number of translations to visualize
        translation_distance (float): Distance to translate in Angstroms
        padding (float): Padding to add around the molecule
        output_dir (str): Directory to save visualization files
        save_format (str): Format to save visualizations - 'html', 'png', or 'both'
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the molecule
    mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
    if mol is None:
        print(f"Error: Could not load molecule from {mol_file}")
        return

    # Extract atom types and positions
    atom_types = []
    atom_coords = []
    origin = [9999.9, 9999.9, 9999.9]
    max_edge = [-9999.9, -9999.9, -9999.9]

    for i in range(mol.GetNumAtoms()):
        atom_types.append(mol.GetAtomWithIdx(i).GetAtomicNum())
        pos = mol.GetConformer().GetAtomPosition(i)
        pos = [pos.x, pos.y, pos.z]
        atom_coords.append(pos)
        for j in range(3):
            origin[j] = min(origin[j], pos[j])
        for j in range(3):
            max_edge[j] = max(max_edge[j], pos[j])

    # Add padding to ensure we capture all electron density
    for i in range(3):
        origin[i] -= padding
        max_edge[i] += padding

    # Convert to numpy arrays for visualization
    atom_types = np.array(atom_types)
    atom_coords = np.array(atom_coords)

    # Calculate cube dimensions for visualization
    cube_length = [max_edge[i] - origin[i] for i in range(3)]
    cell = np.array(
        [
            [cube_length[0], 0.0, 0.0],
            [0.0, cube_length[1], 0.0],
            [0.0, 0.0, cube_length[2]],
        ]
    )

    # Create the original molecule visualization
    fig_original = draw_volume(
        grid_pos=None,
        density=None,
        atom_types=atom_types,
        atom_coord=atom_coords,
        cell=cell,
        origin=origin,
        title="Original Molecule",
        dtype="volume",
    )

    # Save visualization in the requested format(s)
    save_visualization(
        fig_original, os.path.join(output_dir, "original_molecule"), save_format
    )

    # Generate rotation matrices
    rotations = [
        # 90-degree rotation counterclockwise around the x-axis
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        # Identity matrix (no rotation)
        np.eye(3),
    ]

    # Add random rotations
    torch.manual_seed(42)  # For reproducibility
    for _ in range(num_rotations - 2):  # -2 because we already added two rotations
        rotations.append(random_rotation_matrix().numpy())

    # Visualize rotations
    for i, rotation in enumerate(rotations):
        # Apply rotation to atomic coordinates
        rotated_coords = atom_coords @ rotation

        # Create rotation visualization
        rotation_name = (
            "90-degree X-axis" if i == 0 else "Identity" if i == 1 else f"Random {i-1}"
        )
        fig_rotation = draw_volume(
            grid_pos=None,
            density=None,
            atom_types=atom_types,
            atom_coord=rotated_coords,
            cell=cell,
            origin=origin,
            title=f"Rotation: {rotation_name}",
            dtype="volume",
        )

        # Save rotation visualization
        save_visualization(
            fig_rotation, os.path.join(output_dir, f"rotation_{i}"), save_format
        )

        # Create a figure showing both original and rotated molecule
        if i > 0:  # Skip the identity rotation
            fig_comparison = go.Figure()

            # Add original molecule
            fig_comparison.add_trace(
                go.Scatter3d(
                    x=atom_coords[:, 0],
                    y=atom_coords[:, 1],
                    z=atom_coords[:, 2],
                    mode="markers",
                    name="Original",
                    marker=dict(size=6, color="blue", opacity=0.7),
                )
            )

            # Add rotated molecule
            fig_comparison.add_trace(
                go.Scatter3d(
                    x=rotated_coords[:, 0],
                    y=rotated_coords[:, 1],
                    z=rotated_coords[:, 2],
                    mode="markers",
                    name=f"Rotated ({rotation_name})",
                    marker=dict(size=6, color="red", opacity=0.7),
                )
            )

            # Update layout
            axis_dict = dict(
                showgrid=False,
                showbackground=True,
                backgroundcolor="rgb(240, 240, 240)",
                zeroline=False,
            )

            fig_comparison.update_layout(
                title=f"Rotation Comparison: Original vs {rotation_name}",
                scene=dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict),
                width=800,
                height=800,
                margin=dict(l=0, r=0, b=0, t=40),
            )

            # Save comparison visualization
            save_visualization(
                fig_comparison,
                os.path.join(output_dir, f"rotation_comparison_{i}"),
                save_format,
            )

    # Define translations
    translations = [
        # Identity (no translation)
        np.zeros(3),
        # +x direction
        np.array([translation_distance, 0.0, 0.0]),
        # +y direction
        np.array([0.0, translation_distance, 0.0]),
        # +z direction
        np.array([0.0, 0.0, translation_distance]),
    ]

    # Add random translations if needed
    np.random.seed(42)  # For reproducibility
    for _ in range(
        num_translations - 4
    ):  # -4 because we already added four translations
        # Generate random direction in 3D space
        random_dir = np.random.randn(3)
        # Normalize to unit vector
        random_dir = random_dir / np.linalg.norm(random_dir)
        # Scale by translation distance
        random_translation = random_dir * translation_distance
        translations.append(random_translation)

    # Visualize translations
    for i, translation in enumerate(translations):
        # Apply translation to atomic coordinates
        translated_coords = atom_coords + translation

        # Create translation visualization
        if i == 0:
            translation_name = "No Translation"
        elif i == 1:
            translation_name = f"+X ({translation_distance}Å)"
        elif i == 2:
            translation_name = f"+Y ({translation_distance}Å)"
        elif i == 3:
            translation_name = f"+Z ({translation_distance}Å)"
        else:
            translation_name = f"Random {i-3}"

        fig_translation = draw_volume(
            grid_pos=None,
            density=None,
            atom_types=atom_types,
            atom_coord=translated_coords,
            cell=cell,
            origin=origin,
            title=f"Translation: {translation_name}",
            dtype="volume",
        )

        # Save translation visualization
        save_visualization(
            fig_translation, os.path.join(output_dir, f"translation_{i}"), save_format
        )

        # Create a figure showing both original and translated molecule
        if i > 0:  # Skip the identity translation
            fig_comparison = go.Figure()

            # Add original molecule
            fig_comparison.add_trace(
                go.Scatter3d(
                    x=atom_coords[:, 0],
                    y=atom_coords[:, 1],
                    z=atom_coords[:, 2],
                    mode="markers",
                    name="Original",
                    marker=dict(size=6, color="blue", opacity=0.7),
                )
            )

            # Add translated molecule
            fig_comparison.add_trace(
                go.Scatter3d(
                    x=translated_coords[:, 0],
                    y=translated_coords[:, 1],
                    z=translated_coords[:, 2],
                    mode="markers",
                    name=f"Translated ({translation_name})",
                    marker=dict(size=6, color="green", opacity=0.7),
                )
            )

            # Add an arrow to visualize the translation
            mid_point = np.mean(atom_coords, axis=0)
            fig_comparison.add_trace(
                go.Scatter3d(
                    x=[mid_point[0], mid_point[0] + translation[0]],
                    y=[mid_point[1], mid_point[1] + translation[1]],
                    z=[mid_point[2], mid_point[2] + translation[2]],
                    mode="lines+markers",
                    line=dict(width=5, color="black"),
                    marker=dict(size=[0, 5]),
                    name="Translation Vector",
                )
            )

            # Update layout
            axis_dict = dict(
                showgrid=False,
                showbackground=True,
                backgroundcolor="rgb(240, 240, 240)",
                zeroline=False,
            )

            fig_comparison.update_layout(
                title=f"Translation Comparison: Original vs {translation_name}",
                scene=dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict),
                width=800,
                height=800,
                margin=dict(l=0, r=0, b=0, t=40),
            )

            # Save comparison visualization
            save_visualization(
                fig_comparison,
                os.path.join(output_dir, f"translation_comparison_{i}"),
                save_format,
            )


def save_visualization(fig, base_path, save_format="both"):
    """
    Save a plotly figure in the specified format(s).

    Args:
        fig: The plotly figure to save
        base_path: Base file path without extension
        save_format: Format to save - 'html', 'png', or 'both'
    """
    if save_format in ["html", "both"]:
        fig.write_html(f"{base_path}.html")

    if save_format in ["png", "both"]:
        fig.write_image(f"{base_path}.png", width=1200, height=1200, scale=2)


if __name__ == "__main__":
    # Use the same molecule file as in analyze_rotations.py and analyze_translations.py
    mol_file = "caffeine.sdf"

    if not os.path.exists(mol_file):
        print(
            f"Warning: {mol_file} not found. Please make sure the file exists or update the path."
        )
    else:
        plot_molecule_transformations(
            mol_file=mol_file,
            num_rotations=5,
            num_translations=6,
            translation_distance=2.0,
            padding=2.0,
            output_dir="transformation_visualizations",
            save_format="both",  # Save both HTML and PNG
        )
        print(
            "Visualizations created in the 'transformation_visualizations' directory (HTML and PNG formats)."
        )

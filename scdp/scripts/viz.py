import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import torch
import lmdb
import pickle
import argparse


def load_molecule_from_lmdb(
    db_path: str, molecule_name: str
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load molecular structure from LMDB database by molecule name.

    Args:
        db_path: Path to LMDB database
        molecule_name: Name of molecule to load

    Returns:
        Dictionary containing molecular structure data or None if not found
    """
    env = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,  # 2GB
        subdir=False,
        readonly=True,
        lock=False,
    )

    with env.begin() as txn:
        # Get total number of molecules
        length = pickle.loads(txn.get("length".encode()))

        # Search for molecule by name
        for i in range(length):
            data = pickle.loads(txn.get(f"{i}".encode()))
            if data.metadata == molecule_name:
                return {
                    "coords": data.coords,
                    "atom_types": data.atom_types,
                    "edge_index": data.edge_index,
                }

    return None


def visualize_molecule_and_charges(
    results: Dict[str, Any],
    molecule_data: Dict[str, torch.Tensor],
    iso_min: float = 0.1,
    iso_max: float = 2.0,
    n_isosurfaces: int = 4,
    opacity: float = 0.3,
    show_probes: bool = True,
    colorscale: str = "RdBu",  # Red-Blue diverging colorscale
) -> None:
    """
    Visualize 3D charge distribution and molecular structure.

    Args:
        results: Dictionary containing charge predictions
        molecule_data: Dictionary containing molecular structure
        iso_min: Minimum isosurface value
        iso_max: Maximum isosurface value
        n_isosurfaces: Number of isosurfaces to show
        opacity: Opacity of isosurfaces
        show_probes: Whether to show probe points
        colorscale: Plotly colorscale to use
    """
    # Atom colors and sizes
    ATOM_COLORS = {
        1: "#FFFFFF",  # H - White
        6: "#808080",  # C - Gray
        7: "#0000FF",  # N - Blue
        8: "#FF0000",  # O - Red
        9: "#90E050",  # F - Light Green
        16: "#FFC832",  # S - Yellow
    }

    ATOM_SIZES = {
        1: 0.4,  # H
        6: 0.7,  # C
        7: 0.65,  # N
        8: 0.6,  # O
        9: 0.5,  # F
        16: 1.0,  # S
    }

    # Create figure
    fig = go.Figure()

    # Reshape and normalize predictions
    grid_size = 10
    values = results["predictions"].reshape(grid_size, grid_size, grid_size)

    # Normalize values to [-1, 1] range for better visualization
    values = values - np.mean(values)  # Center around 0
    if np.std(values) > 0:
        values = values / np.std(values)  # Scale to unit variance

    print(f"Value range after normalization: [{values.min():.3f}, {values.max():.3f}]")

    # Adjust isosurface values to normalized range
    iso_min = max(-2.0, values.min())  # Limit lower bound
    iso_max = min(2.0, values.max())  # Limit upper bound
    iso_values = np.linspace(iso_min, iso_max, n_isosurfaces)

    # Add isosurfaces
    for iso_val in iso_values:
        fig.add_trace(
            go.Isosurface(
                x=np.arange(grid_size),
                y=np.arange(grid_size),
                z=np.arange(grid_size),
                value=values,
                isomin=iso_val,
                isomax=iso_val,
                surface_count=1,
                colorscale=colorscale,
                opacity=opacity,
                name=f"Density {iso_val:.2f}",
                showscale=True,
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )

    # Add atoms
    coords = molecule_data["coords"].cpu().numpy()
    atom_types = molecule_data["atom_types"].cpu().numpy()

    print(f"Number of atoms: {len(atom_types)}")
    print(f"Unique atom types: {np.unique(atom_types)}")
    print(f"Edge index shape: {molecule_data['edge_index'].shape}")

    # Center and scale the molecule in the grid
    grid_center = np.array([grid_size / 2, grid_size / 2, grid_size / 2])
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    molecule_center = (coords_max + coords_min) / 2

    # Scale the molecule to fit nicely in the grid
    max_extent = np.max(coords_max - coords_min)
    scale_factor = (grid_size * 0.4) / max_extent  # Use 40% of grid size
    coords = (coords - molecule_center) * scale_factor + grid_center

    # Create a dictionary to store atom traces to avoid duplicates
    atom_traces = {}

    # Add atoms (ensuring no duplicates)
    for i, (coord, atom_type) in enumerate(zip(coords, atom_types)):
        if i not in atom_traces:
            fig.add_trace(
                go.Scatter3d(
                    x=[coord[0]],
                    y=[coord[1]],
                    z=[coord[2]],
                    mode="markers",
                    marker=dict(
                        size=ATOM_SIZES.get(atom_type, 0.5) * 20,
                        color=ATOM_COLORS.get(atom_type, "#CCCCCC"),
                        symbol="circle",
                        line=dict(color="#000000", width=1),
                    ),
                    name=f"Atom {atom_type}",
                    showlegend=True,
                )
            )
            atom_traces[i] = True

    # Add bonds (using centered coordinates)
    edge_index = molecule_data["edge_index"].cpu().numpy()
    drawn_bonds = set()

    print("\nDrawing bonds:")
    for i in range(edge_index.shape[1]):
        start_idx = edge_index[0, i]
        end_idx = edge_index[1, i]

        bond = tuple(sorted([int(start_idx), int(end_idx)]))
        if bond not in drawn_bonds:
            drawn_bonds.add(bond)
            print(f"Bond {bond}: {atom_types[start_idx]} - {atom_types[end_idx]}")
            fig.add_trace(
                go.Scatter3d(
                    x=[coords[start_idx, 0], coords[end_idx, 0]],
                    y=[coords[start_idx, 1], coords[end_idx, 1]],
                    z=[coords[start_idx, 2], coords[end_idx, 2]],
                    mode="lines",
                    line=dict(color="#000000", width=2),
                    showlegend=False,
                )
            )

    # Update probe points to use normalized values if showing probes
    if show_probes:
        fig.add_trace(
            go.Scatter3d(
                x=np.arange(grid_size).repeat(grid_size * grid_size),
                y=np.tile(np.arange(grid_size).repeat(grid_size), grid_size),
                z=np.tile(np.arange(grid_size), grid_size * grid_size),
                mode="markers",
                marker=dict(
                    size=10,
                    color=values.flatten(),
                    colorscale=colorscale,
                    showscale=True,
                    opacity=0.5,
                    cmin=-2.0,  # Set consistent color range
                    cmax=2.0,
                ),
                name="Probe points",
            )
        )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        title="3D Charge Distribution with Molecular Structure",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.show()


def main(args):
    # Load data
    db_path = args.db_path
    results = np.load(args.results_path)

    # Get number of molecules from metadata length
    n_molecules = len(results["metadata"])
    print(f"Found {n_molecules} molecules in results")

    # Calculate points per molecule (total predictions / number of molecules)
    grid_size = 10  # Fixed grid size from preprocess.py
    points_per_molecule = grid_size * grid_size * grid_size

    # For each molecule
    for i in range(min(args.num_molecules, n_molecules)):
        molecule_name = results["metadata"][i]
        print(f"\nProcessing molecule {i+1}/{args.num_molecules}: {molecule_name}")

        # Get molecule data from LMDB
        molecule_data = load_molecule_from_lmdb(db_path, molecule_name)

        if molecule_data is not None:
            # Get predictions for this molecule using slice
            start_idx = i * points_per_molecule
            end_idx = start_idx + points_per_molecule
            molecule_predictions = results["predictions"][start_idx:end_idx]

            # Create results dict for this molecule
            molecule_results = {
                "predictions": molecule_predictions,
                "metadata": molecule_name,
            }

            print(f"Grid size: {grid_size}x{grid_size}x{grid_size}")
            print(f"Number of predictions: {len(molecule_predictions)}")

            try:
                visualize_molecule_and_charges(
                    molecule_results,
                    molecule_data,
                    iso_min=0.05,
                    iso_max=1.5,
                    n_isosurfaces=6,
                    opacity=0.2,
                    show_probes=args.show_probes,
                )
            except Exception as e:
                print(f"Failed to visualize molecule {molecule_name}: {str(e)}")
        else:
            print(f"Could not find molecule {molecule_name} in database")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--show_probes", type=bool, default=True)
    parser.add_argument("--num_molecules", type=int, default=1)
    args = parser.parse_args()
    main(args)

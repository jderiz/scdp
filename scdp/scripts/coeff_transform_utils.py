"""
Utility functions for analyzing and predicting coefficient transformations.

This module provides functions to predict how orbital coefficients should
transform under rotation and translation transformations.
"""

import numpy as np
import torch
from e3nn import o3
from torch.backends import cuda

from scdp.model.module import ChgLightningModule


def predict_transformed_coeffs_rotation(
    model: ChgLightningModule,
    coeffs_original: torch.Tensor,
    rotation_matrix: np.ndarray,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Predicts how GTO coefficients should transform under a given rotation R,
    assuming the model's output coefficients live in a space defined by
    the irreps of the model's readout layer.

    Args:
        model: The trained ChgLightningModule instance.
        coeffs_original: The coefficients predicted for the original, unrotated
                         molecule. Shape: [num_atoms, output_dim].
        rotation_matrix: A 3x3 numpy array representing the rotation matrix.
        device: The torch device to use for calculations. If None, uses the
                device of coeffs_original.

    Returns:
        torch.Tensor: The predicted coefficients after applying the theoretical
                      transformation corresponding to the rotation.
                      Shape: [num_atoms, output_dim].
    """
    if device is None:
        device = coeffs_original.device

    coeffs_original = coeffs_original.to(device)
    rotation_matrix_torch = torch.tensor(
        rotation_matrix, dtype=torch.float32, device=device
    )

    # --- 1. Get the Irreducible Representations (Irreps) of the output coefficients ---
    # This defines the space the coefficients live in and how they should transform.
    # We get this from the `orbit_readout` layer's output definition in the eSCN model.
    try:
        # Accessing the underlying eSCN model assuming it's stored in model.model
        orbit_readout_layer = model.model.orbit_readout
        irreps_out = orbit_readout_layer.irreps_out
        # Ensure irreps are on the correct device (though usually device-agnostic)
        # irreps_out = o3.Irreps(str(irreps_out)).to(device) # o3.Irreps doesn't have .to()

    except AttributeError:
        # Fallback: Reconstruct irreps based on model hyperparameters if direct access fails
        print(
            "Warning: Could not directly access orbit_readout.irreps_out. Reconstructing from hparams."
        )
        try:
            # Ensure max_n_orbitals_per_L is a tensor on the correct device
            max_n_orbitals_per_L = model.max_n_orbitals_per_L.to(
                device="cpu"
            ).tolist()  # Convert to list for Irreps constructor
            irreps_out = o3.Irreps(
                [
                    (
                        int(n_orb),
                        (l, 1),
                    )  # Assuming parity (+1, scalar) for coefficients
                    for l, n_orb in enumerate(max_n_orbitals_per_L)
                    if n_orb > 0
                ]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get output irreps from model or hparams: {e}"
            )

    if coeffs_original.shape[1] != irreps_out.dim:
        raise ValueError(
            f"Coefficient dimension mismatch: coeffs_original has dim {coeffs_original.shape[1]}, "
            f"but calculated irreps_out dimension is {irreps_out.dim}. "
            "Check model configuration and coefficient extraction."
        )

    # --- 2. Get the Wigner D matrix for the rotation R in the basis of irreps_out ---
    # D is the matrix representation of the rotation R acting on the space defined by irreps_out
    # Shape: [irreps_out.dim, irreps_out.dim]
    D = irreps_out.D_from_matrix(rotation_matrix_torch)

    # --- 3. Apply the transformation ---
    # We want to compute D @ c for each atom's coefficient vector c.
    # coeffs_original is [num_atoms, irreps_out.dim]
    # D is [irreps_out.dim, irreps_out.dim]
    # We can use einsum: C'_ai = D_ij * C_aj -> C'_ai = C_aj * D_ji
    # Or simply matrix multiply D with the transpose of coeffs_original and transpose back
    # coeffs_predicted_rotated = (D @ coeffs_original.T).T
    # Or using einsum (often clearer): C'_ai = Sum_j D_ij C_aj
    coeffs_predicted_rotated = torch.einsum("ij,aj->ai", D, coeffs_original)

    return coeffs_predicted_rotated


def extract_l_mapping(model, atom_types):
    """
    Extract mapping from coefficient indices to L values from the model.

    This function maps each coefficient to its corresponding angular momentum (L) value,
    which is essential for understanding which types of orbitals (s, p, d, f, etc.)
    are more invariant under rotations or translations.

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


def predict_transformed_coeffs_translation(
    model: ChgLightningModule,
    coeffs_original: torch.Tensor,
    translation_vector: np.ndarray,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Predicts how GTO coefficients should transform under a given translation.
    For true GTO orbitals, coefficients should be invariant under translation.

    Args:
        model: The trained ChgLightningModule instance.
        coeffs_original: The coefficients predicted for the original
                         molecule. Shape: [num_atoms, output_dim].
        translation_vector: A 3-element numpy array representing the translation vector.
        device: The torch device to use for calculations. If None, uses the
                device of coeffs_original.

    Returns:
        torch.Tensor: The predicted coefficients after applying the theoretical
                      transformation corresponding to the translation, which should
                      be identical to the original coefficients for a perfect GTO model.
                      Shape: [num_atoms, output_dim].
    """
    # For true GTO basis functions, orbital coefficients are invariant under translation
    # So theoretically we should just return the original coefficients
    return coeffs_original.clone()


def compare_transformations(
    theoretical_coeffs: torch.Tensor,
    actual_coeffs: torch.Tensor,
    atom_types: torch.Tensor,
    coeff_to_l_mapping: dict,
    transformation_name: str = "Transformation",
):
    """
    Compare theoretically predicted coefficients with actual coefficients
    after a transformation and analyze the results.

    Args:
        theoretical_coeffs: Theoretically predicted coefficients. Shape: [num_atoms, output_dim]
        actual_coeffs: Actual coefficients from the model. Shape: [num_atoms, output_dim]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        transformation_name: Name of the transformation for reporting

    Returns:
        Tuple of:
            - close_mask: Boolean tensor indicating which coefficients are close. Shape: [num_atoms, output_dim]
            - agreement_percentage: Percentage of coefficients that match between theoretical and actual
            - mean_rel_diff: Mean relative difference between theoretical and actual coefficients
            - max_rel_diff: Maximum relative difference between theoretical and actual coefficients
    """
    # Check if coefficients are close within a tolerance
    close_mask = torch.isclose(theoretical_coeffs, actual_coeffs, atol=1e-2, rtol=1e-2)

    # Calculate overall agreement percentage
    agreement_percentage = 100 * close_mask.sum().item() / close_mask.numel()

    # Calculate relative differences
    abs_diff = torch.abs(theoretical_coeffs - actual_coeffs)
    epsilon = 1e-10  # Small constant to prevent division by zero
    rel_diff = abs_diff / (torch.abs(actual_coeffs) + epsilon)

    # Calculate summary statistics
    mean_rel_diff = rel_diff.mean().item()
    max_rel_diff = rel_diff.max().item()

    # Print summary
    print(f"\n{transformation_name} Comparison Summary:")
    print(
        f"Agreement: {agreement_percentage:.2f}% of coefficients match within tolerance"
    )
    print(f"Mean relative difference: {mean_rel_diff:.4e}")
    print(f"Max relative difference: {max_rel_diff:.4e}")

    # Analyze by L value
    l_value_stats = {}

    for i in range(len(atom_types)):
        atom_type = atom_types[i].item()
        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        for j, l in enumerate(l_values):
            if j >= close_mask.shape[1]:
                continue

            if l not in l_value_stats:
                l_value_stats[l] = {"total": 0, "close": 0}

            l_value_stats[l]["total"] += 1
            if close_mask[i, j]:
                l_value_stats[l]["close"] += 1

    # Print L value analysis
    print(f"\n{transformation_name} Agreement by Angular Momentum (L):")
    for l, stats in sorted(l_value_stats.items()):
        percentage = 100 * stats["close"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"L={l}: {stats['close']}/{stats['total']} ({percentage:.2f}%) coefficients match"
        )

    return close_mask, agreement_percentage, mean_rel_diff, max_rel_diff, l_value_stats

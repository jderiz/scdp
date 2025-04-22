from typing import Optional, Tuple

import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R_sci

from scdp.data.data import AtomicData
from scdp.model.module import ChgLightningModule


def predict_transformed_coeffs_rotation(
    model: ChgLightningModule,
    coeffs_original: torch.Tensor,
    rotation_matrix: np.ndarray,
    device: Optional[torch.device] = None,
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
    # Or using einsum: C'_ai = Sum_j D_ij C_aj
    coeffs_predicted_rotated = torch.einsum("ij,aj->ai", D, coeffs_original)

    return coeffs_predicted_rotated


# --- Example Usage ---
# Assuming you have:
# model: Your loaded ChgLightningModule
# original_data: Your AtomicData for the unrotated molecule
# rot_matrix: A 3x3 numpy rotation matrix

# 1. Get coefficients for the original molecule
# model.eval()
# with torch.no_grad():
#     coeffs_original, _ = model.predict_coeffs(original_data.to(model.device))

# 2. Predict how they *should* transform
# coeffs_theo_rotated = predict_transformed_coeffs_rotation(
#     model=model,
#     coeffs_original=coeffs_original,
#     rotation_matrix=rot_matrix
# )

# 3. Get coefficients for the *actually* rotated molecule (from the previous function/analysis script)
# rotated_data = ... # create rotated AtomicData
# with torch.no_grad():
#     coeffs_actual_rotated, _ = model.predict_coeffs(rotated_data.to(model.device))

# 4. Now compare the theoretical prediction with the actual model output
# close_mask = torch.isclose(coeffs_theo_rotated, coeffs_actual_rotated, atol=1e-2, rtol=1e-2)
# print(f"Comparison between theoretical and actual rotated coeffs:")
# print(f"{close_mask.sum() / close_mask.numel() * 100:.2f}% match within tolerance.")

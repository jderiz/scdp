from typing import Optional, Tuple

import numpy as np
import torch
from e3nn import o3  # Still useful for defining Irreps if needed
from scipy.spatial.transform import Rotation as R_sci

# Assuming necessary imports from the scdp project are available
from scdp.data.data import AtomicData
from scdp.model.module import ChgLightningModule
from scdp.model.scn.so3 import SO3_Rotation  # Key class for this transformation


def predict_transformed_embedding_rotation(
    model: ChgLightningModule,
    embedding_original: torch.Tensor,
    rotation_matrix: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Predicts how the internal equivariant embedding (e.g., x.embedding)
    should transform under a given rotation R using Wigner D-matrices via
    the SO3_Rotation class.

    Args:
        model: The trained ChgLightningModule instance (used to get model params).
        embedding_original: The internal embedding tensor predicted for the
                            original, unrotated molecule. Expected shape like
                            [num_atoms, num_internal_coeffs, sphere_channels],
                            where num_internal_coeffs corresponds to the sum
                            of (2l+1) for the relevant L values.
        rotation_matrix: A 3x3 numpy array representing the rotation matrix.
        device: The torch device to use for calculations. If None, uses the
                device of embedding_original.

    Returns:
        torch.Tensor: The predicted embedding after applying the theoretical
                      transformation corresponding to the rotation.
                      Shape matches embedding_original.
    """
    if device is None:
        device = embedding_original.device

    embedding_original = embedding_original.to(device)
    rotation_matrix_torch = torch.tensor(
        rotation_matrix, dtype=torch.float32, device=device
    )

    # --- 1. Determine Parameters for Rotation ---
    try:
        # Get lmax and number of channels from the model's hyperparameters
        # Assuming the internal embedding uses the main lmax_list from config
        lmax_list = model.hparams.model.lmax_list
        sphere_channels = model.hparams.model.sphere_channels
        internal_lmax = max(lmax_list)  # Max L used in the internal representation

        # Infer num_internal_coeffs based on lmax
        # Sum of (2l+1) for l from 0 to internal_lmax
        num_internal_coeffs = (internal_lmax + 1) ** 2

        # Get num_atoms and ensure channels match
        num_atoms, n_coeffs_in = embedding_original.shape

        if n_coeffs_in != num_internal_coeffs:
            raise ValueError(
                f"Input embedding dimension mismatch: embedding has {n_coeffs_in} coefficients, "
                f"but expected {(internal_lmax + 1)**2} based on model lmax={internal_lmax}."
            )
        # Note: sphere_channels in the config might be per resolution.
        # The actual channel dimension might be sphere_channels * num_resolutions.
        # Let's assume embedding_original already has the correct channel dimension.
        # expected_channels = sphere_channels * len(lmax_list) # Need to confirm this structure
        # if n_channels_in != expected_channels:
        #     raise ValueError(
        #          f"Input embedding channel mismatch: embedding has {n_channels_in} channels, "
        #          f"but expected {expected_channels} based on model config."
        #     )

    except AttributeError as e:
        raise RuntimeError(
            f"Failed to get necessary parameters (lmax_list, sphere_channels) from model hparams: {e}"
        )
    except ValueError as e:
        raise ValueError(
            f"Input embedding shape {embedding_original.shape} seems incompatible with model parameters: {e}"
        )

    # --- 2. Instantiate SO3_Rotation ---
    # SO3_Rotation expects a batch of rotation matrices [batch_size, 3, 3].
    # Here, we apply the *same* rotation to all atoms, treating atoms as the batch dim.
    # However, SO3_Rotation is designed for rotating *one* set of features per rotation matrix.
    # We need to apply it atom-wise or adapt. Let's apply it atom-wise for clarity.

    # Alternative: Instantiate once if SO3_Rotation handles batch dimension correctly for the embedding.
    # If SO3_Rotation expects [1, 3, 3] and applies it to [N_atoms, N_coeffs, N_channels],
    # we might not need a loop. Let's try that first based on SO3_Rotation.rotate signature.
    # It takes embedding[batch, coeffs, channels].

    # We treat num_atoms as the batch dimension.
    # SO3_Rotation needs the rotation matrix for the *batch*. Since it's the same rotation
    # for all atoms, we provide it once. The internal Wigner matrix will have batch dim 1.
    # The BMM in rotate might broadcast correctly? Let's assume it does based on e3nn patterns.

    # Need rotation matrix shape [1, 3, 3] for SO3_Rotation init? Let's check SO3_Rotation code.
    # It seems it takes [N_edges, 3, 3]. Let's prepare a single rotation matrix for it.
    rot_mat_batch = rotation_matrix_torch.unsqueeze(0)  # Shape [1, 3, 3]
    so3_rot = SO3_Rotation(rot_mat_batch, internal_lmax)

    # --- 3. Apply the transformation using SO3_Rotation.rotate ---
    # The rotate method applies D(R) to the embedding.
    # Signature: rotate(self, embedding, out_lmax, out_mmax)
    # We want to rotate all components up to internal_lmax.
    # Input embedding: [num_atoms, num_internal_coeffs, sphere_channels]

    # We need to pass the embedding for *one* rotation matrix.
    # If so3_rot.wigner is [1, N_coeffs, N_coeffs], and embedding is [N_atoms, N_coeffs, N_channels],
    # torch.bmm might not work directly as it expects [B, N, M] @ [B, M, P].
    # Let's look at SO3_Rotation.rotate implementation: torch.bmm(wigner, embedding)
    # It seems it expects embedding to be [batch_size, N_coeffs, N_channels] where batch_size matches
    # the batch size of the rotation matrices used to initialize SO3_Rotation.

    # Okay, let's reshape and loop if necessary, or try broadcasting.
    # Simplest might be to process each atom individually if broadcasting is unclear.

    embedding_predicted_rotated_list = []
    for i in range(num_atoms):
        atom_embedding = embedding_original[
            i : i + 1, :, :
        ]  # Shape [1, N_coeffs, N_channels]
        # Apply the rotation defined for batch index 0 (the only rotation we have)
        # We need the Wigner matrix for this rotation
        wigner_matrix = so3_rot.wigner[0]  # Shape [N_coeffs, N_coeffs]
        # Apply manually: D @ emb^T -> transpose result
        rotated_atom_embedding = torch.matmul(
            wigner_matrix, atom_embedding.squeeze(0).T
        ).T.unsqueeze(0)
        # Check if SO3_Rotation.rotate does this:
        # It does bmm(wigner, embedding). wigner needs shape [B, N, N], embedding [B, N, C]
        # Our wigner is [1, N, N]. Our embedding is [N_atoms, N, C].
        # We can try unsqueezing wigner and using bmm.
        # rotated_embedding = torch.bmm(so3_rot.wigner.expand(num_atoms, -1, -1), embedding_original) # Seems wrong

        # Let's stick to the clearer application per atom if SO3_Rotation.rotate isn't directly usable.
        # The code above `rotated_atom_embedding = ...` manually applies the Wigner matrix.
        embedding_predicted_rotated_list.append(rotated_atom_embedding.squeeze(0))

    embedding_predicted_rotated = torch.stack(embedding_predicted_rotated_list, dim=0)

    # --- Sanity Check Shape ---
    if embedding_predicted_rotated.shape != embedding_original.shape:
        print(
            f"Warning: Output shape {embedding_predicted_rotated.shape} "
            f"differs from input shape {embedding_original.shape}"
        )

    return embedding_predicted_rotated


# --- Example Usage ---
# Assuming you have:
# model: Your loaded ChgLightningModule
# original_data: Your AtomicData for the unrotated molecule
# rot_matrix: A 3x3 numpy rotation matrix
# embedding_original: The tensor obtained from your modified predict_coeffs

# 1. Predict how the embedding *should* transform
# embedding_theo_rotated = predict_transformed_embedding_rotation(
#     model=model,
#     embedding_original=embedding_original, # Shape [N_atoms, N_coeffs, N_channels]
#     rotation_matrix=rot_matrix
# )

# 2. Get embedding for the *actually* rotated molecule
# rotated_data = ... # create rotated AtomicData
# with torch.no_grad():
#     _, embedding_actual_rotated = model.predict_coeffs(rotated_data.to(model.device)) # Assuming modified predict_coeffs

# 3. Now compare the theoretical prediction with the actual model output
# close_mask_emb = torch.isclose(embedding_theo_rotated, embedding_actual_rotated, atol=1e-2, rtol=1e-2)
# print(f"Comparison between theoretical and actual rotated embeddings:")
# print(f"{close_mask_emb.sum() / close_mask_emb.numel() * 100:.2f}% match within tolerance.")

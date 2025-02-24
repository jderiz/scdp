import os
import sys
from torch_geometric.datasets import QM9
import argparse
import torch
import torch_geometric
from deepchem.molnet import load_qm9

print("Python version:", sys.version)
print("Importing torch...")

print("Torch version:", torch.__version__)

print("Importing torch_geometric...")

print("Torch geometric version:", torch_geometric.__version__)

# Force CPU only for downloading
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_qm9_dataset(root: str) -> QM9:
    """Load the QM9 dataset from the specified root directory.

    Args:
        root (str): The directory where the dataset will be stored.

    Returns:
        QM9: The loaded QM9 dataset.
    """
    print(f"Attempting to load QM9 dataset to: {root}")
    try:
        dataset = QM9(root)
        print("Dataset loaded successfully")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=os.environ.get("DATA_HOME", "./data") + "/qm9",
    )
    args = parser.parse_args()

    try:
        dataset = load_qm9_dataset(args.out_dir)
        print(dataset)
        print("Load sdf files using deepchem")
        # Load the QM9 dataset
        qm9_tasks, qm9_datasets, qm9_splits = load_qm9(
            data_dir=f"{args.out_dir}/qm9_data"
        )
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        raise

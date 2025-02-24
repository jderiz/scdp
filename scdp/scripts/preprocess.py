from ast import Break
from pathlib import Path
from ase.io.x3d import atom_lines
import lz4, gzip, zlib
import lz4.frame
import lzma
import os
import pickle
import tarfile
import argparse
import multiprocessing as mp

import lmdb
import numpy as np
from pyrho import charge_density
import torch
from tqdm import tqdm

from scdp.data import vnode
from scdp.data.data import AtomicData, AtomicNumberTable
from scdp.data.utils import calculate_grid_pos


def get_atomic_number_table_from_zs(zs) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def decompress_tarmember(tar, tarinfo):
    """Extract compressed tar file member and return a bytes object with the content"""
    bytesobj = tar.extractfile(tarinfo).read()
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(bytesobj)
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(bytesobj)
    elif tarinfo.name.endswith(".gz"):
        filecontent = gzip.decompress(bytesobj)
    elif tarinfo.name.endswith(".xz"):
        filecontent = lzma.decompress(bytesobj)
    else:
        filecontent = bytesobj

    return filecontent


def process_tar_file_to_lmdb(mp_arg):
    z_table, tarpath, db_path, pid, idx, args = mp_arg
    n_device = torch.cuda.device_count()
    device = f"cuda:{pid % n_device}" if args.device == "cuda" else "cpu"
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 16,  # 16GB
        subdir=False,
        meminit=False,
        map_async=True,
    )

    member_list = []
    read_cfg = "r:gz" if tarpath.parts[-1].endswith(".gz") else "r:"
    with tarfile.open(tarpath, read_cfg) as tar:
        for member in tqdm(
            tar.getmembers(), position=pid, desc=f"processing job {pid} on {device}"
        ):
            member_list.append(member)

            filecontent = decompress_tarmember(tar, member)
            fileinfo = member
            data_object = AtomicData.from_file(
                fcontent=filecontent,
                finfo=fileinfo,
                build_method=args.build_method,
                z_table=z_table,
                atom_cutoff=args.atom_cutoff,
                probe_cutoff=args.probe_cutoff,
                vnode_method=args.vnode_method,
                vnode_factor=args.vnode_factor,
                vnode_res=args.vnode_res,
                disable_pbc=args.disable_pbc,
                max_neighbors=args.max_neighbors,
                device=device,
            )

            txn = db.begin(write=True)
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            idx += 1

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return idx


def convert_dict_to_atomic(
    molecule_dict: dict,
    device: str = "cpu",
) -> AtomicData:
    """Convert dictionary of molecule data to AtomicData format."""
    try:
        # Get data from molecule dict
        edge_index = molecule_dict["edge_index"].to(device)
        atom_coords = molecule_dict["pos"].to(device)
        atom_types = molecule_dict["z"].to(device)
        node_attrs = molecule_dict["x"].to(device)
        n_atom = len(atom_coords)

        # Create cell tensor based on molecule size
        # Get molecule bounds with some padding
        min_coords = atom_coords.min(dim=0)[0]
        max_coords = atom_coords.max(dim=0)[0]
        molecule_size = max_coords - min_coords
        padding = molecule_size.max() * 0.2  # 20% padding
        cell_size = molecule_size + padding

        # Create diagonal cell tensor that encompasses the molecule
        cell = torch.diag(cell_size).to(device).unsqueeze(0)

        # Center molecule in cell
        molecule_center = (max_coords + min_coords) / 2
        atom_coords = atom_coords - molecule_center + cell_size / 2

        shifts = torch.zeros((edge_index.shape[1], 3), device=device)
        unit_shifts = torch.zeros_like(shifts, dtype=torch.long, device=device)

        # Create a grid for probe points (20x20x20 is a good starting point)
        grid_size = [10, 10, 10]  # Increased resolution from 10x10x10
        origin = torch.zeros(3, device=device)
        n_probe = grid_size[0] * grid_size[1] * grid_size[2]

        # Create dummy density tensor for grid calculation
        density_dummy = torch.zeros(grid_size, device=device)
        probe_coords = calculate_grid_pos(density_dummy, cell, origin).view(-1, 3)

        z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())
        n_vnode = 0
        atomic_data_dict = {
            "edge_index": edge_index,
            "coords": atom_coords,
            "shifts": shifts,
            "z_table": z_table,
            "atom_cutoff": args.atom_cutoff,
            "probe_cutoff": args.probe_cutoff,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "atom_types": atom_types,
            "node_attrs": node_attrs,
            "n_atom": n_atom,
            "num_nodes": n_atom,
            "vnode_method": "bond",
            "n_vnode": n_vnode,
            "n_probe": n_probe,
            "chg_density": density_dummy,
            "probe_coords": probe_coords,
            "metadata": str(molecule_dict["idx"].item()),
            "build_method": "probe",
        }

        return AtomicData.from_dict(atomic_data_dict, device=device)

    except Exception as e:
        print(f"\nFull error details for molecule:")
        print(f"Molecule data: {molecule_dict}")
        raise e


def process_pt_file_to_lmdb(mp_arg):
    pt_file, db_path, pid, args, start_idx, end_idx = mp_arg
    device = (
        f"cuda:{pid % torch.cuda.device_count()}" if args.device == "cuda" else "cpu"
    )

    # Load the pre-made graphs from the .pt file
    print(f"Loading graphs from {pt_file} for molecules {start_idx} to {end_idx}")
    graphs = torch.load(pt_file, map_location=device)

    # Extract cumulative counts and data
    edge_cumulative_counts = graphs[1]["edge_index"][start_idx : end_idx + 1]
    node_cumulative_counts = graphs[1]["x"][start_idx : end_idx + 1]
    y_cumulative_counts = graphs[1]["y"][start_idx : end_idx + 1]

    # Initialize LMDB
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 16,  # 16GB
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Process molecules
    idx = 0
    edge_start = 0
    node_start = 0
    y_start = 0

    with db.begin(write=True) as txn:
        for i in tqdm(
            range(1, len(edge_cumulative_counts)),  # Skip first empty molecule
            position=pid,
            desc=f"processing job {pid} on {device}",
        ):
            if i < start_idx or i >= end_idx:
                continue

            edge_end = edge_cumulative_counts[i].item()
            node_end = node_cumulative_counts[i].item()
            y_end = y_cumulative_counts[i].item()

            # Create molecule dict with sliced data
            molecule_dict = {
                "x": graphs[0]["x"][node_start:node_end],
                "edge_index": graphs[0]["edge_index"][:, edge_start:edge_end],
                "edge_attr": graphs[0]["edge_attr"][edge_start:edge_end],
                "y": graphs[0]["y"][y_start:y_end],
                "pos": graphs[0]["pos"][node_start:node_end],
                "z": graphs[0]["z"][node_start:node_end],
                "idx": graphs[0]["idx"][i : i + 1],
            }

            try:
                graph = convert_dict_to_atomic(
                    molecule_dict,
                    device=device,
                )
                graph_cpu = graph.cpu()
                txn.put(
                    f"{idx}".encode("ascii"),
                    pickle.dumps(graph_cpu, protocol=-1),
                )
                idx += 1
            except Exception as e:
                print(f"\nError processing molecule {i}:")
                print(str(e))
                raise

            edge_start = edge_end
            node_start = node_end
            y_start = y_end

        txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))

    db.sync()
    db.close()
    return idx


def main_tar(args: argparse.Namespace) -> None:
    tar_files = sorted(list(Path(args.data_path).glob("*.tar*")))
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i) for i in range(len(tar_files))
    ]

    z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            z_table,
            tar_files[i],
            db_paths[i],
            i,
            0,
            args,
        )
        for i in range(len(tar_files))
    ]
    list([*pool.imap(process_tar_file_to_lmdb, mp_args)])


def main_pt(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # For single .pt file, process in chunks using multiple workers
    if data_path.suffix == ".pt":
        # Load data to get total number of molecules
        data = torch.load(data_path, map_location="cpu")
        n_molecules = len(data[0]["idx"])
        # Limit number of molecules if specified
        if args.max_molecules is not None:
            n_molecules = min(n_molecules, args.max_molecules)
            print(f"Processing only first {n_molecules} molecules")

        chunk_size = n_molecules // args.num_workers
        del data  # Free memory

        # Create LMDB paths for each chunk
        db_paths = [
            os.path.join(args.out_path, f"data.{i:04d}.lmdb")
            for i in range(args.num_workers)
        ]

        if args.num_workers == 1:
            # Process without multiprocessing
            process_pt_file_to_lmdb((data_path, db_paths[0], 0, args, 0, n_molecules))
            total_graphs = n_molecules
        else:
            # Create pool and process in parallel
            pool = mp.Pool(args.num_workers)
            mp_args = []

            # Divide molecules among workers
            for i in range(args.num_workers):
                start_idx = i * chunk_size
                end_idx = (
                    start_idx + chunk_size if i < args.num_workers - 1 else n_molecules
                )
                mp_args.append(
                    (
                        data_path,
                        db_paths[i],
                        i,
                        args,
                        start_idx,
                        end_idx,
                    )
                )

            results = list(pool.imap(process_pt_file_to_lmdb, mp_args))
            total_graphs = sum(results)

        print(
            f"Successfully processed {total_graphs} graphs across {args.num_workers} chunks"
        )
    else:
        raise ValueError("Expected a single .pt file containing graphs")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="path to charge density tar files",
    )
    parser.add_argument(
        "--out_path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num_dbs",
        type=int,
        default=32,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="No. of processes to use for feature extraction",
    )
    parser.add_argument(
        "--build_method",
        type=str,
        default="vnode",
        choices=["vnode", "probe"],
        help="Method to use for building graphs",
    )
    parser.add_argument(
        "--atom_cutoff",
        type=float,
        default=4.0,
        help="Cutoff radius for atom graph",
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=None,
        help="Max number of neighbors for each atom in the graph",
    )
    parser.add_argument(
        "--probe_cutoff",
        type=float,
        default=4.0,
        help="Cutoff radius for probe edges",
    )
    parser.add_argument(
        "--vnode_method",
        type=str,
        default="bond",
        choices=["bond", "none"],
        help="Method to use for virtual node generation",
    )
    parser.add_argument(
        "--vnode_factor",
        type=int,
        default=3,
        help="Maximum number of iterations for virtual node generation as a factor of number of atoms",
    )
    parser.add_argument(
        "--vnode_res",
        type=float,
        default=0.8,
        help="vnode resolution in the obb method",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for feature extraction",
    )
    parser.add_argument(
        "--disable_pbc",
        action="store_true",
        help="Disable periodic boundary conditions",
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="tar",
        choices=["tar", "pt"],
        help="Type of files to process: tar archives or PyTorch .pt files",
    )
    parser.add_argument(
        "--max_molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to process. If None, process all molecules",
    )
    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()
    if args.file_type == "tar":
        main_tar(args)
    elif args.file_type == "pt":
        main_pt(args)
    else:
        raise NotImplementedError

import logging
import time
import json
import argparse
import contextlib
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm

import omegaconf
import torch
from lightning.pytorch import seed_everything

from torch.utils.data import Subset
from scdp.common.pyg import DataLoader
from scdp.data.dataset import LmdbDataset
from scdp.data.datamodule import worker_init_fn
from scdp.model.utils import get_nmape
from scdp.model.module import ChgLightningModule
from scdp.data.data import AtomicData

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")  # H200 sepeedup with medium


def get_data_probe_chunk(input_data, indices):
    data = deepcopy(input_data)
    data["chg_labels"] = input_data.chg_labels[indices]
    data["probe_coords"] = input_data.probe_coords[indices]
    data["n_probe"] = len(indices)
    data["sampled"] = True
    return data


def get_probe_chunks(n_probes, max_n_probe_per_pass):
    batch_size = len(n_probes)
    n_per_pass = []
    probes_to_process = []
    n_probes = torch.clone(n_probes)
    probe_indices = torch.arange(n_probes.sum(), device=n_probes.device)

    n_pass = ((n_probes).sum() / max_n_probe_per_pass).ceil().long()
    pass_start_index = 0
    pass_end_index = 0
    for _ in range(n_pass):
        current_pass = torch.zeros(batch_size, dtype=torch.long, device=n_probes.device)
        current_load = 0

        for i in range(batch_size):
            if n_probes[i] == 0:
                continue
            max_points_for_job = max_n_probe_per_pass - current_load
            points_to_process = torch.min(
                torch.tensor([n_probes[i], max_points_for_job])
            )

            current_load += points_to_process
            current_pass[i] = points_to_process
            pass_end_index += int(points_to_process)

            # in-place modification done last
            n_probes[i] -= points_to_process

            if current_load >= max_n_probe_per_pass:
                break
        n_per_pass.append(current_pass)
        probes_to_process.append(probe_indices[pass_start_index:pass_end_index])
        pass_start_index = pass_end_index
        pass_end_index = pass_start_index

    return n_pass, n_per_pass, probes_to_process


def main(
    ckpt_path,
    data_path,
    split_file,
    tag="test",
    max_n_graphs=10000,
    batch_size=4,
    max_n_probe=500000,
    use_last=False,
    save_outputs=None,
):
    seed_everything(42)
    ckpt_path = Path(ckpt_path)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=ckpt_path / f"eval_{tag}.log",
        filemode="w",
    )

    cfg = omegaconf.OmegaConf.load(ckpt_path / "config.yaml")

    if data_path.suffix == ".pt":
        graphs = torch.load(data_path, map_location="cpu")
        dataset = graphs[2](graphs[0])
        print("type of dataset", type(dataset))
        dataset = AtomicData.from_torch_geometric_data(dataset)
    # set up data loader if lmdb
    else:
        print("using LmdbDataset")
        dataset = LmdbDataset(data_path)
        print("type of dataset", type(dataset))
        print("length of dataset", len(dataset))

    # with open(split_file, "r") as fp:
    #     splits = json.load(fp)
    # test_dataset = Subset(dataset, splits["test"])

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init_fn,
    )

    print("\nDataset info:")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Batch size: {batch_size}")

    # Sample first batch to inspect
    sample_batch = next(iter(test_loader))
    print("\nSample batch info:")
    print(f"Available attributes: {sample_batch.__dict__.keys()}")
    print(f"Number of atoms: {len(sample_batch.atom_types)}")
    print(f"Edge index shape: {sample_batch.edge_index.shape}")
    print(f"Positions shape: {sample_batch.coords.shape}")
    print(f"Atom types: {sample_batch.atom_types}")

    # load checkpoint
    storage_dir: str = Path(ckpt_path)
    if (storage_dir / "last.ckpt").exists() and use_last:
        ckpt = storage_dir / "last.ckpt"
    else:
        ckpts = list(storage_dir.glob("*epoch*.ckpt"))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
            )
            ckpt_ix = ckpt_epochs.argsort()[-1]
            ckpt = str(ckpts[ckpt_ix])
        else:
            raise FileNotFoundError(f"No checkpoint found in <{ckpt_path}>.")

    # pylint: disable=E1120
    pylogger.info(f"loaded checkpoint: {ckpt}")
    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=ckpt).to("cuda")
    model.eval()
    model.ema.copy_to(model.parameters())

    # Better load balancing possible.
    pylogger.info("starting testing.")
    with (
        torch.no_grad(),
        contextlib.ExitStack() as context_stack,
        open(storage_dir / f"nmape_{tag}.txt", "w") as f,
    ):
        all_coeffs = []
        all_expo_scaling = []
        all_predictions = []
        all_batch_idx = []

        prog = context_stack.enter_context(
            tqdm(total=min(max_n_graphs, len(test_loader)), disable=None)
        )
        display_bar = context_stack.enter_context(
            tqdm(
                bar_format=(
                    "" if prog.disable else ("{desc:." + str(prog.ncols) + "}")
                ),
                disable=None,
            )
        )

        curr_time = time.time()

        idx = 0
        # all_nmapes = []
        all_metadata = []
        try:
            for batch in test_loader:
                batch = batch.to("cuda")
                coeffs, expo_scaling = model.predict_coeffs(batch)
                all_metadata.append(batch.metadata)  # list of string gdb_1 ... gdb_N

                # Store coefficients and scaling
                all_coeffs.append(coeffs.cpu().numpy())
                # Handle case where expo_scaling is None
                if expo_scaling is not None:
                    all_expo_scaling.append(expo_scaling.cpu().numpy())
                else:
                    # Create zero array of appropriate shape
                    zero_scaling = np.zeros((coeffs.shape[0], model.max_n_Ls))
                    all_expo_scaling.append(zero_scaling)

                n_pass, n_per_pass, probes_to_process = get_probe_chunks(
                    batch.n_probe, max_n_probe
                )
                all_preds = []
                for i_pass in range(n_pass):
                    n_probe = n_per_pass[i_pass]
                    probe_idx = probes_to_process[i_pass]
                    probe_coords = batch.probe_coords[probe_idx]
                    pred = model.orbital_inference(
                        batch, coeffs, expo_scaling, n_probe, probe_coords
                    )
                    all_preds.append(pred)
                all_preds = torch.cat(all_preds, dim=0)

                # Store predictions and corresponding batch indices
                all_predictions.append(all_preds.cpu().numpy())
                all_batch_idx.append(
                    torch.arange(len(batch), device=all_preds.device)
                    .repeat_interleave(batch.n_probe)
                    .cpu()
                    .numpy()
                )
                idx += batch.num_graphs
                if idx >= max_n_graphs:
                    break
        except Exception as e:
            print(e)

            print("Error in batch: batch.num_graphs", batch.num_graphs)
            print("Error in batch: batch.n_probe", batch.n_probe)
            print("Error in batch: batch.n_atom", batch.n_atom)
            # print("Error in batch: batch.n_vnode", batch.n_vnode)
            print("Error in batch: batch.atom_types", batch.atom_types)
            print("Error in batch: batch.coords", batch.coords)
            print("Error in batch: batch.edge_index", batch.edge_index)
            print(sample_batch.__dict__)
            raise e
            # nmape = (
            #     get_nmape(
            #         all_preds,
            #         batch.chg_labels,
            #         torch.arange(len(batch), device=all_preds.device).repeat_interleave(
            #             batch.n_probe
            #         ),
            #     )
            #     .cpu()
            #     .numpy()
            #     .tolist()
            # )
            # all_nmapes.extend(nmape)

            # for item in nmape:
            #     f.write(f"{item}\n")
            # f.flush()
            # prog.update(batch.num_graphs)
            # display_bar.set_description_str(
            #     f"nmape: {np.mean(all_nmapes):.4f} ± {np.std(all_nmapes):.4f}"
            # )

        elapsed_time = time.time() - curr_time
        pylogger.info(f"elapsed time: {elapsed_time:.2f} seconds.")

        # Save all outputs if path provided
        if save_outputs:
            np.savez(
                save_outputs,
                coefficients=np.concatenate(all_coeffs),
                expo_scaling=np.concatenate(all_expo_scaling),
                predictions=np.concatenate(all_predictions),
                batch_indices=np.concatenate(all_batch_idx),
                metadata=np.concatenate(all_metadata),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        help="Path to the split file.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="test",
        help="split to use",
    )
    parser.add_argument(
        "--max_n_graphs",
        type=int,
        default=10000,
        help="max number of data points to do inference for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_n_probe",
        type=int,
        default=180000,
        help="max number of probes to process in one pass.",
    )
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use the last checkpoint.",
    )
    parser.add_argument(
        "--save_outputs",
        type=Path,
        help="Path to save the predictions and coefficients (.npz format)",
        default=None,
    )
    args: argparse.Namespace = parser.parse_args()
    main(
        args.ckpt_path,
        args.data_path,
        args.split_file,
        args.tag,
        args.max_n_graphs,
        args.batch_size,
        args.max_n_probe,
        args.use_last,
        args.save_outputs,
    )

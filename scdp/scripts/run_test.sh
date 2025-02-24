#!/bin/bash

# cleanup old files
echo "Cleaning up old files..."
rm -rf /media/drive/qm9/lmdb_test

echo "Preprocessing data..."
python scdp/scripts/preprocess.py \
    --disable_pbc \
    --data_path "/media/drive/qm9/processed/data_v3.pt" \
    --out_path "/media/drive/qm9/lmdb_test" \
    --file_type "pt" \
    --device "cpu" \
    --atom_cutoff 6 \
    --vnode_method "bond" \
    --num_workers 1 \
    --max_molecules 10

if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

echo "Testing model..."
python scdp/scripts/test.py \
    --data_path "/media/drive/qm9/lmdb_test" \
    --max_n_graphs 10 \
    --max_n_probe 400000 \
    --ckpt_path "/media/drive/scdp/share_models/qm9_none_K4L3_beta_2.0" \
    --save_outputs "out.npz" \
    --batch_size 1

if [ $? -ne 0 ]; then
    echo "Testing failed!"
    exit 1
fi

echo "Visualizing results..."
python scdp/scripts/viz.py \
    --db_path "/media/drive/qm9/lmdb_test/data.0000.lmdb" \
    --results_path "out.npz" \
    --show_probes True \
    --num_molecules 1

if [ $? -ne 0 ]; then
    echo "Visualization failed!"
    exit 1
fi

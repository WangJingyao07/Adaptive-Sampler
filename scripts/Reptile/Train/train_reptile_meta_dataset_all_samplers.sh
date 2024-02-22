#!/bin/bash

source ../env/bin/activate
ulimit -n 50000

cp -r /network/projects/r/ramnath.kumar/meta_dataset/records $SLURM_TMPDIR
cp -r ../data/meta_dataset/records $SLURM_TMPDIR

echo "Finished moving data"

cd .. && python -m src.main --exp_name reptile_meta_dataset --train --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --folder $SLURM_TMPDIR/records --dataset meta_dataset --num-ways 5 --num-shots 1 --use-cuda --num-steps 10 --step-size 0.33 --lr 0.01 --batch-size 16 --num-workers 0 --num-epochs 50 --meta-lr 0.0005 --output-folder ./config/reptile_meta_dataset/$SLURM_ARRAY_TASK_ID/

rm -rf $SLURM_TMPDIR/records

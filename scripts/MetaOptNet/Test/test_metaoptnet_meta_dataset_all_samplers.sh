#!/bin/bash

source ../env/bin/activate
ulimit -n 50000

cp -r <PATH_TO_DATA> $SLURM_TMPDIR

echo "Finished moving data"

cd .. && python -m src.main --exp_name test_metaoptnet_meta_dataset --log-test-tasks --model metaoptnet --runs 1 --folder $SLURM_TMPDIR/records --meta-lr 0.1 --task_sampler $SLURM_ARRAY_TASK_ID --dataset meta_dataset --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 1 --num-workers 0 --num-epochs 60 --output-folder ./config/metaoptnet_meta_dataset/$SLURM_ARRAY_TASK_ID/

rm -rf $SLURM_TMPDIR/records

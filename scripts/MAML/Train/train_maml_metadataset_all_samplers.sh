#!/bin/bash

source ../env/bin/activate
ulimit -n 50000

cp -r <PATH_TO_DATA> $SLURM_TMPDIR

echo "Finished moving data"

cd .. && python -m src.main --exp_name maml_meta_dataset --train --runs 1 --folder $SLURM_TMPDIR/records --task_sampler $SLURM_ARRAY_TASK_ID --dataset meta_dataset --num-ways 5 --num-shots 1 --use-cuda --num-steps 5 --step-size 0.4 --meta-lr 0.001 --batch-size 16 --num-workers 0 --num-epochs 150 --num-adaptation-steps 5 --output-folder ./config/maml_meta_dataset_try_3/$SLURM_ARRAY_TASK_ID/

rm -rf $SLURM_TMPDIR/records

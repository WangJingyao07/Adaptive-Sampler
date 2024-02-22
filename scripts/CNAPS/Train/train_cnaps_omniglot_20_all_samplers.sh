#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name cnaps_omniglot_20 --train --model cnaps --image-size 84 --runs 1 --folder $SLURM_TMPDIR --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --meta-lr 0.001 --num-ways 20 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 16 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_omniglot_20/$SLURM_ARRAY_TASK_ID/

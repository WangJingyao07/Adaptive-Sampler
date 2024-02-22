#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_omniglot_20 --train --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder $SLURM_TMPDIR --num-steps 10 --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 50 --meta-lr 0.0005 --output-folder ./config/reptile_omniglot_20/$SLURM_ARRAY_TASK_ID/

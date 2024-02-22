#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name test_cnaps_omniglot_20 --log-test-tasks --model cnaps --image-size 84 --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --meta-lr 0.001 --num-ways 20 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 16 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_omniglot_20/$SLURM_ARRAY_TASK_ID/

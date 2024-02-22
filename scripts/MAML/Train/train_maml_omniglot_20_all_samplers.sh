#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_omniglot_20 --train --runs 1 --folder ./data --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.1 --meta-lr 0.001 --batch-size 16 --num-workers 8 --num-epochs 150 --num-adaptation-steps 5 --output-folder ./config/maml_omniglot_20/$SLURM_ARRAY_TASK_ID/

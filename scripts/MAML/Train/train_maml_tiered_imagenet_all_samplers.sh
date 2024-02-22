#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_tiered_imagenet --train --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset tiered_imagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --meta-lr 0.001 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_tiered_imagenet/$SLURM_ARRAY_TASK_ID/

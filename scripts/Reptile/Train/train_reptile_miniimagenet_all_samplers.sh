#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_miniimagenet --train --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --folder ./data --dataset miniimagenet --num-ways 5 --num-shots 1 --num-steps 5 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 8 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_miniimagenet_try_2/$SLURM_ARRAY_TASK_ID/

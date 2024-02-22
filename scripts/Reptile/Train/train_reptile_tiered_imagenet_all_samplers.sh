#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_tiered_imagenet --train --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --folder ./data --dataset tiered_imagenet --num-ways 5 --num-shots 1 --use-cuda --num-steps 10 --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 8 --num-epochs 50 --meta-lr 0.0005 --output-folder ./config/reptile_tiered_imagenet_try_2/$SLURM_ARRAY_TASK_ID/

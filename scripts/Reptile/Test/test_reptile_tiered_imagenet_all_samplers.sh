#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name test_reptile_tiered_imagenet --log-test-tasks --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --folder ./data --dataset tiered_imagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 1 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_tiered_imagenet_try_2/$SLURM_ARRAY_TASK_ID/

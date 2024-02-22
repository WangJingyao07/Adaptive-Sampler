#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name test_metaoptnet_tiered_imagenet --log-test-tasks --model metaoptnet --runs 1 --folder ./data --meta-lr 0.1 --task_sampler $SLURM_ARRAY_TASK_ID --dataset tiered_imagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 60 --output-folder ./config/metaoptnet_tiered_imagenet/$SLURM_ARRAY_TASK_ID/

#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name protonet_omniglot --train --model protonet --runs 1 --folder ./data --meta-lr 0.001 --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/protonet_omniglot/$SLURM_ARRAY_TASK_ID/

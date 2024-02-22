#!/bin/bash

source ../env/bin/activate
cd ..
python -m src.main --exp_name test_metaoptnet_omniglot_20 --log-test-tasks --model metaoptnet --runs 1 --folder $SLURM_TMPDIR/data --meta-lr 0.1 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 60 --output-folder ./config/metaoptnet_omniglot_20/$SLURM_ARRAY_TASK_ID/

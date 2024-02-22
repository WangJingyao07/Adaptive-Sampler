#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name test_maml_omniglot_20 --log-test-tasks --runs 1 --folder $SLURM_TMPDIR/data --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_omniglot_20/$SLURM_ARRAY_TASK_ID/

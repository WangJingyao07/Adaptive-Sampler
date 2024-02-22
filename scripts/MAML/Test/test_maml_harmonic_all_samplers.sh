#!/bin/bash

source ../env/bin/activate
cd ..

python -m src.main --exp_name test_maml_harmonic_5 --log-test-tasks --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset harmonic --num-shots 5 --use-cuda --step-size 0.4 --meta-lr 0.001 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_harmonic_5/$SLURM_ARRAY_TASK_ID/
python -m src.main --exp_name test_maml_harmonic_10 --log-test-tasks --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset harmonic --num-shots 10 --use-cuda --step-size 0.4 --meta-lr 0.001 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_harmonic_10/$SLURM_ARRAY_TASK_ID/

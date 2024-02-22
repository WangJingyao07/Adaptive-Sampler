#!/bin/bash

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_sinusoid_line_5 --train --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset sinusoid_line --num-shots 5 --use-cuda --step-size 0.001 --meta-lr 0.001 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_sinusoid_line_5_try2/$SLURM_ARRAY_TASK_ID/

python -m src.main --exp_name maml_sinusoid_line_10 --train --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset sinusoid_line --num-shots 10 --use-cuda --step-size 0.001 --meta-lr 0.001 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_sinusoid_line_10_try2/$SLURM_ARRAY_TASK_ID/

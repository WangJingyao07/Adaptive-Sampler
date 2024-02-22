#!/bin/bash

source ../env/bin/activate
cd ..

python -m src.main --exp_name test_reptile_sinusoid_5 --log-test-tasks --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder ./data --dataset sinusoid --num-shots 5 --num-steps 5 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_sinusoid_5/$SLURM_ARRAY_TASK_ID/
python -m src.main --exp_name test_reptile_sinusoid_10 --log-test-tasks --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder ./data --dataset sinusoid --num-shots 10 --num-steps 5 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_sinusoid_10/$SLURM_ARRAY_TASK_ID/


python -m src.main --exp_name test_reptile_sinusoid_5 --plot --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder ./data --dataset sinusoid --num-shots 5 --num-steps 5 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_sinusoid_5/$SLURM_ARRAY_TASK_ID/
python -m src.main --exp_name test_reptile_sinusoid_10 --plot --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder ./data --dataset sinusoid --num-shots 10 --num-steps 5 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_sinusoid_10/$SLURM_ARRAY_TASK_ID/

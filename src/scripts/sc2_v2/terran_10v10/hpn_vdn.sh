#!/bin/bash

GRES="gpu:1"
mkdir -p ../_log/$SLURM_JOB_ID
SLURM_JOB_PARTITION="gpu1"
cpus_per_task=10

# print sbatch job 
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

for _ in {1..3}; do
    srun --partition=$SLURM_JOB_PARTITION \
        --gres=$GRES \
        --cpus-per-task=$cpus_per_task \
        -o ../_log/%j/%N.out \
        -e ../_log/%j/%N.err \
        python ../../../main.py algo_name=hpn_vdn map_name=terran_10v10 with use_wandb=True group_name=hpn_vdn;
done
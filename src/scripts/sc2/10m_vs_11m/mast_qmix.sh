#!/bin/bash

GRES="gpu:1"
mkdir -p ../_log/$SLURM_JOB_ID
SLURM_JOB_PARTITION="gpu6"
cpus_per_task=15

# print sbatch job 
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

for _ in {1..5}; do
    srun --partition=$SLURM_JOB_PARTITION \
        --gres=$GRES \
        --cpus-per-task=$cpus_per_task \
        -o ../../_log/%j/%N.out \
        -e ../../_log/%j/%N.err \
    python ../../../main.py --config=mast_qmix --env-config=sc2 with env_args.map_name=10m_vs_11m  env_args.use_extended_action_masking=False t_max=1005000 use_wandb=True group_name=mast_qmix;
done
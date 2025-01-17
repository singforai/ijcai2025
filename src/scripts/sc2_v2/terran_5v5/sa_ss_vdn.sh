#!/bin/bash

# SQCA -> self attention 

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="3" python ../../../main.py --config=ss_vdn --env-config=sc2_v2_terran with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    env_args.use_extended_action_masking=True use_wandb=True group_name=sa_ss_vdn use_sqca=False;
done
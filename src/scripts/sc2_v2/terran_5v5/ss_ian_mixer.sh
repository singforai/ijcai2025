#!/bin/bash


for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="2" python ../../../main.py --config=mast_qmix --env-config=sc2_v2_terran with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=ss_ian_mixer mixer=qmix;
done
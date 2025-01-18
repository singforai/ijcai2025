#!/bin/bash

for _ in {3}; do
    CUDA_VISIBLE_DEVICES="3" python ../../../main.py --config=ss_qmix --env-config=sc2_v2_zerg with \
    env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=11 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=ss_qmix;
done
#!/bin/bash

for _ in {1}; do
    python ../../../main.py --config=ss_vdn --env-config=sc2_v2_zerg with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=ss_vdn;
done
#!/bin/bash

for _ in {1..2}; do
    python ../../../main.py --config=ss_vdn --env-config=sc2_v2_zerg with \
    env_args.capability_config.n_units=20 env_args.capability_config.n_enemies=20 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=ss_vdn;
done
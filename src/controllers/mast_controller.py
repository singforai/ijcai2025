#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th

from .basic_controller import BasicMAC

class MASTMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(MASTMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1
        
    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim)
        
    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        raw_obs = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        # assert raw_obs.shape[-1] == self._get_obs_shape()
        obs_component_dim = self._get_obs_component_dim()
        move_feats, enemy_feats, ally_feats, own_feats = th.split(raw_obs, obs_component_dim, dim=-1)
        own_feats = th.cat((own_feats, move_feats), dim=2)
        # use the max_dim (over self, enemy and ally) to init the token layer (to support all maps)

        own_feats = own_feats.reshape(bs * self.n_agents, 1, -1)
        ally_feats = ally_feats.contiguous().view(bs * self.n_agents, self.n_allies, -1)
        enemy_feats = enemy_feats.contiguous().view(bs * self.n_agents, self.n_enemies, -1)

        return bs, own_feats, ally_feats, enemy_feats

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim
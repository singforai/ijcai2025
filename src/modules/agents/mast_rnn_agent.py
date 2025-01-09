import torch as th
import torch.nn as nn

from modules.layer.mast_attention import CrossAttentionBlock, QueryKeyBlock

class MAST_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MAST_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.n_allies = self.args.n_allies
        self.n_enemies = self.args.n_enemies
        self.n_entities = self.n_agents + self.n_enemies
        self.n_actions = self.args.n_actions
        self.n_head = self.args.n_head
        self.hidden_size = self.args.hidden_size
        self.output_normal_actions = self.args.output_normal_actions
        self.use_extended_action_masking = self.args.env_args["use_extended_action_masking"]
        
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]

        self.own_embedding = nn.Linear(self.own_feats_dim, self.hidden_size)
        self.allies_embedding = nn.Linear(self.ally_feats_dim, self.hidden_size) 
        self.enemies_embedding = nn.Linear(self.enemy_feats_dim, self.hidden_size)
        
        self.normal_actions_net = nn.Linear(self.hidden_size, self.output_normal_actions)
        
        self.entity_attention = CrossAttentionBlock(
            d = self.hidden_size, 
            h = self.n_head
        )
        
        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        
        self.action_attention = QueryKeyBlock(
            d = self.hidden_size, 
            h = self.n_head,
        )
        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.own_embedding.weight.new(1, self.hidden_size).zero_()
        
    
    def forward(self, inputs, hidden_state):
        """
        inputs:
            batch
            batch * self.n_agents x 1 x own_feats
            batch * self.n_agents x n_allies x ally_feats
            batch * self.n_agents x n_enemies x enemy_feats
        hidden_state: 
            batch x num_agents x hidden_size    
        """
        
        bs, own_feats, ally_feats, enemy_feats  = inputs 
        
        own_masks = ~th.all(own_feats == 0, dim=-1)
        ally_mask = ~th.all(ally_feats == 0, dim=-1)
        enemy_mask = ~th.all(enemy_feats == 0, dim=-1)

        masks = th.cat((own_masks, ally_mask, enemy_mask), dim=-1).unsqueeze(1)
        
        own_feats = self.own_embedding(own_feats)
        ally_feats = self.allies_embedding(ally_feats)
        enemy_feats = self.enemies_embedding(enemy_feats)
        
        embeddings = th.cat((own_feats, ally_feats, enemy_feats), dim=1) # (bs * n_agents, n_entities, hidden_size)
        action_query = self.entity_attention(embeddings[:, 0].unsqueeze(1), embeddings, masks) # (bs * n_agents, 1, hidden_size)
        
        action_query = action_query.reshape(-1, self.hidden_size) # (bs * n_agents, hidden_size)
    
        hidden_state = hidden_state.reshape(-1, self.hidden_size) # (bs * n_agents, hidden_size)
        hidden_state = self.rnn(action_query, hidden_state)
        
        action_query = hidden_state.unsqueeze(1) # (bs * n_agents, 1, hidden_size)
        
        hidden_state = hidden_state.reshape(bs, self.n_agents, self.hidden_size)

        q_normal = self.normal_actions_net(action_query)
        
    
        if self.use_extended_action_masking:
            action_key = embeddings
            
        elif "sc2" in self.args.env:
            action_key = embeddings[:, self.n_agents:]
            
        q_interact = self.action_attention(action_query, action_key)

        return th.cat((q_normal, q_interact), dim=-1), hidden_state
        
        
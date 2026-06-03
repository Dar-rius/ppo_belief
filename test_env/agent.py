import torch
from torch import nn
from torch import Tensor
from torch.distributions import Categorical

class WorldModel(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh()
                )
        
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

        self.belief_head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, obs_dim)
                )

        self._init_weights()

    def _init_weights(self):
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0.0)
        
        for layer in self.belief_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0.0)

    def foward(self, obs:Tensor):
        feature_extracted = self.feature_extractor(obs)
        actor_logits = self.actor_head(feature_extracted)
        value = self.critic_head(feature_extracted)
        return feature_extracted, actor_logits, value

    def get_action_and_value(self, obs:Tensor, action:Tensor=None):
        z, actor_logits, value = self.foward(obs)
        b_input = torch.cat([z, actor_logits], dim=1)
        belief = self.belief(b_input)
        prob = Categorical(logits=actor_logits)
        if action is None: prob.sample()
        log_prob = prob.log_prob(action)
        dist_ent = prob.entropy()
        return action, log_prob, dist_ent, value, belief, actor_logits

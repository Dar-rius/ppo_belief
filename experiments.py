import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch import Tensor
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo import ppo_func, gae_compute, easy_train_ppo
from zerorl.buffer import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.logger import create_logger
from zerorl.functions import (get_obs_act, vectorize_env, set_seed, processing_state,
                              parse_dict_to_tensor, try_agent, env_step)


def ppo_belief_loss(agent: BaseAgent, params: dict, buffers: dict, hyper_params: AlgoConfig, data: dict, idx: Tensor) -> dict[str, Tensor]:
    states = data["state"][idx]
    actions = data["action"][idx]
    old_log_prob = data["log_prob"][idx]
    old_values = data["value"][idx]
    advantages =  data["adv_norm"][idx]
    returns  = data["return"][idx]
    derivated = data["delta"][idx]
    done = data["terminated"][idx]

    value_coef = hyper_params.value_coef
    ent_coef = hyper_params.ent_coef
    belief_coef = hyper_params.belief_coef
    clip_eps = hyper_params.clip_eps
    clip_vf = getattr(hyper_params, 'clip_vf', False)

    logits, new_values, new_belief = torch.func.functional_call(agent, (params, buffers), (states,))
    dist = agent.build_distribution(logits) #type: ignore[operator]
    new_log_probs, dist_entropy = eval_action(dist, actions)

    idx_adv = advantages.view(-1)
    idx_return = returns.view(-1)
    idx_derivate = derivated
    idx_done = done.view(-1)
    new_values = new_values.view(-1)
    new_belief = new_belief
    old_values = old_values.view(-1)
    old_log_prob = old_log_prob.view(-1)

    logratio = new_log_probs - old_log_prob
    ratio = torch.exp(logratio)

    surr1 = ratio * idx_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * idx_adv
    policy_loss = -torch.min(surr1, surr2).mean()

    if clip_vf:
        value_pred_clipped = old_values + (new_values - old_values).clamp(-clip_eps, clip_eps)
        value_loss = 0.5 * torch.max((idx_return - new_values).pow(2), (value_pred_clipped - idx_return).pow(2)).mean() 
    else:
        value_loss = 0.5 * nn.functional.mse_loss(new_values, idx_return)

    #Belief
    belief_error = (new_belief - idx_derivate).pow(2).mean(dim=-1)
    mask_ = 1.0 - idx_done
    belief_loss = (belief_error * mask_).sum() / mask_.sum()

    entropy_loss = dist_entropy.mean()

    loss = policy_loss + (value_coef * value_loss) + (belief_coef * belief_loss) - (ent_coef * entropy_loss)
    return {'loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'belief_loss': belief_loss,
            'entropy_loss':entropy_loss}

## Agent architecture
class Agent(BaseAgent):
    def __init__(self, obs_n, act_n, hidden_dim=64, is_discrete=False):
        super().__init__()
        self.obs_n = obs_n
        self.act_n = act_n
        self.hidden_dim = hidden_dim
        self.is_discrete = is_discrete
        if not is_discrete:  self.log_std = nn.Parameter(torch.zeros(self.act_n))
        # Feature Extractor
        self.extract_layer = nn.Sequential(
                nn.Linear(self.obs_n, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh()
                )
        # Actor
        self.actor = nn.Linear(self.hidden_dim, self.act_n)
        # Critic
        self.critic = nn.Linear(self.hidden_dim,  1)
        # Belief
        self.belief = nn.Linear(self.hidden_dim, self.obs_n)

        self.apply(self._orthogonal_init)

    def _orthogonal_init(self, module: nn.Module):
        """Apply orthogonal weight initialization with gain based on layer role."""
        if isinstance(module, nn.Linear):
            if module.out_features == self.hidden_dim:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            elif module.out_features == 1:
                nn.init.orthogonal_(module.weight, gain=1.0)
            elif module.out_features == self.obs_n:
                nn.init.orthogonal_(module.weight, gain=1.0)
            else:
                nn.init.orthogonal_(module.weight, gain=0.01)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: Tensor):
        """Forward pass returning (logits, value)."""
        x = self.extract_layer(state)
        logits = self.actor(x)
        value = self.critic(x)
        belief = self.belief(x)
        return (logits, value, belief)

    def build_distribution(self, logits: torch.Tensor):
        """Build a torch distribution from logits (Categorical or Normal)."""
        if self.is_discrete:
            return torch.distributions.Categorical(logits=logits)
        log_std_clamped = torch.clamp(self.log_std, min=-3.0, max=1.0)
        std = log_std_clamped.exp().expand_as(logits)
        return torch.distributions.Normal(logits, std)
    
    def get_action(self, state:Tensor, action:Tensor = None):
        """Sample or evaluate an action, returning action, log_prob, entropy, value."""
        logits, value, belief = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        value = value.squeeze(-1)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}

cfg = TrainConfig(project_name="ppo-belief", model_name="belief-model", timestamp=1_000_000, num_envs=4)
cfg.device = torch.device("cpu")
algo_config = AlgoConfig(belief_coef = 0.2, ent_coef=0.0)
run_seed = 89
seed = set_seed(run_seed, num_envs = cfg.num_envs)
env = vectorize_env("Ant-v5", num_envs = cfg.num_envs)
obs_dim, act_dim, obs_n, act_n, is_discrete = get_obs_act(env)
belief_agent = Agent(obs_n, act_n, is_discrete=is_discrete)
buffer = Buffer(capacity=cfg.rollout_steps,
                num_envs=cfg.num_envs,
                schema={"state": obs_dim, "delta": obs_dim, "action": act_dim,
                      "reward": (), "terminated": (), "entropy": (), "value": (),
                      "return": (), "log_prob": (), "advantage": (), "truncated": ()},
                device=cfg.device)
optimizer = optim.Adam(belief_agent.parameters(), lr=algo_config.lr, eps=1e-5)
scheduler = LambdaLR(optimizer, lambda step_: 1.0 - (step_ / cfg.num_update))
log = create_logger(cfg, algo_config, use_wandb=True)
reward_tensor = torch.zeros(cfg.num_envs, device=cfg.device)
state, _ = env.reset(seed = seed)
episodic_reward = []

# Train PPO-Belief
for step in tqdm(range(cfg.num_update)):
    metrics = {}
    for _ in range(cfg.rollout_steps):
        state_processed = processing_state(state)
        outputs = env_step(env, belief_agent, state_processed)
        outputs["terminated"] = outputs["terminated"] | outputs["truncated"]
        outputs.pop("info")
        outputs = parse_dict_to_tensor(outputs)
        next_state = outputs.pop("next_state")
        outputs["delta"] = (next_state - state_processed)
        buffer.insert(**outputs)
        reward_tensor += outputs["reward"]
        finished = outputs["terminated"] > 0

        if finished.any():
            episodic_reward.extend(reward_tensor[finished].tolist())
            reward_tensor[finished] = 0.0
        state = next_state

    with torch.inference_mode():
        state_processed = processing_state(state)
        last_output = belief_agent.get_action(state_processed)

    data = buffer.get_all()
    gae_compute(data["reward"], data["value"], last_output["value"], data["terminated"], buffer, algo_config)
    losses = ppo_func(belief_agent, optimizer, buffer, algo_config, scheduler, ppo_loss_func=ppo_belief_loss)
    if len(episodic_reward) > 0:
        recent = episodic_reward[-10:]
        mean_reward = float(np.mean(recent))
    else:
        mean_reward = 0.0
    metrics = {"train/mean_episode_reward": mean_reward}
    for k, v in losses.items(): metrics[f"train/{k}"] = v
    log(metrics, step)
    buffer.clear()

env.close()
log.close()
try_agent("Ant-v5", belief_agent, cfg, gif_path="belief-ant.gif")

#Train PPO-Standard
cfg = TrainConfig(project_name="ppo-belief", model_name="ppo-model", timestamp=1_000_000, num_envs=4)
cfg.device = torch.device("cpu")
trainer = easy_train_ppo("Ant-v5", cfg, algo_config, seed=run_seed)
trainer.train(use_wandb=True)
trainer.try_agent(gif_path="ppo-ant.gif")

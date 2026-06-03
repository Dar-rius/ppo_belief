import gymnasium as gym
import numpy as np
import wandb
import tqdm
import torch
from torch import Tensor
from agent import WorldModel
from ppo_belief.ppo_trainer import PPOTrainer
from ppo_belief.common.buffer import Buffer
from torch import nn

def normalize_data(obs:np.array, obs_space:np.array) -> np.array:
    if len(obs.shape)  == 1:
        obs = obs.reshape(1, -1)
    rows, cols = obs.shape
    data_normalized = np.zeros((rows, cols))
    low = obs_space.low
    high = obs_space.high
    for i in range(cols):
        for j in range(rows):
            data_normalized[j, i] = (obs[j, i] - low[i]) / (high[i] - low[i])
    return data_normalized

DEVICE = "cpu"

#Params
LR = 3e-5
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
ENT_COEF = 0.05
VALUE_COEF = 0.5
BELIEF_COEF = 0.1

#Rollout constant
TOTAL_TIMESTAMP = 10_000
BATCH_SIZE = 128
ROLLOUT_STEPS = 2048
NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS

config = {
        'epochs': NUM_UPDATE,
        'lr': LR,
        'gamma': GAMMA,
        'gae_lambda': GAE_LAMBDA,
        'clip_eps': CLIP_EPS,
        'ent_coef': ENT_COEF,
        'value_coef': VALUE_COEF,
        'belief_coef': BELIEF_COEF,
        }

env = gym.make('Acrobot-v1', render_mode="rgb_array")
obs_space = env.observation_space
action_space = env.action_space
buffer = Buffer(ROLLOUT_STEPS, obs_space)
agent = WorldModel(obs_space.shape[0], action_space.n).to(DEVICE)
trainer = PPOTrainer(model=agent,
                     belief_eval_loss=nn.MSELoss(),
                     lr=LR,
                     gamma=GAMMA,
                     gae_lambda=GAE_LAMBDA,
                     clip_eps=CLIP_EPS,
                     value_coef=VALUE_COEF,
                     belief_coef = BELIEF_COEF,
                     ent_coef=ENT_COEF,
                     device=DEVICE)
obs, _ = env.reset()
global_step = 0

with wandb.init(project="ppo-belief", config=config) as run:
    for update in tqdm(range(1, NUM_UPDATE + 1)):
        cumulative_reward = 0.0
        stop = False
        # Collecte phase
        for step in range(ROLLOUT_STEPS):
            global_step += 1
            obs = normalize_data(obs, obs_space)
            obs = torch.from_numpy(obs).unsqueeze(0)
            with torch.inference_mode():
                action, log_prob, entropy, value, belief_logits, _ = agent.get_action_and_value(obs)

            next_obs, reward, done, truncate, info = env.step(action)
            next_obs_normalized = normalize_data(next_obs, obs_space)
            next_obs_normalized = torch.from_numpy(obs)
            target = obs - next_obs_normalized
            buffer.insert(
                obs=obs,
                action=action,
                old_log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
                target=target
            )
            cumulative_reward += reward
            if done or truncate:
                obs, _ = env.reset()
                stop = True
            else:
                obs = next_obs
        if stop:
            last_value = torch.tensor([0.0], device=DEVICE)
        else:
            # Optimisation phase
            with torch.inference_mode():
                obs = obs.unsqueeze(0)
                _, _, _, next_value, _, _, _, _, _ = agent.get_action_and_value(obs.to(DEVICE))
                last_value = torch.tensor([next_value.item()], device=DEVICE)

        reward_list = buffer.reward
        value_list = buffer.value
        done_list = buffer.done
        return_, adv, delta = trainer.compute_gae(reward_list, value_list, last_value, done_list)
        buffer.insert_returns(return_, adv)
        #Compute Belief PPO
        loss, policy_loss, value_loss, belief_loss, entropy = trainer.update(buffer, TOTAL_TIMESTAMP, step, BATCH_SIZE)
        # Clean buffer
        buffer.clear()
        run.log({'loss': loss,
                 'policy loss': policy_loss,
                 'value loss': value_loss,
                 'belief loss': belief_loss,
                 'entropy': entropy,
                 'reward': cumulative_reward})

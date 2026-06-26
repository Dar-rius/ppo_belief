import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm
import torch
from torch import Tensor
from agent import WorldModel
from ppo_belief.ppo_trainer import PPOTrainer
from ppo_belief.common.buffer import Buffer
from torch import nn
import imageio

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

def record_agent_gif(model: torch.nn.Module, env_id: str = 'Acrobot-v1', filename: str = 'agent_performance.gif', device: str = "cpu"):
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    model.eval()
    with torch.inference_mode():
        obs, _ = env.reset()
        frames.append(env.render())
        done = False
        truncate = False
        episode_reward = 0.0
        while not (done or truncate):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _, _, _ = model.get_action_and_value(obs_tensor)
            obs, reward, done, truncate, _ = env.step(action.item())
            episode_reward += reward
            frames.append(env.render())
    env.close()
    print(f"Reward total : {episode_reward}")
    imageio.mimsave(filename, frames, fps=30) 
    print(f" The video is saved: {filename}")
    model.train()

DEVICE = "cpu"

#Params
LR = 3e-4
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VALUE_COEF = 0.2
BELIEF_COEF = 0.2

#Rollout constant
TOTAL_TIMESTAMP = 1_000_000
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
buffer = Buffer(ROLLOUT_STEPS, obs_space.shape[0])
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
            obs_norm = normalize_data(obs, obs_space)
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.inference_mode():
                action, log_prob, entropy, value, _, _ = agent.get_action_and_value(obs_tensor)

            next_obs, reward, done, truncate, info = env.step(action)
            next_obs_norm= normalize_data(next_obs, obs_space)
            target = obs_norm - next_obs_norm
            done = 1 if done else 0
            buffer.insert(
                obs=obs,
                target=target,
                action=action.item(),
                old_log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=done
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
                obs_norm = normalize_data(obs, obs_space)
                obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _, _, _, next_value, _, _ = agent.get_action_and_value(obs_tensor)
                last_value = next_value.squeeze(0)

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

record_agent_gif(
    model=agent,
    env_id='Acrobot-v1',
    filename='mon_agent_acrobot_step_2000.gif',
    device=DEVICE)

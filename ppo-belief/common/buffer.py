import torch
import numpy as np

class Buffer:
    def __init__(self, buffer_space:int):
        self.slice: int = 0
        self.buffer_space = buffer_space
        self.obs = np.zeros(self.buffer_space, 1)
        self.target = np.zeros(self.buffer_space, 1)
        self.actions = np.zeros(self.buffer_space)
        self.old_log_probs = np.zeros(self.buffer_space)
        self.returns = np.zeros(self.buffer_space)
        self.adv = np.zeros(self.buffer_space)
        self.rewards = np.zeros(self.buffer_space)
        self.values = np.zeros(self.buffer_space)
        self.dones = np.zeros(self.buffer_space)

    #Insert datas in buffer
    def insert(self, obs:np.array, target:np.array, action:np.array, old_log_prob:np.array,  reward:np.array, value:np.array, dones:np.array, target_regime:np.array):
        self.obs[self.slice] = obs
        self.target[self.slice] = target
        self.actions[self.slice] = action
        self.old_log_probs[self.slice] = old_log_prob
        self.rewards[self.slice] = reward
        self.values[self.slice] = value
        self.dones[self.slice] = dones
        self.slice += 1

    #Insert return and the advantage in buffer
    def insert_returns(self, returns:np.array, adv:np.array):
        self.returns[:] = returns
        self.adv[:] = adv
    
    # sampling data
    def get_all(self) -> tuple:
        return (self.obs, self.target, self.actions, self.old_log_probs,
                self.returns, self.adv, self.rewards, self.values, self.dones)

    # reset the slicing of arrays
    def clear(self):
        self.slice = 0

    # convert all data from numpy to tensor
    def convert_array_to_tensor(self, device="cpu"):
        self.obs = torch.from_numpy(self.obs).to(device)
        self.target = torch.from_numpy(self.target).to(device)
        self.actions = torch.from_numpy(self.actions).to(device)
        self.old_log_probs = torch.from_numpy(self.old_log_probs).to(device)
        self.rewards = torch.from_numpy(self.rewards).to(device)
        self.values = torch.from_numpy(self.values).to(device)
        self.dones = torch.from_numpy(self.dones).to(device)

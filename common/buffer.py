import torch
from torch import Tensor
import numpy as np

class Buffer:
    """
    0 -> Micro State 
    1 -> Macro State
    2 -> Action
    3 -> Old Log Probs 
    4 -> Returns
    5 -> Advantage
    6 -> Reward
    7 -> Value
    8 -> Dones
    9 -> Target Regime
    """
    def __init__(self, obs_space:int, buffer_space:int, device:str="cpu"):
        self.slice: int = 0
        self.buffer_space = buffer_space
        self.device = device
        self.obs = np.zeros(self.buffer_space, 1)
        #Casting type of actions
        self.actions = np.zeros(self.buffer_space)
        self.old_log_probs = np.zeros(self.buffer_space)
        self.returns = np.zeros(self.buffer_space)
        self.adv = np.zeros(self.buffer_space)
        self.rewards = np.zeros(self.buffer_space)
        self.values = np.zeros(self.buffer_space)
        self.dones = np.zeros(self.buffer_space)

    #Insert datas in buffer
    def insert(self, obs:np.array, action:np.array, old_log_prob:np.array,  reward:np.array, value:np.array, dones:np.array, target_regime:np.array):
        self.obs[self.slice] = obs
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
        return (self.obs, self.actions, self.old_log_probs,
                self.returns, self.adv, self.rewards, 
                self.values, self.dones)

    # Delete all data
    def clear(self):
        self.slice = 0

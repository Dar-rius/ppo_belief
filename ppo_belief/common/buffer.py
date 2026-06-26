import torch
import numpy as np

class Buffer:
    def __init__(self, buffer_space:int, obs_space:int):
        self.slice: int = 0
        self.buffer_space = buffer_space
        self.obs = np.zeros((self.buffer_space, obs_space))
        self.target = np.zeros((self.buffer_space, obs_space))
        self.action = np.zeros(self.buffer_space)
        self.old_log_prob = np.zeros(self.buffer_space)
        self.return_ = np.zeros(self.buffer_space)
        self.adv = np.zeros(self.buffer_space)
        self.reward = np.zeros(self.buffer_space)
        self.value = np.zeros(self.buffer_space)
        self.done = np.zeros(self.buffer_space)

    #Insert datas in buffer
    def insert(self, obs:np.array, target:np.array, action:np.array, old_log_prob:np.array,  reward:np.array, value:np.array, done:np.array):
        self.obs[self.slice] = obs
        self.target[self.slice] = target
        self.action[self.slice] = action
        self.old_log_prob[self.slice] = old_log_prob
        self.reward[self.slice] = reward
        self.value[self.slice] = value
        self.done[self.slice] = done
        self.slice += 1

    #Insert return and the advantage in buffer
    def insert_returns(self, return_:np.array, adv:np.array):
        self.return_[:] = return_
        self.adv[:] = adv
    
    # sampling data
    def get_all(self, device="cpu") -> tuple:
        return (
                torch.tensor(self.obs, dtype=torch.float32, device=device),
                torch.tensor(self.target, dtype=torch.float32, device=device),
                torch.tensor(self.action, dtype=torch.long, device=device),
                torch.tensor(self.old_log_prob, dtype=torch.float32, device=device),
                torch.tensor(self.return_, dtype=torch.float32, device=device),
                torch.tensor(self.adv, dtype=torch.float32, device=device),
                torch.tensor(self.reward, dtype=torch.float32, device=device),
                torch.tensor(self.value, dtype=torch.float32, device=device),
                torch.tensor(self.done, dtype=torch.long, device=device)
                )

    # reset the slicing of arrays
    def clear(self):
        self.slice = 0

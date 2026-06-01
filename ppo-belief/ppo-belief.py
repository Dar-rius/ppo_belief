import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer 
import numpy as np
from .common.buffer import Buffer 

class PPOTrainer:
    def __init__(self,
                model: nn.Module,
                optimizer: Optimizer,
                belief_eval_loss = nn.CrossEntropyLoss(),
                value_eval_loss = nn.MSELoss(),
                lr:float=3e-5,
                gamma:float=0.99,
                gae_lambda:float=0.95,
                clip_eps:float=0.1,
                value_coef:float=0.5,
                belief_coef:float=0.1,
                ent_coef:float=0.02,
                device:str="cpu"
        ):
        self.lr = lr
        self.model = model
        self.optimizer = optimizer
        self.belief_eval_loss = belief_eval_loss
        self.value_eval_loss = value_eval_loss
        # Hyperparams PPO
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        # Total Loss Coefficients
        self.value_coef = value_coef
        self.belief_coef = belief_coef
        self.ent_coef = ent_coef
        #Device where tensor will be run
        self.device = device

    def compute_gae(self,
        rewards:np.array,
        values:np.array,
        last_value:np.array,
        dones:np.array
        ) -> tuple[np.array, np.array]:
        gae: float = 0.0
        mask = 1.0 - dones
        next_values = np.concatenate((values[1:], last_value), axis=0)
        total_size = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        delta = rewards + self.gamma * next_values * mask - values
        for step in reversed(range(total_size)):
            gae = delta[step] + self.gamma * self.gae_lambda * mask[step] * gae
            advantages[step] =  gae
        returns = advantages + values
        return (returns, advantages, delta)

    def _lr_decay(self, lr:float, total_steps:int, step:int, optimizer: Optimizer):
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    # Compute Belief PPO and Update network weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, epochs:int=10):
        self._lr_decay(self.lr, total_steps, step, self.optimizer)
        # the target regime (0 -> Stable, 1 -> Volatility, 2 -> Crisis)
        obs, target, actions, old_log_probs, returns, adv, _, _, _, _ = memory.get_all()
        # Normalize the advantages
        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        dataset_size = actions.shape[0]
        indices = np.arrange(dataset_size)
        size_total = int((dataset_size / batch_size) * epochs)
        epoch_losses = torch.zeros((size_total), dtype=torch.float32, device=self.device)
        epoch_pi_losses = torch.zeros((size_total), device=self.device)
        epoch_v_losses = torch.zeros((size_total), device=self.device)
        epoch_b_losses = torch.zeros((size_total), device=self.device)
        epoch_entropies = torch.zeros((size_total), device=self.device)
        index_loss = 0
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                # Evaluate model again
                _, new_log_probs, dist_entropy, new_values, belief_logits = self.model.get_action_and_value(obs[idx], actions[idx])
                # Compute Ratio (new Policy / old Policy)
                logratio = new_log_probs - old_log_probs[idx]
                ratio = torch.exp(logratio)
                # Loss PPO
                adv = advantages.flatten()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # Loss Value (Critic) - MSE
                value_loss = self.value_eval_loss(new_values.flatten(), returns.flatten())
                # Loss Belief (Auxiliary) - Cross Entropy
                belief_loss = self.belief_eval_loss(belief_logits, target.flatten().long())
                entropy_loss = dist_entropy.mean()
                # Total Loss
                loss = policy_loss + \
                       (self.value_coef * value_loss) + \
                       (self.belief_coef * belief_loss) - \
                       (self.ent_coef * entropy_loss)
                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                epoch_losses[index_loss] = loss
                epoch_pi_losses[index_loss] = policy_loss
                epoch_v_losses[index_loss] = value_loss
                epoch_b_losses[index_loss] = belief_loss
                epoch_entropies[index_loss] = entropy_loss
        return epoch_losses.mean().item(), epoch_pi_losses.mean().item(), epoch_v_losses.mean().item(), epoch_b_losses.mean().item(), epoch_entropies.mean().item()

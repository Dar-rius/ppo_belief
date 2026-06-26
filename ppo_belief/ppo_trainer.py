import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch import optim
from .common.buffer import Buffer 
from torch.nn.modules.loss import _Loss

class PPOTrainer:
    def __init__(self,
                model: nn.Module,
                belief_eval_loss:_Loss = nn.CrossEntropyLoss(),
                value_eval_loss: _Loss = nn.MSELoss(),
                world_lr:float=1e-3,
                agent_lr:float=3e-4,
                optimizer: Optimizer = None,
                gamma:float=0.99,
                gae_lambda:float=0.95,
                clip_eps:float=0.1,
                value_coef:float=0.5,
                belief_coef:float=1.,
                ent_coef:float=0.01,
                device:str="cpu"
        ):
        #self.lr = lr
        self.model = model
        if optimizer is None:
            world_params = list(model.feature_extractor.parameters()) + \
                           list(model.belief_head.parameters()) + \
                           list(model.reward_head.parameters()) + \
                           list(model.transition_model.parameters()) # Si tu as un réseau pour le delta
            self.world_optimizer = optim.Adam(world_params, lr=world_lr)
            
            # Optimiseur de l'Agent (Actor, Critic)
            agent_params = list(model.actor_head.parameters()) + \
                           list(model.critic_head.parameters())
            self.agent_optimizer = optim.Adam(agent_params, lr=agent_lr)
        else:
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
        rewards:Tensor,
        values:Tensor,
        last_value:Tensor,
        dones:Tensor
        ) -> tuple[Tensor, Tensor]:
        gae: float = 0.0
        mask = 1.0 - dones
        next_values = np.concatenate((values[1:], last_value), axis=0)
        total_size = rewards.shape[0]
        advantages = torch.zeros(rewards)
        delta = rewards + self.gamma * next_values - values
        for step in reversed(range(total_size)):
            gae = delta[step] + self.gamma * self.gae_lambda * gae
            advantages[step] =  gae
        returns = advantages + values
        return (returns.detach(), advantages.detach(), delta.detach())

    """
    def _lr_decay(self, lr:float, total_steps:int, step:int, optimizer: Optimizer):
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
    """

    # Compute Belief PPO and Update network weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, horizon:int=15, epochs:int=10):
        #self._lr_decay(self.lr, total_steps, step, self.optimizer)
        # the target regime (0 -> Stable, 1 -> Volatility, 2 -> Crisis)
        obs, target, action, old_log_prob, return_, adv, _, _, _ = memory.get_all()
        # Normalize the advantages
        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        dataset_size = action.size(0)
        size_total = int((dataset_size / batch_size) * epochs)
        epoch_losses = torch.zeros((size_total), dtype=torch.float32, device=self.device)
        epoch_pi_losses = torch.zeros((size_total), device=self.device)
        epoch_v_losses = torch.zeros((size_total), device=self.device)
        epoch_b_losses = torch.zeros((size_total), device=self.device)
        epoch_entropies = torch.zeros((size_total), device=self.device)
        index_loss = 0
        for _ in range(epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                if idx.numel() == 0: continue
                # Evaluate model again
                _, new_log_probs, dist_entropy, new_values, belief_logits, _ = self.model.get_action_and_value(obs[idx], action[idx])
                # Compute Ratio (new Policy / old Policy)
                logratio = new_log_probs - old_log_prob[idx]
                ratio = torch.exp(logratio)
                # Loss PPO
                adv = advantages[idx].flatten()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # Loss Value (Critic) - MSE
                value_loss = self.value_eval_loss(new_values.flatten(), return_[idx].flatten())
                # Loss Belief (Auxiliary)
                belief_loss = self.belief_eval_loss(belief_logits.flatten(), target[idx].flatten().long())
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

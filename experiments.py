import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import copy
import json
import random
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from zerorl.helpers.factory import ActorCriticAgent
from zerorl.helpers.agent import BaseAgent, eval_action
from zerorl.algorithms.ppo import ppo_func, gae_compute
from zerorl.buffer import Buffer
from zerorl.config import AlgoConfig, TrainConfig
from zerorl.logger import create_logger
from zerorl.functions import (get_obs_act, vectorize_env, processing_state,
                              parse_dict_to_tensor, try_agent, env_step, to_env_action)


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

    logits, new_values, new_belief = torch.func.functional_call(agent, (params, buffers), (states, actions))
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
        self.belief = nn.Sequential(
                nn.Linear(self.hidden_dim + self.act_n, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.obs_n))

        self.apply(self._orthogonal_init)

    def _orthogonal_init(self, module: nn.Module):
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

    def forward(self, state: Tensor, action: Tensor = None):
        x = self.extract_layer(state)
        logits = self.actor(x)
        value = self.critic(x)
        belief = None
        if action is not None:
            if self.is_discrete:
                action_features = F.one_hot(action.long().view(-1),num_classes=self.act_n).to(device=x.device,dtype=x.dtype)
            else:
                action_features = action.to(device=x.device, dtype=x.dtype).reshape(action.shape[0], -1)
            belief_input = torch.cat([x, action_features],dim=-1)
            belief = self.belief(belief_input)
        return (logits, value, belief)

    def build_distribution(self, logits: torch.Tensor):
        if self.is_discrete:
            return torch.distributions.Categorical(logits=logits)
        log_std_clamped = torch.clamp(self.log_std, min=-3.0, max=1.0)
        std = log_std_clamped.exp().expand_as(logits)
        return torch.distributions.Normal(logits, std)

    def get_action(self, state:Tensor, action:Tensor = None):
        logits, value, belief = self.forward(state)
        dist = self.build_distribution(logits)
        if action is None: action = dist.sample()
        log_prob, dist_entropy = eval_action(dist, action)
        value = value.squeeze(-1)
        return {"action": action, "log_prob": log_prob, "entropy":dist_entropy, "value":value}


#config seeds
@dataclass(frozen=True)
class PairSeeds:
    init_seed: int
    belief_seed: int
    train_seed: int
    env_seed: int
    eval_seed: int

def make_pair_seeds(master_seed: int, pair_id: int) -> PairSeeds:
    sequence = np.random.SeedSequence([master_seed, pair_id])
    children = sequence.spawn(5)
    seeds = [int(child.generate_state(1)[0]) for child in children]
    return PairSeeds(*seeds)

def reset_global_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env_seeds(seed: int, num_envs: int) -> list[int]:
    sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(num_envs)]

def make_eval_seeds(base_seed: int, checkpoint: int, episodes: int) -> list[int]:
    sequence = np.random.SeedSequence([base_seed, checkpoint])
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(episodes)]

# init W with same params
@torch.no_grad()
def copy_shared_weights(ppo_agent: ActorCriticAgent, belief_agent: Agent):
    belief_agent.extract_layer.load_state_dict(ppo_agent.extract_layer.state_dict())
    belief_agent.actor.load_state_dict(ppo_agent.actor.state_dict())
    belief_agent.critic.load_state_dict(ppo_agent.critic.state_dict())
    if hasattr(ppo_agent, "log_std"): belief_agent.log_std.copy_(ppo_agent.log_std)

#Build all agent
def build_paired_agents(obs_n: int, act_n: int, is_discrete: bool, seeds: PairSeeds):
    # PPO initialization
    reset_global_rng(seeds.init_seed)
    ppo_agent = ActorCriticAgent(obs_n, act_n, is_discrete, hidden_dim=64)

    # Independent initialization for Belief architecture
    reset_global_rng(seeds.belief_seed)
    belief_agent = Agent(obs_n, act_n, hidden_dim=64, is_discrete=is_discrete)

    # Replace PPO part by EXACTLY the same parameters.
    copy_shared_weights(ppo_agent, belief_agent)
    return ppo_agent, belief_agent

#Action deterministic for evals
@torch.no_grad()
def deterministic_action(agent: BaseAgent, state: Tensor):
    output = agent.forward(state)

    # PPO -> (logits, value)
    # Belief -> (logits, value, belief)
    logits = output[0]
    dist = agent.build_distribution(logits)
    if isinstance(dist, torch.distributions.Categorical):
        return torch.argmax(dist.probs, dim=-1)
    return dist.mean

#Eval Policy
def evaluate_policy(agent: BaseAgent, env_id: str, seeds: list[int], device: torch.device):
    env = vectorize_env(env_id, num_envs=1)
    was_training = agent.training
    agent.eval()
    episode_returns = []

    for seed in seeds:
        obs, _ = env.reset(seed=[seed])
        total_reward = 0.0
        finished = False

        while not finished:
            state = processing_state(obs, device=device)
            action = deterministic_action(agent, state)
            obs, reward, terminated, truncated, _ = env.step(to_env_action(action, env))
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            finished = bool(terminated[0] or truncated[0])

        episode_returns.append(total_reward)

    env.close()
    if was_training: agent.train()
    values = np.asarray(episode_returns, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "returns": values.tolist()}

# Calcul of AUC
def compute_auc(env_steps: list[int], returns: list[float]):
    x = np.asarray(env_steps, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)

    if len(x) < 2: return float("nan"), float("nan")

    auc = np.trapezoid(y, x)
    budget = x[-1] - x[0]
    normalized_auc = (auc / budget if budget > 0 else float("nan"))
    return (float(auc), float(normalized_auc))

#Maximum DrawnDown
def compute_max_drawdown(returns: list[float]):
    values = np.asarray(returns, dtype=np.float64)
    running_best = np.maximum.accumulate(values)
    drawdowns = (running_best - values)
    return float(drawdowns.max())

# Learning Curve
@dataclass
class EvalCurve: 
    env_steps: list[int] = field(default_factory=list)
    mean_returns: list[float] = field(default_factory=list)
    std_returns: list[float] = field(default_factory=list)
    raw_returns: list[list[float]] = field(default_factory=list)

    def add(self, env_steps: int, evaluation: dict):
        self.env_steps.append(env_steps)
        self.mean_returns.append(evaluation["mean"])
        self.std_returns.append(evaluation["std"])
        self.raw_returns.append(evaluation["returns"])

    def summary(self):
        auc, auc_norm = compute_auc(self.env_steps, self.mean_returns)
        return {"final_return": self.mean_returns[-1], "auc": auc,
                "auc_normalized": auc_norm, "max_drawdown": compute_max_drawdown(self.mean_returns)
                }

cfg = TrainConfig(project_name="ppo-belief", model_name="belief-model", timestamp=5_000_000, num_envs=10)
algo_config = AlgoConfig(belief_coef = 0.2)
run_seed = 89

# Train  all PPO
def train_one_run(*, algorithm: str, agent: BaseAgent, env_id: str, cfg: TrainConfig,
                    algo_config: AlgoConfig, seeds: PairSeeds, eval_every: int = 10, eval_episodes: int = 20):
    assert algorithm in {"ppo", "belief"}
    reset_global_rng(seeds.train_seed)
    agent = agent.to(cfg.device)
    env = vectorize_env(env_id, num_envs=cfg.num_envs)
    obs_dim, act_dim, _, _, _ = (get_obs_act(env))

    # Same buffer layout for both algorithms.
    buffer = Buffer(
        capacity=cfg.rollout_steps,
        num_envs=cfg.num_envs,
        schema={
            "state": obs_dim,
            "delta": obs_dim,
            "action": act_dim,
            "reward": (),
            "terminated": (),
            "entropy": (),
            "value": (),
            "return": (),
            "log_prob": (),
            "advantage": (),
            "truncated": (),
        },
        device=cfg.device,
    )

    optimizer = optim.Adam(agent.parameters(), lr=algo_config.lr, eps=1e-5)
    scheduler = LambdaLR(optimizer, lambda step_: (1.0 - step_ / cfg.num_update))
    log = create_logger(cfg, algo_config, use_wandb=True)
    env_seeds = make_env_seeds(seeds.env_seed, cfg.num_envs)
    state, _ = env.reset(seed=env_seeds)
    reward_tensor = torch.zeros(cfg.num_envs, device=cfg.device)
    episodic_reward = []
    curve = EvalCurve()

    # ------------------------------------
    # Evaluation BEFORE training
    # ------------------------------------

    initial_eval = evaluate_policy(agent, env_id, make_eval_seeds(
                                                    seeds.eval_seed,
                                                    checkpoint=0,
                                                    episodes=eval_episodes),
                                    cfg.device)

    curve.add(0, initial_eval)
    log({"eval/mean_return": initial_eval["mean"],
        "eval/std_return": initial_eval["std"],
        "global/env_steps": 0}, 0)

    # ------------------------------------
    # Training
    # ------------------------------------

    for step in tqdm(range(1, cfg.num_update + 1), desc=f"{algorithm}"):
        metrics = {}
        for _ in range(cfg.rollout_steps):
            state_processed = processing_state(state, device=cfg.device)
            outputs = env_step(env, agent, state_processed)
            outputs["terminated"] = outputs["terminated"] | outputs["truncated"]
            outputs.pop("info")
            outputs = parse_dict_to_tensor(outputs, cfg.device)
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
            state_processed = processing_state(state, device=cfg.device)
            last_output = agent.get_action(state_processed)

        data = buffer.get_all()
        gae_compute(data["reward"], data["value"], last_output["value"], data["terminated"], buffer, algo_config)
        if algorithm == "belief":
            losses = ppo_func(agent, optimizer, buffer, algo_config, scheduler, ppo_loss_func=ppo_belief_loss)
        else:
            losses = ppo_func(agent, optimizer, buffer, algo_config, scheduler)

        if episodic_reward:
            recent = episodic_reward[-10:]
            mean_reward = float(np.mean(recent))
        else:
            mean_reward = 0.0
        env_steps = (step * cfg.rollout_steps * cfg.num_envs)
        metrics["train/mean_episode_reward"] = mean_reward
        metrics["global/env_steps"] = env_steps
        for k, v in losses.items(): metrics[f"train/{k}"] = v
        # --------------------------------
        # Independent evaluation
        # --------------------------------
        should_eval = (step % eval_every == 0 or step == cfg.num_update)
        if should_eval:
            evaluation = evaluate_policy(
                agent,
                env_id,
                make_eval_seeds(
                    seeds.eval_seed,
                    checkpoint=step,
                    episodes=eval_episodes,
                ),
                cfg.device,
            )
            curve.add(env_steps, evaluation)
            metrics["eval/mean_return"] = evaluation["mean"]
            metrics["eval/std_return"] = evaluation["std"]

        log(metrics, step)
        buffer.clear()
    env.close()
    log.close()
    return {"summary": curve.summary(), "curve": asdict(curve)}


def make_config(algorithm: str, pair_id: int):
    cfg = TrainConfig(project_name="ppo-belief", model_name=f"{algorithm}-pair-{pair_id}", timestamp=5_000_000, num_envs=10)
    return cfg


def run_pair(pair_id: int, env_id: str, master_seed: int = 2026, try_agents: bool = False):
    seeds = make_pair_seeds(master_seed, pair_id)

    # Probe environment only to get dimensions.
    probe_env = vectorize_env(env_id, num_envs=1)

    _, _, obs_n, act_n, is_discrete = (get_obs_act(probe_env))
    probe_env.close()
    ppo_agent, belief_agent = (build_paired_agents(obs_n, act_n, is_discrete, seeds))

    ppo_cfg = make_config("ppo", pair_id,)
    belief_cfg = make_config("belief", pair_id)

    # Separate copies in case config is mutated.
    ppo_algo = copy.deepcopy(algo_config)
    belief_algo = copy.deepcopy(algo_config)

    ppo_result = train_one_run(
        algorithm="ppo",
        agent=ppo_agent,
        env_id= env_id,
        cfg=ppo_cfg,
        algo_config=ppo_algo,
        seeds=seeds,
    )

    belief_result = train_one_run(
        algorithm="belief",
        agent=belief_agent,
        env_id= env_id,
        cfg=belief_cfg,
        algo_config=belief_algo,
        seeds=seeds,
    )

    if try_agents:
        os.makedirs(
            "results/gifs",
            exist_ok=True,
        )

        try_agent(
            env_id,
            ppo_agent,
            ppo_cfg,
            iterations=1,
            gif_path=(
                f"results/gifs/"
                f"ppo_pair_{pair_id}"
            ),
        )

        try_agent(
            env_id,
            belief_agent,
            belief_cfg,
            iterations=1,
            gif_path=(
                f"results/gifs/"
                f"belief_pair_{pair_id}"
            ),
        )

    result = {
        "pair_id": pair_id,
        "seeds": asdict(seeds),
        "ppo": ppo_result,
        "belief": belief_result,
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/pair_{pair_id}.json", "w") as f: json.dump(result, f, indent=2)
    return result

ENV_ID = "Ant-v5"
results = [ run_pair(i, ENV_ID, try_agents=(i == 0)) for i in range(10)]

def paired_delta(results, metric,):
    deltas = []
    for result in results:
        ppo = (result["ppo"] ["summary"] [metric])
        belief = (result["belief"] ["summary"] [metric])

        if metric == "max_drawdown":
            # Lower drawdown is better.
            delta = ppo - belief
        else:
            # Higher is better.
            delta = belief - ppo

        deltas.append(delta)

    return np.asarray(deltas, dtype=np.float64)


def paired_bootstrap_ci( deltas: np.ndarray, *, bootstrap_samples: int = 10_000, confidence: float = 0.95, seed: int = 123):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    bootstrap_means = (deltas[indices].mean(axis=1))
    alpha = (1.0 - confidence) / 2.0
    low = np.quantile(bootstrap_means, alpha)
    high = np.quantile(bootstrap_means, 1.0 - alpha)

    return {"mean": float(deltas.mean()),
            "ci_low": float(low),
            "ci_high": float(high)}

def paired_probability_improvement(deltas: np.ndarray,):
    wins = (deltas > 0).astype(np.float64)
    ties = (deltas == 0).astype(np.float64)
    return float((wins + 0.5 * ties).mean())


def analyse_metric(results, metric):
    deltas = paired_delta(results, metric)
    bootstrap = paired_bootstrap_ci(deltas)
    probability = (paired_probability_improvement(deltas))

    return {"metric": metric,
        "paired_deltas": deltas.tolist(),
        "mean_improvement": bootstrap["mean"],
        "95_ci": [bootstrap["ci_low"], bootstrap["ci_high"]],
        "probability_improvement": probability}

for metric in ["final_return", "auc_normalized", "max_drawdown"]:
    report = analyse_metric(results, metric)
    print(report)

#try_agent("Ant-v5", agent, cfg, gif_path="belief-ant.gif")

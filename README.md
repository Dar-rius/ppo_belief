# PPO-Belief

> **Status:** Active research project. A research write-up describing the
> methodology, experiments, and results is in progress.

PPO-Belief investigates a simple question:

> **Can an auxiliary transition-prediction objective change how PPO learns in
> continuous-control environments?**

The project extends PPO with an auxiliary prediction head trained to predict
changes in observations between consecutive environment states.

The auxiliary objective updates the shared representation used by
the actor and critic.

The goal is to study whether learning simple environment dynamics alongside
the PPO objective affects learning dynamics, sample efficiency, and final
policy performance.

## Core Idea

Standard PPO optimizes a policy and value function from collected trajectories.

PPO-Belief adds an auxiliary task:

$$
\Delta o_t = o_{t+1} - o_t
$$

Given the shared representation of the current observation and the action
taken at time \(t\), the auxiliary head predicts this observation delta:

$$
\hat{\Delta o_t} = f_{\text{belief}}(z_t, a_t)
$$

where:

- $$(o_t\)$$ is the current observation,
- $$(o_{t+1}\)$$ is the next observation,
- $$(a_t\)$$ is the action,
- $$(z_t\)$$ is the representation produced by the shared encoder,
- $$(\hat{\Delta o_t}\)$$ is the predicted change in observation.

The prediction head is not used to select actions during evaluation. Its
effect on the policy comes through the gradients it contributes to the shared
representation during training.

## Architecture

The PPO-Belief agent contains:

- **Shared feature extractor** — two 64-unit `Tanh` layers
- **Actor head** — produces policy outputs
- **Critic head** — estimates the state value
- **Belief head** — predicts observation deltas from the shared features and action

For continuous-control environments, the actor parameterizes a Normal
distribution with a learned state-independent log standard deviation.

Conceptually:

```text
                       ┌──────────── Actor ────────> Policy
Observation ─> Encoder ┤
                       ├──────────── Critic ───────> Value
                       │
Action ────────────────┴─> Belief Head ───────────> Δ Observation
```

## Licence

MIT - see [LICENSE](LICENSE) for details.

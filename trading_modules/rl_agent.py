"""
RL Agent Module — PPO for Trading
====================================

Reinforcement Learning agent using PPO (Proximal Policy Optimization)
for automated trading decisions.

Usage:
    from trading_modules.rl_agent import RLAgent, TradingEnv

    env = TradingEnv(df, window=20)
    agent = RLAgent(state_dim=env.observation_dim, action_dim=3)
    agent.train(env, n_episodes=100)
    action, confidence = agent.predict(current_features)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Try importing torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — RL agent disabled. Install: pip install torch")


# ═══════════════════════════════════════════════════════════════
# Config & Metrics (always available, no torch dependency)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    n_updates: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class TrainingMetrics:
    """Training metrics for one episode."""
    episode: int = 0
    total_reward: float = 0.0
    portfolio_value: float = 1.0
    n_trades: int = 0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "total_reward": round(self.total_reward, 4),
            "portfolio_value": round(self.portfolio_value, 4),
            "n_trades": self.n_trades,
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "policy_loss": round(self.policy_loss, 6),
            "value_loss": round(self.value_loss, 6),
            "entropy": round(self.entropy, 4),
        }


# ═══════════════════════════════════════════════════════════════
# Trading Environment (no torch dependency)
# ═══════════════════════════════════════════════════════════════

class TradingEnv:
    """
    Gymnasium-style trading environment.

    Observation: window of technical features (window_size × n_features)
    Action: 0=flat, 1=long, 2=short
    Reward: position_return - transaction_cost, with volatility penalty
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 20,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        reward_type: str = "sharpe",
    ):
        from .ml_models import build_features

        self.window = window
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_type = reward_type

        self.features = build_features(df)
        self.returns = df['close'].pct_change().fillna(0).values

        valid_mask = ~self.features.iloc[:, 0].isna()
        self.features = self.features[valid_mask].reset_index(drop=True)
        self.returns = self.returns[valid_mask.values]

        self.n_features = self.features.shape[1]
        self.n_steps = len(self.features) - window - 1

        self.current_step = 0
        self.position = 0
        self.position_history = []
        self.return_history = []

    @property
    def observation_dim(self) -> int:
        return self.window * self.n_features

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self) -> np.ndarray:
        self.current_step = self.window
        self.position = 0
        self.position_history = []
        self.return_history = []
        return self._get_observation()

    def step(self, action: int) -> tuple:
        new_position = {0: 0, 1: 1, 2: -1}[action]

        cost = 0.0
        if new_position != self.position:
            cost = self.transaction_cost

        self.position = new_position
        self.position_history.append(self.position)

        bar_return = self.returns[self.current_step] if self.current_step < len(self.returns) else 0.0
        strategy_return = self.position * bar_return - cost
        self.return_history.append(strategy_return)

        if self.reward_type == "sharpe" and len(self.return_history) > 10:
            recent = np.array(self.return_history[-20:])
            std = recent.std()
            reward = recent.mean() / std if std > 1e-8 else recent.mean()
        else:
            reward = strategy_return

        self.current_step += 1
        done = self.current_step >= self.n_steps

        info = {
            "position": self.position,
            "bar_return": bar_return,
            "strategy_return": strategy_return,
            "cost": cost,
            "step": self.current_step,
        }

        return self._get_observation(), float(reward), done, info

    def _get_observation(self) -> np.ndarray:
        start = max(0, self.current_step - self.window)
        end = self.current_step
        obs = self.features.iloc[start:end].values
        obs = obs.flatten().astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def get_portfolio_value(self) -> float:
        if not self.return_history:
            return 1.0
        return float(np.prod(1 + np.array(self.return_history)))


# ═══════════════════════════════════════════════════════════════
# PyTorch-dependent classes (only if torch available)
# ═══════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        """Actor-Critic network for PPO."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, state):
            features = self.feature_extractor(state)
            action_logits = self.actor(features)
            state_value = self.critic(features)
            return action_logits, state_value

        def get_action(self, state):
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, value.squeeze()

        def evaluate(self, states, actions):
            logits, values = self.forward(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, values.squeeze(), entropy


    class RLAgent:
        """
        PPO Reinforcement Learning agent for trading.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int = 3,
            config: Optional[PPOConfig] = None,
            device: str = "cpu",
        ):
            self.config = config or PPOConfig()
            self.device = torch.device(device)

            self.network = ActorCritic(state_dim, action_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.is_trained = False
            self.metrics_history: list[TrainingMetrics] = []

        def train(self, env: TradingEnv, n_episodes: int = 100, verbose: bool = True) -> list:
            self.metrics_history = []

            for ep in range(n_episodes):
                state = env.reset()
                done = False
                states, actions, rewards, log_probs, values = [], [], [], [], []

                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action, log_prob, value = self.network.get_action(state_tensor)
                    next_state, reward, done, info = env.step(action)
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    log_probs.append(log_prob.item())
                    values.append(value.item())
                    state = next_state

                advantages, returns = self._compute_gae(rewards, values)
                metrics = self._update(
                    states, actions, log_probs, values,
                    advantages, returns, ep, env,
                )
                self.metrics_history.append(metrics)

                if verbose and (ep % 10 == 0 or ep == n_episodes - 1):
                    print(
                        f"  Episode {ep:3d} | "
                        f"Reward: {metrics.total_reward:.2f} | "
                        f"Portfolio: {metrics.portfolio_value:.4f} | "
                        f"Trades: {metrics.n_trades} | "
                        f"Sharpe: {metrics.sharpe:.2f}"
                    )

            self.is_trained = True
            return self.metrics_history

        def predict(self, state: np.ndarray) -> tuple:
            if not self.is_trained:
                return 0, 0.33
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.network.forward(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = probs.argmax(dim=-1).item()
                confidence = probs[0, action].item()
            return action, confidence

        def predict_batch(self, states: np.ndarray) -> tuple:
            if not self.is_trained:
                n = len(states)
                return np.zeros(n, dtype=int), np.full(n, 0.33)
            states_tensor = torch.FloatTensor(states).to(self.device)
            with torch.no_grad():
                logits, _ = self.network.forward(states_tensor)
                probs = torch.softmax(logits, dim=-1)
                actions = probs.argmax(dim=-1).cpu().numpy()
                confidences = probs.max(dim=-1).values.cpu().numpy()
            return actions, confidences

        def save(self, path: str) -> None:
            torch.save({
                "network_state": self.network.state_dict(),
                "config": self.config.__dict__,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "is_trained": self.is_trained,
            }, path)
            logger.info(f"RL agent saved to {path}")

        def load(self, path: str) -> None:
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint["network_state"])
            self.is_trained = checkpoint["is_trained"]
            logger.info(f"RL agent loaded from {path}")

        def _compute_gae(self, rewards, values):
            rewards = np.array(rewards)
            values = np.array(values + [0])
            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                advantages[t] = gae
            returns = advantages + values[:-1]
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return advantages, returns

        def _update(self, states, actions, old_log_probs, values, advantages, returns, episode, env):
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_updates = 0

            n_samples = len(states)
            batch_size = min(self.config.batch_size, n_samples)

            for _ in range(self.config.n_updates):
                perm = torch.randperm(n_samples)
                batches = perm.split(batch_size)

                for batch_idx in batches:
                    s = states_tensor[batch_idx]
                    a = actions_tensor[batch_idx]
                    old_lp = old_log_probs_tensor[batch_idx]
                    adv = advantages_tensor[batch_idx]
                    ret = returns_tensor[batch_idx]

                    log_probs, critic_values, entropy = self.network.evaluate(s, a)

                    ratio = torch.exp(log_probs - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.MSELoss()(critic_values, ret)
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy_loss.item()
                    n_updates += 1

            returns_hist = np.array(env.return_history)
            n_trades = sum(1 for i in range(1, len(env.position_history))
                          if env.position_history[i] != env.position_history[i-1])

            if len(returns_hist) > 1 and returns_hist.std() > 1e-8:
                sharpe = float(returns_hist.mean() / returns_hist.std() * np.sqrt(252))
            else:
                sharpe = 0.0

            cumulative = np.cumprod(1 + returns_hist)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            return TrainingMetrics(
                episode=episode,
                total_reward=float(returns_hist.sum()),
                portfolio_value=env.get_portfolio_value(),
                n_trades=n_trades,
                sharpe=sharpe,
                max_drawdown=max_dd,
                policy_loss=total_policy_loss / max(n_updates, 1),
                value_loss=total_value_loss / max(n_updates, 1),
                entropy=total_entropy / max(n_updates, 1),
            )

        def get_training_summary(self) -> dict:
            if not self.metrics_history:
                return {"trained": False}
            last = self.metrics_history[-1]
            best = max(self.metrics_history, key=lambda m: m.sharpe)
            return {
                "trained": self.is_trained,
                "n_episodes": len(self.metrics_history),
                "final_sharpe": round(last.sharpe, 4),
                "final_portfolio": round(last.portfolio_value, 4),
                "best_sharpe": round(best.sharpe, 4),
                "best_episode": best.episode,
                "final_max_dd": round(last.max_drawdown, 4),
                "final_n_trades": last.n_trades,
            }

else:
    # Critical #7 fix: stub that returns None from a factory function instead
    # of raising ImportError on instantiation. This allows modules that
    # import RLAgent to load without PyTorch installed — they just get None
    # when they try to create an agent, which they can handle gracefully.
    class RLAgent:
        """Stub RLAgent — PyTorch not installed.

        Use create_rl_agent() factory instead of direct instantiation.
        Returns None if PyTorch is unavailable.
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch\n"
                "RL agent requires PyTorch for neural network training.\n"
                "Alternatively, use create_rl_agent() which returns None "
                "instead of raising."
            )


def create_rl_agent(*args, **kwargs):
    """Critical #7 fix: factory function that returns None (instead of
    raising ImportError) when PyTorch is not available. Callers should
    check for None before using the agent.

    Usage:
        agent = create_rl_agent(state_dim=10, action_dim=3)
        if agent is not None:
            agent.train(...)
    """
    if not TORCH_AVAILABLE:
        return None
    return RLAgent(*args, **kwargs)

import random
from collections import defaultdict
from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta
from gymnasium import spaces

from models.DPT import DPT, OptimizedPortfolio
from utils.polars import sliding_window


class DPTEnv(gym.Env):
    DPT_WINDOW: int = 194  # ie: 12months *16years + 1month (for logR) + 1month for forecasting

    def __init__(
        self, full_returns: pl.DataFrame, index_ticker: str, discount_factor: float = 0.003, liabilities_window: int = 5
    ):
        super().__init__()
        self.full_returns: pl.DataFrame = full_returns
        self.index_ticker: str = index_ticker
        self.discount_factor: float = discount_factor
        self.liabilities_window: int = liabilities_window

        # internal attributes
        self.liabilities: np.ndarray[np.float32] | None = None
        self.sliding_returns: pl.DataFrame | None = None
        self.ptf_value: float | None = None
        self.ptf_expected_return: float | None = None

        # --- Action space ---
        # (min, max)
        S = (5, 50)
        L = (0.01, 0.2)
        M = (0.05, 0.5)
        betas = ([0.01] * 24, [2.0] * 24)
        alphas = ([0.01] * 24, [1.0] * 24)

        self.action_space = spaces.Box(
            low=np.array([S[0], L[0], M[0], *betas[0], *alphas[0]], dtype=np.float32),
            high=np.array([S[1], L[1], M[1], *betas[1], *alphas[1]], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Observation space ---
        self.observation_space = spaces.Dict(
            {
                "ptf_value": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "liabilities": spaces.Box(low=0, high=np.inf, shape=(2 * self.liabilities_window,), dtype=np.float32),
            }
        )

    def step(self, action: np.array):
        params = self._transform_action(action)
        try:
            returns = next(self.sliding_returns)
        except StopIteration:
            raise RuntimeError("No more sliding returns available.") from None
        obs_returns = returns[:-1]
        dpt = DPT(data=obs_returns, index_ticker=self.index_ticker)
        dpt.calculate_signals()
        future_returns = returns[-1].to_dicts()[0]
        optptf: OptimizedPortfolio = dpt.solve(mu=future_returns, **params)

        # Update the portfolio value
        ptf_arithmetic_return = np.exp(optptf.ptf_return)  # log return -> arithmetic return
        self.ptf_value *= 1 + ptf_arithmetic_return

        # Update the observation (this deducts liabilities that have come to maturity)
        observation = self._get_obs()

        # Calculate the reward
        reward = self._calculate_reward(optptf)

        # Check if portfolio value is below or equal to 0 or if all liabilities have matured
        done = self.ptf_value <= 0 or self.liabilities[self.liabilities[:, 0] > 0].size == 0

        # Return the updated observation, reward, done flag, and info
        info = {
            "ptf_value": self.ptf_value,
            "step_return": ptf_arithmetic_return,
            "remaining_liabilities": self.liabilities[:, 1].sum(),
        }
        return observation, reward, done, info

    def reset(self, seed: int | None = None, options: dict = None):
        super().reset(seed=seed)

        options = options or {}
        self.ptf_value = options.get("start_amount", 100.0)
        self.ptf_expected_return = options.get("ptf_expected_return", 0.3)
        # TODO: add a random date generator respecting available data constraints (past>16years)
        start_date = options.get("start_date", date(2020, 1, 31))
        self.sliding_returns = sliding_window(self.full_returns, start_date, self.DPT_WINDOW)

        liabilities_dict = options.get(
            "liabilities",
            self._generate_dict_liabilities(
                start_date + relativedelta(months=6),
                start_date + relativedelta(years=2),
                self.ptf_value * (1 + self.ptf_return),
                seed=seed,
            ),
        )
        self.liabilities = self._dict_liabilities_to_observation(liabilities_dict, start_date)
        return self._get_obs(), {}

    def _calculate_reward(self, optptf: OptimizedPortfolio) -> float:
        reward = 0.0
        # 1. Portfolio return
        reward += np.exp(optptf.ptf_return)
        # 2. Liabilities management penalty
        discounted_liab = (self.liabilities[:, 1] / (1 + self.discount_factor) ** self.liabilities[:, 0]).sum()
        if self.ptf_value < discounted_liab:
            reward -= 10 * (discounted_liab - self.ptf_value)  # Penalize unmet liabilities
        # 3. Risk penalty
        risk_penalty = 0.01 * np.std(self.sliding_returns)  # Example risk factor
        reward -= risk_penalty
        # 4. Survival bonus
        if self.ptf_value > 0:
            reward += 0.1  # Small bonus for staying solvent
        return reward

    def _get_obs(self) -> dict[str, np.ndarray[np.float32]]:
        # Decrement liabilities (time to maturity decreases by 1 month)
        decrementer = np.array([-1.0, 0.0] * len(self.liabilities)).reshape(-1, 2)
        self.liabilities += decrementer

        # Filter liabilities that are still in view (positive months remaining)
        inview_liab = self.liabilities[self.liabilities[:, 0] > 0][: self.liabilities_window]
        inview_liab = inview_liab.flatten()

        # Handle liabilities that have matured (months remaining == 0)
        if (current_liab := self.liabilities[self.liabilities[:, 0] == 0]).size > 0:
            self.ptf_value += current_liab[:, 1].sum()

        # Return the observation as a dictionary
        observation = {
            "ptf_value": np.array([self.ptf_value], dtype=np.float32),
            "liabilities": np.pad(
                inview_liab, (0, 2 * self.liabilities_window - len(inview_liab)), constant_values=0
            ).astype(np.float32),
        }
        return observation

    def render(self, mode="human"):
        print(f"Portfolio Value: {self.ptf_value}")
        print(f"Liabilities: {self.liabilities}")

    def seed(self, seed: int | None = None):
        random.seed(seed)
        np.random.seed(seed)

    def _transform_action(self, action: np.array) -> dict[str, Any]:
        if len(action) != 3 + 24 + 24:  # S, L, M, 24 betas, 24 alphas
            raise ValueError(f"Invalid action length: {len(action)}. Expected 51 elements.")
        S = round(action[0])
        L = action[1]
        M = action[2]
        C_betas = dict(zip(np.arange(1, 24 + 1), action[3 : 3 + 24], strict=False))
        C_alphas = dict(zip(np.arange(1, 24 + 1), action[3 + 24 :], strict=False))
        return dict(S=S, C_betas=C_betas, C_alphas=C_alphas, L=L, M=M)

    def _generate_dict_liabilities(
        self,
        min_date: date,
        max_date: date,
        total_amount: float,
        nb_min: int = 3,
        nb_max: int = 10,
        seed: int | None = None,
    ) -> dict[date, float]:
        """
        Note:
            - The sum of amounts may differ slighty from the parameter due do rounding error.
            - The number of liabilities may be lower to the parameter due to merging issue.
        """
        total_amount = -total_amount if total_amount > 0 else total_amount
        random.seed(seed)

        num_liabilities = random.randint(nb_min, nb_max)

        weights = [random.random() for _ in range(num_liabilities)]
        weight_sum = sum(weights)
        amounts = [round(total_amount * w / weight_sum, 2) for w in weights]

        # Generate liabilities
        days_range = (max_date - min_date).days
        liabilities = defaultdict(lambda: 0.0)
        for amount in amounts:
            offset = random.randint(0, days_range)
            date = min_date + relativedelta(days=offset)
            liabilities[date] += amount
        return dict(sorted(liabilities.items()))

    def _dict_liabilities_to_observation(
        self, liab_dict: dict[date, float], current_date: date
    ) -> np.ndarray[np.float32]:
        months_between = lambda d1, d2: (d2.year - d1.year) * 12 + (d2.month - d1.month)  # noqa: E731

        array = defaultdict(lambda: 0.0)
        for k, v in sorted(liab_dict.items()):
            delta = months_between(current_date, k)
            array[delta] += v

        return np.array(list(zip(array.keys(), array.values(), strict=False)), dtype=np.float32)

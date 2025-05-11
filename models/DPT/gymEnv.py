import logging
import random
from collections import defaultdict
from collections.abc import Generator
from datetime import date
from numbers import Number
from typing import Any

import gymnasium as gym
import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta
from gymnasium import spaces

from models.DPT import DPT, OptimizedPortfolio
from utils.polars import sliding_window

logger = logging.getLogger(__name__)

START_DATE = date(2021, 1, 31)
DISCOUNT_FACTOR = 0.035
START_AMOUNT = 1000.0
LIAB_NB_MIN = 6
LIAB_NB_MAX = 10
LIAB_DELTA_MIN = 2
LIAB_DELTA_MAX = 15
LIAB_PERC_RETURN = 1.4


class DPTEnv(gym.Env):
    DPT_WINDOW: int = 194  # ie: 12months *16years + 1month (for logR) + 1month for forecasting

    def __init__(self, full_returns: pl.DataFrame, index_ticker: str, liabilities_window: int = 5):
        super().__init__()
        self.full_returns: pl.DataFrame = full_returns
        self.index_ticker: str = index_ticker

        self.liabilities_window: int = liabilities_window  # Number of liabilities visible simultaneously by the agent

        # internal attributes
        self.liabilities: np.ndarray[np.float32] | None = None
        self.decrementer: np.ndarray[np.float32] | None = None
        self.returns_slider: Generator | None = None
        self.sliding_returns: pl.DataFrame | None = None
        self.start_amount: Number | None = None
        self.ptf_value: Number | None = None
        self.dpt: DPT | None = None
        self.info: dict[str, Any] = {}
        self.discount_factor_m: Number | None = None  # discount factor to apply to futures liabilities
        self.retry_step: bool = False  # Catch non feaseable solution in the DPT framework

        # --- Action space ---
        # (min, max)
        S = (5, 50)
        L = (0.01, 0.2)
        M = (0.05, 0.4)
        betas = ([0.45] * 24, [2.0] * 24)
        alphas = ([0.15] * 24, [1.0] * 24)

        self.action_space = spaces.Box(
            low=np.array([S[0], L[0], M[0], *betas[0], *alphas[0]], dtype=np.float32),
            high=np.array([S[1], L[1], M[1], *betas[1], *alphas[1]], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Observation space ---
        self.observation_space = spaces.Dict(
            {
                "ptf_value": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "liabilities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.liabilities_window, 2), dtype=np.float32
                ),
            }
        )
        logger.info("Initialisation completed.")

    def step(self, action: np.array):
        params = self._transform_action(action)
        if not self.retry_step:
            try:
                self.sliding_returns = next(self.returns_slider)
            except StopIteration:
                raise RuntimeError("No more sliding returns available.") from None
            obs_returns = self.sliding_returns[:-1]
            self.dpt = DPT(data=obs_returns, index_ticker=self.index_ticker)
            self.dpt.calculate_signals()

        future_returns = self.sliding_returns[-1].to_dicts()[0]
        try:
            optptf: OptimizedPortfolio = self.dpt.solve(mu=future_returns, **params)
            self.retry_step = False
        except ValueError as e:
            self.retry_step = True
            logger.warning(e)
            reward = -10
            terminated = False
            self.info["status"] = 0
            return self._get_obs(), reward, terminated, False, self.info

        # Update PTF value
        # --- Handle liabilities that have matured (months remaining == 0) ---
        if (current_liab := self.liabilities[self.liabilities[:, 0] == 0]).size > 0:
            self.ptf_value += current_liab[:, 1].sum()

        ptf_arithmetic_return = np.exp(optptf.ptf_return) - 1  # log return -> arithmetic return
        self.ptf_value *= 1 + ptf_arithmetic_return

        # Decrement liabilities (time to maturity decreases by 1 month)
        self.liabilities += self.decrementer

        # Calculate the reward
        reward = self._calculate_reward(optptf)

        # Check if portfolio value is below or equal to 0 or if all liabilities have matured
        if self.ptf_value <= 0:
            terminated = True
            logger.info("Terminated due to negative portfolio value.")
        elif self.liabilities[self.liabilities[:, 0] > 0].size == 0:
            terminated = True
            logger.info("Terminated due to no liabilities left.")
        else:
            terminated = False
        # Return the updated observation, reward, done flag, and info
        self.info = {
            "status": 1,
            "ptf_value": self.ptf_value,
            "step_return": ptf_arithmetic_return,
            "remaining_liabilities": self.liabilities[:, 1].sum(),
        }
        return self._get_obs(), reward, terminated, False, self.info

    def reset(self, seed: int | None = None, options: dict = None):
        super().reset(seed=seed)

        options = options or {}
        self.start_amount = options.get("start_amount", START_AMOUNT)
        self.ptf_value = self.start_amount

        discount_factor_annual = options.get("discount_factor", DISCOUNT_FACTOR)
        self.discount_factor_m = (1 + discount_factor_annual) ** (1 / 12) - 1
        # TODO: add a random date generator respecting available data constraints (past>16years)
        start_date = options.get("start_date", START_DATE)
        self.returns_slider = sliding_window(self.full_returns, start_date, self.DPT_WINDOW)

        liabilities_dict = options.get(
            "liabilities",
            self._generate_dict_liabilities(
                min_date=start_date + relativedelta(months=LIAB_DELTA_MIN),
                max_date=start_date + relativedelta(months=LIAB_DELTA_MAX),
                total_amount=self.ptf_value * (1 + options.get("ptf_expected_return", LIAB_PERC_RETURN)),
                nb_min=LIAB_NB_MIN,
                nb_max=LIAB_NB_MAX,
                seed=seed,
            ),
        )
        self.liabilities = self._dict_liabilities_to_observation(liabilities_dict, start_date)
        self.decrementer = np.array([-1.0, 0.0] * len(self.liabilities)).reshape(-1, 2)
        return self._get_obs(), {}

    def render(self, mode="human"):
        logger.info(f"Portfolio Value-: {self.ptf_value}")
        logger.info(f"Discounted Liabilities left: {self._get_discounted_liab_val()}")
        logger.info(f"Next liability: {self.liabilities[self.liabilities[:, 0] >= 0][0]}")
        logger.info(f"Months lefts: {self.liabilities[-1, 0]}")

    def seed(self, seed: int | None = None):
        random.seed(seed)
        np.random.seed(seed)

    def _get_discounted_liab_val(self):
        liab_left = self.liabilities[self.liabilities[:, 0] > 0]
        return (liab_left[:, 1] / (1 + self.discount_factor_m) ** liab_left[:, 0]).sum()

    def _calculate_reward(self, optptf: OptimizedPortfolio) -> float:
        reward = 0.0
        # 1. Portfolio return
        reward += 10 * (1 + optptf.ptf_return)
        discounted_liab_val = self._get_discounted_liab_val()
        # 2. Liabilities management penalty
        if (pnl := self.ptf_value + discounted_liab_val) < 0:
            reward -= pnl / 10  # Penalize unmet liabilities
        # 3. Survival bonus
        brut_return = self.ptf_value / self.start_amount
        if brut_return < 0.5:
            reward -= 100
        elif brut_return < 0.8:
            reward -= 50
        elif brut_return < 1:
            reward -= 20
        elif brut_return < 1.2:
            reward += 20
        elif brut_return > 1.4:
            reward += 30

        # TODO: add reward for high variance between C_betas and C_alphas
        return reward

    def _get_obs(self) -> dict[str, np.ndarray[np.float32]]:
        # Filter liabilities that are still in view (positive months remaining)
        inview_liab = self.liabilities[self.liabilities[:, 0] > 0][: self.liabilities_window]

        # Pad liabilities
        padding = np.zeros((max(0, self.liabilities_window - len(inview_liab)), 2))
        obs_liab = np.concatenate((inview_liab, padding), axis=0, dtype=np.float32)

        logger.info(f"ptf_value shape: {np.array([self.ptf_value], dtype=np.float32).shape}")
        logger.info(f"liabilities shape: {obs_liab.shape}")

        # Return the observation as a dictionary
        observation = {"ptf_value": np.array([self.ptf_value], dtype=np.float32), "liabilities": obs_liab}
        # TODO: Add and returns csd to observation
        return observation

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

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pulp as plp
from scipy.signal import coherence, csd, welch

from utils.polars import TimesSeriesPolars

logger = logging.getLogger(__name__)


@dataclass
class OptimizedPortfolio:
    weights: pl.DataFrame
    returns: pl.DataFrame
    betas: pl.DataFrame
    alphas: pl.DataFrame
    R: pl.DataFrame
    scaling_factor: float

    @property
    def ptf_return(self) -> float:
        return sum((self.weights * self.returns).row(0))

    @property
    def ptf_betas(self) -> pl.DataFrame:
        return self._calc_ptf_alphas_betas(self.betas, "ptf_betas")

    @property
    def ptf_alphas(self) -> pl.DataFrame:
        return self._calc_ptf_alphas_betas(self.alphas, "ptf_alphas")

    def _calc_ptf_alphas_betas(self, theta_like: pl.DataFrame, colname: str) -> pl.DataFrame:
        return pl.DataFrame(
            (theta_like.to_numpy() * self.R.to_numpy() * self.weights.to_numpy() * self.scaling_factor).sum(axis=1),
            schema=[colname],
        )


@dataclass
class DPT(TimesSeriesPolars):
    logR: pl.DataFrame = field(default=None, init=False)
    R: pl.DataFrame = field(default=None, init=False)
    csd: pl.DataFrame = field(default=None, init=False)
    coherence: pl.DataFrame = field(default=None, init=False)
    theta: pl.DataFrame = field(default=None, init=False)
    cos_theta: pl.DataFrame = field(default=None, init=False)
    sin_theta: pl.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data must be provided")
        logger.info("Initializing DPT with provided data.")
        self.data = self.data.with_columns(
            (pl.col(col).log() - pl.col(col).log().shift(1)).alias(col.replace("_Close", "_logR"))
            for col in self.data.columns
            if col.endswith("Close")
        )
        sorted_columns = ["Date"] + sorted(self.get(include_date=False, include_index=True).columns)
        self.data = self.data.select(sorted_columns)
        self.logR = self.get("logR", include_index=True, include_date=False)
        logger.debug("Initializing DPT with provided data.")

    def calculate_signals(self, T: int = 48, dt: int = 1):
        logger.info("Calculating signals with T=%d and dt=%d.", T, dt)
        fs = 1 / dt
        spectral_params = dict(fs=fs, nperseg=T, noverlap=T // 2, window="boxcar")

        index_logR = self.logR[self.index_ticker]
        assets_logR = self.logR.select(pl.all().exclude(self.index_ticker))

        # Calculate Fourier Transform
        _, R = welch(assets_logR.transpose(), **spectral_params)
        self.R = pl.DataFrame(np.sqrt(R[:, 1:]), schema=assets_logR.columns)  # Remove the first element (frequency 0)
        logger.debug("Fourier Transform calculated successfully.")

        # Calculate phase-shift
        ## Cross-Spectral Density
        _, csd_np = csd(assets_logR.transpose(), index_logR, **spectral_params)
        self.csd = pl.DataFrame(csd_np[:, 1:], schema=assets_logR.columns)  # see above
        ## Coherence
        _, coherence_np = coherence(assets_logR.transpose(), index_logR, **spectral_params)
        self.coherence = pl.DataFrame(coherence_np[:, 1:], schema=assets_logR.columns)  # see above
        ## Phase shift from cross-spectrum, use to convert complex to real
        self.theta = pl.DataFrame(np.angle(csd_np[:, 1:]), schema=assets_logR.columns)
        self.cos_theta = self.theta.select(pl.all().cos())
        self.sin_theta = self.theta.select(pl.all().sin())
        logger.debug("Phase-shift and coherence calculated successfully.")

    def solve(
        self,
        mu: dict[str, float],
        S: int = 15,
        C_betas: dict[int, float] | float = 1.2,
        C_alphas: dict[int, float] | float = 0.5,
        L: dict[str, float] | float = 0.02,
        M: dict[str, float] | float = 0.2,
        scaling_factor: float = 10.0,
    ) -> OptimizedPortfolio:
        """
        Solves the Dynamic Portfolio Theory (DPT) optimization problem to construct an optimal portfolio
        based on expected returns, risk constraints, and portfolio size.

        Parameters:
        ----------
        mu : dict[str, float]
            A dictionary mapping each security identifier (str) to its expected return (float).
        S : int, optional
            The desired number of securities to include in the portfolio. Default is 10.
        C_betas : dict[int, float] | float, optional
            A dictionary mapping each period (int) to the upper bound for systematic risk (float),
            or a single float value applied to all periods. Default is 1.2.
        C_alphas : dict[int, float] | float, optional
            A dictionary mapping each period (int) to the upper bound for unsystematic risk (float),
            or a single float value applied to all periods. Default is 0.5.
        L : dict[str, float] | float, optional
            A dictionary mapping each security identifier (str) to its minimum investment fraction (float),
            or a single float value applied to all securities. Default is 0.002.
        M : dict[str, float] | float, optional
            A dictionary mapping each security identifier (str) to its maximal investment fraction (float),
            or a single float value applied to all securities. Default is 0.02.
        scaling_factor : float, optional
        A scaling factor applied to risk constraints to normalize their values. Default is 10.0.

        Returns:
        -------
        None

        Notes:
        -----
        - The optimization problem is formulated as a Mixed-Integer Linear Programming (MILP) problem.
        - The objective is to maximize the expected portfolio return while satisfying constraints on:
            - Budget (weights sum to 1).
            - Systematic risk (beta) and unsystematic risk (alpha) for each period.
            - Portfolio size (number of selected securities equals `S`).
            - Minimum investment fractions for selected securities.
        - The function uses the `pulp` library to define and solve the optimization problem.
        - Results include:
            - `ptf_returns`: The optimal portfolio return.
            - `ptf_composition`: A dictionary of selected securities and their weights.
            - `ptf_betas`: Systematic risk coefficients for each period.
            - `ptf_alphas`: Unsystematic risk coefficients for each period.
        - If the solver fails to find an optimal solution, a `ValueError` is raised with the solver's status.
        """
        logger.info("Starting portfolio optimization")
        # --- 1. Variable transformations
        securities = self.csd.columns

        R = self._wrap_polars_to_dict(self.R)
        cos_theta = self._wrap_polars_to_dict(self.cos_theta)
        sin_theta = self._wrap_polars_to_dict(self.sin_theta)

        K = range(1, self.R.shape[0] + 1)
        if isinstance(C_betas, float | int):
            C_betas = dict(zip(K, np.repeat(C_betas, len(K)), strict=True))
        if isinstance(C_alphas, float | int):
            C_alphas = dict(zip(K, np.repeat(C_alphas, len(K)), strict=True))
        if isinstance(L, float | int):
            L = dict(zip(securities, np.repeat(L, len(securities)), strict=True))
        if isinstance(M, float | int):
            M = dict(zip(securities, np.repeat(M, len(securities)), strict=True))

        C_betas = {k: v / scaling_factor for k, v in C_betas.items()}
        C_alphas = {k: v / scaling_factor for k, v in C_alphas.items()}
        if scaling_factor != 1.0:
            logger.debug("Risk constraints scaled by factor of %.2f.", scaling_factor)

        # --- 2. Init of pulp variables
        dpt_problem = plp.LpProblem("DPT_Optimization", plp.LpMaximize)
        # Continuous weight variables (w_j)
        weights = plp.LpVariable.dicts("w", securities, lowBound=0, cat="Continuous")
        # Binary variables (z_j) to indicate if a security is selected
        selection = plp.LpVariable.dicts("z", securities, cat="Binary")  # z_j = 0 or 1
        logger.debug("Pulp variables initialized.")

        # --- 3. Define Objective Function ---
        # Maximize expected portfolio return
        dpt_problem += plp.lpSum(mu[j] * weights[j] for j in securities), "Total Expected Return"
        logger.debug("Objective function defined.")

        # --- 4. Define Constraints ---
        # Budget Constraint: Sum of weights = 1
        dpt_problem += plp.lpSum(weights[j] for j in securities) == 1, "Budget Constraint"
        logger.debug("Budget constraint added.")

        # Risk Constraints (Systematic and Unsystematic for each period k)
        for k in K:
            # Systematic Risk (Beta)
            dpt_problem += (
                plp.lpSum(weights[j] * R[k, j] * cos_theta[k, j] for j in securities) <= C_betas[k],
                f"Systematic_Risk_Upper_{k}",
            )
            dpt_problem += (
                plp.lpSum(weights[j] * R[k, j] * cos_theta[k, j] for j in securities) >= -C_betas[k],
                f"Systematic_Risk_Lower_{k}",
            )

            # Unsystematic Risk (Alpha)
            dpt_problem += (
                plp.lpSum(weights[j] * R[k, j] * sin_theta[k, j] for j in securities) <= C_alphas[k],
                f"Unsystematic_Risk_Upper_{k}",
            )
            dpt_problem += (
                plp.lpSum(weights[j] * R[k, j] * sin_theta[k, j] for j in securities) >= -C_alphas[k],
                f"Unsystematic_Risk_Lower_{k}",
            )
        logger.debug("Risk constraints added for all periods.")

        # Portfolio Size Constraint: Sum of selection variables = S
        dpt_problem += plp.lpSum(selection[j] for j in securities) == S, "Portfolio Size"
        logger.debug("Portfolio size constraint added.")

        # Linking Constraints and Minimum Holding
        for j in securities:
            # w_j <= z_j (Simplified as w_j <= 1 * z_j since max w_j is 1)
            dpt_problem += weights[j] <= selection[j], f"Link_Upper_{j}"
            # w_j >= L_j * z_j
            dpt_problem += weights[j] >= L[j] * selection[j], f"Link_Lower_Min_Holding_{j}"
            dpt_problem += weights[j] <= M[j] * selection[j], f"Link_Upper_Max_Holding_{j}"
        logger.debug("Linking constraints added for all securities.")

        # --- 6. Solve the Problem ---
        logger.info("All constraints have been initialized.")
        logger.info("Solving the optimization problem...")
        dpt_problem.solve()

        # --- 7. Results ---
        if dpt_problem.status != plp.LpStatusOptimal:
            raise ValueError(f"Solver finised with non-optimal status: {plp.LpStatus[dpt_problem.status]}")

        logger.info("Solver finished successfully with status: %s", plp.LpStatus[dpt_problem.status])

        ptf_weights = {j: weights[j].varValue for j in securities if selection[j].varValue}
        ptf_returns = {k: mu[k] for k in ptf_weights.keys() & mu.keys()}
        ptf_betas = self.cos_theta.select(pl.col(col) for col in ptf_weights.keys())
        ptf_alphas = self.sin_theta.select(pl.col(col) for col in ptf_weights.keys())
        ptf_R = self.R.select(pl.col(col) for col in ptf_weights.keys())

        ptf_total_return = plp.value(dpt_problem.objective)
        logger.info("Optimal portfolio return: %.4f", ptf_total_return)
        logger.info("Optimal portfolio weigths: %s", ptf_weights)

        return OptimizedPortfolio(
            weights=pl.DataFrame(ptf_weights),
            returns=pl.DataFrame(ptf_returns),
            betas=ptf_betas,
            alphas=ptf_alphas,
            R=ptf_R,
            scaling_factor=scaling_factor,
        )

    @staticmethod
    def _wrap_polars_to_dict(df: pl.DataFrame) -> dict[tuple[int, str], float]:
        K = pl.Series("Kth-harmonics", np.arange(1, df.shape[0] + 1))
        df = df.with_columns(K)
        df_pivot = df.unpivot(df.columns, index="Kth-harmonics", variable_name="Securities", value_name="psd")
        keys = tuple(zip(df_pivot["Kth-harmonics"].to_list(), df_pivot["Securities"].to_list(), strict=True))
        return dict(zip(keys, df_pivot["psd"].to_list(), strict=True))

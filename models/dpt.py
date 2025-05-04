from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pulp as plp
from scipy.signal import coherence, csd, welch

from utils.polars import TimesSeriesPolars


@dataclass
class DPT(TimesSeriesPolars):
    R: np.ndarray = field(default=None, init=False)
    csd: np.ndarray = field(default=None, init=False)
    coherence: np.ndarray = field(default=None, init=False)
    phase_shift: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data must be provided")
        self.data = self.data.with_columns(
            (pl.col(col).log() - pl.col(col).log().shift(1)).alias(col.replace("_Close", "_logR"))
            for col in self.data.columns
            if col.endswith("Close")
        )

    def calculate_signals(self, T: int = 48, dt: int = 1):
        fs = 1 / dt
        spectral_params = dict(fs=fs, nperseg=T, noverlap=T // 2, window="boxcar")

        # Calculate Fourier Transform
        log_returns = self.get("logR", include_index=True, include_date=False)
        _, self.R = welch(log_returns.transpose(), **spectral_params)
        self.R = self.R[:, 1:]  # Remove the first element (frequency 0)
        self.tickers = [col.removesuffix("_logR") for col in log_returns.columns]

        # Calculate phase-shift
        index_logR = log_returns[self.index_ticker]
        assets_logR = log_returns.select(pl.all().exclude(self.index_ticker)).transpose()
        ## Cross-Spectral Density
        _, self.csd = csd(assets_logR, index_logR, **spectral_params)
        self.csd = self.csd[:, 1:]  # Remove the first element (frequency 0)
        ## Coherence
        _, self.coherence = coherence(assets_logR, index_logR, **spectral_params)
        self.coherence = self.coherence[:, 1:]  # Remove the first element (frequency 0)
        ## Phase shift
        self.theta = np.angle(self.csd)  # Phase from cross-spectrum, use to convert complex to real

    def solve(
        self, C_betas: dict[int, float], C_alphas: dict[int, float], L: dict[int, float], S: int, mu: dict[str, float]
    ) -> None:
        # K: Number of periodic components (e.g., 24 for T=48 months)
        # securities: List of security identifiers [j=0 to N-1]
        # periods: List of period identifiers [k=0 to K-1]
        # mu: Dictionary {j: expected_return_j}
        # R: Dictionary {(k, j): R_kj} (Std deviation of kth period return for security j)
        # cos_theta: Dictionary {(k, j): cos(theta_kmj)} (Systematic component factor)
        # sin_theta: Dictionary {(k, j): sin(theta_kmj)} (Unsystematic component factor)

        # C_betas: Dictionary {k: C_beta_k} (RHS for systematic risk constraints)
        # C_alphas: Dictionary {k: C_alpha_k} (RHS for unsystematic risk constraints)
        # S: Desired number of securities in the portfolio
        # L: Dictionary {j: L_j} (Minimum investment fraction for security j)

        dpt_problem = plp.LpProblem("DPT_Optimization", plp.LpMaximize)
        cos_theta = {k: np.cos(self.theta[k]) for k in range(self.R.shape[0])}
        sin_theta = {k: np.sin(self.theta[k]) for k in range(self.R.shape[0])}
        periods = range(self.R.shape[0])  # Assuming R is a 2D array with shape (K, N)
        securities = range(len(self.tickers))

        # Continuous weight variables (w_j)
        weights = plp.LpVariable.dicts("w", securities, lowBound=0, cat="Continuous")

        # Binary variables (z_j) to indicate if a security is selected
        selection = plp.LpVariable.dicts("z", securities, cat="Binary")  # z_j = 0 or 1

        # --- 4. Define Objective Function ---
        # Maximize expected portfolio return
        dpt_problem += plp.lpSum(mu[j] * weights[j] for j in securities), "Total Expected Return"

        # --- 5. Define Constraints ---
        # Budget Constraint: Sum of weights = 1
        dpt_problem += plp.lpSum(weights[j] for j in securities) == 1, "Budget Constraint"

        # Risk Constraints (Systematic and Unsystematic for each period k)
        for k in periods:
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

        # Portfolio Size Constraint: Sum of selection variables = S [cite: 330]
        dpt_problem += plp.lpSum(selection[j] for j in securities) == S, "Portfolio Size"

        # Linking Constraints and Minimum Holding [cite: 330]
        for j in securities:
            # w_j <= z_j (Simplified as w_j <= 1 * z_j since max w_j is 1)
            dpt_problem += weights[j] <= selection[j], f"Link_Upper_{j}"
            # w_j >= L_j * z_j
            dpt_problem += weights[j] >= L[j] * selection[j], f"Link_Lower_Min_Holding_{j}"

        # --- 6. Solve the Problem ---
        # You might need to specify a solver that handles MILP, e.g., CBC (default), GLPK, CPLEX, Gurobi
        # dpt_problem.solve(pulp.PULP_CBC_CMD(msg=1)) # Example using CBC
        dpt_problem.solve()

        # --- 7. Output Results ---
        print("Status:", plp.LpStatus[dpt_problem.status])
        print("Optimal Portfolio Return:", plp.value(dpt_problem.objective))
        print("Optimal Weights:")
        for j in securities:
            if selection[j].varValue > 0.5:  # Check if security j is selected
                print(f"  Security {j}: {weights[j].varValue:.4f}")
        print("Number of Securities in Portfolio:", sum(selection[j].varValue for j in securities))

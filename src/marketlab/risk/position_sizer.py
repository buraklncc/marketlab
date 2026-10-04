"""
position_sizer.py — Position sizing calculations.

Implements:
  - Kelly Criterion (half-Kelly)
  - CPPI (Constant Proportion Portfolio Insurance)
  - Becker-Parkinson volatility estimator
  - Discrete 5-level sizing
  - Combined sizing logic per strategy
"""

import numpy as np
from typing import Optional


class PositionSizer:
    """
    Computes position sizes using multiple methods.
    """

    def __init__(self,
                 max_position_pct: float = 0.10,
                 cppi_floor_pct: float = 0.80,
                 cppi_multiplier: float = 3.0):
        self.max_position_pct = max_position_pct
        self.cppi_floor_pct = cppi_floor_pct
        self.cppi_multiplier = cppi_multiplier

    # ── Kelly Criterion ──────────────────────────────────────────

    @staticmethod
    def kelly_fraction(mu: float, sigma: float) -> float:
        """
        Compute Kelly fraction.

        f = mu / sigma^2

        Parameters
        ----------
        mu : float     expected return (mean of return distribution)
        sigma : float  standard deviation of returns

        Returns
        -------
        float : Kelly fraction (can be > 1 or negative)
        """
        if sigma <= 0 or np.isnan(sigma):
            return 0.0
        f = mu / (sigma ** 2)
        return float(f)

    @staticmethod
    def half_kelly(mu: float, sigma: float) -> float:
        """Half-Kelly: f_actual = 0.5 * Kelly fraction, clamped to [0, 1]."""
        f = PositionSizer.kelly_fraction(mu, sigma)
        half_f = 0.5 * f
        return float(np.clip(half_f, 0.0, 1.0))

    # ── CPPI ─────────────────────────────────────────────────────

    def cppi_allocation(self, total_capital: float) -> float:
        """
        Compute CPPI risky allocation.

        floor = cppi_floor_pct * total_capital
        cushion = capital - floor
        risky_allocation = multiplier * cushion

        Returns
        -------
        float : max capital to risk (in USDT)
        """
        floor = self.cppi_floor_pct * total_capital
        cushion = max(0.0, total_capital - floor)
        risky = self.cppi_multiplier * cushion
        # Can't risk more than total capital
        return min(risky, total_capital)

    # ── Becker-Parkinson Volatility ──────────────────────────────

    @staticmethod
    def becker_parkinson(highs: np.ndarray, lows: np.ndarray) -> float:
        """
        Becker-Parkinson volatility estimator.

        BP = sqrt(1 / (4 * n * ln(2)) * sum((ln(H_i/L_i))^2))

        Parameters
        ----------
        highs : array of high prices
        lows : array of low prices

        Returns
        -------
        float : BP volatility estimate
        """
        n = len(highs)
        if n == 0:
            return 0.0

        valid = (highs > 0) & (lows > 0) & (lows <= highs)
        h = highs[valid]
        l = lows[valid]

        if len(h) == 0:
            return 0.0

        log_hl_sq = np.log(h / l) ** 2
        bp = np.sqrt(np.sum(log_hl_sq) / (4 * len(h) * np.log(2)))
        return float(bp)

    # ── Discrete Levels ──────────────────────────────────────────

    @staticmethod
    def discrete_level(probability: float) -> float:
        """
        Map probability to 5 discrete position levels.

        level 1: 0.2x (prob 0.4-0.52)
        level 2: 0.4x (prob 0.52-0.64)
        level 3: 0.6x (prob 0.64-0.76)
        level 4: 0.8x (prob 0.76-0.88)
        level 5: 1.0x (prob >= 0.88)
        """
        if probability < 0.4:
            return 0.0
        elif probability < 0.52:
            return 0.2
        elif probability < 0.64:
            return 0.4
        elif probability < 0.76:
            return 0.6
        elif probability < 0.88:
            return 0.8
        else:
            return 1.0

    # ── Combined Sizing ──────────────────────────────────────────

    def compute_size_cointegration(self, mu: float, sigma: float,
                                   total_capital: float,
                                   price: float) -> float:
        """
        Position size for cointegration strategy.

        Uses Half-Kelly constrained by CPPI allocation.

        Returns quantity (in base asset units).
        """
        # Half-Kelly fraction of capital
        hk_frac = self.half_kelly(mu, sigma)
        kelly_size_usd = hk_frac * total_capital

        # CPPI constraint
        cppi_max = self.cppi_allocation(total_capital)

        # Max position constraint
        max_pos_usd = self.max_position_pct * total_capital

        # Take the minimum
        size_usd = min(kelly_size_usd, cppi_max, max_pos_usd)
        size_usd = max(0.0, size_usd)

        # Convert to quantity
        if price <= 0:
            return 0.0
        return size_usd / price

    def compute_size_vpin(self, bp_volatility: float,
                          total_capital: float,
                          price: float) -> float:
        """
        Position size for VPIN strategy.

        size = base_capital / (BP * price) — inverse volatility sizing.
        """
        if bp_volatility <= 0 or price <= 0:
            return 0.0

        # Base capital per trade
        base_capital = self.max_position_pct * total_capital

        # Inverse volatility sizing
        size_usd = base_capital / bp_volatility
        # But cap at max position
        size_usd = min(size_usd, self.max_position_pct * total_capital)
        size_usd = max(0.0, size_usd)

        return size_usd / price

    def compute_size_gap(self, signal_strength: float,
                         total_capital: float,
                         price: float,
                         max_position_usd: Optional[float] = None) -> float:
        """
        Position size for gap strategy.

        Uses 5-level discrete sizing based on signal strength.
        """
        if price <= 0:
            return 0.0

        max_pos = max_position_usd or (self.max_position_pct * total_capital)

        # signal_strength is already a discrete level (0.2, 0.4, 0.6, 0.8, 1.0)
        size_usd = signal_strength * max_pos
        return size_usd / price


"""
strategy1_cointegration.py — Cointegrated Spread Mean Reversion (BTC/ETH).

Implementation:
  1. Kalman Filter for dynamic hedge ratio h_t
  2. Spread: S_t = log(BTC) - h_t * log(ETH)
  3. Half-life via Ornstein-Uhlenbeck OLS
  4. Z-score with rolling window = half_life
  5. Entry/exit on Z-score thresholds
  6. ADF test for cointegration validation
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from statsmodels.tsa.stattools import adfuller

from marketlab.strategies.base import Strategy, Signal, ExitSignal


class KalmanHedgeRatio:
    """
    Kalman Filter for dynamic hedge ratio estimation.

    State: h_t (hedge ratio)
    Observation: log(BTC_t) = h_t * log(ETH_t) + epsilon_t

    State transition: h_t = h_{t-1}  (random walk)
    Observation: y_t = H_t * h_t + v_t

    where y_t = log(BTC_t), H_t = log(ETH_t)
    """

    def __init__(self, delta: float = 1e-4, ve: float = 1e-3):
        """
        Parameters
        ----------
        delta : float  state transition variance (controls adaptation speed)
        ve : float     observation noise variance
        """
        self.delta = delta
        self.ve = ve

        # State estimate
        self.h: float = 0.0           # hedge ratio
        self.P: float = 1.0           # state covariance
        self._initialized = False

    def update(self, y: float, x: float) -> float:
        """
        Update the Kalman filter with a new observation.

        Parameters
        ----------
        y : float   log(BTC price)
        x : float   log(ETH price)

        Returns
        -------
        float : updated hedge ratio h_t
        """
        if not self._initialized:
            # Initialize with simple OLS-like estimate
            self.h = y / x if x != 0 else 1.0
            self.P = 1.0
            self._initialized = True
            return self.h

        # Prediction step
        h_pred = self.h
        P_pred = self.P + self.delta

        # Update step
        # Innovation
        y_hat = h_pred * x
        e = y - y_hat                    # innovation / residual

        # Innovation covariance
        S = x * P_pred * x + self.ve

        # Kalman gain
        K = P_pred * x / S if S != 0 else 0

        # State update
        self.h = h_pred + K * e
        self.P = (1 - K * x) * P_pred

        return self.h

    def reset(self) -> None:
        self.h = 0.0
        self.P = 1.0
        self._initialized = False


class CointegrationStrategy(Strategy):
    """
    Strategy 1: Cointegrated spread mean reversion on BTC/ETH.
    """

    def __init__(self,
                 symbol_y: str = "BTCUSDT",
                 symbol_x: str = "ETHUSDT",
                 z_entry: float = 2.0,
                 z_exit: float = 0.0,
                 z_stop: float = 3.0,
                 adf_window: int = 100,
                 adf_pvalue: float = 0.05,
                 min_half_life: int = 5,
                 max_half_life: int = 100,
                 kalman_delta: float = 1e-4,
                 kalman_ve: float = 1e-3):
        super().__init__("cointegration")

        self.symbol_y = symbol_y
        self.symbol_x = symbol_x
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.adf_window = adf_window
        self.adf_pvalue = adf_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

        # Kalman filter
        self.kalman = KalmanHedgeRatio(delta=kalman_delta, ve=kalman_ve)

        # State
        self._spreads: list[float] = []
        self._hedge_ratios: list[float] = []
        self._z_scores: list[float] = []
        self._half_life: float = 20.0
        self._cointegrated: bool = True
        self._current_z: float = 0.0
        self._position_side: Optional[str] = None  # "long_spread" | "short_spread"
        self._position_id: Optional[str] = None

        # Prices
        self._last_y: Optional[float] = None
        self._last_x: Optional[float] = None
        self._last_bar_time: Optional[str] = None

    async def warmup(self, df_y: pd.DataFrame, df_x: pd.DataFrame):
        """Warm up the Kalman filter using historical data."""
        common_idx = df_y.index.intersection(df_x.index)
        if len(common_idx) < 30:
            return

        from marketlab.logging_.logger import get_logger
        log = get_logger()
        log.simple("system", f"Warming up Cointegration strategy with {len(common_idx)} historical bars...")

        for ts in common_idx:
            price_y = float(df_y.loc[ts, "close"])
            price_x = float(df_x.loc[ts, "close"])
            self._process_pair(price_y, price_x)
            self._last_bar_time = str(ts)
            self._last_y = price_y
            self._last_x = price_x

    async def on_bar(self, symbol: str, bar: dict,
                     bars_history: Any = None, **kwargs) -> Optional[Signal | ExitSignal]:
        """
        Process a new bar for either BTC or ETH.
        Signals are only generated when both prices are updated.
        """
        price = float(bar["close"])

        if symbol == self.symbol_y:
            self._last_y = price
        elif symbol == self.symbol_x:
            self._last_x = price
        else:
            return None

        # Need both prices and a new bar time
        if self._last_y is None or self._last_x is None:
            return None

        ts = bar.get("timestamp")
        if ts == self._last_bar_time:
            return None

        self._last_bar_time = ts
        return self._process_pair(self._last_y, self._last_x)

    async def on_tick(self, symbol: str, tick: dict) -> Optional[Signal | ExitSignal]:
        """Cointegration strategy operates on bars, not ticks."""
        return None

    def _process_pair(self, price_y: float, price_x: float) -> Optional[Signal | ExitSignal]:
        """Core logic: update Kalman, compute spread, Z-score, generate signals."""

        log_y = np.log(price_y)
        log_x = np.log(price_x)

        # Step 1: Update Kalman filter for hedge ratio
        h = self.kalman.update(log_y, log_x)
        self._hedge_ratios.append(h)

        # Step 2: Compute spread
        spread = log_y - h * log_x
        self._spreads.append(spread)

        # Need enough data for half-life estimation
        if len(self._spreads) < 30:
            return None

        # Step 3: Estimate half-life via OU process
        self._half_life = self._estimate_half_life()

        # Clamp half_life
        hl = max(self.min_half_life, min(self.max_half_life, int(self._half_life)))

        # Step 4: Z-score
        spread_arr = np.array(self._spreads)
        if len(spread_arr) < hl:
            return None

        rolling_mean = np.mean(spread_arr[-hl:])
        rolling_std = np.std(spread_arr[-hl:], ddof=1)

        if rolling_std < 1e-10:
            self._current_z = 0.0
        else:
            self._current_z = (spread - rolling_mean) / rolling_std

        self._z_scores.append(self._current_z)

        # ADF test on recent spreads for cointegration check
        if len(self._spreads) >= self.adf_window:
            recent = np.array(self._spreads[-self.adf_window:])
            try:
                adf_result = adfuller(recent, maxlag=1, regression="c")
                self._cointegrated = bool(adf_result[1] < self.adf_pvalue)
            except Exception:
                self._cointegrated = False

        z = self._current_z

        # ── Exit logic (check first) ────────────────────────────
        if self._position_side is not None and self._position_id is not None:
            # Stop loss: |Z| >= z_stop
            if abs(z) >= self.z_stop:
                signal_type = "z_exit_stop"
                exit_sig = ExitSignal(
                    strategy=self.name,
                    symbol=self.symbol_y,
                    position_id=self._position_id,
                    signal_type=signal_type,
                    exit_price=price_y,
                    reason="Z-score exceeded stop threshold",
                    reasoning=f"Z-score: {z:.2f} (stop: {self.z_stop}), Half-life: {hl}",
                    metadata={"z_score": float(z), "half_life": int(hl)},
                )
                self._position_side = None
                self._position_id = None
                return exit_sig

            # Take profit: Z crosses 0
            if self._position_side == "long_spread" and z >= self.z_exit:
                exit_sig = ExitSignal(
                    strategy=self.name,
                    symbol=self.symbol_y,
                    position_id=self._position_id,
                    signal_type="z_exit_profit",
                    exit_price=price_y,
                    reason="Spread reverted to mean",
                    reasoning=f"Z-score: {z:.2f} (exit: {self.z_exit}), Half-life: {hl}",
                    metadata={"z_score": z, "half_life": hl},
                )
                self._position_side = None
                self._position_id = None
                return exit_sig

            if self._position_side == "short_spread" and z <= self.z_exit:
                exit_sig = ExitSignal(
                    strategy=self.name,
                    symbol=self.symbol_y,
                    position_id=self._position_id,
                    signal_type="z_exit_profit",
                    exit_price=price_y,
                    reason="Spread reverted to mean",
                    reasoning=f"Z-score: {z:.2f} (exit: {self.z_exit}), Half-life: {hl}",
                    metadata={"z_score": float(z), "half_life": int(hl)},
                )
                self._position_side = None
                self._position_id = None
                return exit_sig

            # Regime exit: ADF failed
            if not self._cointegrated:
                exit_sig = ExitSignal(
                    strategy=self.name,
                    symbol=self.symbol_y,
                    position_id=self._position_id,
                    signal_type="z_exit_stop",
                    exit_price=price_y,
                    reason="Pair lost cointegration",
                    reasoning=f"ADF test failed (p > {self.adf_pvalue})",
                    metadata={"z_score": float(z), "cointegrated": False},
                )
                self._position_side = None
                self._position_id = None
                return exit_sig

            return None  # Hold position

        # ── Entry logic ─────────────────────────────────────────
        if not self._cointegrated:
            return None  # Don't enter if not cointegrated

        if self._position_side is not None:
            return None  # Already in a position

        # Compute signal strength based on how far Z is beyond threshold
        strength = min(1.0, (abs(z) - self.z_entry) / (self.z_stop - self.z_entry)) if abs(z) > self.z_entry else 0.0

        if z < -self.z_entry:
            # Long spread: Buy BTC, Sell ETH
            return Signal(
                strategy=self.name,
                symbol=self.symbol_y,
                side="buy",
                signal_type="z_score_long",
                strength=strength,
                entry_price=price_y,
                reasoning=f"Z-score: {z:.2f} (entry: -{self.z_entry}), Hedge: {h:.4f}",
                metadata={
                    "z_score": float(z),
                    "hedge_ratio": float(h),
                    "half_life": int(hl),
                    "spread": float(spread),
                    "pair_symbol": self.symbol_x,
                    "pair_price": float(price_x),
                    "pair_side": "sell",
                    "cointegrated": bool(self._cointegrated),
                },
            )

        if z > self.z_entry:
            # Short spread: Sell BTC, Buy ETH
            return Signal(
                strategy=self.name,
                symbol=self.symbol_y,
                side="sell",
                signal_type="z_score_short",
                strength=strength,
                entry_price=price_y,
                reasoning=f"Z-score: {z:.2f} (entry: {self.z_entry}), Hedge: {h:.4f}",
                metadata={
                    "z_score": float(z),
                    "hedge_ratio": float(h),
                    "half_life": int(hl),
                    "spread": float(spread),
                    "pair_symbol": self.symbol_x,
                    "pair_price": float(price_x),
                    "pair_side": "buy",
                    "cointegrated": bool(self._cointegrated),
                },
            )

        return None

    def _estimate_half_life(self) -> float:
        """
        Estimate the half-life of the spread via OU process.

        dS = lambda * (mu - S) * dt + sigma * dW
        Regression: delta_S ~ lambda * S_lagged
        half_life = -log(2) / lambda
        """
        spread_arr = np.array(self._spreads)
        n = len(spread_arr)
        if n < 10:
            return 20.0

        # Use last 200 bars or available
        window = min(n, 200)
        s = spread_arr[-window:]

        # OLS: delta_s = alpha + beta * s_lagged
        s_lag = s[:-1]
        delta_s = np.diff(s)

        # Add intercept
        X = np.column_stack([np.ones(len(s_lag)), s_lag])
        try:
            beta = np.linalg.lstsq(X, delta_s, rcond=None)[0]
            lam = beta[1]  # mean-reversion speed

            if lam >= 0:
                # Not mean-reverting
                return self.max_half_life

            half_life = -np.log(2) / lam
            return max(1.0, half_life)
        except Exception:
            return 20.0

    def set_position(self, side: str, position_id: str) -> None:
        """Called after executor opens a position."""
        self._position_side = side
        self._position_id = position_id

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "hedge_ratio": self._hedge_ratios[-1] if self._hedge_ratios else None,
            "current_z": round(self._current_z, 4),
            "half_life": round(self._half_life, 1),
            "cointegrated": self._cointegrated,
            "spread_count": len(self._spreads),
            "position_side": self._position_side,
            "symbol_y": self.symbol_y,
            "symbol_x": self.symbol_x,
        }


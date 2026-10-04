"""
strategy3_gap.py — Intraday Buy-on-Gap strategy.

Implementation:
  1. Gap detection at UTC 00:00
  2. Momentum filter (20-day MA)
  3. Random Forest meta-labeling for exit prediction
  4. 5-level discrete position sizing
  5. Weekly drawdown tracking
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

from marketlab.strategies.base import Strategy, Signal, ExitSignal
from marketlab.config import MODELS_DIR


class GapStrategy(Strategy):
    """
    Strategy 3: Intraday Buy-on-Gap with RF meta-labeling.
    """

    def __init__(self,
                 symbols: Optional[list[str]] = None,
                 gap_std_multiplier: float = 1.0,
                 ma_period: int = 20,
                 daily_return_window: int = 90,
                 time_exit_hours: int = 24,
                 rf_exit_threshold: float = 0.4,
                 weekly_drawdown_limit: float = 0.05,
                 retrain_interval_days: int = 7):
        super().__init__("gap")

        self.symbols = symbols or []
        self.gap_std_multiplier = gap_std_multiplier
        self.ma_period = ma_period
        self.daily_return_window = daily_return_window
        self.time_exit_hours = time_exit_hours
        self.rf_exit_threshold = rf_exit_threshold
        self.weekly_drawdown_limit = weekly_drawdown_limit
        self.retrain_interval_days = retrain_interval_days

        # RF model
        self._model: Optional[RandomForestClassifier] = None
        self._model_path = MODELS_DIR / "strategy3_rf.pkl"
        self._last_retrain: Optional[datetime] = None
        self._load_model()

        # Per-symbol state
        self._daily_data: dict[str, pd.DataFrame] = {}  # symbol → daily OHLCV
        self._position_ids: dict[str, str] = {}           # symbol → position_id
        self._entry_times: dict[str, datetime] = {}
        self._entry_prices: dict[str, float] = {}
        self._last_bar_times: dict[str, str] = {}

        # Weekly drawdown
        self._weekly_pnl: float = 0.0
        self._week_start: Optional[str] = None
        self._disabled_until_next_week: bool = False

        # Training data accumulator
        self._training_data: list[dict] = []

    def _load_model(self) -> None:
        """Load pre-trained RF model if available."""
        if self._model_path.exists():
            try:
                self._model = joblib.load(self._model_path)
            except Exception:
                self._model = None

    def _save_model(self) -> None:
        """Save trained RF model to disk."""
        if self._model is not None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self._model_path)

    def set_daily_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Inject daily OHLCV data for gap detection."""
        self._daily_data[symbol] = df.copy()

    async def on_bar(self, symbol: str, bar: dict,
                     bars_history: Any = None, **kwargs) -> Optional[Signal | ExitSignal]:
        """
        Process a new bar (15m or daily).

        On daily bars: check for gap signals.
        On 15m bars: check for time exits and RF re-evaluation.
        """
        # Check weekly drawdown
        self._check_weekly_reset()
        if self._disabled_until_next_week:
            return None

        # Check for exit on existing positions
        if symbol in self._position_ids:
            return self._check_exit(symbol, bar)

        # Ensure we only process a new bar
        ts = bar.get("timestamp")
        if ts and self._last_bar_times.get(symbol) == ts:
            return None
        self._last_bar_times[symbol] = ts

        # Only check gap at the start of the day (first bar)
        if bars_history is not None and isinstance(bars_history, pd.DataFrame):
            return self._check_gap_entry(symbol, bar, bars_history, turbo=kwargs.get("turbo", False))

        return None

    async def on_tick(self, symbol: str, tick: dict) -> Optional[Signal | ExitSignal]:
        """Gap strategy operates on bars, not ticks."""
        return None

    def _check_gap_entry(self, symbol: str, bar: dict,
                         daily_df: pd.DataFrame, turbo: bool = False) -> Optional[Signal]:
        """
        Check for a gap-down entry signal.

        gap = (prev_low - current_open) / prev_low
        gap_signal = gap > std_90 * gap_std_multiplier
        """
        if len(daily_df) < self.daily_return_window + 1:
            return None

        current_open = float(bar.get("open", bar.get("close", 0)))
        current_price = float(bar["close"])

        # Previous day's low
        prev_low = float(daily_df.iloc[-2]["low"]) if len(daily_df) >= 2 else None
        if prev_low is None or prev_low <= 0:
            return None

        # Gap calculation
        gap = (prev_low - current_open) / prev_low

        # Standard deviation of daily returns
        closes = daily_df["close"].astype(float).values
        returns = np.diff(np.log(closes))
        if len(returns) < self.daily_return_window:
            return None
        std_90 = np.std(returns[-self.daily_return_window:])

        # Gap must be positive (gap down) and exceed threshold
        gap_signal = gap > self.gap_std_multiplier * std_90

        if not gap_signal:
            return None

        # Momentum filter: price > 20-day MA
        if len(closes) < self.ma_period:
            return None
        ma_20 = np.mean(closes[-self.ma_period:])
        if current_price <= ma_20:
            return None

        # RF meta-labeling for position sizing
        rf_prob = self._predict_probability(symbol, gap, current_price,
                                            closes, daily_df)

        # Discrete position sizing (5 levels)
        strength = self._discrete_level(rf_prob)

        # Check RF exit threshold
        if rf_prob < self.rf_exit_threshold:
            return None  # RF says don't enter

        return Signal(
            strategy=self.name,
            symbol=symbol,
            side="buy",
            signal_type="gap_entry",
            strength=strength,
            entry_price=current_price,
            reasoning=f"Gap: {gap:.2%} (std_90: {std_90:.4f}), RF Prob: {rf_prob:.2f}, Level: {self._strength_to_level(strength)}",
            metadata={
                "gap": round(gap, 6),
                "gap_std": round(std_90, 6),
                "ma_20": round(ma_20, 2),
                "rf_probability": round(rf_prob, 4),
                "position_level": self._strength_to_level(strength),
            },
        )

    def _check_exit(self, symbol: str, bar: dict) -> Optional[ExitSignal]:
        """Check time exit and RF re-evaluation."""
        position_id = self._position_ids.get(symbol)
        if position_id is None:
            return None

        current_price = float(bar["close"])
        entry_time = self._entry_times.get(symbol)

        # Time exit: 24h
        if entry_time:
            now = datetime.now(timezone.utc)
            hours_held = (now - entry_time).total_seconds() / 3600
            if hours_held >= self.time_exit_hours:
                return self._create_exit(symbol, current_price, "gap_exit_time",
                                         "24-hour time limit reached")

        # RF re-evaluation
        daily_df = self._daily_data.get(symbol)
        if daily_df is not None and self._model is not None:
            entry_price = self._entry_prices.get(symbol, current_price)
            gap = (entry_price - current_price) / entry_price if entry_price > 0 else 0
            closes = daily_df["close"].astype(float).values
            rf_prob = self._predict_probability(symbol, gap, current_price,
                                                closes, daily_df)
            if rf_prob < self.rf_exit_threshold:
                return self._create_exit(symbol, current_price, "gap_exit_model",
                                         "Model predicts unfavorable outcome")

        return None

    def _create_exit(self, symbol: str, price: float,
                     signal_type: str, reason: str) -> ExitSignal:
        position_id = self._position_ids.pop(symbol, "")
        self._entry_times.pop(symbol, None)

        entry_price = self._entry_prices.pop(symbol, price)
        pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
        self._weekly_pnl += pnl_pct

        return ExitSignal(
            strategy=self.name,
            symbol=symbol,
            position_id=position_id,
            signal_type=signal_type,
            exit_price=price,
            reason=reason,
            reasoning=f"Gap Exit: {reason} | Weekly PnL: {self._weekly_pnl:.2%}",
            metadata={"weekly_pnl": self._weekly_pnl},
        )

    def set_position(self, symbol: str, position_id: str,
                     entry_price: float) -> None:
        """Called after executor opens a position."""
        self._position_ids[symbol] = position_id
        self._entry_times[symbol] = datetime.now(timezone.utc)
        self._entry_prices[symbol] = entry_price

    # ── RF Meta-labeling ────────────────────────────────────────

    def _predict_probability(self, symbol: str, gap: float,
                             current_price: float,
                             closes: np.ndarray,
                             daily_df: pd.DataFrame) -> float:
        """Predict probability of successful gap trade using RF."""
        features = self._extract_features(gap, current_price, closes, daily_df)

        if self._model is None:
            # No model yet — return neutral probability
            return 0.5

        try:
            X = np.array([features])
            prob = self._model.predict_proba(X)[0][1]  # prob of class 1 (success)
            return float(prob)
        except Exception:
            return 0.5

    def _extract_features(self, gap: float, current_price: float,
                          closes: np.ndarray,
                          daily_df: pd.DataFrame) -> list[float]:
        """
        Extract features for RF model.

        Features:
          1. gap magnitude
          2. distance from MA20
          3. 5-day momentum
          4. 20-day volatility
          5. volume ratio (today vs 20d avg)
          6. hour of day
          7. day of week
        """
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        dist_ma = (current_price - ma_20) / ma_20 if ma_20 > 0 else 0

        # 5-day momentum
        if len(closes) >= 5:
            mom_5d = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
        else:
            mom_5d = 0

        # 20-day volatility
        if len(closes) >= 20:
            returns_20 = np.diff(np.log(closes[-21:]))
            vol_20 = np.std(returns_20) if len(returns_20) > 0 else 0
        else:
            vol_20 = 0

        # Volume ratio
        if "volume" in daily_df.columns and len(daily_df) >= 20:
            vols = daily_df["volume"].astype(float).values
            today_vol = vols[-1] if len(vols) > 0 else 0
            avg_vol = np.mean(vols[-20:])
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1
        else:
            vol_ratio = 1.0

        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()

        return [gap, dist_ma, mom_5d, vol_20, vol_ratio, hour, day_of_week]

    def train_model(self, trade_records: list[dict]) -> bool:
        """
        Train/retrain the Random Forest model on historical trades.

        Parameters
        ----------
        trade_records : list of dicts with at least:
            features (list[float]), label (0 or 1)

        Returns
        -------
        bool : True if training succeeded
        """
        if len(trade_records) < 20:
            return False

        try:
            X = np.array([r["features"] for r in trade_records])
            y = np.array([r["label"] for r in trade_records])

            self._model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(X, y)
            self._save_model()
            self._last_retrain = datetime.now(timezone.utc)
            return True
        except Exception:
            return False

    # ── Position sizing ─────────────────────────────────────────

    @staticmethod
    def _discrete_level(rf_probability: float) -> float:
        """
        Map RF probability to 5 discrete position levels.

        level 1: 0.2x (prob 0.4-0.52)
        level 2: 0.4x (prob 0.52-0.64)
        level 3: 0.6x (prob 0.64-0.76)
        level 4: 0.8x (prob 0.76-0.88)
        level 5: 1.0x (prob 0.88+)
        """
        if rf_probability < 0.4:
            return 0.0
        elif rf_probability < 0.52:
            return 0.2
        elif rf_probability < 0.64:
            return 0.4
        elif rf_probability < 0.76:
            return 0.6
        elif rf_probability < 0.88:
            return 0.8
        else:
            return 1.0

    @staticmethod
    def _strength_to_level(strength: float) -> int:
        """Convert strength (0-1) to level (1-5)."""
        if strength <= 0.2:
            return 1
        elif strength <= 0.4:
            return 2
        elif strength <= 0.6:
            return 3
        elif strength <= 0.8:
            return 4
        else:
            return 5

    # ── Weekly drawdown ─────────────────────────────────────────

    def _check_weekly_reset(self) -> None:
        """Reset weekly P&L at start of new week."""
        now = datetime.now(timezone.utc)
        week_key = f"{now.year}-W{now.isocalendar()[1]}"

        if self._week_start != week_key:
            self._week_start = week_key
            self._weekly_pnl = 0.0
            self._disabled_until_next_week = False

        if abs(self._weekly_pnl) > self.weekly_drawdown_limit:
            self._disabled_until_next_week = True

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "symbols_tracked": list(self._position_ids.keys()),
            "active_positions": len(self._position_ids),
            "weekly_pnl": round(self._weekly_pnl, 4),
            "disabled_until_next_week": self._disabled_until_next_week,
            "model_loaded": self._model is not None,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "training_samples": len(self._training_data),
        }


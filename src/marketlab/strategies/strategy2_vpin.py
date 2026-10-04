"""
strategy2_vpin.py — Volume-Synchronized Momentum (VPIN).

Implementation:
  1. Uses volume bars from VolumeBarBuilder
  2. VPIN = sum(|V_buy - V_sell|) over n bars / (n * V)
  3. Signal when VPIN > 90th percentile + directional bias
  4. Triple barrier exit: profit target, stop loss, time barrier
  5. Position sizing via inverse Becker-Parkinson volatility
"""

import numpy as np
from typing import Optional, Any
from collections import deque

from marketlab.strategies.base import Strategy, Signal, ExitSignal
from marketlab.data_feed.volume_bars import VolumeBarBuilder, VolumeBar


class VPINStrategy(Strategy):
    """
    Strategy 2: Volume-Synchronized Probability of Informed Trading.
    """

    def __init__(self,
                 symbol: str = "BTCUSDT",
                 vpin_window: int = 50,
                 percentile_window: int = 200,
                 percentile_threshold: float = 90.0,
                 directional_window: int = 5,
                 bp_window: int = 20,
                 profit_multiplier: float = 2.0,
                 stop_multiplier: float = 1.0,
                 time_barrier_bars: int = 50,
                 max_trades_24h: int = 3,
                 adv_lookback_days: int = 20,
                 bar_divisor: int = 50):
        super().__init__("vpin")

        self.symbol = symbol
        self.vpin_window = vpin_window
        self.percentile_window = percentile_window
        self.percentile_threshold = percentile_threshold
        self.directional_window = directional_window
        self.bp_window = bp_window
        self.profit_multiplier = profit_multiplier
        self.stop_multiplier = stop_multiplier
        self.time_barrier_bars = time_barrier_bars
        self.max_trades_24h = max_trades_24h
        self.adv_lookback_days = adv_lookback_days
        self.bar_divisor = bar_divisor

        # Volume bar builder (initialized with default; updated with ADV)
        self.bar_builder = VolumeBarBuilder(bar_volume=1000.0)
        self._adv_initialized = False

        # VPIN history
        self._vpin_values: deque[float] = deque(maxlen=percentile_window + 50)
        self._completed_bars: list[VolumeBar] = []

        # Position tracking
        self._position_id: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._entry_bar_idx: int = 0
        self._stop_loss: Optional[float] = None
        self._take_profit: Optional[float] = None
        self._position_side: Optional[str] = None

        # Trade counter (24h window)
        self._trade_timestamps: list[float] = []

        # Current state
        self._current_vpin: float = 0.0
        self._current_bp: float = 0.0
        self._threshold: float = 0.0

    def initialize_bar_volume(self, average_daily_volume: float) -> None:
        """Set bar volume = ADV / bar_divisor."""
        bar_vol = average_daily_volume / self.bar_divisor
        self.bar_builder.update_bar_volume(bar_vol)
        self._adv_initialized = True

    async def on_bar(self, symbol: str, bar: dict,
                     bars_history: Any = None) -> Optional[Signal | ExitSignal]:
        """VPIN strategy operates on ticks, not time-based bars."""
        return None

    async def on_tick(self, symbol: str, tick: dict) -> Optional[Signal | ExitSignal]:
        """
        Process each incoming tick from aggTrade stream.

        tick: { price, quantity, is_buyer_maker, timestamp }
        """
        if symbol != self.symbol:
            return None

        price = float(tick["price"])
        quantity = float(tick["quantity"])
        is_buyer_maker = tick.get("is_buyer_maker", False)
        timestamp = int(tick.get("timestamp", 0))

        # Feed to volume bar builder
        completed_bar = self.bar_builder.on_tick(
            price, quantity, is_buyer_maker, timestamp
        )

        if completed_bar is None:
            return None

        # New volume bar completed
        self._completed_bars.append(completed_bar)

        # Check position exit (triple barrier)
        if self._position_id is not None:
            return self._check_exit(completed_bar, price)

        # Compute VPIN
        return self._compute_and_signal(price)

    def _compute_and_signal(self, current_price: float) -> Optional[Signal]:
        """Compute VPIN and generate entry signals."""
        bars = self._completed_bars
        n = self.vpin_window

        if len(bars) < n:
            return None

        recent_bars = bars[-n:]

        # Step 3: VPIN calculation
        bar_vol = self.bar_builder.bar_volume
        imbalances = [abs(b.buy_volume - b.sell_volume) for b in recent_bars]
        vpin = sum(imbalances) / (n * bar_vol) if bar_vol > 0 else 0.0
        self._current_vpin = vpin
        self._vpin_values.append(vpin)

        # Compute threshold (90th percentile)
        if len(self._vpin_values) < 20:
            return None
        vpin_arr = np.array(list(self._vpin_values))
        self._threshold = float(np.percentile(vpin_arr, self.percentile_threshold))

        # Check 24h trade limit
        import time
        now = time.time()
        self._trade_timestamps = [t for t in self._trade_timestamps if now - t < 86400]
        if len(self._trade_timestamps) >= self.max_trades_24h:
            return None

        # Signal condition: VPIN > threshold
        if vpin <= self._threshold:
            return None

        # Directional bias from last N bars
        dir_bars = bars[-self.directional_window:]
        net_flow = sum(b.buy_volume - b.sell_volume for b in dir_bars)

        if net_flow == 0:
            return None

        # Becker-Parkinson volatility
        bp_vol = self._becker_parkinson(bars[-self.bp_window:])
        self._current_bp = bp_vol

        if bp_vol <= 0:
            return None

        # Determine side
        if net_flow > 0:
            side = "buy"
            signal_type = "vpin_long"
            stop_loss = current_price * (1 - self.stop_multiplier * bp_vol)
            take_profit = current_price * (1 + self.profit_multiplier * bp_vol)
        else:
            side = "sell"
            signal_type = "vpin_short"
            stop_loss = current_price * (1 + self.stop_multiplier * bp_vol)
            take_profit = current_price * (1 - self.profit_multiplier * bp_vol)

        # Signal strength: how far VPIN exceeds threshold
        strength = min(1.0, (vpin - self._threshold) / self._threshold) if self._threshold > 0 else 0.5

        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            side=side,
            signal_type=signal_type,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"VPIN: {vpin:.4f} > threshold: {self._threshold:.4f}, Net Flow: {net_flow:.1f}, BP Vol: {bp_vol:.4f}",
            metadata={
                "vpin": round(vpin, 6),
                "threshold": round(self._threshold, 6),
                "net_flow": net_flow,
                "bp_volatility": round(bp_vol, 6),
                "bar_count": len(bars),
            },
        )

    def _check_exit(self, bar: VolumeBar, current_price: float) -> Optional[ExitSignal]:
        """Check triple barrier exit conditions."""
        if self._position_id is None:
            return None

        bar_idx = len(self._completed_bars)
        bars_since_entry = bar_idx - self._entry_bar_idx

        # 1. Profit target
        if self._position_side == "long" and current_price >= self._take_profit:
            return self._create_exit("vpin_exit_profit", current_price, "Profit target hit")

        if self._position_side == "short" and current_price <= self._take_profit:
            return self._create_exit("vpin_exit_profit", current_price, "Profit target hit")

        # 2. Stop loss
        if self._position_side == "long" and current_price <= self._stop_loss:
            return self._create_exit("vpin_exit_stop", current_price, "Stop loss hit")

        if self._position_side == "short" and current_price >= self._stop_loss:
            return self._create_exit("vpin_exit_stop", current_price, "Stop loss hit")

        # 3. Time barrier
        if bars_since_entry >= self.time_barrier_bars:
            return self._create_exit("vpin_exit_time", current_price, "Time barrier reached")

        return None

    def _create_exit(self, signal_type: str, price: float, reason: str) -> ExitSignal:
        exit_sig = ExitSignal(
            strategy=self.name,
            symbol=self.symbol,
            position_id=self._position_id,
            signal_type=signal_type,
            exit_price=price,
            reason=reason,
            reasoning=f"Exit triggering: {reason}. Current price: {price}, Stop: {self._stop_loss}, TP: {self._take_profit}",
            metadata={
                "vpin": self._current_vpin,
                "bp_volatility": self._current_bp,
            },
        )
        self._position_id = None
        self._entry_price = None
        self._position_side = None
        return exit_sig

    def set_position(self, side: str, position_id: str,
                     entry_price: float, stop_loss: float,
                     take_profit: float) -> None:
        """Called after executor opens a position."""
        import time
        self._position_side = side
        self._position_id = position_id
        self._entry_price = entry_price
        self._entry_bar_idx = len(self._completed_bars)
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._trade_timestamps.append(time.time())

    @staticmethod
    def _becker_parkinson(bars: list[VolumeBar]) -> float:
        """
        Becker-Parkinson volatility estimator.

        BP = sqrt(1 / (4 * n * ln(2)) * sum((ln(H_i/L_i))^2))
        """
        if not bars:
            return 0.0
        n = len(bars)
        log_hl_sq = []
        for b in bars:
            if b.low > 0 and b.high > 0:
                log_ratio = np.log(b.high / b.low)
                log_hl_sq.append(log_ratio ** 2)

        if not log_hl_sq:
            return 0.0

        bp = np.sqrt(np.sum(log_hl_sq) / (4 * n * np.log(2)))
        return float(bp)

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "symbol": self.symbol,
            "current_vpin": round(self._current_vpin, 6),
            "threshold": round(self._threshold, 6),
            "bp_volatility": round(self._current_bp, 6),
            "completed_bars": len(self._completed_bars),
            "position_id": self._position_id,
            "position_side": self._position_side,
            "trades_24h": len(self._trade_timestamps),
            "adv_initialized": self._adv_initialized,
        }


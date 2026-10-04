"""
volume_bars.py — Tick aggregation into volume bars.

Aggregates incoming ticks (from Binance aggTrade stream) into
volume-based bars for VPIN calculation.

A new bar closes when cumulative volume reaches the bar_volume threshold.
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class VolumeTick:
    """A single tick from aggTrade."""
    price: float
    quantity: float
    is_buyer_maker: bool
    timestamp: int  # epoch ms


@dataclass
class VolumeBar:
    """A completed volume bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    tick_count: int
    open_time: int    # epoch ms
    close_time: int   # epoch ms
    vwap: float

    def to_dict(self) -> dict:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "tick_count": self.tick_count,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "vwap": self.vwap,
        }


class VolumeBarBuilder:
    """
    Aggregates ticks into volume bars.

    Parameters
    ----------
    bar_volume : float
        Volume threshold for each bar (e.g. ADV / 50).
    """

    def __init__(self, bar_volume: float):
        self.bar_volume = bar_volume
        self._completed_bars: list[VolumeBar] = []

        # Current bar accumulation
        self._reset_current()

        # Tick classification state
        self._last_price: Optional[float] = None
        self._last_direction: int = 1  # 1 = buy, -1 = sell

    def _reset_current(self) -> None:
        """Reset the current bar accumulator."""
        self._cur_open: Optional[float] = None
        self._cur_high: float = -float("inf")
        self._cur_low: float = float("inf")
        self._cur_close: float = 0.0
        self._cur_volume: float = 0.0
        self._cur_buy_vol: float = 0.0
        self._cur_sell_vol: float = 0.0
        self._cur_tick_count: int = 0
        self._cur_open_time: Optional[int] = None
        self._cur_close_time: int = 0
        self._cur_pv_sum: float = 0.0  # price * volume sum for VWAP

    def classify_tick(self, price: float) -> int:
        """
        Classify a tick as buy (+1) or sell (-1) using the tick rule.

        If price > previous → buy volume
        If price < previous → sell volume
        If equal → same as previous classification
        """
        if self._last_price is None:
            self._last_price = price
            return self._last_direction

        if price > self._last_price:
            direction = 1
        elif price < self._last_price:
            direction = -1
        else:
            direction = self._last_direction

        self._last_price = price
        self._last_direction = direction
        return direction

    def on_tick(self, price: float, quantity: float,
                is_buyer_maker: bool, timestamp: int) -> Optional[VolumeBar]:
        """
        Process a single tick.

        Parameters
        ----------
        price : float          trade price
        quantity : float       trade quantity
        is_buyer_maker : bool  from Binance aggTrade 'm' field
        timestamp : int        epoch ms

        Returns
        -------
        VolumeBar if a bar was completed, else None.
        """
        # Classify using tick rule
        direction = self.classify_tick(price)

        # Update current bar
        if self._cur_open is None:
            self._cur_open = price
            self._cur_open_time = timestamp

        self._cur_high = max(self._cur_high, price)
        self._cur_low = min(self._cur_low, price)
        self._cur_close = price
        self._cur_close_time = timestamp
        self._cur_volume += quantity
        self._cur_tick_count += 1
        self._cur_pv_sum += price * quantity

        # Classify volume
        if direction > 0:
            self._cur_buy_vol += quantity
        else:
            self._cur_sell_vol += quantity

        # Check if bar is complete
        if self._cur_volume >= self.bar_volume:
            bar = self._close_bar()
            return bar

        return None

    def _close_bar(self) -> VolumeBar:
        """Finalize the current bar and reset."""
        vwap = self._cur_pv_sum / self._cur_volume if self._cur_volume > 0 else self._cur_close

        bar = VolumeBar(
            open=self._cur_open,
            high=self._cur_high,
            low=self._cur_low,
            close=self._cur_close,
            volume=self._cur_volume,
            buy_volume=self._cur_buy_vol,
            sell_volume=self._cur_sell_vol,
            tick_count=self._cur_tick_count,
            open_time=self._cur_open_time,
            close_time=self._cur_close_time,
            vwap=vwap,
        )

        self._completed_bars.append(bar)
        self._reset_current()
        return bar

    def get_current_partial(self) -> dict:
        """Return the current (incomplete) bar state."""
        vol = self._cur_volume if self._cur_volume > 0 else 0
        return {
            "open": self._cur_open,
            "high": self._cur_high if self._cur_high > -float("inf") else None,
            "low": self._cur_low if self._cur_low < float("inf") else None,
            "close": self._cur_close,
            "volume": vol,
            "buy_volume": self._cur_buy_vol,
            "sell_volume": self._cur_sell_vol,
            "tick_count": self._cur_tick_count,
            "fill_pct": (vol / self.bar_volume * 100) if self.bar_volume > 0 else 0,
        }

    def get_completed_bars(self, n: Optional[int] = None) -> list[VolumeBar]:
        """Return the last n completed bars (or all if n is None)."""
        if n is None:
            return list(self._completed_bars)
        return list(self._completed_bars[-n:])

    def update_bar_volume(self, new_bar_volume: float) -> None:
        """Update the bar volume threshold (e.g., when ADV changes)."""
        self.bar_volume = new_bar_volume

    @property
    def total_bars(self) -> int:
        return len(self._completed_bars)


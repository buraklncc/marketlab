"""
base.py — Abstract base class for all strategies.

Every strategy must implement on_bar(), on_tick(), and get_state().
Defines the Signal dataclass used for all inter-module communication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Signal:
    """A trading signal emitted by a strategy."""
    strategy: str          # "cointegration" | "vpin" | "gap"
    symbol: str            # e.g. "BTCUSDT"
    side: str              # "buy" | "sell"
    signal_type: str       # e.g. "z_score_long", "vpin_short", "gap_entry"
    strength: float        # 0.0 to 1.0
    entry_price: float     # suggested entry price
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    reasoning: str = ""
    logs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "signal_type": self.signal_type,
            "strength": self.strength,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantity": self.quantity,
            "reasoning": self.reasoning,
            "logs": self.logs,
            "metadata": self.metadata,
        }


@dataclass
class ExitSignal:
    """An exit signal for an existing position."""
    strategy: str
    symbol: str
    position_id: str
    signal_type: str       # e.g. "z_exit_profit", "vpin_exit_time"
    exit_price: float
    reason: str
    reasoning: str = ""
    logs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "position_id": self.position_id,
            "signal_type": self.signal_type,
            "exit_price": self.exit_price,
            "reason": self.reason,
            "reasoning": self.reasoning,
            "logs": self.logs,
            "metadata": self.metadata,
        }


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.enabled: bool = True

    @abstractmethod
    async def on_bar(self, symbol: str, bar: dict,
                     bars_history: Any = None) -> Optional[Signal | ExitSignal]:
        """
        Called when a new OHLCV bar is completed.

        Parameters
        ----------
        symbol : str
        bar : dict with keys: open, high, low, close, volume, open_time
        bars_history : DataFrame or list of historical bars

        Returns
        -------
        Signal, ExitSignal, or None
        """
        ...

    @abstractmethod
    async def on_tick(self, symbol: str, tick: dict) -> Optional[Signal | ExitSignal]:
        """
        Called on each incoming tick (for tick-based strategies like VPIN).

        Parameters
        ----------
        symbol : str
        tick : dict with keys: price, quantity, is_buyer_maker, timestamp

        Returns
        -------
        Signal, ExitSignal, or None
        """
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """Return the internal state of this strategy for logging/display."""
        ...

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


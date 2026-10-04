"""
drawdown_tracker.py — Per-strategy and global drawdown tracking.

Tracks peak equity and current drawdown for each strategy
and globally across the portfolio.
"""

from typing import Optional
from datetime import datetime, timezone


class DrawdownTracker:
    """
    Tracks drawdown for individual strategies and globally.
    """

    def __init__(self):
        # Per-strategy tracking: strategy_name → {peak, current, drawdown, max_drawdown}
        self._strategies: dict[str, dict] = {}
        # Global tracking
        self._global_peak: float = 0.0
        self._global_equity: float = 0.0
        self._global_max_drawdown: float = 0.0

    def update_strategy(self, strategy_name: str, equity: float) -> dict:
        """
        Update drawdown tracking for a specific strategy.

        Parameters
        ----------
        strategy_name : str
        equity : float  current equity for this strategy

        Returns
        -------
        dict with keys: peak, equity, drawdown, drawdown_pct, max_drawdown, max_drawdown_pct
        """
        if strategy_name not in self._strategies:
            self._strategies[strategy_name] = {
                "peak": equity,
                "equity": equity,
                "drawdown": 0.0,
                "drawdown_pct": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "last_update": datetime.now(timezone.utc).isoformat(),
            }

        state = self._strategies[strategy_name]

        # Update peak
        if equity > state["peak"]:
            state["peak"] = equity

        state["equity"] = equity

        # Current drawdown
        state["drawdown"] = state["peak"] - equity
        state["drawdown_pct"] = (state["drawdown"] / state["peak"] * 100) if state["peak"] > 0 else 0.0

        # Max drawdown
        if state["drawdown"] > state["max_drawdown"]:
            state["max_drawdown"] = state["drawdown"]
            state["max_drawdown_pct"] = state["drawdown_pct"]

        state["last_update"] = datetime.now(timezone.utc).isoformat()
        return dict(state)

    def update_global(self, total_equity: float) -> dict:
        """
        Update global portfolio drawdown.

        Parameters
        ----------
        total_equity : float  total portfolio value

        Returns
        -------
        dict with drawdown info
        """
        if total_equity > self._global_peak:
            self._global_peak = total_equity

        self._global_equity = total_equity
        drawdown = self._global_peak - total_equity
        drawdown_pct = (drawdown / self._global_peak * 100) if self._global_peak > 0 else 0.0

        if drawdown > self._global_max_drawdown:
            self._global_max_drawdown = drawdown

        return {
            "peak": self._global_peak,
            "equity": total_equity,
            "drawdown": drawdown,
            "drawdown_pct": round(drawdown_pct, 2),
            "max_drawdown": self._global_max_drawdown,
            "max_drawdown_pct": round(
                (self._global_max_drawdown / self._global_peak * 100)
                if self._global_peak > 0 else 0.0, 2
            ),
        }

    def get_strategy_drawdown(self, strategy_name: str) -> Optional[dict]:
        return self._strategies.get(strategy_name)

    def get_all_drawdowns(self) -> dict:
        """Return all strategy drawdowns plus global."""
        return {
            "strategies": {k: dict(v) for k, v in self._strategies.items()},
            "global": self.update_global(self._global_equity),
        }

    def reset_strategy(self, strategy_name: str) -> None:
        """Reset drawdown tracking for a strategy."""
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]

    def reset_all(self) -> None:
        """Reset all drawdown tracking."""
        self._strategies.clear()
        self._global_peak = 0.0
        self._global_equity = 0.0
        self._global_max_drawdown = 0.0


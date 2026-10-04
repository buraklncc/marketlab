"""
backtest_engine.py — Run strategies on historical data.

Replays OHLCV data through strategies and paper trading simulator
to produce performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timezone

from marketlab.strategies.base import Signal, ExitSignal
from marketlab.strategies.strategy1_cointegration import CointegrationStrategy
from marketlab.strategies.strategy2_vpin import VPINStrategy
from marketlab.strategies.strategy3_gap import GapStrategy
from marketlab.risk.position_sizer import PositionSizer


class BacktestResult:
    """Results from a backtest run."""

    def __init__(self):
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.initial_balance: float = 0
        self.final_balance: float = 0

    @property
    def total_return(self) -> float:
        if self.initial_balance <= 0:
            return 0
        return (self.final_balance - self.initial_balance) / self.initial_balance

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        if np.std(returns) == 0:
            return 0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    @property
    def avg_trade_pnl(self) -> float:
        if not self.trades:
            return 0
        return np.mean([t.get("pnl", 0) for t in self.trades])

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return * 100, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "avg_trade_pnl": round(self.avg_trade_pnl, 2),
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 2),
            "trades": self.trades[-50:],  # Last 50 trades
            "equity_curve": self.equity_curve[::max(1, len(self.equity_curve) // 200)],
        }


class BacktestEngine:
    """
    Replays historical data through strategies.
    """

    def __init__(self, initial_balance: float = 10000.0,
                 fee_rate: float = 0.001, slippage: float = 0.0005):
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.sizer = PositionSizer()

    async def run_cointegration(self, bars_y: pd.DataFrame,
                                bars_x: pd.DataFrame,
                                symbol_y: str = "BTCUSDT",
                                symbol_x: str = "ETHUSDT") -> BacktestResult:
        """Backtest cointegration strategy on two price series."""
        result = BacktestResult()
        result.initial_balance = self.initial_balance
        balance = self.initial_balance

        strategy = CointegrationStrategy(symbol_y=symbol_y, symbol_x=symbol_x)
        position = None
        entry_price = 0
        entry_qty = 0
        entry_side = ""

        min_len = min(len(bars_y), len(bars_x))
        result.equity_curve.append(balance)

        for i in range(min_len):
            bar_y = {
                "open": float(bars_y.iloc[i]["open"]),
                "high": float(bars_y.iloc[i]["high"]),
                "low": float(bars_y.iloc[i]["low"]),
                "close": float(bars_y.iloc[i]["close"]),
                "volume": float(bars_y.iloc[i]["volume"]),
            }
            bar_x = {
                "close": float(bars_x.iloc[i]["close"]),
            }

            # Feed both bars
            await strategy.on_bar(symbol_x, bar_x)
            sig = await strategy.on_bar(symbol_y, bar_y)

            if sig is None:
                result.equity_curve.append(balance)
                continue

            price = float(bars_y.iloc[i]["close"])

            if isinstance(sig, ExitSignal) and position is not None:
                # Close
                exit_price = price * (1 - self.slippage if entry_side == "long" else 1 + self.slippage)
                if entry_side == "long":
                    pnl = (exit_price - entry_price) * entry_qty
                else:
                    pnl = (entry_price - exit_price) * entry_qty
                fee = exit_price * entry_qty * self.fee_rate
                pnl -= fee
                balance += pnl

                result.trades.append({
                    "symbol": symbol_y, "side": entry_side,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "quantity": entry_qty, "pnl": round(pnl, 2),
                    "fee": round(fee, 4), "reason": sig.signal_type,
                })
                position = None

            elif isinstance(sig, Signal) and position is None:
                # Open
                qty_usd = min(self.sizer.max_position_pct * balance, balance * 0.1)
                entry_price = price * (1 + self.slippage if sig.side == "buy" else 1 - self.slippage)
                entry_qty = qty_usd / entry_price if entry_price > 0 else 0
                entry_side = "long" if sig.side == "buy" else "short"
                fee = entry_price * entry_qty * self.fee_rate
                balance -= fee
                position = sig
                strategy.set_position("long_spread" if sig.side == "buy" else "short_spread", f"bt_{i}")

            result.equity_curve.append(balance)

        result.final_balance = balance
        return result

    async def run_gap(self, daily_bars: pd.DataFrame,
                      symbol: str = "BTCUSDT") -> BacktestResult:
        """Backtest gap strategy on daily data."""
        result = BacktestResult()
        result.initial_balance = self.initial_balance
        balance = self.initial_balance

        strategy = GapStrategy(symbols=[symbol])
        position = None
        entry_price = 0
        entry_qty = 0
        entry_idx = 0
        result.equity_curve.append(balance)

        for i in range(2, len(daily_bars)):
            bar = {
                "open": float(daily_bars.iloc[i]["open"]),
                "high": float(daily_bars.iloc[i]["high"]),
                "low": float(daily_bars.iloc[i]["low"]),
                "close": float(daily_bars.iloc[i]["close"]),
                "volume": float(daily_bars.iloc[i]["volume"]),
            }
            price = bar["close"]
            history = daily_bars.iloc[:i + 1]

            if position is not None:
                # Check time exit (simulate 24h = next bar on daily)
                if i - entry_idx >= 1:
                    exit_price = price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * entry_qty
                    fee = exit_price * entry_qty * self.fee_rate
                    pnl -= fee
                    balance += pnl
                    result.trades.append({
                        "symbol": symbol, "side": "long",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "quantity": entry_qty, "pnl": round(pnl, 2),
                        "fee": round(fee, 4), "reason": "gap_exit_time",
                    })
                    position = None
            else:
                sig = await strategy.on_bar(symbol, bar, history)
                if isinstance(sig, Signal):
                    qty_usd = sig.strength * self.sizer.max_position_pct * balance
                    entry_price = price * (1 + self.slippage)
                    entry_qty = qty_usd / entry_price if entry_price > 0 else 0
                    entry_idx = i
                    fee = entry_price * entry_qty * self.fee_rate
                    balance -= fee
                    position = sig

            result.equity_curve.append(balance)

        result.final_balance = balance
        return result


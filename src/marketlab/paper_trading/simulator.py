"""
simulator.py — Paper trading engine.

Simulates order execution with realistic fee and slippage modeling.
Uses real market prices but fake balance.
"""

import uuid
import math
import time
from typing import Optional
from datetime import datetime, timezone

from marketlab.bot_state import AppState, Position
from marketlab.logging_.logger import get_logger

logger = get_logger()


class PaperSimulator:
    """
    Paper trading simulator with fee + slippage modeling.

    Config:
      fee_rate = 0.001 (0.1% taker fee)
      slippage = base_slippage + (order_size / avg_volume) * impact_factor
      base_slippage = 0.0005
      impact_factor = 0.1
    """

    def __init__(self,
                 fee_rate: float = 0.001,
                 base_slippage: float = 0.0005,
                 impact_factor: float = 0.1):
        self.fee_rate = fee_rate
        self.base_slippage = base_slippage
        self.impact_factor = impact_factor

    def compute_slippage(self, order_size_usd: float,
                         avg_volume_usd: float) -> float:
        """
        Compute slippage based on order size relative to average volume.

        slippage = base_slippage + (order_size / avg_volume) * impact_factor
        """
        if avg_volume_usd <= 0:
            return self.base_slippage

        market_impact = (order_size_usd / avg_volume_usd) * self.impact_factor
        return self.base_slippage + market_impact

    def compute_fee(self, notional: float) -> float:
        """Compute trading fee on notional value."""
        return notional * self.fee_rate

    def execute_order(self, state: AppState, symbol: str, side: str,
                      quantity: float, price: float,
                      avg_volume_usd: float = 1_000_000,
                      strategy: str = "",
                      signal_type: str = "",
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None,
                      metadata: Optional[dict] = None,
                      logs: Optional[list[str]] = None) -> Optional[Position]:
        """
        Execute a simulated order.

        Applies slippage to the price and deducts fees from balance.

        Parameters
        ----------
        state : AppState
        symbol : str           e.g. "BTCUSDT"
        side : str             "buy" | "sell" (for opening position: "long" or "short")
        quantity : float       base asset quantity
        price : float          current market price
        avg_volume_usd : float average daily volume in USD
        strategy : str         strategy name
        signal_type : str      signal identifier
        stop_loss : float      optional stop loss price
        take_profit : float    optional take profit price
        metadata : dict        additional signal metadata

        Returns
        -------
        Position if opened successfully, else None
        """
        if side not in ("buy", "long", "sell", "short"):
            raise ValueError("Side must be buy, long, sell or short")
        if not all(math.isfinite(value) and value > 0 for value in (quantity, price)):
            raise ValueError("Quantity and price must be positive finite numbers")
        notional = quantity * price
        slippage = self.compute_slippage(notional, avg_volume_usd)
        fee = self.compute_fee(notional)

        # Apply slippage to fill price
        if side in ("buy", "long"):
            fill_price = price * (1 + slippage)  # buy at slightly higher
            position_side = "long"
        else:
            fill_price = price * (1 - slippage)  # sell at slightly lower
            position_side = "short"

        # Check if we have enough balance
        notional = quantity * fill_price
        fee = self.compute_fee(notional)
        cost = notional + fee
        if cost > state.balance:
            # Reduce quantity to fit
            if state.balance <= 0:
                return None
            quantity = state.balance / (fill_price * (1 + self.fee_rate))
            notional = quantity * fill_price
            fee = self.compute_fee(notional)
            cost = notional + fee

        # Create position object first
        position_id = f"paper_{uuid.uuid4().hex[:12]}"
        pos = Position(
            id=position_id,
            strategy=strategy,
            symbol=symbol,
            side=position_side,
            entry_price=fill_price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                **(metadata or {}),
                "signal_type": signal_type,
                "slippage": slippage,
                "fee": fee,
                "fill_price": fill_price,
                "market_price": price,
            },
        )
        if logs:
            pos.logs = logs

        # Only deduct from balance if we successfully created the position
        state.balance -= cost
        logger.simple("simulator", f"Balance updated: {state.balance + cost:.2f} -> {state.balance:.2f} (Trade {symbol})")
        
        state.add_position(pos)
        return pos

    def close_order(self, state: AppState, position_id: str,
                    current_price: float,
                    avg_volume_usd: float = 1_000_000,
                    exit_reason: str = "",
                    exit_reasoning: str = "") -> Optional[dict]:
        """
        Close an existing paper position.

        Applies slippage + fee to the close price.

        Returns trade record dict or None.
        """
        pos = next((p for p in state.positions if p.id == position_id), None)
        if pos is None:
            return None

        notional = pos.quantity * current_price
        slippage = self.compute_slippage(notional, avg_volume_usd)
        fee = self.compute_fee(notional)

        # Apply slippage to exit price
        if pos.side == "long":
            exit_price = current_price * (1 - slippage)  # sell at slightly lower
        else:
            exit_price = current_price * (1 + slippage)  # buy back at slightly higher
        fee = self.compute_fee(pos.quantity * exit_price)

        # Close the position in state (handles P&L, trade history)
        record = state.close_position(
            position_id=position_id,
            exit_price=exit_price,
            reason=exit_reason,
            fee=fee,
        )

        if record:
            if exit_reasoning:
                record["logs"] = [exit_reasoning]
            return record
        return None

    def get_account_summary(self, state: AppState) -> dict:
        """Return a summary of the paper trading account."""
        open_pnl = sum(p.pnl for p in state.positions)
        return {
            "balance": round(state.balance, 2),
            "equity": round(state.balance + sum(p.entry_price * p.quantity for p in state.positions) + open_pnl, 2),
            "open_pnl": round(open_pnl, 2),
            "total_pnl": round(state.total_pnl, 2),
            "daily_pnl": round(state.daily_pnl, 2),
            "open_positions": len(state.positions),
            "total_trades": len(state.trade_history),
            "mode": "paper",
        }


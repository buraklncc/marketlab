import os
import json
import uuid
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from marketlab.config import DATA_DIR, LOGS_DIR

class Position:
    def __init__(self, id: str, strategy: str, symbol: str, side: str, 
                 entry_price: float, quantity: float, 
                 stop_loss: Optional[float] = None, 
                 take_profit: Optional[float] = None,
                 entry_time: Optional[str] = None,
                 metadata: Optional[dict] = None):
        self.id = id or str(uuid.uuid4())
        self.strategy = strategy
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}
        self.current_price = entry_price
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.logs = []

    def update_pnl(self, current_price: float):
        self.current_price = current_price
        if self.side == "long":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
        cost = self.entry_price * self.quantity
        self.pnl_pct = (self.pnl / cost * 100) if cost > 0 else 0

    def to_dict(self):
        return {
            "id": self.id, "strategy": self.strategy, "symbol": self.symbol,
            "side": self.side, "entry_price": self.entry_price, "current_price": self.current_price,
            "quantity": self.quantity, "pnl": self.pnl, "pnl_pct": self.pnl_pct,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit,
            "entry_time": self.entry_time, "metadata": self.metadata, "logs": self.logs
        }

class AppState:
    def __init__(self, mode: str = "paper"):
        if mode != "paper":
            raise ValueError("MarketLab only supports paper state")
        self.mode = mode
        self.bot_running = False
        self.turbo_mode = False
        self.aggression = 0.5
        self.initial_balance = 10000.0
        self.balance = 10000.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.positions: List[Position] = []
        self.trade_history = []
        self.scanner_results = []
        self.regime = {}
        self.prices = {}
        self.start_time = None
        self.last_update = datetime.now(timezone.utc)
        self.daily_kill_switch = False
        self.cooldown_until = None
        self.consecutive_losses = 0
        self.last_trade_times = {}
        self.ws_clients = set()
        self._on_change_callbacks = []
        self._load_mode_state()

    def register_on_change(self, callback):
        if callback not in self._on_change_callbacks:
            self._on_change_callbacks.append(callback)

    def trigger_change(self):
        # Wrap the state in a standard message format for the WebSocket
        message = {"type": "state", "payload": self.to_dict()}
        for cb in self._on_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(message))
                else:
                    cb(message)
            except Exception:
                pass

    def reset_daily_pnl(self):
        self.daily_pnl = 0.0
        self.save_to_file()
        self.trigger_change()

    def set_turbo(self, enabled: bool):
        self.turbo_mode = enabled
        self.trigger_change()

    def _get_file_path(self):
        return DATA_DIR / f"state_{self.mode}.json"

    def _load_mode_state(self):
        path = self._get_file_path()
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self.balance = data.get("balance", self.initial_balance)
                    self.initial_balance = data.get("initial_balance", 10000.0)
                    self.total_pnl = data.get("total_pnl", 0.0)
                    self.daily_pnl = data.get("daily_pnl", 0.0)
                    self.trade_history = data.get("trade_history", [])
                    self.daily_kill_switch = data.get("daily_kill_switch", False)
                    self.cooldown_until = data.get("cooldown_until", None)
                    self.consecutive_losses = data.get("consecutive_losses", 0)
                    self.last_trade_times = data.get("last_trade_times", {})
                    if "positions" in data:
                        self.positions = []
                        for p in data.get("positions", []):
                            # Filter out fields that are NOT in the constructor
                            valid_keys = ("id", "strategy", "symbol", "side", "entry_price", "quantity", "stop_loss", "take_profit", "entry_time", "metadata")
                            filtered_p = {k: v for k, v in p.items() if k in valid_keys}
                            pos = Position(**filtered_p)
                            # Manually restore fields that are updated after creation
                            if "pnl" in p: pos.pnl = p["pnl"]
                            if "pnl_pct" in p: pos.pnl_pct = p["pnl_pct"]
                            if "current_price" in p: pos.current_price = p["current_price"]
                            if "logs" in p: pos.logs = p["logs"]
                            self.positions.append(pos)
            except Exception:
                pass

    def save_to_file(self):
        path = self._get_file_path()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "balance": self.balance, "initial_balance": self.initial_balance,
            "total_pnl": self.total_pnl, "daily_pnl": self.daily_pnl,
            "daily_kill_switch": self.daily_kill_switch,
            "cooldown_until": self.cooldown_until,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_times": self.last_trade_times,
            "trade_history": self.trade_history, "positions": [p.to_dict() for p in self.positions]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def update_price(self, symbol: str, price: float):
        self.prices[symbol] = price
        for pos in self.positions:
            if pos.symbol == symbol:
                pos.update_pnl(price)

        self.trigger_change()

    def add_position(self, pos: Position):
        self.positions.append(pos)
        self.save_to_file()
        from marketlab.logging_.logger import logger
        logger.simple("trade_open", f"Neural Entry: {pos.symbol} {pos.side.upper()} at ${pos.entry_price:.2f}")
        self.trigger_change()

    def clear_positions(self):
        self.positions = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_kill_switch = False
        self.cooldown_until = None
        self.consecutive_losses = 0
        self.last_trade_times = {}
        self.save_to_file()
        self.trigger_change()

    def close_position(self, position_id: str, exit_price: float, reason: str = "", fee: float = 0.0):
        pos = next((p for p in self.positions if p.id == position_id), None)
        if pos:
            self.positions = [p for p in self.positions if p.id != position_id]
            # Calculate PnL from position data
            if pos.side == "long":
                pnl = (exit_price - pos.entry_price) * pos.quantity - fee
                # Refund principal + PnL
                self.balance += (pos.quantity * exit_price) - fee
            else:
                pnl = (pos.entry_price - exit_price) * pos.quantity - fee
                # For shorts, we return the collateral plus/minus PnL
                # Initial deduction was (entry_price * quantity) + fee
                # Result is initial_cost + pnl
                self.balance += (pos.quantity * pos.entry_price) + pnl
            pnl -= pos.metadata.get("fee", 0.0)
            
            self.total_pnl += pnl
            self.daily_pnl += pnl
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            # Record last trade time for cooldown
            self.last_trade_times[pos.symbol] = datetime.now(timezone.utc).isoformat()
            record = {
                "id": pos.id, "symbol": pos.symbol, "side": pos.side,
                "entry": pos.entry_price, "exit": exit_price, "quantity": pos.quantity,
                "pnl": round(pnl, 4), "fee": round(fee, 4), "reason": reason,
                "time": datetime.now(timezone.utc).isoformat()
            }
            self.trade_history.append(record)
            self.save_to_file()
            from marketlab.logging_.logger import logger
            status = "PROFIT ✓" if pnl > 0 else "LOSS ✗"
            logger.simple("trade_close", f"Trade Closed: {pos.symbol} PnL: ${pnl:.2f} ({status})")
            self.trigger_change()
            return record
        return None

    def to_dict(self):
        return {
            "mode": self.mode, "bot_running": self.bot_running, "turbo_mode": self.turbo_mode,
            "aggression": self.aggression,
            "balance": self.balance, "initial_balance": self.initial_balance,
            "daily_pnl": self.daily_pnl, "total_pnl": self.total_pnl,
            "positions": [p.to_dict() for p in self.positions],
            "trade_history": self.trade_history[-20:],
            "scanner_results": self.scanner_results, "regime": self.regime,
            "prices": self.prices, "start_time": self.start_time
        }

_state_instance = None

def get_state(mode: str = None) -> AppState:
    global _state_instance
    if _state_instance is None:
        _state_instance = AppState(mode or "paper")
    elif mode and _state_instance.mode != mode:
        _state_instance.save_to_file()
        _state_instance = AppState(mode)
    return _state_instance


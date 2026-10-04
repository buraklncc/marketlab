"""
trade_guard.py — Risk rules enforcement.

Implements:
  - Stop loss checking
  - Daily loss kill switch (5% of capital)
  - Consecutive loss cooldown (3 losses → 2h pause)
  - Trade interval cooldown (15 min between trades on same symbol)
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from marketlab.bot_state import AppState


class TradeGuard:
    """
    Enforces risk management rules before trade execution.
    """

    def __init__(self,
                 stop_loss_pct: float = 0.02,
                 daily_loss_limit_pct: float = 0.05,
                 max_consecutive_losses: int = 3,
                 cooldown_hours: float = 2.0,
                 trade_interval_minutes: int = 15):
        self.stop_loss_pct = stop_loss_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_hours = cooldown_hours
        self.trade_interval_minutes = trade_interval_minutes

    def can_trade(self, state: AppState, symbol: str) -> tuple[bool, str]:
        """
        Check all risk rules. Returns (allowed, reason).
        """
        # Daily kill switch
        allowed, reason = self.check_daily_limit(state)
        if not allowed:
            return False, reason

        # Check if already in position for this symbol
        for pos in state.positions:
            if pos.symbol == symbol:
                return False, "already_in_position"

        # Cooldown from consecutive losses
        allowed, reason = self.check_cooldown(state)
        if not allowed:
            return False, reason

        # Consecutive losses
        allowed, reason = self.check_consecutive_losses(state)
        if not allowed:
            return False, reason

        # Trade interval
        allowed, reason = self.check_trade_interval(state, symbol)
        if not allowed:
            return False, reason

        return True, "ok"

    def check_daily_limit(self, state: AppState) -> tuple[bool, str]:
        """
        If daily_loss > daily_loss_limit_pct of capital → kill switch.
        """
        if state.daily_kill_switch:
            return False, "daily_kill_switch"

        if state.initial_balance <= 0:
            return True, "ok"

        daily_loss_pct = abs(min(0, state.daily_pnl)) / state.initial_balance

        if daily_loss_pct >= self.daily_loss_limit_pct:
            state.daily_kill_switch = True
            state.save_to_file()
            return False, "daily_kill_switch"

        return True, "ok"

    def check_cooldown(self, state: AppState) -> tuple[bool, str]:
        """Check if we're in a cooldown period from consecutive losses."""
        if state.cooldown_until:
            try:
                cooldown_end = datetime.fromisoformat(state.cooldown_until)
                if datetime.now(timezone.utc) < cooldown_end:
                    return False, "cooldown_active"
                else:
                    # Cooldown expired
                    state.cooldown_until = None
                    state.save_to_file()
            except (ValueError, TypeError):
                state.cooldown_until = None

        return True, "ok"

    def check_consecutive_losses(self, state: AppState) -> tuple[bool, str]:
        """
        If consecutive_losses >= max → pause for cooldown_hours.
        """
        if state.consecutive_losses >= self.max_consecutive_losses:
            # Set cooldown
            cooldown_end = datetime.now(timezone.utc) + timedelta(hours=self.cooldown_hours)
            state.cooldown_until = cooldown_end.isoformat()
            state.consecutive_losses = 0  # reset after applying cooldown
            state.save_to_file()
            return False, "cooldown_active"

        return True, "ok"

    def check_trade_interval(self, state: AppState, symbol: str) -> tuple[bool, str]:
        """
        Minimum trade_interval_minutes between trades on same symbol.
        """
        last_trade = state.last_trade_times.get(symbol)
        if last_trade:
            try:
                last_time = datetime.fromisoformat(last_trade)
                elapsed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
                if elapsed < self.trade_interval_minutes:
                    remaining = self.trade_interval_minutes - elapsed
                    return False, f"trade_cooldown_{int(remaining)}m"
            except (ValueError, TypeError):
                pass

        return True, "ok"

    def check_stop_loss(self, entry_price: float, current_price: float,
                        side: str) -> bool:
        """
        Check if stop loss has been hit.

        Returns True if stop loss triggered.
        """
        if side == "long":
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price

        return loss_pct >= self.stop_loss_pct

    def compute_stop_price(self, entry_price: float, side: str,
                           custom_pct: Optional[float] = None) -> float:
        """Compute the stop loss price."""
        pct = custom_pct or self.stop_loss_pct
        if side == "long":
            return entry_price * (1 - pct)
        else:
            return entry_price * (1 + pct)


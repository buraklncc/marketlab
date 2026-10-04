"""
translator.py — Maps technical signals to human-friendly messages.

Every signal type has 2-3 phrasings that rotate randomly.
No jargon — the user should understand every message without trading knowledge.
"""

import random
from typing import Optional


# Signal type → list of human-friendly message templates
# {coins}, {symbol}, {pnl}, {price} are available placeholders
_MESSAGES: dict[str, list[str]] = {
    # ── Cointegration (Strategy 1) ──────────────────────────────
    "z_score_long": [
        "BTC and ETH drifted apart. We're buying the gap.",
        "The price gap between BTC and ETH is unusually wide. Entering a trade to catch the bounce back.",
        "BTC looks cheap compared to ETH right now. Taking a position.",
    ],
    "z_score_short": [
        "BTC got too expensive vs ETH. Selling the difference.",
        "The gap between BTC and ETH flipped the other way. We're trading the correction.",
        "BTC is overpriced relative to ETH. Positioning for them to come back together.",
    ],
    "z_exit_profit": [
        "Prices came back together. We took our profit. ✓",
        "The gap closed as expected. Nice trade! ✓",
        "BTC and ETH realigned. Profit locked in. ✓",
    ],
    "z_exit_stop": [
        "Gap kept growing instead of closing. We exited safely.",
        "The trade moved against us too far. Closed to protect your money.",
        "Safety limit hit on this trade. Exited to minimize losses.",
    ],

    # ── VPIN (Strategy 2) ───────────────────────────────────────
    "vpin_long": [
        "Big players seem to be buying. We're joining them.",
        "Heavy buying activity detected. Riding the momentum upward.",
        "Smart money is flowing in. We're following the move.",
    ],
    "vpin_short": [
        "Big players seem to be selling. We're going with it.",
        "Heavy selling pressure detected. Positioning for a drop.",
        "Institutional selling spotted. Following the trend down.",
    ],
    "vpin_exit_profit": [
        "Move played out as expected. Profit taken. ✓",
        "The momentum trade hit its target. Cashed out. ✓",
        "Great timing on that one. Profit secured. ✓",
    ],
    "vpin_exit_stop": [
        "Momentum reversed. Stopped out to protect capital.",
        "The move didn't go our way. Cut the loss early.",
        "Safety exit triggered on momentum trade.",
    ],
    "vpin_exit_time": [
        "Trade took too long. We closed it to stay safe.",
        "Time limit reached on this trade. Closing to free up capital.",
        "This trade ran out of time. Moving on to new opportunities.",
    ],

    # ── Gap (Strategy 3) ────────────────────────────────────────
    "gap_entry": [
        "Price dropped sharply overnight. History says it bounces.",
        "Big overnight dip detected. Entering for the expected recovery.",
        "Gap-down spotted. Statistically, these tend to recover.",
    ],
    "gap_exit_time": [
        "24 hours passed. We closed the trade.",
        "Time's up on this gap trade. Closing the position.",
        "The daily window ended. Trade closed as planned.",
    ],
    "gap_exit_model": [
        "Our system sensed the bounce may not come. Exited early.",
        "The analysis suggests this gap won't close. Playing it safe.",
        "Conditions changed. Closing early to avoid a loss.",
    ],

    # ── Regime ──────────────────────────────────────────────────
    "regime_trending": [
        "Market has clear direction. Momentum strategy active.",
        "Strong trend detected. Switching to momentum-based trading.",
        "The market is moving with purpose. Following the trend.",
    ],
    "regime_ranging": [
        "Market is moving sideways. Pair strategy active.",
        "No clear trend right now. Using our mean-reversion approach.",
        "Sideways market detected. Looking for pairs that drift apart.",
    ],
    "regime_extreme_vol": [
        "Market is too chaotic right now. Bot is waiting. ⚠️",
        "Extreme volatility detected. Staying out until it calms down. ⚠️",
        "Wild price swings — too risky to trade. Pausing all strategies. ⚠️",
    ],

    # ── Risk events ─────────────────────────────────────────────
    "daily_kill_switch": [
        "We hit today's safety limit. Bot paused until tomorrow. 🛑",
        "Daily loss limit reached. All trading stopped for today. 🛑",
        "Safety first — today's loss cap was hit. Resuming tomorrow. 🛑",
    ],
    "cooldown_active": [
        "Taking a short break before the next trade. ⏸️",
        "A few trades didn't work out. Cooling down before trying again. ⏸️",
        "Brief pause to reset. We'll be back trading shortly. ⏸️",
    ],

    # ── Scanner ─────────────────────────────────────────────────
    "scanner_update": [
        "Best coins right now: {coins}",
        "Top opportunities spotted: {coins}",
        "Market scan complete. Interesting coins: {coins}",
    ],
    "scanner_trend": [
        "Scanner found a strong trend. Taking a position. 🎯",
        "High momentum detected by scanner. Entering trade. 🎯",
        "A top opportunity just met our criteria. We're in. 🎯",
    ],

    # ── Turbo ───────────────────────────────────────────────────
    "turbo_toggle": [
        "Turbo mode toggled.",
    ],

    # ── Idle ────────────────────────────────────────────────────
    "no_signal": [
        "Watching the market. Nothing interesting yet.",
        "All quiet. Monitoring for opportunities.",
        "Scanning... no good setups right now.",
    ],

    # ── Connection ──────────────────────────────────────────────
    "connected": [
        "Connected to the market. Ready to trade. ✅",
        "Market connection established. All systems go. ✅",
        "We're live and watching the markets. ✅",
    ],
    "disconnected": [
        "Lost connection to the market. Trying to reconnect... 🔄",
        "Connection dropped. Reconnecting... 🔄",
        "Market feed interrupted. Working on it... 🔄",
    ],
    "reconnected": [
        "Back online! Resuming market watch. ✅",
        "Reconnected successfully. Picking up where we left off. ✅",
    ],
    "error_api_key": [
        "API key problem. Check your keys in Settings. ❌",
        "Can't connect — your API key seems wrong. Please check Settings. ❌",
    ],

    # ── Bot lifecycle ───────────────────────────────────────────
    "bot_started": [
        "Bot is running! Watching the markets for you. 🚀",
        "Started up and ready to go. Let's find some trades! 🚀",
    ],
    "bot_stopped": [
        "Bot stopped. Your positions are safe.",
        "Trading paused. Everything is saved.",
    ],
}


class Translator:
    """Translates technical signal types to human-readable messages."""

    def __init__(self):
        self._last_index: dict[str, int] = {}  # track rotation per type

    def translate(self, signal_type: str, **context) -> str:
        """
        Get a human-friendly message for a signal type.

        Parameters
        ----------
        signal_type : str   e.g. "z_score_long", "daily_kill_switch"
        **context : dict    template variables (coins, symbol, pnl, price)

        Returns
        -------
        str : human-readable message
        """
        templates = _MESSAGES.get(signal_type)
        if not templates:
            return f"Something happened: {signal_type}"

        # Rotate through templates (not purely random — avoids repeats)
        last = self._last_index.get(signal_type, -1)
        available = list(range(len(templates)))
        if len(available) > 1 and last in available:
            available.remove(last)
        idx = random.choice(available)
        self._last_index[signal_type] = idx

        message = templates[idx]

        # Apply context variables
        try:
            message = message.format(**context)
        except (KeyError, IndexError):
            pass

        return message

    def get_color(self, signal_type: str) -> str:
        """Return a color category for the signal type."""
        if any(x in signal_type for x in ("profit", "started", "connected", "reconnected", "entry", "long", "short")):
            return "green"
        if any(x in signal_type for x in ("stop", "kill", "error", "extreme")):
            return "red"
        if any(x in signal_type for x in ("cooldown", "time", "ranging", "no_signal", "scanner", "model")):
            return "yellow"
        return "blue"

    def get_icon(self, signal_type: str) -> str:
        """Return an emoji icon for the signal type."""
        icons = {
            "green": "📈",
            "red": "🔴",
            "yellow": "⏳",
            "blue": "ℹ️",
        }
        return icons.get(self.get_color(signal_type), "ℹ️")


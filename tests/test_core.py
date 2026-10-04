import asyncio
import importlib
import math
import numpy as np
import pandas as pd
import pytest
from marketlab.risk.position_sizer import PositionSizer
from marketlab.backtesting.backtest_engine import BacktestEngine
from marketlab.data_feed.volume_bars import VolumeBarBuilder

@pytest.fixture
def paper_state(tmp_path, monkeypatch):
    from marketlab import config
    from marketlab import bot_state
    monkeypatch.setattr(bot_state, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    return bot_state.AppState()

@pytest.mark.parametrize("side", ["long", "short"])
def test_flat_roundtrip_accounts_for_both_fees(paper_state, side):
    from marketlab.paper_trading.simulator import PaperSimulator
    sim = PaperSimulator(fee_rate=.001, base_slippage=0, impact_factor=0)
    pos = sim.execute_order(paper_state, "DEMO", side, 10, 100)
    assert pos is not None
    assert sim.get_account_summary(paper_state)["equity"] == pytest.approx(9999)
    record = sim.close_order(paper_state, pos.id, 100)
    assert record["pnl"] == pytest.approx(-2)
    assert paper_state.balance == pytest.approx(9998)
    assert paper_state.total_pnl == pytest.approx(-2)
    assert paper_state.positions == []

def test_oversized_order_cannot_spend_more_than_balance(paper_state):
    from marketlab.paper_trading.simulator import PaperSimulator
    sim = PaperSimulator()
    pos = sim.execute_order(paper_state, "DEMO", "long", 10000, 100)
    assert pos is not None
    assert paper_state.balance >= -1e-8
    assert pos.entry_price * pos.quantity * (1 + sim.fee_rate) == pytest.approx(10000)

@pytest.mark.parametrize("quantity,price", [(-1, 100), (1, 0), (math.nan, 100)])
def test_invalid_orders_do_not_mutate_balance(paper_state, quantity, price):
    from marketlab.paper_trading.simulator import PaperSimulator
    with pytest.raises(ValueError):
        PaperSimulator().execute_order(paper_state, "DEMO", "long", quantity, price)
    assert paper_state.balance == 10000 and not paper_state.positions

def test_live_state_cannot_be_created():
    from marketlab.bot_state import AppState
    with pytest.raises(ValueError, match="paper"):
        AppState("live")

def test_daily_loss_gate_blocks_new_positions(paper_state):
    from marketlab.risk.trade_guard import TradeGuard
    paper_state.daily_pnl = -500
    allowed, reason = TradeGuard().can_trade(paper_state, "DEMO")
    assert not allowed and reason == "daily_kill_switch"

def test_position_sizing_caps_allocated_capital():
    size = PositionSizer(max_position_pct=.1).compute_size_cointegration(mu=.05, sigma=.1, total_capital=10000, price=100)
    assert 0 <= size * 100 <= 1000

def test_volume_bar_conserves_volume():
    builder = VolumeBarBuilder(5)
    assert builder.on_tick(100, 2, False, 1000) is None
    bar = builder.on_tick(101, 3, False, 2000)
    assert bar.volume == 5
    assert bar.buy_volume + bar.sell_volume == bar.volume
    assert bar.vwap == pytest.approx(100.6)

def test_replay_is_reproducible_without_network():
    rng = np.random.default_rng(42)
    x = 100 + np.cumsum(rng.normal(0, .25, 150))
    y = 2 * x + rng.normal(0, .4, 150)
    def frame(a):
        return pd.DataFrame({"open": a, "high": a + .1, "low": a - .1, "close": a, "volume": 1000.0})
    engine = BacktestEngine()
    first = asyncio.run(engine.run_cointegration(frame(y), frame(x)))
    second = asyncio.run(engine.run_cointegration(frame(y), frame(x)))
    assert first.to_dict() == second.to_dict()
    assert math.isfinite(first.final_balance)


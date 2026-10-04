"""Deterministic synthetic-data example for the replay engine."""
import argparse
import asyncio
import json
import tempfile
import os

def main():
    parser = argparse.ArgumentParser(description="Offline market replay using synthetic prices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=300)
    args = parser.parse_args()
    if args.bars < 120:
        parser.error("--bars must be at least 120")
    with tempfile.TemporaryDirectory(prefix="marketlab-") as scratch:
        os.environ.setdefault("MARKETLAB_HOME", scratch)
        import numpy as np
        import pandas as pd
        from marketlab.backtesting.backtest_engine import BacktestEngine
        rng = np.random.default_rng(args.seed)
        x = 100 + np.cumsum(rng.normal(0, .25, args.bars))
        y = 2 * x + rng.normal(0, .4, args.bars)
        make = lambda a: pd.DataFrame({"open": a, "high": a + .1, "low": a - .1, "close": a, "volume": 1000.0})
        result = asyncio.run(BacktestEngine().run_cointegration(make(y), make(x), "SYNTH-Y", "SYNTH-X"))
        print(json.dumps({"input": "synthetic", "seed": args.seed, "bars": args.bars, "result": result.to_dict()}, indent=2))

if __name__ == "__main__":
    main()


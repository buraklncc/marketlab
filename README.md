# MarketLab

Offline market-research primitives: historical strategy replay, position sizing, risk gates, volume bars and paper execution.

Development started: **27 April 2026**. This date is based on the earliest recorded project commit. This repository presents the current backend snapshot.

## Quick start

Python 3.10+.

```sh
python -m pip install -e ".[dev]"
marketlab --seed 42 --bars 300
python -m pytest
```

The CLI uses deterministic synthetic prices and emits a JSON replay result. It performs no exchange requests. Experimental state, logs and model paths are configured through `MARKETLAB_HOME`.

## Components

- Cointegration, VPIN and gap strategy experiments.
- Historical replay with fee and slippage parameters.
- Half-Kelly/CPPI sizing primitives and loss/cooldown gates.
- Volume-bar aggregation and a paper ledger.

## Boundaries

These are research prototypes. Replay statistics are not verified trading performance. The replay engine simplifies fills and equity accounting; the strategies require independent validation. No trained model is bundled. The gap strategy's optional model must be supplied or trained by the caller.

MIT license.


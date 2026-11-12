import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import black_scholes as bs  # noqa: E402
import hedging  # noqa: E402

ARGS = dict(S=100.0, K=100.0, T=0.25, r=0.03)


def test_simulated_paths_have_the_requested_volatility():
    paths = hedging.simulate_paths(100.0, 1.0, 0.0, 0.3, steps=252, paths=4_000, seed=0)
    log_returns = np.diff(np.log(paths), axis=1)
    assert log_returns.std() * np.sqrt(252) == pytest.approx(0.3, rel=0.02)


def test_fair_hedge_has_near_zero_mean_pnl():
    pnl = hedging.delta_hedge_pnl(**ARGS, implied_vol=0.2, realised_vol=0.2, steps=100, paths=20_000, seed=1)
    premium = float(bs.price(100.0, 100.0, 0.25, 0.03, 0.2))
    assert abs(pnl.mean()) < 0.03 * premium


def test_hedging_error_shrinks_with_square_root_of_rebalances():
    coarse = hedging.delta_hedge_pnl(**ARGS, implied_vol=0.2, realised_vol=0.2, steps=25, paths=20_000, seed=2)
    fine = hedging.delta_hedge_pnl(**ARGS, implied_vol=0.2, realised_vol=0.2, steps=100, paths=20_000, seed=2)
    assert fine.std() / coarse.std() == pytest.approx(0.5, rel=0.15)


def test_selling_volatility_above_realised_makes_money_on_average():
    rich = hedging.delta_hedge_pnl(**ARGS, implied_vol=0.30, realised_vol=0.20, steps=100, paths=20_000, seed=3)
    cheap = hedging.delta_hedge_pnl(**ARGS, implied_vol=0.15, realised_vol=0.20, steps=100, paths=20_000, seed=3)
    assert rich.mean() > 0
    assert cheap.mean() < 0


def test_unhedged_short_option_is_far_riskier():
    hedged = hedging.delta_hedge_pnl(**ARGS, implied_vol=0.2, realised_vol=0.2, steps=50, paths=10_000, seed=4)
    paths = hedging.simulate_paths(100.0, 0.25, 0.03, 0.2, steps=50, paths=10_000, seed=4)
    premium = float(bs.price(100.0, 100.0, 0.25, 0.03, 0.2))
    unhedged = premium * np.exp(0.03 * 0.25) - np.maximum(paths[:, -1] - 100.0, 0.0)
    assert hedged.std() < 0.3 * unhedged.std()

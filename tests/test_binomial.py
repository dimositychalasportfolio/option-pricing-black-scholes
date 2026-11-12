import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import binomial  # noqa: E402
import black_scholes as bs  # noqa: E402

# Standard American put tree from Hull, "Options, Futures, and Other Derivatives"
# (binomial trees chapter): S=50, K=50, r=10%, sigma=40%, T=5 months, 5 steps -> 4.49.
HULL_TREE = dict(S=50.0, K=50.0, T=5 / 12, r=0.10, sigma=0.40)


def test_matches_hull_american_put_tree():
    assert binomial.crr_price(**HULL_TREE, kind="put", steps=5, american=True) == pytest.approx(4.49, abs=0.005)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_converges_to_black_scholes(kind):
    args = dict(S=100.0, K=95.0, T=0.75, r=0.04, sigma=0.3, kind=kind, q=0.01)
    exact = float(bs.price(**args))
    coarse = abs(binomial.crr_price(**args, steps=50) - exact)
    fine = abs(binomial.crr_price(**args, steps=2000) - exact)
    assert fine < 0.005
    assert fine < coarse


def test_american_call_without_dividends_equals_european():
    """Never optimal to exercise a call early on a non-dividend stock."""
    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, kind="call", q=0.0, steps=400)
    assert binomial.crr_price(**args, american=True) == pytest.approx(binomial.crr_price(**args), abs=1e-10)


def test_american_put_is_worth_at_least_european_and_intrinsic():
    args = dict(S=80.0, K=100.0, T=1.0, r=0.06, sigma=0.25, kind="put", steps=400)
    american = binomial.crr_price(**args, american=True)
    assert american > binomial.crr_price(**args)
    assert american >= 100.0 - 80.0


def test_early_exercise_premium_is_positive_for_deep_puts_and_zero_for_calls():
    assert binomial.early_exercise_premium(70.0, 100.0, 1.0, 0.08, 0.2, "put") > 0.5
    assert binomial.early_exercise_premium(100.0, 100.0, 1.0, 0.05, 0.2, "call") == pytest.approx(0.0, abs=1e-10)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        binomial.crr_price(**HULL_TREE, steps=0)
    with pytest.raises(ValueError):
        binomial.crr_price(S=100, K=100, T=10, r=0.9, sigma=0.01, steps=1)  # p outside (0, 1)

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import black_scholes as bs  # noqa: E402
import monte_carlo as mc  # noqa: E402

ARGS = dict(S=100.0, K=105.0, T=0.5, r=0.03, sigma=0.25, q=0.01)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("method", [{}, {"antithetic": True}, {"control_variate": True},
                                    {"antithetic": True, "control_variate": True}])
def test_estimate_is_within_four_standard_errors_of_black_scholes(kind, method):
    exact = float(bs.price(**ARGS, kind=kind))
    result = mc.mc_price(**ARGS, kind=kind, paths=200_000, seed=1, **method)
    assert abs(result.price - exact) < 4 * result.std_error


def test_variance_reduction_shrinks_the_standard_error():
    plain = mc.mc_price(**ARGS, paths=100_000, seed=2)
    antithetic = mc.mc_price(**ARGS, paths=100_000, seed=2, antithetic=True)
    control = mc.mc_price(**ARGS, paths=100_000, seed=2, control_variate=True)
    assert antithetic.std_error < plain.std_error
    assert control.std_error < 0.6 * plain.std_error
    both = mc.mc_price(**ARGS, paths=100_000, seed=2, antithetic=True, control_variate=True)
    assert both.std_error <= control.std_error * 1.05  # combining must not undo the control variate


def test_standard_error_halves_when_paths_quadruple():
    small = mc.mc_price(**ARGS, paths=50_000, seed=3)
    large = mc.mc_price(**ARGS, paths=200_000, seed=3)
    assert large.std_error / small.std_error == pytest.approx(0.5, rel=0.05)


def test_same_seed_gives_same_result():
    assert mc.mc_price(**ARGS, seed=7).price == mc.mc_price(**ARGS, seed=7).price


def test_confidence_interval_is_centred_on_the_estimate():
    result = mc.mc_price(**ARGS, paths=10_000, seed=4)
    low, high = result.ci95
    assert low < result.price < high
    assert high - low == pytest.approx(2 * 1.96 * result.std_error)

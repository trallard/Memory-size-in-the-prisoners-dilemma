import numpy as np

import opt_mo


def test_simulate_match_with_defector():
    assert np.isclose(
        opt_mo.simulate_match((0, 0, 0, 0), (0, 0, 0, 0)), 1, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((0, 0, 0, 0), (1, 1, 1, 1)), 5, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((0, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)),
        3,
        atol=10 ** -2,
    )


def test_simulate_match_with_cooperator():
    assert np.isclose(
        opt_mo.simulate_match((1, 1, 1, 1), (0, 0, 0, 0)), 0, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((1, 1, 1, 1), (1, 1, 1, 1)), 3, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5)),
        1.5,
        atol=10 ** -2,
    )


def test_simulate_spatial_tournament():
    opponents = [(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)]
    player = (1, 0, 1, 0)

    score = opt_mo.simulate_spatial_tournament(
        player, opponents, turns=200, repetitions=5
    )

    assert np.isclose(score, 2.5, atol=10 ** -2)

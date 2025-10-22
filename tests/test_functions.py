# Test Functions and Endpoints here

import numpy as np

from functions.odds import calc_prob_matrix


def test_calc_prob_matrix():
    hg_exp, ag_exp, max_goals, draw_lambda, f_half_perc = 1.5, 1.3, 9, 0.05, 0.445
    ft_matrix, _, _ = calc_prob_matrix(
        hg_exp, ag_exp, max_goals, draw_lambda, f_half_perc
    )
    # Assert it's a NumPy array
    assert isinstance(ft_matrix, np.ndarray), "ft_matrix should be a numpy array"
    assert ft_matrix.size > 0, "Matrix is empty"
    assert ft_matrix.ndim == 2, "Matrix should be 2D"

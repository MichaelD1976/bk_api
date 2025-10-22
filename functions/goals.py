import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


########################  FUNCTION: CALC_EXP_TEAM_GLS_FROM_1X2_OU()  ############################
# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import poisson

"""
Returns home goals exp and away goals exp from marginated 1x2 and ou2.5
Args: home odds (eg 1.5); draw odds (eg 3.5); away odds (eg. 3.5); over odds (eg. 1.7); under odds (eg. 2.1)
Returns: 2 outputs. HG exp and AG exp
Output as numpy array. Extract as: hg, ag = calc_exp_team_gls_from_1x2_ou(1.5, 3.5, 4.2, 1.5, 2.5)
"""


def calc_exp_team_gls_from_1x2_ou(
    home_odds, draw_odds, away_odds, over_2_5_odds, under_2_5_odds
):
    def odds_to_probs(odds):
        raw_probs = np.array([1 / o for o in odds])
        return raw_probs / raw_probs.sum()

    def poisson_matrix(lam_home, lam_away, max_goals=7):
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                matrix[i, j] = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
        return matrix

    def outcome_probs(matrix):
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))

        total_goals = np.add.outer(
            np.arange(matrix.shape[0]), np.arange(matrix.shape[1])
        )
        over_2_5 = matrix[total_goals > 2.5].sum()
        under_2_5 = matrix[total_goals <= 2.5].sum()

        return [home_win, draw, away_win, over_2_5, under_2_5]

    def loss(params, target_probs):
        lam_home, lam_away = params
        matrix = poisson_matrix(lam_home, lam_away)
        model_probs = outcome_probs(matrix)
        return sum((m - t) ** 2 for m, t in zip(model_probs, target_probs))

    # Step 1: Convert odds to target probabilities
    prob_1x2 = odds_to_probs([home_odds, draw_odds, away_odds])
    prob_ou = odds_to_probs([over_2_5_odds, under_2_5_odds])
    target_probs = list(prob_1x2) + list(prob_ou)

    # Step 2: Run optimization
    initial_guess = [1.2, 1.2]
    bounds = [(0.1, 5), (0.1, 5)]
    res = minimize(loss, initial_guess, args=(target_probs,), bounds=bounds)
    lam_home, lam_away = res.x

    hg_exp, ag_exp = round(lam_home, 2), round(lam_away, 2)

    return hg_exp, ag_exp

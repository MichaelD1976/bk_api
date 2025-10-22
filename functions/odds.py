import numpy as np
import math

# from scipy.stats import poisson


from functions.goals import calc_exp_team_gls_from_1x2_ou


########################  FUNCTION TO RETURN CORRECT SCORE PROBABILITY MATRICES (FT,1H & 2H)  ###########################
# import math
# import numpy

"""
Returns probability matrices for FT, 1H and 2H
Args: hg_exp (eg 1.6); ag_exp (eg 1.2); max_goals (eg 9); draw_lambda (0-0.3); f_half_perc (eg 0.44)
Output as numpy array
"""


def calc_prob_matrix(
    hg_exp, ag_exp, max_goals, draw_lambda, f_half_perc
):  # lam0 adjusts the draw likelihood for higher draw leagues

    # Calculate Home and Away Goals Expected 1H
    hg1h = round((hg_exp / 100) * f_half_perc, 2)
    ag1h = round((ag_exp / 100) * f_half_perc, 2)

    s_half_perc = 100 - f_half_perc

    # Calculate Home and Away Goals Expected 2H
    hg2h = round((hg_exp / 100) * s_half_perc, 2)
    ag2h = round((ag_exp / 100) * s_half_perc, 2)

    # Function to calculate Bivariate Poisson probability matrix
    def bivariate_poisson(lam0, lam1, lam2, max_goals):
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                s = 0
                for m in range(0, min(i, j) + 1):
                    term = (
                        (lam0**m)
                        * (lam1 ** (i - m))
                        * (lam2 ** (j - m))
                        / (
                            math.factorial(m)
                            * math.factorial(i - m)
                            * math.factorial(j - m)
                        )
                    )
                    s += term
                matrix[i, j] = np.exp(-(lam0 + lam1 + lam2)) * s
        matrix /= matrix.sum()  # Normalize
        return matrix

    # Set shared lambda (lam0) to induce correlation
    # You can adjust lam0 depending on how correlated you want the teams to be
    # so to increase draw scoreline percentages (make more likely) - increase lambda values and vice versa
    lam0_ft = draw_lambda
    lam0_1h = lam0_ft / 2
    lam0_2h = lam0_ft / 2

    # Calculate lambda1 and lambda2 for each period
    lam1_ft = hg_exp - lam0_ft
    lam2_ft = ag_exp - lam0_ft

    lam1_1h = hg1h - lam0_1h
    lam2_1h = ag1h - lam0_1h

    lam1_2h = hg2h - lam0_2h
    lam2_2h = ag2h - lam0_2h

    # Calculate probability matrices
    prob_matrix_ft = bivariate_poisson(lam0_ft, lam1_ft, lam2_ft, max_goals)
    prob_matrix_1h = bivariate_poisson(lam0_1h, lam1_1h, lam2_1h, max_goals)
    prob_matrix_2h = bivariate_poisson(lam0_2h, lam1_2h, lam2_2h, max_goals)

    # Tiny manual adjustment just on the FT matrix
    prob_matrix_ft[1, 1] *= 1.09  # Boost 1-1 % (decrease odds)
    prob_matrix_ft[0, 0] *= 0.98  # Lower 0-0
    prob_matrix_ft[2, 2] *= 1.01  # Lower 2-2

    # --- Floor and cap probabilities ---
    min_prob = 1e-4  # floor at 0.0001
    max_prob = 0.9999  # cap at 0.9999

    for matrix in [prob_matrix_ft, prob_matrix_1h, prob_matrix_2h]:
        np.clip(matrix, min_prob, max_prob, out=matrix)
        matrix /= matrix.sum()  # re-normalize after clipping

    return prob_matrix_ft, prob_matrix_1h, prob_matrix_2h


#########################################################################################

# FUNCTION GET DERIVATIVE MARKETS & SELECTIONS FROM WDW & OU


def get_markets_and_true_probabilities(
    home_odds, draw_odds, away_odds, over_2_5_odds, under_2_5_odds
):  # parse bk 1x2/ou odds to generate market prices
    """
    Parse marginated wdw and ou odds to generate chosen markets with true probabilities - uncomment selected market calculation and return selection
    """

    hg_exp, ag_exp = calc_exp_team_gls_from_1x2_ou(
        home_odds, draw_odds, away_odds, over_2_5_odds, under_2_5_odds
    )
    prob_matrix_ft, prob_matrix_1h, prob_matrix_2h = calc_prob_matrix(
        hg_exp, ag_exp, max_goals=9, draw_lambda=0.05, f_half_perc=0.445
    )

    # Calculate Win-Draw-Win market
    home_win_prob = np.sum(
        np.tril(prob_matrix_ft, -1)
    )  # Home win (lower triangle of the matrix excluding diagonal)
    draw_prob = np.sum(np.diag(prob_matrix_ft))  # Draw (diagonal of the matrix)
    away_win_prob = np.sum(
        np.triu(prob_matrix_ft, 1)
    )  # Away win (upper triangle of the matrix excluding diagonal)

    # # Calculate Over/Under
    # max_goals = 9
    # # Calculate Over/Under 0.5 goals
    # over_0_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 0])
    # under_0_5 = 1 - over_0_5

    # # Calculate Over/Under 1.5 goals
    # over_1_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 1])
    # under_1_5 = 1 - over_1_5

    # # Calculate Over/Under 2.5 goals
    # over_2_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 2])
    # under_2_5 = 1 - over_2_5

    # # Calculate Over/Under 3.5 goals
    # over_3_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 3])
    # under_3_5 = 1 - over_3_5

    # # Calculate Over/Under 4.5 goals
    # over_4_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 4])
    # under_4_5 = 1 - over_4_5

    # # dictionary to reference values in selection and return probabilities
    # lines_to_select_ov_un = {"2.5": [over_2_5, under_2_5],
    #                          "1.5": [over_1_5, under_1_5],
    #                          "3.5": [over_3_5, under_3_5],
    #                          "4.5": [over_4_5,under_4_5],
    #                          "0.5": [over_0_5,under_0_5],
    # }

    # # Both Teams to Score (Yes/No)
    # btts_yes = np.sum(prob_matrix_ft[1:, 1:])
    # btts_no = 1 - btts_yes

    # # Double Chance market
    # home_or_draw = home_win_prob + draw_prob
    # away_or_draw = away_win_prob + draw_prob
    # home_or_away = home_win_prob + away_win_prob

    # # DNB
    # dnb_home = home_win_prob / (home_win_prob + away_win_prob)
    # dnb_away = 1 - dnb_home

    # # Calculate Home Win to Nil and Away Win to Nil
    # home_win_to_nil = sum(prob_matrix_ft[i, 0] for i in range(1, max_goals+1))
    # away_win_to_nil = sum(prob_matrix_ft[0, j] for j in range(1, max_goals+1))

    # Calculate Win-Draw-Win market - 1H
    home_win_prob_fh = np.sum(
        np.tril(prob_matrix_1h, -1)
    )  # Home win (lower triangle of the matrix excluding diagonal)
    draw_prob_fh = np.sum(np.diag(prob_matrix_1h))  # Draw (diagonal of the matrix)
    away_win_prob_fh = np.sum(
        np.triu(prob_matrix_1h, 1)
    )  # Away win (upper triangle of the matrix excluding diagonal)

    # # Handicap -1
    # # Home Win by 2 or More Goals
    # home_win_by_2_or_more = np.sum(prob_matrix_ft[i, j] for i in range(2, max_goals+1) for j in range(i-1))
    # # Tie (Home Wins by Exactly 1 Goal)
    # tie_home_win_by_1 = np.sum(prob_matrix_ft[i, j] for i in range(1, max_goals+1) for j in range(i) if i-j == 1)
    # # Away Win or Draw
    # away_win_or_draw = draw_prob + away_win_prob

    # # Handicap +1
    # # Home Win or Draw (if home wins or if match is a draw, considering the handicap)
    # home_win_or_draw = home_win_prob + draw_prob  # Sum of match draw + home win percentage
    # # Tie (Away Wins by Exactly 1 Goal, which nullifies the -1 handicap)
    # tie_away_win_by_1 = np.sum(prob_matrix_ft[i, j] for j in range(1, max_goals+1) for i in range(j) if j-i == 1)
    # # Away Win by 2 or More Goals (considering the handicap of -1)
    # away_win_by_2_or_more = np.sum(prob_matrix_ft[i, j] for j in range(2, max_goals+1) for i in range(j-1))

    # Away wins 2nd half probability
    away_win_2h = (
        prob_matrix_2h[0, 1]
        + prob_matrix_2h[0, 2]
        + prob_matrix_2h[0, 3]
        + prob_matrix_2h[0, 4]
        + prob_matrix_2h[0, 5]
        + prob_matrix_2h[0, 6]
        + prob_matrix_2h[0, 7]
        + prob_matrix_2h[0, 8]
        + prob_matrix_2h[0, 9]
#        + prob_matrix_2h[0, 10]
        + prob_matrix_2h[1, 2]
        + prob_matrix_2h[1, 3]
        + prob_matrix_2h[1, 4]
        + prob_matrix_2h[1, 5]
        + prob_matrix_2h[1, 6]
        + prob_matrix_2h[1, 7]
        + prob_matrix_2h[1, 8]
        + prob_matrix_2h[2, 3]
        + prob_matrix_2h[2, 4]
        + prob_matrix_2h[2, 5]
        + prob_matrix_2h[2, 6]
        + prob_matrix_2h[2, 7]
        + prob_matrix_2h[3, 4]
        + prob_matrix_2h[3, 5]
        + prob_matrix_2h[3, 6]
        + prob_matrix_2h[3, 7]
        + prob_matrix_2h[4, 5]
        + prob_matrix_2h[4, 6]
        + prob_matrix_2h[4, 7]
        + prob_matrix_2h[5, 6]
        + prob_matrix_2h[5, 7]
    )

    # Draw 1h/2h probability
    # draw_1h = prob_matrix_1h[0,0] + prob_matrix_1h[1,1] + prob_matrix_1h[2,2] + prob_matrix_1h[3,3] + prob_matrix_1h[4,4] + prob_matrix_1h[5,5]
    draw_2h = (
        prob_matrix_2h[0, 0]
        + prob_matrix_2h[1, 1]
        + prob_matrix_2h[2, 2]
        + prob_matrix_2h[3, 3]
        + prob_matrix_2h[4, 4]
        + prob_matrix_2h[5, 5]
    )

    # Home 2h probability
    home_win_2h = (
        prob_matrix_2h[1, 0]
        + prob_matrix_2h[2, 0]
        + prob_matrix_2h[3, 0]
        + prob_matrix_2h[4, 0]
        + prob_matrix_2h[5, 0]
        + prob_matrix_2h[6, 0]
        + prob_matrix_2h[7, 0]
        + prob_matrix_2h[8, 0]
        + prob_matrix_2h[9, 0]
#        + prob_matrix_2h[10, 0]
        + prob_matrix_2h[2, 1]
        + prob_matrix_2h[3, 1]
        + prob_matrix_2h[4, 1]
        + prob_matrix_2h[5, 1]
        + prob_matrix_2h[6, 1]
        + prob_matrix_2h[7, 1]
        + prob_matrix_2h[8, 1]
        + prob_matrix_2h[9, 1]
        + prob_matrix_2h[3, 2]
        + prob_matrix_2h[4, 2]
        + prob_matrix_2h[5, 2]
        + prob_matrix_2h[6, 2]
        + prob_matrix_2h[7, 2]
        + prob_matrix_2h[8, 2]
        + prob_matrix_2h[4, 3]
        + prob_matrix_2h[5, 3]
        + prob_matrix_2h[6, 3]
        + prob_matrix_2h[7, 3]
        + prob_matrix_2h[5, 4]
        + prob_matrix_2h[6, 4]
        + prob_matrix_2h[7, 4]
        + prob_matrix_2h[6, 5]
        + prob_matrix_2h[7, 5]
    )

    # Create grouped 'win by' probabilities to aid HT-FT calculations

    home_win_1h_by_1 = (
        prob_matrix_1h[1, 0]
        + prob_matrix_1h[2, 1]
        + prob_matrix_1h[3, 2]
        + prob_matrix_1h[4, 3]
        + prob_matrix_1h[5, 4]
        + prob_matrix_1h[6, 5]
    )
    home_win_1h_by_2 = (
        prob_matrix_1h[2, 0]
        + prob_matrix_1h[3, 1]
        + prob_matrix_1h[4, 2]
        + prob_matrix_1h[5, 3]
        + prob_matrix_1h[6, 4]
        + prob_matrix_1h[7, 5]
    )
    home_win_1h_by_3 = (
        prob_matrix_1h[3, 0]
        + prob_matrix_1h[4, 1]
        + prob_matrix_1h[5, 2]
        + prob_matrix_1h[6, 3]
        + prob_matrix_1h[7, 4]
        + prob_matrix_1h[8, 5]
    )
    home_win_1h_by_4 = (
        prob_matrix_1h[4, 0]
        + prob_matrix_1h[5, 1]
        + prob_matrix_1h[6, 2]
        + prob_matrix_1h[7, 3]
        + prob_matrix_1h[8, 4]
        + prob_matrix_1h[9, 5]
    )

    away_win_1h_by_1 = (
        prob_matrix_1h[0, 1]
        + prob_matrix_1h[1, 2]
        + prob_matrix_1h[2, 3]
        + prob_matrix_1h[3, 4]
        + prob_matrix_1h[4, 5]
        + prob_matrix_1h[5, 6]
    )
    away_win_1h_by_2 = (
        prob_matrix_1h[0, 2]
        + prob_matrix_1h[1, 3]
        + prob_matrix_1h[2, 4]
        + prob_matrix_1h[3, 5]
        + prob_matrix_1h[4, 6]
        + prob_matrix_1h[5, 7]
    )
    away_win_1h_by_3 = (
        prob_matrix_1h[0, 3]
        + prob_matrix_1h[1, 4]
        + prob_matrix_1h[2, 5]
        + prob_matrix_1h[3, 6]
        + prob_matrix_1h[4, 7]
        + prob_matrix_1h[5, 8]
    )
    away_win_1h_by_4 = (
        prob_matrix_1h[0, 4]
        + prob_matrix_1h[1, 5]
        + prob_matrix_1h[2, 6]
        + prob_matrix_1h[3, 7]
        + prob_matrix_1h[4, 8]
        + prob_matrix_1h[5, 9]
    )

    home_win_2h_by_1 = (
        prob_matrix_2h[1, 0]
        + prob_matrix_2h[2, 1]
        + prob_matrix_2h[3, 2]
        + prob_matrix_2h[4, 3]
        + prob_matrix_2h[5, 4]
        + prob_matrix_2h[6, 5]
    )
    home_win_2h_by_2 = (
        prob_matrix_2h[2, 0]
        + prob_matrix_2h[3, 1]
        + prob_matrix_2h[4, 2]
        + prob_matrix_2h[5, 3]
        + prob_matrix_2h[6, 4]
        + prob_matrix_2h[7, 5]
    )
    home_win_2h_by_3 = (
        prob_matrix_2h[3, 0]
        + prob_matrix_2h[4, 1]
        + prob_matrix_2h[5, 2]
        + prob_matrix_2h[6, 3]
        + prob_matrix_2h[7, 4]
        + prob_matrix_2h[8, 5]
    )
    home_win_2h_by_4 = (
        prob_matrix_2h[4, 0]
        + prob_matrix_2h[5, 1]
        + prob_matrix_2h[6, 2]
        + prob_matrix_2h[7, 3]
        + prob_matrix_2h[8, 4]
        + prob_matrix_2h[9, 5]
    )

    away_win_2h_by_1 = (
        prob_matrix_2h[0, 1]
        + prob_matrix_2h[1, 2]
        + prob_matrix_2h[2, 3]
        + prob_matrix_2h[3, 4]
        + prob_matrix_2h[4, 5]
        + prob_matrix_2h[5, 6]
    )
    away_win_2h_by_2 = (
        prob_matrix_2h[0, 2]
        + prob_matrix_2h[1, 3]
        + prob_matrix_2h[2, 4]
        + prob_matrix_2h[3, 5]
        + prob_matrix_2h[4, 6]
        + prob_matrix_2h[5, 7]
    )
    away_win_2h_by_3 = (
        prob_matrix_2h[0, 3]
        + prob_matrix_2h[1, 4]
        + prob_matrix_2h[2, 5]
        + prob_matrix_2h[3, 6]
        + prob_matrix_2h[4, 7]
        + prob_matrix_2h[5, 8]
    )
    away_win_2h_by_4 = (
        prob_matrix_2h[0, 4]
        + prob_matrix_2h[1, 5]
        + prob_matrix_2h[2, 6]
        + prob_matrix_2h[3, 7]
        + prob_matrix_2h[4, 8]
        + prob_matrix_2h[5, 9]
    )

    # Probabilities for each HT-FT outcome

    HHp = 1 / (1 / home_win_prob_fh * (1 / (draw_2h + home_win_2h))) * 1.02
    DHp = draw_prob_fh * home_win_2h
    AHp = home_win_prob - HHp - DHp

    HDp = (
        1 / (1 / home_win_1h_by_1 * 1 / away_win_2h_by_1)
        + 1 / (1 / home_win_1h_by_2 * 1 / away_win_2h_by_2)
        + 1 / (1 / home_win_1h_by_3 * 1 / away_win_2h_by_3)
        + 1 / (1 / home_win_1h_by_4 * 1 / away_win_2h_by_4)
    ) * 1.10

    ADp = (
        1 / (1 / away_win_1h_by_1 * 1 / home_win_2h_by_1)
        + 1 / (1 / away_win_1h_by_2 * 1 / home_win_2h_by_2)
        + 1 / (1 / away_win_1h_by_3 * 1 / home_win_2h_by_3)
        + 1 / (1 / away_win_1h_by_4 * 1 / home_win_2h_by_4)
    ) * 1.10
    # DDp = draw_prob - HDp - ADp

    AAp = 1 / (1 / away_win_prob_fh * (1 / (draw_2h + away_win_2h))) * 1.03
    DAp = 1 / (1 / draw_prob_fh * (1 / away_win_2h))
    HAp = away_win_prob - AAp - DAp

    ## Calculate Clean sheet

    # # Calculate Clean Sheet Probabilities
    # home_clean_sheet_prob = np.sum(prob_matrix_ft[0, :])
    # away_clean_sheet_prob = np.sum(prob_matrix_ft[:, 0])

    # # Calculate Next Goal
    # home_next_goal = hg_exp/hg_exp + ag_exp * (1-prob_matrix_ft[0,0])
    # away_next_goal = ag_exp/hg_exp + ag_exp * (1-prob_matrix_ft[0,0])
    # no_next_goal = prob_matrix_ft[0,0]

    # # Calculate Win Either Half
    # home_either_half = ((home_win_prob_fh * away_win_2h) + (home_win_prob_fh * draw_2h) + (home_win_prob_fh * home_win_2h) +
    #                     (draw_prob_fh * home_win_2h) + (away_win_prob_fh * home_win_2h))
    # away_either_half = ((away_win_prob_fh * home_win_2h) + (away_win_prob_fh * draw_2h) + (away_win_prob_fh * away_win_2h) +
    #                     (draw_prob_fh * away_win_2h) + (home_win_prob_fh * away_win_2h))

    # # Calculate Win Both Halves
    # home_both_halves = home_win_prob_fh * home_win_2h
    # away_both_halves = away_win_prob_fh * away_win_2h

    # # Calculate Asian Lines
    # h_p_0_25 = 1 / ((1 - draw_prob / 2) / (home_win_prob + draw_prob / 2))
    # a_m_0_25 = 1 / ((1 - draw_prob / 2) / away_win_prob)
    # a_p_0_25 = 1 / ((1 - draw_prob / 2) / (away_win_prob + draw_prob / 2))
    # h_m_0_25 = 1 / ((1 - draw_prob / 2) / home_win_prob)

    # h_p_0_5 = home_or_draw
    # a_m_0_5 = away_win_prob
    # a_p_0_5 = away_or_draw
    # h_m_0_5 = home_win_prob

    # awb1 = prob_matrix_ft[0,1] + prob_matrix_ft[1,2] + prob_matrix_ft[2,3] + prob_matrix_ft[3,4] + prob_matrix_ft[4,5] + prob_matrix_ft[5,6]
    # hwb1 = prob_matrix_ft[1,0] + prob_matrix_ft[2,1] + prob_matrix_ft[3,2] + prob_matrix_ft[4,3] + prob_matrix_ft[5,4] + prob_matrix_ft[6,5]

    # h_p_0_75 = 1 / ((1 - awb1 / 2) / (1 - away_win_prob))
    # a_m_0_75 = 1 / ((1 - awb1 / 2) / (away_win_prob - (awb1 / 2)))
    # a_p_0_75 = 1 / ((1 - hwb1 / 2) / (1 - home_win_prob))
    # h_m_0_75 = 1 / ((1 - hwb1 / 2) / (home_win_prob - (hwb1 / 2)))

    # h_p_1_0 = 1 / ((1 - awb1) / (1 - away_win_prob))
    # a_m_1_0 = 1 / ((1 - awb1) / (away_win_prob - awb1))
    # a_p_1_0 = 1 / ((1 - hwb1) / (1 - home_win_prob))
    # h_m_1_0 = 1 / ((1 - hwb1) / (home_win_prob - hwb1))

    # h_p_1_25 = 1 / ((1 - (awb1 / 2)) / (1 - away_win_prob + (awb1 / 2)))
    # a_m_1_25 = 1 / ((1 - (awb1 / 2)) / (away_win_prob - awb1))
    # a_p_1_25 = 1 / ((1 - (hwb1 / 2)) / (1 - home_win_prob + (hwb1 / 2)))
    # h_m_1_25 = 1 / ((1 - (hwb1 / 2)) / (home_win_prob - hwb1))

    # h_p_1_5 = 1 / (1 / (1 - away_win_prob + awb1))
    # a_m_1_5 = 1 / (1 / (away_win_prob - awb1))
    # a_p_1_5 = 1 / (1/ (1 - home_win_prob + hwb1))
    # h_m_1_5 = 1 / (1 / (home_win_prob - hwb1))

    # awb2 = prob_matrix_ft[0,2] + prob_matrix_ft[1,3] + prob_matrix_ft[2,4] + prob_matrix_ft[3,5] + prob_matrix_ft[4,6] + prob_matrix_ft[5,7]
    # hwb2 = prob_matrix_ft[2,0] + prob_matrix_ft[3,1] + prob_matrix_ft[4,2] + prob_matrix_ft[5,3] + prob_matrix_ft[6,4] + prob_matrix_ft[7,5]

    # h_p_1_75 = 1 / ((1 - awb2 / 2) / (1 - away_win_prob + awb1))
    # a_m_1_75 = 1 / ((1 - awb2 / 2) / (away_win_prob - awb1 - (awb2 / 2)))
    # a_p_1_75 = 1 / ((1 - hwb2 / 2) / (1 - home_win_prob + hwb1))
    # h_m_1_75 = 1 / ((1 - hwb2 / 2) / (home_win_prob - hwb1 - (hwb2 / 2)))

    # h_p_2_0 = 1 / ((1 - awb2) / (1 - away_win_prob + awb1))
    # a_m_2_0 = 1 / ((1 - awb2) / (away_win_prob - awb1 - awb2))
    # a_p_2_0 = 1 / ((1 - hwb2) / (1 - home_win_prob + hwb1))
    # h_m_2_0 = 1 / ((1 - hwb2) / (home_win_prob - hwb1 - hwb2))

    # h_p_2_25 = 1 / ((1 - awb2 / 2) / (1 - away_win_prob + awb1 + (awb2 / 2)))
    # a_m_2_25 = 1 / ((1 - awb2 / 2) / (away_win_prob - awb1 - awb2))
    # a_p_2_25 = 1 / ((1 - hwb2 / 2) / (1 - home_win_prob + hwb1 + (hwb2 / 2)))
    # h_m_2_25 = 1 / ((1 - hwb2 / 2) / (home_win_prob - hwb1 - hwb2))

    # h_p_2_5 = 1 / (1 / (1 - away_win_prob + awb1 + awb2))
    # a_m_2_5 = 1 / (1 / (away_win_prob - awb1 - awb2))
    # a_p_2_5 = 1 / (1 / (1 - home_win_prob + hwb1 + hwb2))
    # h_m_2_5 = 1 / (1 / (home_win_prob - hwb1 - hwb2))

    """
    Continue adding hacaps
    """

    # # dictionary to reference values in selection and return probabilities
    # lines_to_select = {"0": [dnb_home,dnb_away],
    #                     "+0.25": [h_p_0_25, a_m_0_25],
    #                     "-0.25": [h_m_0_25, a_p_0_25],
    #                     "+0.5": [h_p_0_5, a_m_0_5],
    #                     "-0.5": [h_m_0_5, a_p_0_5],
    #                     "+0.75": [h_p_0_75, a_m_0_75],
    #                     "-0.75": [h_m_0_75, a_p_0_75],
    #                     "+1.0": [h_p_1_0, a_m_1_0],
    #                     "-1.0": [h_m_1_0, a_p_1_0],
    #                     "+1.25": [h_p_1_25, a_m_1_25],
    #                     "-1.25": [h_m_1_25, a_p_1_25],
    #                     "+1.50": [h_p_1_5, a_m_1_5],
    #                     "-1.50": [h_m_1_5, a_p_1_5],
    #                     "+1.75": [h_p_1_75, a_m_1_75],
    #                     "-1.75": [h_m_1_75, a_p_1_75],
    #                     "+2.00": [h_p_2_0, a_m_2_0],
    #                     "-2.00": [h_m_2_0, a_p_2_0],
    #                     "+2.25": [h_p_2_25, a_m_2_25],
    #                     "-2.25": [h_m_2_25, a_p_2_25],
    #                     "+2.50": [h_p_2_5, a_m_2_5],
    #                     "-2.50": [h_m_2_5, a_p_2_5],
    # }

    # # Home No Bet
    # home_no_bet_draw = draw_prob / (draw_prob + away_win_prob)
    # home_no_bet_away = away_win_prob / (draw_prob + away_win_prob)

    # # Away No Bet
    # away_no_bet_draw = draw_prob / (draw_prob + home_win_prob)
    # away_no_bet_home = home_win_prob / (draw_prob + home_win_prob)

    # # Half Most Goals
    # def calculate_highest_scoring_half(prob_matrix_1h, prob_matrix_2h):
    #     prob_first_half_higher = 0
    #     prob_second_half_higher = 0
    #     prob_draw = 0

    #     # Iterate over all possible goal combinations for both halves
    #     for i in range(len(prob_matrix_1h)):
    #         for j in range(len(prob_matrix_1h[i])):
    #             for k in range(len(prob_matrix_2h)):
    #                 for l in range(len(prob_matrix_2h[k])):
    #                     prob_1h = prob_matrix_1h[i][j]
    #                     prob_2h = prob_matrix_2h[k][l]
    #                     total_goals_1h = i + j
    #                     total_goals_2h = k + l

    #                     if total_goals_1h > total_goals_2h:
    #                         prob_first_half_higher += prob_1h * prob_2h
    #                     elif total_goals_1h < total_goals_2h:
    #                         prob_second_half_higher += prob_1h * prob_2h
    #                     else:
    #                         prob_draw += prob_1h * prob_2h

    #     return prob_first_half_higher, prob_second_half_higher, prob_draw

    # # Odd/Even
    # def calculate_even_and_odd_probabilities(prob_matrix_ft):
    #     even_probability = 0
    #     for i in range(len(prob_matrix_ft)):
    #         for j in range(len(prob_matrix_ft[i])):
    #             if (i + j) % 2 == 0:  # Check if the sum of goals is even
    #                 even_probability += prob_matrix_ft[i][j]
    #     odd_probability = 1 - even_probability
    #     return even_probability, odd_probability

    # # Home Score Both Halves
    # def calculate_home_to_score_both_halves(prob_matrix_1h, prob_matrix_2h):
    #     # For the first half, sum probabilities where the home team scores 1 or more goals (i ≥ 1),
    #     # while the away team's score (j) can be any value (from 0 to maximum).
    #     prob_home_first_half = sum(prob_matrix_1h[i][j]
    #                             for i in range(1, len(prob_matrix_1h))   # i = 1, 2, ..., max home goals
    #                             for j in range(len(prob_matrix_1h[i])))    # j = 0, 1, 2, ... (all away outcomes)

    #     # Similarly, for the second half:
    #     prob_home_second_half = sum(prob_matrix_2h[i][j]
    #                                 for i in range(1, len(prob_matrix_2h))
    #                                 for j in range(len(prob_matrix_2h[i])))

    #     prob_home_score_both_halves = prob_home_first_half * prob_home_second_half
    #     # Multiply the summed probabilities to get the overall probability
    #     return prob_home_score_both_halves

    # # Away Score Both Halves
    # def calculate_away_to_score_both_halves(prob_matrix_1h, prob_matrix_2h):
    #     # For the first half, sum probabilities where the away team scores 1 or more goals (j ≥ 1),
    #     # while the home team's score (i) can be any value (from 0 to maximum).
    #     prob_away_first_half = sum(prob_matrix_1h[i][j]
    #                             for i in range(len(prob_matrix_1h))   # i = 0, 1, ..., max home goals
    #                             for j in range(1, len(prob_matrix_1h[i])))    # j = 1, 2, ... (away team scores at least 1)

    #     # Similarly, for the second half:
    #     prob_away_second_half = sum(prob_matrix_2h[i][j]
    #                                 for i in range(len(prob_matrix_2h))
    #                                 for j in range(1, len(prob_matrix_2h[i])))

    #     prob_away_score_both_halves = prob_away_first_half * prob_away_second_half
    #     # Multiply the summed probabilities to get the overall probability
    #     return prob_away_score_both_halves

    # # -----------  1 Up Functions ------------------------------

    # def calculate_win_given_one_nil(hg, ag, minute_of_goal=29):
    #     """
    #     Estimate P(Win | 1-0 lead at a given minute) for both home and away teams.

    #     Parameters:
    #     - hg: Expected goals for the home team (full match)
    #     - ag: Expected goals for the away team (full match)
    #     - minute_of_goal: The minute at which the 1-0 lead is taken (default: 29)

    #     Returns:
    #     - (P(Home wins | 1-0), P(Away wins | 0-1))
    #     """

    #     minutes_remaining = 93 - minute_of_goal

    #     home_rate = hg / 93
    #     away_rate = ag / 93

    #     rem_home_xg = home_rate * minutes_remaining
    #     rem_away_xg = away_rate * minutes_remaining

    #     # Adjust for 1-0 scenario
    #     adj_home_xg_lead = rem_home_xg * 0.95
    #     adj_away_xg_trail = rem_away_xg * 1.05

    #     # Adjust for 0-1 scenario
    #     adj_home_xg_trail = rem_home_xg * 1.05
    #     adj_away_xg_lead = rem_away_xg * 0.95

    #     max_goals = 7

    #     win_prob_home = 0.0
    #     win_prob_away = 0.0

    #     for i in range(max_goals + 1):  # goals after lead
    #         for j in range(max_goals + 1):

    #             # Home leads 1-0
    #             final_home_1_0 = 1 + i
    #             final_away_1_0 = j
    #             prob_home_lead = poisson.pmf(i, adj_home_xg_lead) * poisson.pmf(j, adj_away_xg_trail)
    #             if final_home_1_0 > final_away_1_0:
    #                 win_prob_home += prob_home_lead

    #             # Away leads 0-1
    #             final_home_0_1 = i
    #             final_away_0_1 = 1 + j
    #             prob_away_lead = poisson.pmf(i, adj_home_xg_trail) * poisson.pmf(j, adj_away_xg_lead)
    #             if final_away_0_1 > final_home_0_1:
    #                 win_prob_away += prob_away_lead

    #     return win_prob_home, win_prob_away

    # w_pb_given_1_up_h, w_pb_given_1_up_a = calculate_win_given_one_nil(hg_exp, ag_exp, minute_of_goal=29)

    # # --- 1 Up Main Calculation ---
    # # Prob (bet wins) = P(1-0 at anytime) + [ Prob(Home win - P(1-0 and HW) ]
    # def calculate_one_up(home_next_goal, home_win_prob, w_pb_given_1_up_h, away_next_goal, away_win_prob, w_pb_given_1_up_a):
    #     home_1_up = home_next_goal + home_win_prob - (home_next_goal * w_pb_given_1_up_h)
    #     away_1_up = away_next_goal + away_win_prob - (away_next_goal * w_pb_given_1_up_a)

    #     return home_1_up, away_1_up

    # # ------------------  2 UP FUNCTIONS  --------------------

    # def calculate_win_given_two_nil(hg, ag, minute_of_second_goal=42):
    #     """
    #     Estimate P(Win | 2-0 lead at a given minute) for both home and away teams
    #     using adjusted Poisson models.

    #     Parameters:
    #     - hg: Expected goals for the home team (full match)
    #     - ag: Expected goals for the away team (full match)
    #     - minute_of_second_goal: Minute at which team takes a 2-0 lead (default: 42)

    #     Returns:
    #     - Tuple: (home_win_given_2_0, away_win_given_0_2)
    #     """
    #     minutes_remaining = 93 - minute_of_second_goal

    #     home_rate = hg / 93
    #     away_rate = ag / 93

    #     rem_home_xg = home_rate * minutes_remaining
    #     rem_away_xg = away_rate * minutes_remaining

    #     # Game state adjustments: more defensive when 2-0 up
    #     adj_home_xg = rem_home_xg * 0.90
    #     adj_away_xg = rem_away_xg * 1.10

    #     adj_away_lead_home_xg = rem_home_xg * 1.10
    #     adj_away_lead_away_xg = rem_away_xg * 0.90

    #     max_goals = 7
    #     home_win_prob_given_2_0 = 0.0
    #     away_win_prob_given_0_2 = 0.0

    #     # Home team leading 2–0
    #     for i in range(max_goals + 1):  # home goals after 2-0
    #         for j in range(max_goals + 1):  # away goals
    #             prob = poisson.pmf(i, adj_home_xg) * poisson.pmf(j, adj_away_xg)
    #             final_home = 2 + i
    #             final_away = j
    #             if final_home > final_away:
    #                 home_win_prob_given_2_0 += prob

    #     # Away team leading 0–2
    #     for i in range(max_goals + 1):  # home goals after 0-2
    #         for j in range(max_goals + 1):  # away goals
    #             prob = poisson.pmf(i, adj_away_lead_home_xg) * poisson.pmf(j, adj_away_lead_away_xg)
    #             final_home = i
    #             final_away = 2 + j
    #             if final_away > final_home:
    #                 away_win_prob_given_0_2 += prob

    #     return home_win_prob_given_2_0, away_win_prob_given_0_2

    # w_pb_given_2_up_h, w_pb_given_2_up_a = calculate_win_given_two_nil(hg_exp, ag_exp, minute_of_second_goal=42)

    # # ---- 2 Up - Main Calculation  ----
    # # Prob (bet wins) = P(2-0 at anytime) + [ Prob(Home win - P(2-0 and HW) ]
    # def calculate_two_up(home_next_goal, home_win_prob, w_pb_given_2_up_h, away_next_goal, away_win_prob, w_pb_given_2_up_a):
    #     # calc initial going 2-0 up
    #     home_2_0_initial = home_next_goal * home_next_goal * 0.95
    #     away_2_0_initial = away_next_goal * away_next_goal * 0.95

    #     home_2_up = home_2_0_initial + home_win_prob - (home_2_0_initial * w_pb_given_2_up_h)
    #     away_2_up = away_2_0_initial + away_win_prob - (away_2_0_initial * w_pb_given_2_up_a)

    #     return home_2_up, away_2_up

    # HTEP (Half time early payout)
    htep_h = home_win_prob + HDp + HAp
    htep_a = away_win_prob + ADp + AHp
    htep_x = draw_prob + DHp + DAp

    return (
        hg_exp, 
        ag_exp,
        home_win_prob,
        away_win_prob,
        draw_prob,
        htep_h,
        htep_x,
        htep_a,
    )

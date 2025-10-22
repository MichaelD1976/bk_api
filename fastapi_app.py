from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
# import numpy as np
# from typing import List
# from functions.goals import calc_exp_team_gls_from_1x2_ou
from functions.odds import get_markets_and_true_probabilities



'''
uvicorn fastapi_app:app --reload
'''


app = FastAPI(
    title="Football Betting Analytics API",
    description="""
This API exposes football betting related functions:

- `/htep` → generates half-time early payout true odds

All endpoints accept JSON payloads and return JSON responses.
""",
)

# ----------------------------------------
# Common Error Responses with explanations
# ----------------------------------------

common_responses = {
    404: {
        "description": "Resource not found — the requested endpoint or item does not exist",
        "content": {
            "application/json": {
                "example": {"detail": "Not Found"}
            }
        },
    },
    422: {
        "description": "Validation error — input values are missing or invalid",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "some_field"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        },
    },
    500: {
        "description": "Internal server error — unexpected failure on the server",
        "content": {
            "application/json": {
                "example": {"detail": "Internal Server Error"}
            }
        },
    },
}


# Initial Load - redirect to docs page
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# --------------------------------
# Post request 1 - htep
# --------------------------------

# Create a model to validate and parse the request
# fast api integrates pydantic BaseModel to validate, convert and parse correct data types & provide fastapi swagger docs schemas
class HTEPRequest(BaseModel):
    home_odds: float
    draw_odds: float
    away_odds: float
    over_2_5_odds: float
    under_2_5_odds: float

# class HTEPResponse(BaseModel):
#     htep_h: float
#     htep_a: float
#     htep_x: float

@app.post(
        "/htep",
        summary="Calculate HTEP full-time odds",
        description="""
            This endpoint generates the true odds for home, away, draw half-time early pay-out
            based on 1X2 odds and Over/Under 2.5 goals odds inputs.

            *Parameters:*
            - `home_odds`: Odds for the home team
            - `draw_odds': Odds for a draw
            - `away_odds`: Odds for the away team
            - `over_2_5_odds`: Odds for over 2.5 goals
            - `under_2_5_odds`: Odds for under 2.5 goals

            *Returns:*  
            - `htep_h`: Home EP true odds (float)
            - `htep_x`: Away EP true odds (float)
            - `htep_a`: Away EP true odds (float)
            """,
        # response_model=HTEPResponse,
        responses={
        200: {"description": "Expected goals calculated successfully"},
        **common_responses
                },
) 
def htep_true_odds(odds: HTEPRequest):
    hgexp, agexp, home_win_prob, away_win_prob, draw_prob,htep_h, htep_x, htep_a = get_markets_and_true_probabilities(
    odds.home_odds, 
    odds.draw_odds, 
    odds.away_odds, 
    odds.over_2_5_odds, 
    odds.under_2_5_odds
    )
    return {hgexp, agexp, home_win_prob, away_win_prob, draw_prob,htep_h, htep_x, htep_a}

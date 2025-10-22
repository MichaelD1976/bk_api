## BK Prototype Trading/Pricing API  

### Various BK market pricing functions with fastapi integration

#### To Add
- HTUP
- Chancemix
- Pen Yes/No
- GG_NG or OU
- Stats Mkts? - SOT,Fouls,Offsides

---
#### Functions added

odds.py:
- calc_prob_matrix():  
    args - hg_exp, ag_exp  
    returns - prob_matrix_ft, prob_matrix_1h, prob_matrix_2h, 

- get_markets_and_true_probailities()  
    args - home_odds, draw_odds, away_odds, over_2_5_odds, under_2_5_odds  
    returns - any market (uncomment calculation from function & add to return) - currently just HTEP configured

goals.py:
- calc_team_goals_from_1x2_ou:  
    args - 1,x,2,ov,un  
    returns - hg_exp, ag_exp


#### Test directory
- test_functions:  
    calc_prob_matrix

powershell: make test / make format / make lint

---
#### Installed Packages
pip install ...
- pandas
- fastapi uvicorn
- scipy
- pytest pylint black
- jupyter ipykernel
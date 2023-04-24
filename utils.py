import pandas as pd
import numpy as np

def advanced_data_prediction(data):

    POSSESSION = (data["2_PTS"] + data["3_PTS"]) + (0.44 * data["LF"]) - data["Offensive Rebound"] + data['Turnover']
    TOV = (data['Turnover'] / POSSESSION) * 100
    eFG = (((data["2_PTS_M"] + data["3_PTS_M"]) + (0.5 * data["3_PTS_M"])) / (data["2_PTS"] + data["3_PTS"])) * 100
    OREB = (data["Offensive Rebound"] / (data["Offensive Rebound"] + data["ADV_Defensive Rebound"])) * 100
    DREB = (data["Defensive Rebound"] / (data["Defensive Rebound"] + data["ADV_Offensive Rebound"])) * 100

    OREB_ADV = (data["ADV_Offensive Rebound"] / (data["ADV_Offensive Rebound"] + data["Defensive Rebound"])) * 100
    DREB_ADV = (data["ADV_Defensive Rebound"] / (data["ADV_Defensive Rebound"] + data["Offensive Rebound"])) * 100

    FT_Rate = (data["LF"] / (data["2_PTS"] + data["3_PTS"])) * 100

    POSSESSION_ADV = (data["ADV_2_PTS"] + data["ADV_3_PTS"]) + (0.44 * data["ADV_LF"]) - data["ADV_Offensive Rebound"] + data['ADV_Turnover']
    TOV_ADV = (data['ADV_Turnover'] / POSSESSION_ADV) * 100
    eFG_ADV = (((data["ADV_2_PTS_M"] + data["ADV_3_PTS_M"]) + (0.5 * data["ADV_3_PTS_M"])) / (data["ADV_2_PTS"] + data["ADV_3_PTS"])) * 100
    FT_Rate_ADV = (data["ADV_LF"] / (data["ADV_2_PTS"] + data["ADV_3_PTS"])) * 100

    OFF100 = 100 * (data['Points'] / POSSESSION)
    DEF100 = 100 * (data['PTS ENCAISSES'] / POSSESSION_ADV)
    eFG2 = (data["2_PTS_M"] / data["2_PTS"]) * 100
    eFG3 = (data["3_PTS_M"] / data["3_PTS"]) * 100
    POTENTIAL = (data["2_PTS"] * 2) + (data["3_PTS"] * 3) + data["LF"]

    eFG2_ADV = (data["ADV_2_PTS_M"] / data["ADV_2_PTS"]) * 100
    eFG3_ADV = (data["ADV_3_PTS_M"] / data["ADV_3_PTS"]) * 100
    POTENTIAL_ADV = (data["ADV_2_PTS"] * 2) + (data["ADV_3_PTS"] * 3) + data["LF"]

    DIFF_POTENTIAL = POTENTIAL - POTENTIAL_ADV

    ASSIST = (data["Assist"] / (data["2_PTS_M"] + data["3_PTS_M"])) * 100
    ASSIST_ADV = (data["ADV_Assist"] / (data["2_PTS_M"] + data["3_PTS_M"])) * 100

    STEAL = (data["Steal"] / POSSESSION) * 100
    STEAL_ADV = (data["ADV_Steal"] / POSSESSION_ADV) * 100

    PIR = data["PIR"]
    PIR_ADV = data["ADV_PIR"]
    # Pas sur qu'on utilise
    # PROP_3_PTS = (data["3_PTS"] / (data["2_PTS"] + data["3_PTS"])) * 100
    POSSESSION = (POSSESSION + POSSESSION_ADV)  / 2

    Team = data['Team']
    Adversaire = data["Adversaire"]
    Top8 = data["Top8"]

    # OFF100, DEF100, "OFF100", "DEF100", PIR, PIR_ADV, "PIR", , "PIR_ADV", eFG,  "eFG", "eFG_ADV",  eFG_ADV, POSSESSION, "POSSESSION",

    advanced_data = pd.DataFrame(list(zip(Team, Adversaire, Top8, TOV, eFG2, eFG3, OREB, DREB, FT_Rate, # ASSIST, STEAL, 
                                         TOV_ADV, eFG2_ADV, eFG3_ADV, OREB_ADV, DREB_ADV, FT_Rate_ADV, #ASSIST_ADV, STEAL_ADV,
                                         # DIFF_POTENTIAL
                                           )), 
                                           columns= ["Team", "Adversaire", "Top8", "TOV", "eFG2", "eFG3", "OREB", "DREB", "FT_Rate", #"ASSIST", "STEAL",
                                            "TOV_ADV", "eFG2_ADV", "eFG3_ADV", "OREB_ADV", "DREB_ADV", "FT_Rate_ADV", #"ASSIST_ADV", # "STEAL_ADV",
                                          #  "DIFF_POTENTIAL"
                                              ])
    

    return advanced_data


def predict_matchup(HomeTeam, AwayTeam, CHomeFar, CHomePaint, CHomeAgressivity, CAwayFar, CAwayPaint, CAwayAgressivity, model, data): 

    TeamHome = data[(data['Team'] == HomeTeam) & (data['Top8'] == 1) ]
    TeamAway = data[(data['Adversaire'] == AwayTeam) & (data['Top8'] == 1)]

    data = advanced_data_prediction(data)

    TeamHome = advanced_data_prediction(TeamHome)
    TeamAway = advanced_data_prediction(TeamAway)

    HomeFar = [] 
    HomeAgressivity = []
    AwayFar = []
    AwayAgressivity = []

    for i in ['TOV', "eFG3"] :
        HomeFar.append((np.std(TeamHome[i]) + np.std(TeamAway[i+str('_ADV')])) / 2)

    HomePaint = (np.std(TeamHome["eFG2"]) + np.std(TeamAway["eFG2"])) / 2
    AwayPaint =(np.std(TeamAway["eFG2_ADV"]) + np.std(TeamHome["eFG2_ADV"])) / 2

    for i in ['TOV_ADV', "eFG3_ADV"] :
        AwayFar.append((np.std(TeamAway[i]) + np.std(TeamHome[i])) / 2 )

    for i in ["OREB", "DREB", "FT_Rate"] :
        HomeAgressivity.append((np.std(TeamHome[i]) + np.std(TeamAway[i])) / 2)

    for i in ["OREB_ADV", "DREB_ADV", "FT_Rate_ADV"] :
        AwayAgressivity.append((np.std(TeamAway[i]) + np.std(TeamHome[i])) / 2)

    TeamHome =  TeamHome.groupby(['Team']).mean()
    TeamAway =  TeamAway.groupby(['Adversaire']).mean()

    Confrontation1 = data[(data['Team'] == HomeTeam) & (data['Adversaire'] == AwayTeam)]

    ConfrontationInversed = data[(data['Team'] == AwayTeam) & (data['Adversaire'] == HomeTeam)]
    ConfrontationReversed = ConfrontationInversed.copy()
    ConfrontationReversed[["Team", "TOV", 'eFG2', 'eFG3', 'OREB', 'DREB', 'FT_Rate', #"DIFF_POTENTIAL",
                        "TOV_ADV", 'eFG2_ADV', 'eFG3_ADV', 'OREB_ADV', 'DREB_ADV', 'FT_Rate_ADV' #'ASSIST_ADV', 'STEAL_ADV',
                        ]] = ConfrontationInversed[["Adversaire", "TOV_ADV", 
                        'eFG2_ADV', 'eFG3_ADV', 'OREB_ADV', 'DREB_ADV', 'FT_Rate_ADV', #"DIFF_POTENTIAL",
                        "TOV", 'eFG2', 'eFG3', 'OREB', 'DREB', 'FT_Rate', #'ASSIST', 'STEAL'
                        ]]

    # ConfrontationReversed['DIFF_POTENTIAL'] = ConfrontationReversed["DIFF_POTENTIAL"] * -1

    ConfrontationTop8 = ConfrontationReversed.copy()
    columns = ["TOV", 'eFG2', 'eFG3', 'OREB', 'DREB', 'FT_Rate', "TOV_ADV", 'eFG2_ADV', 'eFG3_ADV', 'OREB_ADV', 'DREB_ADV', 'FT_Rate_ADV']
    for i in columns : 
        ConfrontationTop8[i] = (TeamHome.reset_index()[i].values + TeamAway.reset_index()[i].values)/2

    Game = pd.concat([Confrontation1, ConfrontationReversed, ConfrontationTop8], ignore_index=True)
    Game = Game.groupby(["Team"]).mean()

    Game.reset_index(inplace=True)
    Game.drop(columns=['Team', "Top8"], axis=0, inplace=True)

    Game["TOV"] = Game["TOV"] - (CHomeFar * HomeFar[0])
    Game["eFG3"] = Game["eFG3"] + (CHomeFar * HomeFar[1])
    # Game["ASSIST"] = Game["ASSIST"] + (CHomeEfficiency * HomeEfficiency[3])

    Game["eFG2"] = Game["eFG2"] + (CHomePaint * HomePaint)
    Game["eFG2_ADV"] = Game["eFG2_ADV"] + (CAwayPaint * AwayPaint)

    Game["TOV_ADV"] = Game["TOV_ADV"] - (CAwayFar * AwayFar[0])
    Game["eFG3_ADV"] = Game["eFG3_ADV"] + (CAwayFar * AwayFar[1])
    # Game["ASSIST_ADV"] = Game["ASSIST_ADV"] + (CAwayEfficiency * AwayEfficiency[3])

    Game["OREB"] = Game["OREB"] + (CHomeAgressivity * HomeAgressivity[0])
    Game["DREB"] = Game["DREB"] + (CHomeAgressivity * HomeAgressivity[1])
    Game["FT_Rate"] = Game["FT_Rate"] + (CHomeAgressivity * HomeAgressivity[2])
    # Game["STEAL"] = Game["STEAL"] + (CHomeAgressivity * HomeAgressivity[3])

    Game["OREB_ADV"] = Game["OREB_ADV"] + (CAwayAgressivity * AwayAgressivity[0])
    Game["DREB_ADV"] = Game["DREB_ADV"] + (CAwayAgressivity * AwayAgressivity[1])
    Game["FT_Rate_ADV"] = Game["FT_Rate_ADV"] + (CAwayAgressivity * AwayAgressivity[2])
    # Game["STEAL_ADV"] = Game["STEAL_ADV"] + (CAwayAgressivity * AwayAgressivity[3])

    # print("The probability of ", str(HomeTeam), "- " + str(AwayTeam), "is : ", LR.predict_proba(Game), LR.predict(Game))

    return model.predict_proba(Game)[0][0] * 100, model.predict_proba(Game)[0][1] * 100, model.predict(Game)
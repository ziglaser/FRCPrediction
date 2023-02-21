# This work is licensed under the Creative Commons Attribution Non-Commercial 3.0 Unported License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/3.0/

import tbapy
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import brier_score_loss
import time
import random
import tensorflow # never explicitly called, but is a necessary dependency for Keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense

# This is an important switch. Once you train the model on your computer, I'd turn it off, then if you want to
# make adjustments to the neural network, you don't have to recalculate the underlying training data
data_collection = True

# Allows me to access TBA API
tba = tbapy.TBA("x9bbuY4qH2k7jo0YTynCWKMWUGnMLqhI5mBXnfq3OaIB8HPiJPJrbTCSFw6CliKK")

# If you've already initialized the model, you can use the np.load line, else you need
# To start from a blank team dictionary
dict_team_OPRs = {}
#dict_team_OPRs = np.load('OPRs_2019.npy').item()

# Scrapes data from Caleb Sykes' ELO data into a dictionary
dict_team_ELOs = {}
ELO_csv = open('C:\\Users\\zglas\\Documents\\2019ELO.csv', "r", encoding='utf-8-sig')
for line in ELO_csv:
    line.strip()
    team, elo = line.split(",")
    dict_team_ELOs["frc"+str(team)] = int(elo)

def rank(index, array):
    # Finds the rank of a certain index value wise in an array
    # Used in the event simulation function to find average seed
    target = array[index]
    sorted_array = sorted(array, reverse=True)
    return sorted_array.index(target) + 1


def load_model():
    global OPR_model, ELO_model
    # load json and create model
    OPR_json_file = open('OPR_model.json', 'r')
    # This json file stores all of the information about the model that we need, it stores nodes, dropout, etc
    loaded_OPR_model_json = OPR_json_file.read()
    OPR_json_file.close()
    # Using the Keras function model_from_json to reconstruct the model in order to make predictions
    OPR_model = model_from_json(loaded_OPR_model_json)
    # load weights for each neuron into new model
    OPR_model.load_weights("OPR_model.h5")


def ELO_probability(diff):
    # This is the formula for probability from ELO, https://en.wikipedia.org/wiki/Elo_rating_system
    return 1/(1+10**(-diff/400))


def Prob_From_Alliances(red_alliance, blue_alliance):
    OPR_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ELO_data = 0
    for j in red_alliance:
        for q in range(7):
            OPR_data[q] += int(float(dict_team_OPRs[j][q]))
        ELO_data += dict_team_ELOs[j]
    for j in blue_alliance:
        for q in range(7):
            OPR_data[q + 7] += int(float(dict_team_OPRs[j][q]))
        ELO_data -= dict_team_ELOs[j]

    # Like above, we have to reformat the data set using numpy into a vector in order to use keras's predict_proba function
    b = []
    for stat in OPR_data:
        b.append([stat])
    b = np.array(b).T
    OPR_prob = OPR_model.predict_proba(b)[0][0]
    ELO_prob = ELO_probability(ELO_data)
    return (OPR_prob+ELO_prob)/2


def calculate_opr(event):
    global dict_team_OPRs

    # Retrieves our lists of matches and teamsfrom the TBA API
    teams = tba.event_teams(event, keys=True)
    matches = tba.event_matches(event, keys=True)

    # Finding just quals by removing quarters, semis, and finals matches from the list
    start_location = matches.index(event + "_qm1")
    try:
        end_location = matches.index(event + "_sf1m1")
    except:
        end_location = None
    matches = matches[start_location:end_location]

    r = 0
    while r < len(matches):
        match_dict = tba.match(matches[r])
        # print match_dict
        if match_dict['alliances']['red']['score'] == -1:
            matches.remove(matches[r])
        else:
            r += 1
    # Removes no shows
    all_matches = ""
    # Creates a list of all the mentions in the event
    for i in matches:
        all_matches += str(tba.match(i))
    # If a team is not mentioned for the duration of the event, they are removed from the team list
    p = 0
    while p < len(teams):
        if teams[p] not in all_matches:
            # print teams[p] + "was removed"
            teams.remove(teams[p])
        p += 1

    # initializes sparse matrix and stats
    sparse_matrix = []
    stats = [[], [], [], [], [], [], [], [], []]
    stats_names = ["totalPoints", "teleopPoints", "rp", "sandStormBonusPoints", "cargoPoints", "hatchPanelPoints",
                   "habClimbPoints", "completeRocketRankingPoint", "habDockingRankingPoint"]

    # Lets scrape some web!
    for i in matches:
        # Creates the dictionary with all information pertaining to the match
        working_match = tba.match(i)
        # Finds, indexes and then records score for red and blue alliance in the data
        def find_a_stat(name, alliance):
            # Score is stored in a different dictionary than the rest of collected stats
            try:
                return int(working_match['score_breakdown'][alliance][name])
            except:
                return 0

        # Puts the scores in the result scores vector
        for i in range(len(stats)):
            stats[i].append([find_a_stat(stats_names[i], 'red')])
            stats[i].append([find_a_stat(stats_names[i], 'blue')])
            # creates the array out of alliances, runs through a team list, puts a 1 if they are in the alliance, 0 if they are not
        row_red = []
        row_blue = []
        for j in teams:
            if j in working_match['alliances']['red']['team_keys']:
                row_red.append(1)
            else:
                row_red.append(0)
            if j in working_match['alliances']['blue']['team_keys']:
                row_blue.append(1)
            else:
                row_blue.append(0)
                # Puts all of the rows into the event matrix. It is important to make sure that red and blue stay in the same order so we calculate the correct statistics
        sparse_matrix.append(row_red)
        sparse_matrix.append(row_blue)

    # Makes sure when we do operations to the matrix that they work fine since I am unsure of what would happen with ints
    for i in range(len(stats)):
        for j in range(len(stats[i])):
            stats[i][j][0] = float(stats[i][j][0])

    # We need to convert every list into a numpy array in order to use matrix algebra
    for i in range(len(stats)):
        stats[i] = np.array(stats[i])

    # Makes sure everything carries out smoothly with floats once again
    for i in range(len(sparse_matrix)):
        for j in range(len(sparse_matrix[i])):
            sparse_matrix[i][j] = float(sparse_matrix[i][j])

    # Makes everything fine and numpy
    sp = np.array(sparse_matrix)

    # Finds At*A, so that we can invert in and multiply the score vector by its inverse (spls = sparse matrix least squares)
    spls = np.dot(sp.T, sp)
    array_of_god = []
    # Goal is x = ((AtA)^-1)(At)(scores)
    # This is where all of the complex linear algebra happens
    for i in range(len(stats)):
        array_of_god.append(np.dot(inv(spls), np.dot(sp.T, stats[i])))

    # This loop puts all of the statistics into the "teams array" so that we can put that into the OPRs dictionary
    for i in range(len(teams)):
        teams[i] = [teams[i]]
        for j in range(len(stats)):
            teams[i].append(round(float(array_of_god[j][i][0]),2))

    # This allows us to create a dictionary of teams and their component OPRs that we can efficiently query, as well as save and load them
    for i in teams:
        # Checks to see if the team already has statistics in the dictionary
        current = dict_team_OPRs.get(i[0], 0)
        if current == 0:
            # If it doens't, we just add in our newly calculated statistics
            dict_team_OPRs[i[0]] = i[1:]
        else:
            # If it does, we average all of our new statistics with the existing values in the dictionary, and replaces the old values with those
            average_values = []
            for j in range(len(current)):
                average_values.append((current[j] + i[1:][j]) / 2)
            dict_team_OPRs[i[0]] = average_values

    # Now we can save the new dictionary using numpy (isn't it great) to be reloaded as neccesary
    np.save('OPRs_2019.npy', dict_team_OPRs)


def Alliance_Selection():
    global dict_team_OPRs
    # dict_team_OPRs = np.load('OPRs.npy').item()

    print("Starting Aliance Selection... This should take 3-10 minutes")
    # loads our model from storage so we don't have to retrain
    load_model()
    # Part of the CLI to enable specificity in Alliance Selection
    number = input("What team are you on? \n")
    event = input("What Event are you competing at (enter event code i.e. 2019mnmi) \n")

    calculate_opr(event)

    team = 'frc' + number
    team_list = tba.event_teams(event, keys=True)
    ind = 0
    # Removes teams from the team list that never collected data, these are the teams we removed from calculating OPRs becuase they didn't show up for any matches
    while ind < len(team_list):
        if dict_team_OPRs.get(team_list[ind], 0) == 0:
            team_list.remove(team_list[ind])
        ind += 1
    # Generating Sample Opposing Alliance Data
    opp_research = []
    # We want to create 1000 random alliances and collect their stats to input into our prediction model
    for i in range(1000):
        opp_alliance = [random.choice(team_list), random.choice(team_list), random.choice(team_list)]
        opp_alliance_OPR_data = [0, 0, 0, 0, 0, 0, 0]
        opp_alliance_ELO_data = 0
        for robot in opp_alliance:
            # This takes the sum of the stats from all of the robots and puts them into our list of data
            for stat in range(len(dict_team_OPRs[robot])):
                opp_alliance_OPR_data[stat] += int(float(dict_team_OPRs[robot][stat]))
            opp_alliance_ELO_data -= dict_team_ELOs[robot]
        # We want all the data in the larger list so we can iterate through each alliance
        opp_research.append([opp_alliance_ELO_data,opp_alliance_OPR_data])

    for i in range(len(team_list)):
        team_list[i] = str(team_list[i])

    teams_picked = []
    include = input("do you wish to include a team? (True/False) ")
    if include == "True" or "T":
        team_code = 'frc' + input("input team number: ")
        teams_picked.append(team_code)
    loops = int(input("how many teams do you wish to exclude? "))
    for i in range(loops):
        teams_picked.append('frc' + input("input team number: "))

    for taken in teams_picked:
        team_list.remove(taken)
    team_list.remove(team)
    possible_alliance_results = []
    if include == "True":
        for i in team_list:
            start_test = time.time()
            alliance = [team, team_code, i]
            alliance_OPR_data = [0, 0, 0, 0, 0, 0, 0]
            alliance_ELO_data = 0
            for robot in alliance:
                for stat in range(len(dict_team_OPRs[robot])):
                    # Adds the sum of the component stats for each robot together and puts them in the alliance list
                    alliance_OPR_data[stat] += int(float(dict_team_OPRs[robot][stat]))
                alliance_ELO_data += dict_team_ELOs[robot]

            # We want figure out how many wins the alliance gets against the random slate, so we iterate through and count
            alliance_wins = 0
            for opposing_alliance_data in opp_research:
                input_data = alliance_OPR_data + opposing_alliance_data[1]
                # Input data contains the data we want to input into the model now: 1 alliance appended to their opponents, but we also need to reshape it into a vector in order for predict_proba to work
                b = []
                for stat in input_data:
                    b.append([stat])
                b = np.array(b).T
                # If we add all of the probabilities, we get the "expected wins" on average
                OPR_proba = OPR_model.predict_proba(b)[0][0]
                ELO_proba = ELO_probability(alliance_ELO_data+opposing_alliance_data[0])
                alliance_wins += (OPR_proba+ELO_proba)/2
            alliance_win_percentage = round(alliance_wins / 10, 2)
            possible_alliance_results.append([alliance, alliance_win_percentage])
            end_test = time.time()
            # Gives progress given this takes a long time
            print("alliance", alliance, "tested in", end_test - start_test, "seconds")
    else:
        for i in team_list:
            # We don't want to have a team on the alliance twice, so we remove them from consideration for the 2nd robot
            temp_list = team_list
            temp_list.remove(i)
            for j in temp_list:
                start_test = time.time()
                alliance = [team, i, j]
                alliance_OPR_data = [0, 0, 0, 0, 0, 0, 0]
                alliance_ELO_data = 0
                for robot in alliance:
                    for stat in range(len(dict_team_OPRs[robot])):
                        # Adds the sum of the component stats for each robot together and puts them in the alliance list
                        alliance_OPR_data[stat] += int(float(dict_team_OPRs[robot][stat]))
                    alliance_ELO_data += dict_team_ELOs[robot]

                # We want figure out how many wins the alliance gets against the random slate, so we iterate through and count
                alliance_wins = 0
                for opposing_alliance_data in opp_research:
                    input_data = alliance_OPR_data + opposing_alliance_data[1]
                    # Input data contains the data we want to input into the model now: 1 alliance appended to their opponents, but we also need to reshape it into a vector in order for predict_proba to work
                    b = []
                    for stat in input_data:
                        b.append([stat])
                    b = np.array(b).T
                    # If we add all of the probabilities, we get the "expected wins" on average
                    OPR_proba = OPR_model.predict_proba(b)[0][0]
                    ELO_proba = ELO_probability(alliance_ELO_data + opposing_alliance_data[0])
                    alliance_wins += (OPR_proba + ELO_proba) / 2
                alliance_win_percentage = round(alliance_wins / 5, 2)
                possible_alliance_results.append([alliance, alliance_win_percentage])
                end_test = time.time()
                # Gives progress given this takes a long time
                print("alliance", alliance, "tested in", end_test - start_test, "seconds")
    # This is a rudimentary bubble sort that will allow us to only examine the best alliances (especially since too many print statements overflow the screen and make some inaccessible)
    for i in range(len(possible_alliance_results)):
        for j in range(1, len(possible_alliance_results)):
            if possible_alliance_results[j][1] >= possible_alliance_results[j - 1][1]:
                possible_alliance_results[j], possible_alliance_results[j - 1] = possible_alliance_results[j - 1], \
                                                                                 possible_alliance_results[j]
    # We can see the top 50 alliances since that's roughly the amount needed to ensure that no matter what happens in alliance selection, we'll have data on the best possible remaining alliance
    print(possible_alliance_results[:50])
    # After every function we want the user to be able to continue using the program, so we recurse back to the main menu
    Main_Menu()


def Component_OPRs():
    global dict_team_OPRs

    print("Starting process of retriving OPRs, should be instantaneous")
    # Part of the CLI, enables the user more options about the volume and specificity of data they want to receive
    state = input("Do you want an individual team's stats or an event's stats? (input 'team' or 'event') \n")
    if state == 'team':
        # We want to input and format the team so we can query the dictionary
        team = input("Enter team number \n")
        team_code = 'frc' + team

        results = dict_team_OPRs.get(team_code, 0)
        # We want to take a shortcut if we can and just pull the data from the dictionary,
        # unless it's not listed in which case we need to calculate data for their event, data we need from the user
        if results == 0:
            last_event = input("What event were they last at? (input event code i.e. 2019mnmi) \n")
            calculate_opr(last_event)
            print(team, dict_team_OPRs.get(team_code, 0))
        else:
            print(team, results)

    else:
        event = input("What event are you looking at? (input event code i.e. 2019mnmi) \n")
        # We can't tell if an event has been previously calculated, so we recalculate
        calculate_opr(event)
        team_list = tba.event_teams(event, keys=True)

        # This is a relatively readable format to return data, each team and their respective stats, with labels at the top
        for i in team_list:
            print(i, dict_team_OPRs.get(i, []))
    # Again, recursing back to the main page to ensure continuous use
    Main_Menu()


def Match_Probability(match_code, model_loaded):
    # This function is a relatively pointless CLI version of Prob_From_Alliances
    global dict_team_OPRs, OPR_model
    # loads our model from storage so we don't have to retrain
    if not model_loaded:
        load_model()
    # This gives the dictionary with all of the data from the match in quesiton
    working_match = tba.match(match_code)
    # The dictionaries are nested, so we have to work our way through the keys to get the teams involved in the match
    red_alliance = working_match['alliances']['red']['team_keys']
    blue_alliance = working_match['alliances']['blue']['team_keys']

    return Prob_From_Alliances(red_alliance,blue_alliance)


def Predict_Event_Matches(event):
  global dict_team_OPRs
  load_model()
  calculate_opr(event)
  matches = tba.event_matches(event, keys = True)
  predictions = []
  for match in matches:
    prob = Match_Probability(match, True)
    if prob >= 0.5:
      prediction = [match, round(prob, 4), 'red']
    else:
      prediction = [match, round(1-prob, 4), 'blue']
    predictions.append(prediction)
    print("predicted match", match)
  print(predictions)
  Main_Menu()


def Event_Simulation(event):
    load_model()
    calculate_opr(event)
    trials = 50
    teams = tba.event_teams(event, keys=True)
    # ranks is where we're going to store the seeds from every trial
    ranks = [[]]*len(teams)

    matches = tba.event_matches(event, keys = True)
    matches = matches[matches.index(event+"_qm1"):matches.index(event+"_sf1m1")] # only qual matches
    rp_temp = [0]*len(teams)

    for trial in range(trials):
        print(trial)
        rps = rp_temp
        for match in matches:
            match_dict = tba.match(match)
            red_alliance, blue_alliance = match_dict['alliances']['red']['team_keys'], match_dict['alliances']['blue']['team_keys']
            # We're only making predictions for unplayed matches mid-event
            if True or match_dict['alliances']['red']['score'] == -1 or match_dict['alliances']['blue']['score'] == -1:
                red_rp = 0
                blue_rp = 0
                red_win = Prob_From_Alliances(red_alliance, blue_alliance)
                red_rocket_prob = 0
                blue_rocket_prob = 0
                red_hab_prob = 0
                blue_hab_prob = 0
                for team in red_alliance:
                    red_rocket_prob += dict_team_OPRs[team][7] # this takes the calculated contributions for the various RPs
                    red_hab_prob += dict_team_OPRs[team][8] # 10/10 best way to find RP probabilities
                for team in blue_alliance:
                    blue_rocket_prob += dict_team_OPRs[team][7]
                    blue_hab_prob += dict_team_OPRs[team][8]
                win_num = random.random()
                red_rp += (random.random() < red_rocket_prob) + (random.random() < red_hab_prob) + 2*(win_num < red_win)
                blue_rp += (random.random() < blue_rocket_prob) + (random.random() < blue_hab_prob) + 2 * (win_num > red_win)
                for team in red_alliance:
                    rps[teams.index(team)] += red_rp
                for team in blue_alliance:
                    rps[teams.index(team)] += blue_rp
            else:
                # if the match is played we just get data from TBA
                red_rp = match_dict['score_breakdown']['red']['rp']
                blue_rp = match_dict['score_breakdown']['blue']['rp']
                for team in red_alliance:
                    rps[teams.index(team)] += red_rp
                for team in blue_alliance:
                    rps[teams.index(team)] += blue_rp
        for j in range(len(teams)):
            ranks[j] = ranks[j]+[rank(j, rps)]
    # Sort teams by average seed
    for m in range(len(ranks)):
        for n in range(len(ranks)-1):
            if sum(ranks[n])/len(ranks[n]) > sum(ranks[n+1])/len(ranks[n+1]):
                ranks[n], ranks[n+1] = ranks[n+1], ranks[n]
                teams[n], teams[n+1] = teams[n+1], teams[n]

    for p in range(len(teams)):
        print(teams[p], sum(ranks[p])/len(ranks[p]))


def Initialize_Model():
    global dict_team_OPRs
    print("Reinitializing model... Should take a loooong time (>10 minutes)")
    if data_collection:
        # Splitting the data into a training and testing pool ensures that we don't overfit the model by testing its results on the same data we used to train it
        # This is an important step in ensuring prediction quality
        event_list = []
        for event in tba.events("2019", keys=True):
            if tba.event(event)['week'] != None and tba.event(event)['week'] >= 0:
                event_list.append([event, tba.event(event)['week']])

        for i in range(len(event_list)):
            for j in range(len(event_list) - 1):
                if event_list[j][1] > event_list[j + 1][1]:
                    event_list[j], event_list[j + 1] = event_list[j + 1], event_list[j]
        for i in range(len(event_list)):
            event_list[i] = event_list[i][0]

        test_event_list = ['2019hop', '2019carv', '2019gal', '2019tur', '2019new', '2019roe', '2019cars','2019tes','2019cur','2019dar','2019dal','2019arc']
        for thing in test_event_list:
            event_list.append(thing)

        X_OPR = []
        X_ELO = []
        Y = []
        X_OPR_test = []
        X_ELO_test = []
        Y_test = []

        for code in event_list:
            # For our reinitialization we want to recalculate OPRs too to make sure they're up to date
            matches = tba.event_matches(code, keys=True)
            # Finding just quals by eliminating eliminations matches
            try:
                start_location = matches.index(code + "_qm1")
            except:
                continue
            end_location = matches.index(code + "_sf1m1")
            matches = matches[start_location:end_location]
            # We finally get to data collection
            # The if/else statement allows us to sort between adding data to the training and testing lists
            if code not in test_event_list:
                try:
                    calculate_opr(code)
                except:
                    print("event", code, "failed")
                    continue
                for i in matches:
                    # Creates dictionary with match information
                    working_match = tba.match(i)
                    winner = working_match['winning_alliance']
                    # Indicates if red or blue won, mapping red win as 1, blue win as 0
                    if winner == "red":
                        Y.append(1)
                    else:
                        Y.append(0)
                    # Now that we have Y, X is going to be tough to get
                    # The first 10 indecies of the Xdata list represent the red alliance stats, while the second half keeps track of blue
                    Xdata_OPR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    Xdata_ELO = 0
                    red_alliance = working_match['alliances']['red']['team_keys']
                    blue_alliance = working_match['alliances']['blue']['team_keys']
                    # For every team in the match we pull their stats from the dictionary and add it to our data
                    for j in red_alliance:
                        for q in range(7):
                            Xdata_OPR[q] += int(float(dict_team_OPRs[j][q]))
                        Xdata_ELO += dict_team_ELOs[j]
                    for j in blue_alliance:
                        for q in range(7):
                            Xdata_OPR[q + 7] += int(float(dict_team_OPRs[j][q]))
                        Xdata_ELO -= dict_team_ELOs[j]
                    # Collecting all data in one large array allows us to input it into the Keras model
                    X_OPR.append(Xdata_OPR[0])
                    X_ELO.append(Xdata_ELO)
            else:
                for i in matches:
                    # We can use the exact same method as above, but we want to add data to the testing set instead
                    working_match = tba.match(i)
                    winner = working_match['winning_alliance']
                    if winner == "red":
                        Y_test.append(1)
                    else:
                        Y_test.append(0)
                    Xdata_OPR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    Xdata_ELO = 0
                    red_alliance = working_match['alliances']['red']['team_keys']
                    blue_alliance = working_match['alliances']['blue']['team_keys']
                    # For every team in the match we pull their stats from the dictionary and add it to our data
                    for j in red_alliance:
                        for q in range(7):
                            Xdata_OPR[q] += int(float(dict_team_OPRs[j][q]))
                        Xdata_ELO += dict_team_ELOs[j]
                    for j in blue_alliance:
                        for q in range(7):
                            Xdata_OPR[q + 7] += int(float(dict_team_OPRs[j][q]))
                        Xdata_ELO -= dict_team_ELOs[j]
                    # Collecting all data in one large array allows us to input it into the Keras model
                    X_OPR_test.append(Xdata_OPR[0])
                    X_ELO_test.append(Xdata_ELO)
                    # Updates on progress computing the arrays
            print(len(X_OPR), len(X_ELO))

        X_OPR = np.array(X_OPR)
        X_OPR_test = np.array(X_OPR_test)
        # We once again use the numpy library to save the files so we don't have to recalculate them
        np.save("X_OPR_test_2019.npy", X_OPR_test)
        np.save("X_ELO_test_2019.npy", X_ELO_test)
        np.save("X_OPR_2019.npy", X_OPR)
        np.save("X_ELO_2019.npy", X_ELO)
        np.save("Y_test_2019.npy", Y_test)
        np.save("Y_2019.npy", Y)

    X_OPR_test = np.load("X_OPR_test_2019.npy")
    X_ELO_test = np.load("X_ELO_test_2019.npy")
    X_OPR = np.load("X_OPR_2019.npy")
    X_ELO = np.load("X_ELO_2019.npy")
    Y_test = np.load("Y_test_2019.npy")
    Y = np.load("Y_2019.npy")

    # Setting a consistent random seed ensures our training is consistent
    np.random.seed(42)

    OPR_model = Sequential()
    # Creates the neural net with 3 hidden layers
    OPR_model.add(Dense(13, input_dim=14, activation='relu'))
    OPR_model.add(Dense(14, activation='relu'))
    OPR_model.add(Dense(1, activation='sigmoid'))
    # Compiles the model using loss functions and an optimizer from the Keras Library
    OPR_model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
    # Fits the model over 1000 forward and back propogations
    OPR_model.fit(X_OPR, Y, epochs=1000, batch_size=len(X_OPR), verbose=2)
    #acc = model.evaluate(X, Y)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], acc[1] * 100))

    # serialize model to JSON
    OPR_model_json = OPR_model.to_json()
    with open("OPR_model.json", "w+") as json_file:
        json_file.write(OPR_model_json)
    # serialize weights to HDF5
    OPR_model.save_weights("OPR_model.h5")

    X_OPR_test = np.array(X_OPR_test)
    X_ELO_test = np.array(X_ELO_test)
    sample_probabilities = []
    # this loop goes through all of the matches we want to predict, appends the prediction into an array, sample_probabilities
    for match in range(len(X_OPR_test)):
        # Once last time, we need to reformat the data in order to use predict_proba
        OPR_b = []
        OPR_a = np.array(X_OPR_test[match])
        for stat in OPR_a:
            OPR_b.append([stat])
        OPR_b = np.array(OPR_b).T

        ELO_prob = ELO_probability(X_ELO_test[match])
        OPR_prob = OPR_model.predict_proba(OPR_b)[0][0]
        sample_probabilities.append(round((OPR_prob+ELO_prob)/2, 2))

    # calculates the brier score of my predictions given the real results I pull
    # brier score is the average squared difference of prediction to outcome, outcome being 0 or 1. 0.5 is a coin flip.
    print("The brier score is " + str(round(brier_score_loss(Y_test, sample_probabilities), 4)))
    # To get stats on pure percent correct, we want to turn our prediction into a boolean, and then count the number correct vs total predictions
    overconfident = 0.
    number_correct = 0.
    for match in range(len(sample_probabilities)):
        if sample_probabilities[match] >= 0.95:
            overconfident+=1
        if round(sample_probabilities[match],0) == Y_test[match]:
            number_correct += 1
    print("The % correct with absolute guesses (ie 0% or 100%) is " + str(round(100 * number_correct / len(sample_probabilities), 2)))
    print("The % of predictions over 95% is", round(100 * overconfident / len(sample_probabilities), 2))
    # Back to main menu we go!
    Main_Menu()


def Main_Menu():
    # Sets up the main branch of our CLI, calling all of our major functions that we want the user to access
    purpose = input("What brings you here today? \n A. Alliance Selection \n B. Match Prediction \n C. Component OPRs \n D. Model Initialization \n E. Event Match Predictions \n F. Event Seeding Predictions \n")
    if purpose == "A":
        Alliance_Selection()
    elif purpose == "B":
        event_code = input("Event the match is at (ie 2019mnmi): ")
        match_number = input("Specific match code (ie q1, qf1m1,sf1m2, f1m3")
        print(Match_Probability(event_code+"_"+match_number, False))
    elif purpose == "C":
        Component_OPRs()
    elif purpose == "D":
        Initialize_Model()
    elif purpose == "E":
        event_code = input("Event to predict: ")
        Predict_Event_Matches(event_code)
    elif purpose == "F":
        event_code = input("Event to simulate: ")
        Event_Simulation(event_code)
    else:
        # If things go haywire, we always want to go back to Main Menu
        Main_Menu()

# Starts it all!!!!!
load_model()

Main_Menu()

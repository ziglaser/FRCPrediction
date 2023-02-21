import csv
import numpy as np
import tbapy
from sklearn.linear_model import LogisticRegression
import pandas
import matplotlib.pyplot as plt

tba = tbapy.TBA("x9bbuY4qH2k7jo0YTynCWKMWUGnMLqhI5mBXnfq3OaIB8HPiJPJrbTCSFw6CliKK")

#dict_team_climbs = {}
dict_team_climbs = np.load('climb_data.npy').item()
def clean_match_list(event):
    teams = tba.event_teams(event, keys=True)
    matches = tba.event_matches(event, keys=True)
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

    return matches


def endgame_into_points(state):
    key = {'Hang': 25, 'Park': 5, 'None': 0}
    return key[state]


def write_endgame(event):
    global dict_team_climbs
    teams = tba.event_teams(event, keys=True)
    matches = clean_match_list(event)

    team_climbs = [0] * len(teams)

    for match in matches:
        match_dict = tba.match(match)
        red_alliance, blue_alliance = match_dict['alliances']['red']['team_keys'], match_dict['alliances']['blue'][
            'team_keys']

        for robot in range(len(red_alliance)):
            endgame_state = match_dict['score_breakdown']['red']['endgameRobot' + str(robot + 1)]
            climb_points = endgame_into_points(endgame_state)

            num_climbing = 0
            for state in ['endgameRobot1', 'endgameRobot2', 'endgameRobot3']:
                if match_dict['score_breakdown']['red'][state] == 'Hung':
                    num_climbing += 1
            if match_dict['score_breakdown']['red']['endgameRungIsLevel'] == "IsLevel" and endgame_state == 'Hung':
                climb_points += 15 / num_climbing

            index = teams.index(red_alliance[robot])
            team_climbs[index] += climb_points

        for robot in range(len(blue_alliance)):
            endgame_state = match_dict['score_breakdown']['blue']['endgameRobot' + str(robot + 1)]
            climb_points = endgame_into_points(endgame_state)

            num_climbing = 0
            for state in ['endgameRobot1', 'endgameRobot2', 'endgameRobot3']:
                if match_dict['score_breakdown']['blue'][state] == 'Hung':
                    num_climbing += 1
            if match_dict['score_breakdown']['blue']['endgameRungIsLevel'] == "IsLevel" and endgame_state == 'Hung':
                climb_points += 15 / num_climbing

            index = teams.index(blue_alliance[robot])
            team_climbs[index] += climb_points

        matches_per_team = int(round((len(matches) * 6) / len(teams), 0))

        for index in range(len(teams)):
            dict_team_climbs[teams[index]] = team_climbs[index] / (matches_per_team)

        np.save('climb_data.npy', dict_team_climbs)

'''event_list = []
for event in tba.events("2020", keys=True):
    if tba.event(event)['week'] == 0 or tba.event(event)['week'] == 1:
        event_list.append(event)

X = []
Y = []
for event in event_list:
    print(event)
    try:
        matches = tba.event_matches(event, keys = True)
        start_location = matches.index(event + "_qm1")
        try:
            end_location = matches.index(event + "_sf1m1")
        except:
            end_location = None
        matches = matches[start_location:end_location]

        for match in matches:
            match_dict = tba.match(match)

            red_climb = 0
            blue_climb = 0

            red_alliance = match_dict['alliances']['red']['team_keys']
            blue_alliance = match_dict['alliances']['blue']['team_keys']

            for robot in red_alliance:
                red_climb += dict_team_climbs[robot]
            for robot in blue_alliance:
                blue_climb += dict_team_climbs[robot]

            red_climb_rp = int(match_dict['score_breakdown']['red']['shieldOperationalRankingPoint'])
            blue_climb_rp = int(match_dict['score_breakdown']['blue']['shieldOperationalRankingPoint'])

            X.append(red_climb)
            Y.append(red_climb_rp)
            X.append(blue_climb)
            Y.append(blue_climb_rp)
    except:
        print("done")
        break

X = np.reshape(X, (len(X), 1))
Y = np.reshape(Y, (len(Y), 1))

np.save("X_climb_regression.npy", X)
np.save("Y_climb_regression.npy", Y)'''

X = np.load("X_climb_regression.npy")
Y = np.load("Y_climb_regression.npy")

lr = LogisticRegression()
lr.fit(X, Y)

print(lr.coef_, lr.intercept_)

def predict(climb_points):
    exponent = 0.10643065*climb_points - 5.21505723
    return 1 / (1 + np.exp(-exponent))

brier_total = 0
correct = 0

for i in range(len(X)):
    brier_total += (predict(X[i])[0]-Y[i][0])**2
    if abs(predict(X[i])[0]-Y[i][0]) < 0.5 and :
        correct += 1

print(brier_total/len(X))
print(correct/len(X))
'''
plt.figure(1, figsize=(4, 3))
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
    return 1 / (1 + np.exp(-x))


X_test = np.linspace(-5, 100, 600)
loss = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, loss, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')
plt.axvline(x=X_test[123], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0, 100)
plt.show()'''




import csv
import math
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import DataFrame
from pandas import Series
import matplotlib.pyplot as plt
from cycler import cycler


# define k factor assumptions
def k_factor(matches_played):
    K = 250
    offset = 5
    shape = 0.4
    return K / (matches_played + offset) ** shape


# winning a match regardless the number of sets played = 1
score = 1


# define a function for calculating the expected score of player_A
# expected score of player_B = 1 - expected score of player
def calc_exp_score(playerA_rating, playerB_rating):
    exp_score = 1 / (1 + (10 ** ((playerB_rating - playerA_rating) / 400)))
    return exp_score


# define a function for calculating new elo
def update_elo(old_elo, k, actual_score, expected_score):
    new_elo = old_elo + k * (actual_score - expected_score)
    return new_elo


# read player CSV file and store important columns into lists
with open('atp_players_new.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    col_index = [0, 1, 2, 5]
    all_players = []
    for row in readCSV:
        player_info = []
        for i in col_index:
            player_info.append(row[i])
        all_players.append(player_info)

# Column headers for player dataframe

player_col_header = ['player_id', 'last_name', 'first_name', 'country']
print(player_info)
# Create a dataframe for keeping track of player info
# every player starts with an elo rating of 1500
players = DataFrame(all_players, columns=player_col_header)
players['current_elo'] = Series(1500, index=players.index)
players['last_tourney_date'] = Series('N/A', index=players.index)
players['matches_played'] = Series(0, index=players.index)
players['peak_elo'] = Series(1500, index=players.index)
players['peak_elo_date'] = Series('N/A', index=players.index)

# Convert objects within dataframe to numeric
players = players.infer_objects()
print(players.columns)
# Create an empty dataframe to store time series elo for top 10 players based on peak elo rating
# Use player_id as the column header of the dataframe
# Top ten players consist of: Djokovic, Federer, McEnroe, Nadal, Borg, Lendl, Becker, Murray, Sampras, Connors
# elo_timeseries_col_header = [104925, 103819, 100581, 104745, 100437, 100656, 101414, 104918, 101948, 100284]

elo_timeseries_col_header = players['player_id'].unique()
nameList = players['first_name'].unique()
elo_timeseries = DataFrame(columns=elo_timeseries_col_header)

# read through matches file for each year to update players data frame
# starting from current_year
current_year = 2018

for i in range((2023 - 2018) + 1):
    current_year_file_name = 'atp_matches_' + str(current_year) + '.csv'

    # read match CSV file and store important columns into lists
    with open(current_year_file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        col_index = [0, 5, 7, 15, 25, 26, 27, 28, 29]
        # 0, 5, 7, 17, 27, 28, 29
        all_matches = []
        for row in readCSV:
            match_info = []
            for i in col_index:
                match_info.append(row[i])
            all_matches.append(match_info)

    # separate column names and match info
    header_info = all_matches[0]
    all_matches = all_matches[1:]

    # Create a dataframe to store match info
    matches = DataFrame(all_matches, columns=header_info)

    # Convert objects within matches dataframe to numeric
    matches = matches.infer_objects()
    print(matches)
    # Sort matches dataframe by tourney_date and then by round
    sorter = ['RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']

    # matches.round = matches.round.astype('category')
    # matches['round'] = matches['round'].astype('category')
    matches['round'] = matches['round'].astype('category').cat.set_categories(sorter, ordered=True)

    matches['round'].cat.set_categories(sorter)
    # matches.round.cat.set_categories(sorter, inplace=True)

    matches = matches.sort_values(['tourney_date', 'round'], ascending=[True, True])

    for index, row in matches.iterrows():
        winner_id = row['winner_id']
        if winner_id not in players['player_id'].values:
            continue
        # print(winner_id)
        # print("--------------")
        # print(players['player_id'])
        loser_id = row['loser_id']
        tourney_date = row['tourney_date']

        if loser_id not in players['player_id'].values:
            continue
        index_winner = players[players['player_id'] == winner_id].index.tolist()
        index_loser = players[players['player_id'] == loser_id].index.tolist()

        old_elo_winner = players.loc[index_winner[0], 'current_elo']
        old_elo_loser = players.loc[index_loser[0], 'current_elo']
        exp_score_winner = calc_exp_score(old_elo_winner, old_elo_loser)
        exp_score_loser = 1 - exp_score_winner
        matches_played_winner = players.loc[index_winner[0], 'matches_played']
        matches_played_loser = players.loc[index_loser[0], 'matches_played']
        new_elo_winner = update_elo(old_elo_winner, k_factor(matches_played_winner), score, exp_score_winner)
        new_elo_loser = update_elo(old_elo_loser, k_factor(matches_played_loser), score - 1, exp_score_loser)
        players.loc[index_winner[0], 'current_elo'] = new_elo_winner
        players.loc[index_winner[0], 'last_tourney_date'] = tourney_date
        players.loc[index_winner[0], 'matches_played'] = players.loc[index_winner[0], 'matches_played'] + 1
        players.loc[index_loser[0], 'current_elo'] = new_elo_loser
        players.loc[index_loser[0], 'last_tourney_date'] = tourney_date
        players.loc[index_loser[0], 'matches_played'] = players.loc[index_loser[0], 'matches_played'] + 1
        if new_elo_winner > players.loc[index_winner[0], 'peak_elo']:
            players.loc[index_winner[0], 'peak_elo'] = new_elo_winner
            players.loc[index_winner[0], 'peak_elo_date'] = row['tourney_date']

        # Convert tourney_date to a time stamp, then update elo_timeseries data frame
        tourney_date_timestamp = pandas.to_datetime(tourney_date, format='%Y%m%d')
        if tourney_date_timestamp not in elo_timeseries.index:
            elo_timeseries.loc[tourney_date_timestamp, elo_timeseries_col_header] = np.nan

        if (winner_id in elo_timeseries_col_header) and (loser_id in elo_timeseries_col_header):
            elo_timeseries.loc[tourney_date_timestamp, winner_id] = new_elo_winner
            elo_timeseries.loc[tourney_date_timestamp, loser_id] = new_elo_loser
        elif winner_id in elo_timeseries_col_header:
            elo_timeseries.loc[tourney_date_timestamp, winner_id] = new_elo_winner
        elif loser_id in elo_timeseries_col_header:
            elo_timeseries.loc[tourney_date_timestamp, loser_id] = new_elo_loser

    ##Uncomment to output year end elo_rankings for every year between 1968 and 2015
    # output_file_name = str(current_year) + '_yr_end_elo_ranking.csv'
    # players.to_csv(output_file_name)

    current_year = current_year + 1

print(players)
players.to_csv('2015_yr_end_elo_ranking.csv')
players = pandas.read_csv('2015_yr_end_elo_ranking.csv')
# Print all-time top 10 peak_elo
print(players.sort_values(by='peak_elo', ascending=False).head(20))

# print(elo_timeseries.dtypes)

# print(elo_timeseries['104925'].unique())
# print(elo_timeseries['103819'].value_counts())
# Save elo_timeseries dataframe for plotting purposes
elo_timeseries.to_pickle('elo_timeseries.pkl')

# Open saved pickle file and save into a dataframe
elo_timeseries = pandas.read_pickle('elo_timeseries.pkl')

# Convert objects within elo_timeseries dataframe to numeric
elo_timeseries = elo_timeseries.apply(pandas.to_numeric, errors='coerce')
# print("-----------")
# print(elo_timeseries)

# Use linear interpolation for elo_ratings
elo_timeseries = elo_timeseries.interpolate(method='linear')

# Store the indices in the elo_timeseries in a list
index_timestamp = list(elo_timeseries.index.values)

# Get rid of elo ratings since known last_tourney_date
# print(elo_timeseries_col_header)

for player in elo_timeseries_col_header:
    # print(players['player_id'])
    player_index = players[players['player_id'] == player].index.tolist()

    if player_index:
        player_last_played = players.loc[player_index[0], 'last_tourney_date']
    else:
        continue  # 或其他适当的处理

    if pandas.isna(player_last_played):
        print("nanananaananana")
        print(player)
        print("nanananaananana")
        player_last_played = pandas.to_datetime('19700101', format='%Y%m%d')
        player_last_played_timestamp = np.datetime64(pandas.to_datetime(player_last_played, format='%Y%m%d'))

    else:
        # 如果不是NaN，您可以将其转换为日期时间对象
        player_last_played = pandas.to_datetime(player_last_played, format='%Y%m%d')
        player_last_played_timestamp = np.datetime64(pandas.to_datetime(player_last_played, format='%Y%m%d'))
    # players['player_last_played'] = players['player_last_played'].apply(
    # 	lambda x: pandas.to_datetime(x, format='%Y%m%d', errors='coerce'))
    if player_last_played_timestamp not in index_timestamp:
        # 如果不在列表中，将默认时间添加到列表中
        index_timestamp.append(player_last_played_timestamp)
    elo_ratings_remove = index_timestamp[index_timestamp.index(player_last_played_timestamp) + 1:]

    for i in elo_ratings_remove:
        elo_timeseries.loc[i, player] = np.nan

style.use('stylesheet.mplstyle')

for playerID in elo_timeseries_col_header:
    if playerID == 'player_id': continue
    plt.plot(elo_timeseries.index, elo_timeseries[playerID], label=players.loc[players['player_id'] == playerID, 'first_name'].values[0])
    # player_name = players.loc[players['player_id'] == playerID, 'first_name'].values[0]
    # print(playerID)
    # x_max = elo_timeseries.index[0]  # 最后一个数据点的 x 坐标
    # y_max = elo_timeseries[playerID].iloc[2]  # 最后一个数据点的 y 坐标
    # print(x_max, y_max)
    # plt.text(x_max, y_max, player_name, fontsize=12, verticalalignment='center',
    #          bbox=dict(facecolor='white', alpha=0.7))

# plt.plot(elo_timeseries.index, elo_timeseries[104925]) #Djokovic
# plt.plot(elo_timeseries.index, elo_timeseries[103819]) #Federer
# plt.plot(elo_timeseries.index, elo_timeseries[100581]) #McEnroe
# plt.plot(elo_timeseries.index, elo_timeseries[104745]) #Nadal
# plt.plot(elo_timeseries.index, elo_timeseries[100437]) #Borg
# plt.plot(elo_timeseries.index, elo_timeseries[100656]) #Lendl
# plt.plot(elo_timeseries.index, elo_timeseries[101414]) #Becker
# plt.plot(elo_timeseries.index, elo_timeseries[104918]) #Murray
# plt.plot(elo_timeseries.index, elo_timeseries[101948]) #Sampras
# plt.plot(elo_timeseries.index, elo_timeseries[100284]) #Connors
# plt.legend(loc='upper right', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(0, 0.7), fontsize=10)
plt.title("Historical elo ratings for top 10 ATP players", fontsize=25, y=1.1, weight='bold')
plt.xlabel("Years starting in the Open-Era", labelpad=25)
plt.ylabel("Elo rating", labelpad=32)
plt.axhline(1200, color='grey', linewidth=5)
plt.show()

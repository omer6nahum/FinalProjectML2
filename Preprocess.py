import numpy as np
import pandas as pd
import pickle


def create_squad(team_in_season_df, k=None, prefix_path=''):
    """
    :param team_in_season_df: dataframe contains team's data in a season
    :param k: a dictionary with k per position_category. e.g.: {'gk': 3}
    :param prefix_path: prefix path to data directory
    :return: squad representation
    """
    if k is None:
        k = {'gk': 2,
             'def': 6,
             'mid': 5,
             'att': 5}

    features_pickle_path = prefix_path + 'pickles/data/features_dict.pkl'
    with open(features_pickle_path, 'rb') as f:
        features_dict = pickle.load(f)

    data = team_in_season_df.drop(['club_name', 'sofifa_id', 'best_position', 'short_name'], axis=1)

    gk_data, _, __ = create_position_category_data(data, 'gk', k['gk'], features_dict['field_skills'])
    def_data, def_rest, __ = create_position_category_data(data, 'def', k['def'], features_dict['gk_skills'])
    mid_data, _, num_returned = create_position_category_data(data, 'mid', k['mid'], features_dict['gk_skills'])
    att_data, att_rest, __ = create_position_category_data(data, 'att', k['att'], features_dict['gk_skills'])

    if num_returned < k['mid']:
        rest_data = pd.concat((def_rest, att_rest)).sort_values(by='overall', ascending=False)
        rest_data = np.array(rest_data.head(k['mid'] - num_returned).values.flatten())
        mid_data = np.hstack((mid_data, rest_data))

    return np.concatenate((gk_data, def_data, mid_data, att_data))


def create_position_category_data(data, position_category, k, features_to_drop):
    res = data[data['position_category'] == position_category] \
         .drop(['position_category'] + features_to_drop, axis=1) \
         .sort_values(by='overall', ascending=False)

    return np.array(res.head(k).values.flatten()), res.tail(res.shape[0] - k), min(res.shape[0], k)


def create_x_y_z_approach1(year, prefix_path=''):
    table_path = prefix_path + f'data/tables/table_{year}.csv'
    table = pd.read_csv(table_path)

    x = []  # x_i is a squad representation
    y = []  # y_i is number of points
    z = []  # z_i is team name

    path = prefix_path + f'data/players/players_{year}.csv'
    data = pd.read_csv(path)

    for i, row in table.iterrows():
        x.append(create_squad(data[data['club_name'] == row['team_name']], prefix_path=prefix_path))  # x_i is a team's squad
        y.append(float(row['PTS']))  # y_i is number of points at the end of the season
        z.append(row['team_name'])
    return np.vstack(x), np.squeeze(np.vstack(y)), np.vstack(z)


def create_x_y_z_approach2(year, prefix_path=''):
    all_matches_path = prefix_path + f'data/matches/matches_{year}.csv'
    all_matches = pd.read_csv(all_matches_path)

    x = []  # x_i is 2-squads representation
    y = []  # y_i is label from {'H', 'D', 'A'}
    z = []  # z_i is 2d vector: [home_team_name, away_team_name]

    path = prefix_path + f'data/players/players_{year}.csv'
    data = pd.read_csv(path)

    for i, row in all_matches.iterrows():
        home_squad = create_squad(data[data['club_name'] == row['HomeTeam']], prefix_path=prefix_path)
        away_squad = create_squad(data[data['club_name'] == row['AwayTeam']], prefix_path=prefix_path)
        squads = np.concatenate((home_squad, away_squad))
        x.append(squads)
        y.append(row['FTR'])
        z.append(np.array([row['HomeTeam'], row['AwayTeam']]))

    return np.vstack(x), np.squeeze(np.vstack(y)), np.vstack(z)


def create_sequence_matrix(row, home):
    """
    Transform the required team sequence from row into a matrix of shape (5, 3).
    Each Row represents a match result (Win/Draw/Lose). Overall 5 last matches.
    The oreder of the rows in this matrix is:
    matrix[-1] is the last match, matrix[-2] is the second last match, and so on.
    :param row: one row from csv, representing a single match
    :param home: True if home_team is required, False if away_team is.
    :return: The transformed matrix
    """
    if home:
        team = 'HomeTeam'
    else:
        team = 'AwayTeam'
    columns = [f'{team}_p{i}{result}' for i in range(5, 0, -1) for result in ['Win', 'Draw', 'Lose']]

    return row[columns].values.astype(np.float).reshape((5, 3))


def create_x_y_z_approach2_advanced(year, prefix_path=''):
    all_matches_path = prefix_path + f'data/matches/adv_matches_{year}.csv'
    all_matches = pd.read_csv(all_matches_path)
    all_matches = all_matches.sort_values(by='Date', ascending=True)

    x = []  # x_i is a *list*:
            # x_i[0] is 2-squads representation
            # x_i[1] is a (5, 3) shaped array of 5 last matches of home team
            # x_i[2] is a (5, 3) shaped array of 5 last matches of away team
    y = []  # y_i is label from {'H', 'D', 'A'}
    z = []  # z_i is 2d vector: [home_team_name, away_team_name]

    path = prefix_path + f'data/players/players_{year}.csv'
    data = pd.read_csv(path)

    for i, row in all_matches.iterrows():
        home_squad = create_squad(data[data['club_name'] == row['HomeTeam']], prefix_path=prefix_path)
        away_squad = create_squad(data[data['club_name'] == row['AwayTeam']], prefix_path=prefix_path)
        squads = np.concatenate((home_squad, away_squad))
        home_sequence = create_sequence_matrix(row, home=True)
        away_sequence = create_sequence_matrix(row, home=False)
        x.append([])
        x[-1].append(squads)  # x_i[0]
        x[-1].append(home_sequence)  # x_i[0]
        x[-1].append(away_sequence)  # x_i[0]
        y.append(row['FTR'])
        z.append(np.array([row['HomeTeam'], row['AwayTeam']]))

    return x, np.squeeze(np.vstack(y)), np.vstack(z)


def create_x_y_z(year, approach, part='basic', prefix_path=''):
    """
    :param year: 2 digits
    :param approach: can be one of {1, 2}
    :param part: can be one of {'basic', 'advanced'}
    :param prefix_path: prefix path to data directory
    :return:
    """
    if part == 'basic':
        if approach == 1:
            return create_x_y_z_approach1(year, prefix_path)
        elif approach == 2:
            return create_x_y_z_approach2(year, prefix_path)
    elif part == 'advanced':
        return create_x_y_z_approach2_advanced(year, prefix_path)
    else:
        raise ValueError('Not an approach')


def load_train_test(test_year, approach, part='basic', prefix_path='', dir_path=None):
    if dir_path is None:
        dir_path = 'pickles/data/'

    years = range(15, 22)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    z_train = []
    z_test = []
    for year in years:
        x, y, z = load_x_y_z_pickle(year, approach, part, prefix_path + dir_path)
        if year == test_year:
            x_test.append(x)
            y_test.append(y)
            z_test.append(z)
        else:
            x_train.append(x)
            y_train.append(y)
            z_train.append(z)
    if part == 'basic':
        x_train = np.vstack(x_train)
        x_test = np.vstack(x_test)
    else:
        x_train_res = []
        for item in x_train:
            x_train_res += item
        x_train = x_train_res
        x_test_res = []
        for item in x_test:
            x_test_res += item
        x_test = x_test_res
    y_train = np.hstack(y_train)
    z_train = np.vstack(z_train)
    y_test = np.hstack(y_test)
    z_test = np.vstack(z_test)

    return x_train, x_test, y_train, y_test, z_train, z_test


def dump_x_y_z_pickle(year, approach, part='basic', dir_path=None):
    if dir_path is None:
        dir_path = 'pickles/data/'
    path = dir_path + f'year_{year}_approach_{approach}_{part}.pkl'
    data = create_x_y_z(year, approach, part)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_x_y_z_pickle(year, approach, part='basic', dir_path=None):
    if dir_path is None:
        dir_path = 'pickles/data/'
    path = dir_path + f'year_{year}_approach_{approach}_{part}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

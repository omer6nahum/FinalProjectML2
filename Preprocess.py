import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def create_squad(team_name, year, k=None):
    """
    :param team_name:
    :param year: season year (2 figures, e.g: 15)
    :param k: a dictionary with k per position_category. e.g.: {'gk': 3}
    :return: squad representation
    """

    if k is None:
        # k = {'gk': 2,
        #      'def': 6,
        #      'mid': 4,
        #      'att': 6}
        k = {'gk': 1,
             'def': 2,
             'mid': 2,
             'att': 2}

    features_pickle_path = 'data/pickles/features_dict.pkl'
    with open(features_pickle_path, 'rb') as f:
        features_dict = pickle.load(f)

    path = f'data/players/players_{year}.csv'
    data = pd.read_csv(path)

    data = data[data['club_name'] == team_name].drop(['club_name', 'sofifa_id', 'best_position', 'short_name'], axis=1)

    gk_data = create_position_category_data(data, 'gk', k['gk'], features_dict['field_skills'])
    def_data = create_position_category_data(data, 'def', k['def'], features_dict['gk_skills'])
    mid_data = create_position_category_data(data, 'mid', k['mid'], features_dict['gk_skills'])
    att_data = create_position_category_data(data, 'att', k['att'], features_dict['gk_skills'])

    # todo: complete from another category if head is not enough players

    return np.concatenate((gk_data, def_data, mid_data, att_data))


def create_position_category_data(data, position_category, k, features_to_drop):
    return np.array(data[data['position_category'] == position_category] \
        .drop(['position_category'] + features_to_drop, axis=1) \
        .sort_values(by='overall', ascending=False) \
        .head(k).values.flatten())


def create_x_y_z_approach1(year):
    table_path = f'data/tables/table_{year}.csv'
    table = pd.read_csv(table_path)

    x = []  # x_i is a squad representation
    y = []  # y_i is number of points
    z = []  # z_i is team name

    for i, row in table.iterrows():
        x_i = create_squad(row['team_name'], year)  # x_i is a team's squad
        if len(x_i) == 0:
            print(row['team_name'])
        x.append(x_i)
        y.append(float(row['PTS']))  # y_i is number of points at the end of the season
        z.append(row['team_name'])
    return np.vstack(x), np.vstack(y), np.vstack(z)


def create_x_y_z_approach2(year):
    all_matches_path = f'data/matches/matches_{year}.csv'
    all_matches = pd.read_csv(all_matches_path)

    x = []  # x_i is 2-squads representation
    y = []  # y_i is label from {'H', 'D', 'A'}
    z = []  # z_i is 2d vector: [home_team_name, away_team_name]

    for i, row in all_matches.iterrows():
        home_squad = create_squad(row['HomeTeam'], year)
        away_squad = create_squad(row['AwayTeam'], year)
        squads = np.concatenate((home_squad, away_squad))
        x.append(squads)
        y.append(row['FTR'])
        z.append(np.array([row['HomeTeam'], row['AwayTeam']]))

    return np.vstack(x), np.vstack(y), np.vstack(z)


def create_x_y_z(year, approach):
    """
    :param year: 2 digits
    :param approach: can be one of {1, 2}
    :return:
    """

    if approach == 1:
        return create_x_y_z_approach1(year)
    elif approach == 2:
        return create_x_y_z_approach2(year)
    else:
        raise ValueError('Not an approach')


def create_train_test(test_year, approach):
    years = range(15, 22)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    z_train = []
    z_test = []
    for year in tqdm(years):
        x, y, z = create_x_y_z(year, approach)
        if year == test_year:
            x_test.append(x)
            y_test.append(y)
            z_test.append(z)
        else:
            x_train.append(x)
            y_train.append(y)
            z_train.append(z)
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    z_train = np.vstack(z_train)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)
    z_test = np.vstack(z_test)

    return x_train, x_test, y_train, y_test, z_train, z_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test(test_year=21, approach=1)
    print(f'x={x_train.shape}, y={y_train.shape}, z={z_train.shape}')
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test(test_year=21, approach=2)
    print(f'x={x_train.shape}, y={y_train.shape}, z={z_train.shape}')


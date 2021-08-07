import numpy as np
import pandas as pd
import pickle


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
    # todo: document x,y,z
    x = []
    y = []
    z = []
    for i, row in table.iterrows():
        x_i = create_squad(row['team_name'], year)  # x_i is a team's squad
        if len(x_i) == 0:
            print(row['team_name'])
        x.append(x_i)
        y.append(float(row['PTS']))  # y_i is number of points at the end of the season
        z.append(row['team_name'])
    return np.vstack(x), np.vstack(y), np.vstack(z)


def create_train_test_approach1(test_year):
    years = range(15, 22)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    z_train = []
    z_test = []
    for year in years:
        x, y, z = create_x_y_z_approach1(year)
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


def create_x_y_z_approach2(year):
    # todo: finish
    all_matches_path = f'data/matches/matches_{year}.csv'
    all_matches = pd.read_csv(all_matches_path)
    teams = list(set(all_matches['HomeTeam']))


def create_train_test_approach2(test_years):
    # todo: finish
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test_approach1(21)



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
        k = {'gk': 2,
             'def': 6,
             'mid': 4,
             'att': 6}

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

    return np.concatenate((gk_data, def_data, mid_data, att_data))


def create_position_category_data(data, position_category, k, features_to_drop):
    return data[data['position_category'] == position_category] \
        .drop(['position_category'] + features_to_drop, axis=1) \
        .sort_values(by='overall', ascending=False) \
        .head(k).values.flatten()


if __name__ == '__main__':
    print(create_squad('Liverpool', 21).shape)

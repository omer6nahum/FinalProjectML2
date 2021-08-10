import numpy as np
import pandas as pd
from Preprocess import load_train_test
from models.LogReg import LogReg
from models.LinReg import LinReg
from FirstApproach import FirstApproach
from SecondApproach import SecondApproach

# TODO: implement evaluation ranking metrics
TEAM_NAME, PTS, RANK, ADJ_RANK = 'team_name', 'PTS', 'rank', 'adj_rank'


def add_adjusted_ranks(table1, table2):
    # TODO: add documentation.
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    table1_adj = table1.copy()
    table2_adj = table2.copy()
    adj_ranks = [1] + \
                [2, 2.25, 2.5] + \
                [3, 3.25, 3.5] + \
                list(np.arange(4, 5.555, 1/6)) + \
                [6, 6.25, 6.5]
    table1_adj[ADJ_RANK] = adj_ranks
    table2_adj[ADJ_RANK] = adj_ranks
    return table1_adj, table2_adj


def add_regular_ranks(table1, table2):
    # TODO: add documentation.
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    table1_adj = table1.copy()
    table2_adj = table2.copy()
    table1_adj[RANK] = range(1, 21)
    table2_adj[RANK] = range(1, 21)
    return table1_adj, table2_adj


def weighted_hamming(predicted_table, ground_truth):
    # TODO: add documentation.
    assert set(predicted_table[TEAM_NAME]) == set(ground_truth[TEAM_NAME])
    teams = set(predicted_table[TEAM_NAME])
    dist = 0
    for team in teams:
        predicted_idx = pd.Index(predicted_table[TEAM_NAME]).get_loc(team)
        ground_truth_index = pd.Index(ground_truth[TEAM_NAME]).get_loc(team)
        curr_dist = abs(predicted_idx - ground_truth_index)
        normalized_dist = curr_dist / max(ground_truth_index, 20 - ground_truth_index)
        dist += normalized_dist
    return dist


def hamming(table1, table2):
    # TODO: add documentation.
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    teams = set(table1[TEAM_NAME])
    dist = 0
    for team in teams:
        dist += abs(pd.Index(table1[TEAM_NAME]).get_loc(team) - pd.Index(table2[TEAM_NAME]).get_loc(team))
    return dist


if __name__ == '__main__':
    gd_table = pd.read_csv('data/tables/table_21.csv')[['team_name', 'PTS']]
    print(gd_table)
    print()

    x_train_1, x_test_1, y_train_1, \
    y_test_1, z_train_1, z_test_1 = load_train_test(test_year=21,
                                                    approach=1,
                                                    prefix_path='')
    lin_reg = LinReg()
    lin_reg.load_params('pickles/models/lin_reg_ver1.pkl')
    tbl1 = FirstApproach(lin_reg).predict_table(x_test_1, z_test_1)
    print(tbl1)
    print()

    x_train_2, x_test_2, y_train_2, \
    y_test_2, z_train_2, z_test_2 = load_train_test(test_year=21,
                                                    approach=2,
                                                    prefix_path='')
    log_reg = LogReg()
    log_reg.load_params('pickles/models/log_reg_ver1.pkl')
    tbl2 = SecondApproach(log_reg).predict_table(x_test_2, z_test_2, ranking_method='expectation')
    print(tbl2)
    print()

    # Test hamming eval functions:
    print(f'hamming(ground truth vs. lin_reg): {hamming(tbl1, gd_table)}')
    print(f'hamming(ground truth vs. log_reg): {hamming(tbl2, gd_table)}')
    print(f'weighted_hamming(ground truth vs. lin_reg): {weighted_hamming(tbl1, gd_table)}')
    print(f'weighted_hamming(ground truth vs. log_reg): {weighted_hamming(tbl2, gd_table)}')



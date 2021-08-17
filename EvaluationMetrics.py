import numpy as np
import pandas as pd
from Preprocess import load_train_test
from models.LogReg import LogReg
from models.LinReg import LinReg
from FirstApproach import FirstApproach
from SecondApproach import SecondApproach

# TODO: implement evaluation ranking metrics
TEAM_NAME, PTS, RANK, ADJ_RANK = 'team_name', 'PTS', 'rank', 'adj_rank'

def diff_in_val(table1, table2, team, param):
    return table1[table1[TEAM_NAME] == team][param].iloc[0] - table2[table2[TEAM_NAME] == team][param].iloc[0]

def mul_in_val(table1, table2, team, param, mean_normalization=0):
    return (table1[table1[TEAM_NAME] == team][param].iloc[0] - mean_normalization) *\
           (table2[table2[TEAM_NAME] == team][param].iloc[0] - mean_normalization)

def add_adjusted_ranks(table1, table2, return_rank=False):
    # TODO: add documentation.
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    table1_adj = table1.copy()
    table2_adj = table2.copy()
    adj_ranks = [1] + \
                [2, 2.25, 2.5] + \
                [3, 3.25, 3.5] + \
                list(np.arange(4, 5.555, 1/6)) + \
                [6, 6.25, 6.5]
    adj_ranks = [(x-1)/(6.5-1)*19+1 for x in adj_ranks]
    table1_adj[ADJ_RANK] = adj_ranks
    table2_adj[ADJ_RANK] = adj_ranks
    if return_rank:
        return table1_adj, table2_adj, adj_ranks
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

def hamming_normalized(table1, table2):
    # TODO: add documentation.
    return (10*20-hamming(table1, table2)) / (10*20)

def adj_hamming(table1, table2, return_rank=False):
    adj_table1, adj_table2, rank = add_adjusted_ranks(table1, table2, return_rank=True)
    dist = 0
    for team in set(table1[TEAM_NAME]):
        dist += np.abs(diff_in_val(adj_table1, adj_table2, team, ADJ_RANK))
    if return_rank:
        return dist, rank
    return dist

def adj_hamming_normalized(table1, table2):
    # TODO: add documentation.
    hamming, rank = adj_hamming(table1, table2, return_rank=True)
    return (sum(rank)-hamming) / sum(rank)


def adj_MAP(table, ground_truth_table):
    assert set(table[TEAM_NAME]) == set(ground_truth_table[TEAM_NAME])
    map = 0
    for i in range(1,len(table[TEAM_NAME])+1):
        p_at_i = len(set(table[TEAM_NAME][:i]).intersection(ground_truth_table[TEAM_NAME][:i]))/i
        map += p_at_i/20
    return map

def adj_MAP_normalized(table, ground_truth_table):
    assert set(table[TEAM_NAME]) == set(ground_truth_table[TEAM_NAME])
    map_norm = 0
    for i in range(1,20):
        min_at_i = max(0,i-10)*2
        p_at_i = (len(set(table[TEAM_NAME][:i]).intersection(ground_truth_table[TEAM_NAME][:i]))-min_at_i)/(i-min_at_i)
        map_norm += p_at_i/19
    return map_norm

def spearman(table1, table2):
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    adj_table1, adj_table2, rank = add_adjusted_ranks(table1, table2, return_rank=True)
    sum_Di = 0
    mean_rank = sum(rank)/len(rank)
    for team in set(table1[TEAM_NAME]):
        sum_Di += mul_in_val(adj_table1, adj_table2, team, ADJ_RANK, mean_normalization=mean_rank)
    squre_rank = sum([(x-mean_rank)**2 for x in rank])
    return sum_Di/squre_rank

def points_error(table1, table2):
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    error = 0
    for team in set(table1[TEAM_NAME]):
         error += np.abs(diff_in_val(table1, table2, team, PTS))
    return error

def points_regression(table1, table2):
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    SR = 0
    for team in set(table1[TEAM_NAME]):
         SR += diff_in_val(table1, table2, team, PTS) ** 2
    return np.sqrt(SR)


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
    all_metrics = [hamming,
                   hamming_normalized,
                   adj_hamming,
                   adj_hamming_normalized,
                   weighted_hamming,
                   adj_MAP,
                   adj_MAP_normalized,
                   spearman,
                   #points_regression,
                   points_error,
                   ]
    all_tables = {"lin_reg": tbl1, "log_reg": tbl2, "gt": gd_table}
    metrics = pd.DataFrame({m.__name__: [m(t, gd_table) for t in all_tables.values()] for m in all_metrics},
                           index=all_tables.keys())
    print(metrics.T)
    # print(f'hamming(ground truth vs. lin_reg): {hamming(tbl1, gd_table)}')
    # print(f'hamming(ground truth vs. log_reg): {hamming(tbl2, gd_table)}')
    # print(f'hamming_normalized(ground truth vs. lin_reg): {hamming_normalized(tbl1, gd_table)}')
    # print(f'hamming_normalized(ground truth vs. log_reg): {hamming_normalized(tbl2, gd_table)}')
    # print(f'weighted_hamming(ground truth vs. lin_reg): {weighted_hamming(tbl1, gd_table)}')
    # print(f'weighted_hamming(ground truth vs. log_reg): {weighted_hamming(tbl2, gd_table)}')



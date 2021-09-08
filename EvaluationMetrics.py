import numpy as np
import pandas as pd

TEAM_NAME, PTS, RANK, ADJ_RANK = 'team_name', 'PTS', 'rank', 'adj_rank'


def diff_in_val(table1, table2, team, param):
    return table1[table1[TEAM_NAME] == team][param].iloc[0] - table2[table2[TEAM_NAME] == team][param].iloc[0]


def mul_in_val(table1, table2, team, param, mean_normalization=0):
    return (table1[table1[TEAM_NAME] == team][param].iloc[0] - mean_normalization) *\
           (table2[table2[TEAM_NAME] == team][param].iloc[0] - mean_normalization)


def add_adjusted_ranks(table1, table2, return_rank=False):
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
    assert set(table1[TEAM_NAME]) == set(table2[TEAM_NAME])
    table1_adj = table1.copy()
    table2_adj = table2.copy()
    table1_adj[RANK] = range(1, 21)
    table2_adj[RANK] = range(1, 21)
    return table1_adj, table2_adj


def weighted_hamming(predicted_table, ground_truth):
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


def adj_hamming(table1, table2, return_rank=False):
    adj_table1, adj_table2, rank = add_adjusted_ranks(table1, table2, return_rank=True)
    dist = 0
    for team in set(table1[TEAM_NAME]):
        dist += np.abs(diff_in_val(adj_table1, adj_table2, team, ADJ_RANK))
    if return_rank:
        return dist, rank
    return dist


def adj_hamming_normalized(table1, table2):
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
    return error/20


def correct_champion(table1, table2):
    return int(table1.iloc[0][TEAM_NAME] == table2.iloc[0][TEAM_NAME])
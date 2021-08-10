import pandas as pd
from Preprocess import load_train_test
from models.LogReg import LogReg
from models.LinReg import LinReg
from FirstApproach import FirstApproach
from SecondApproach import SecondApproach

# TODO: implement evaluation ranking metrics
TEAM_NAME, PTS = 'team_name', 'PTS'


def hamming(tabl1, tabl2):
    # TODO: add documentation.
    assert set(tabl1[TEAM_NAME]) == set(tabl2[TEAM_NAME])
    teams = set(tabl1[TEAM_NAME])
    dist = 0
    for team in teams:
        dist += abs(pd.Index(tabl1[TEAM_NAME]).get_loc(team) - pd.Index(tabl2[TEAM_NAME]).get_loc(team))
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

    # Test hamming eval function:
    print(f'hamming(ground truth vs. lin_reg): {hamming(gd_table, tbl1)}')
    print(f'hamming(ground truth vs. log_reg): {hamming(gd_table, tbl2)}')

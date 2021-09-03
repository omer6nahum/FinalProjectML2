from FirstApproach import FirstApproach
from SecondApproach import SecondApproach
from models.LogReg import LogReg
from models.LinReg import LinReg
from models.BasicNN import BasicNN
from models.AdvancedNN import AdvancedNN
from models.OrdLogReg import OrdLogReg
from models.OrdBasicNN import OrdBasicNN
from EvaluationMetrics import adj_hamming_normalized, adj_MAP_normalized, spearman, points_error
from Preprocess import load_train_test, load_x_y_z_pickle
import pandas as pd
import numpy as np
import pickle


def second_approach_cv(model, metrics, test_years):
    """
    # todo:
    :param model:
    :param metrics:
    :return:
    """
    assert type(model) in [LogReg, BasicNN, OrdLogReg, OrdBasicNN]
    assert not model.is_fitted

    mid_res = {'expectation': [], 'simulation': []}
    for test_year in test_years:
        for ranking_method in mid_res.keys():
            mid_res[ranking_method].append([])
        x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=test_year,
                                                                            approach=2,
                                                                            prefix_path='')
        model.fit(x_train, y_train)
        second_app = SecondApproach(model)

        true_table = pd.read_csv(f'data/tables/table_{test_year}.csv')[['team_name', 'PTS']]
        for ranking_method in mid_res.keys():
            pred_table, (acc, adj_acc) = second_app.predict_table(x_test, z_test, ranking_method, y_test)
            for metric_name, metric in metrics:
                mid_res[ranking_method][-1].append(metric(pred_table, true_table))
            mid_res[ranking_method][-1].append(acc)
            mid_res[ranking_method][-1].append(adj_acc)

    res = {}
    metric_names = [metric_name for metric_name, _ in metrics] + ['accuracy', 'adj_accuracy']
    for ranking_method in mid_res.keys():
        mean_cv_values = np.array(mid_res[ranking_method]).mean(axis=0)
        res[ranking_method] = {metric_name: val for val, metric_name in zip(mean_cv_values, metric_names)}
    return res


def second_approach_cv_advanced(model, metrics, test_years):
    """
    # todo:
    :param model:
    :param metrics:
    :return:
    """
    assert type(model) == AdvancedNN
    assert not model.is_fitted
    np.random.seed(5)
    mid_res = []
    for test_year in test_years:
        mid_res.append([])
        x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=test_year,
                                                                            approach=2,
                                                                            part='advanced',
                                                                            prefix_path='')
        idxs = np.random.randint(0, y_train.shape[0], y_train.shape[0] // 5)
        y_train = y_train[idxs]
        z_train = z_train[idxs]
        x_train = [x_train[i] for i in idxs]

        model.fit(x_train, y_train)
        second_app = SecondApproach(model)

        true_table = pd.read_csv(f'data/tables/table_{test_year}.csv')[['team_name', 'PTS']]
        pred_table, (acc, adj_acc) = second_app.predict_table(x_test, z_test, 'advanced_simulation', y_test)
        for metric_name, metric in metrics:
            mid_res[-1].append(metric(pred_table, true_table))
        mid_res[-1].append(acc)
        mid_res[-1].append(adj_acc)

    res = {}
    metric_names = [metric_name for metric_name, _ in metrics] + ['accuracy', 'adj_accuracy']
    mean_cv_values = np.array(mid_res).mean(axis=0)
    res = {metric_name: val for val, metric_name in zip(mean_cv_values, metric_names)}
    return res


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    test_years = range(15, 22)
    metrics = [('adj_hamming', adj_hamming_normalized), ('adj_MAP', adj_MAP_normalized),
               ('Spearman', spearman), ('L1_distance', points_error)]

    # Basic Part
    # -- First Approach
    first_app_results = []  # list of metrics names
    for test_year in test_years:
        first_app_results.append([])
        x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=test_year,
                                                                            approach=1,
                                                                            prefix_path='')
        lin_reg = LinReg()
        lin_reg.fit(x_train, y_train)
        first_app = FirstApproach(model=lin_reg)
        pred_table = first_app.predict_table(x_test, z_test)
        true_table = pd.read_csv(f'data/tables/table_{test_year}.csv')[['team_name', 'PTS']]
        for metric_name, metric in metrics:
            first_app_results[-1].append(metric(pred_table, true_table))

    mean_cv_values = np.array(first_app_results).mean(axis=0)
    first_app_results = {metric_name: val for val, (metric_name, _) in zip(mean_cv_values, metrics)}
    final_df = pd.DataFrame(first_app_results, index=['LinReg'])

    # -- Second approach
    input_shape = load_x_y_z_pickle(15, 2)[0].shape[1]
    log_reg = LogReg()
    ord_log_reg = OrdLogReg()
    # todo: num_epochs should be 50
    basic_nn = BasicNN(input_shape=input_shape, lr=1e-3, num_epochs=5, batch_size=128, num_units=4150)
    second_app_models = [('LogReg', log_reg), ('OrdLogReg', ord_log_reg)]
    # second_app_models = [('LogReg', log_reg), ('BasicNN', basic_nn), ('OrdLogReg', ord_log_reg)]

    for model_name, model in second_app_models:
        second_app_model_result = second_approach_cv(model, metrics, test_years)
        for ranking_method, results in sorted(second_app_model_result.items()):
            row_model_name = f'{model_name}_{ranking_method}'
            row_values = results
            final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    # Advanced Part
    # advanced_nn = AdvancedNN(input_shape=input_shape, hidden_lstm_dim=20, hidden_first_fc_dim=input_shape//2, num_epochs=5,
    #                          batch_size=128, lr=1e-3, optimizer=None, num_units=None)
    # advanced_model_result = second_approach_cv_advanced(advanced_nn, metrics, test_years)
    # row_model_name = 'AdvancedNN'
    # row_values = advanced_model_result
    # final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    print(final_df)
    # with open('final_df_advanced.pkl', 'wb') as f:
    #     pickle.dump(final_df, f)

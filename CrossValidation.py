from FirstApproach import FirstApproach
from SecondApproach import SecondApproach
from models.LogReg import LogReg
from models.BasicNN import BasicNN
from models.AdvancedNN import AdvancedNN
from models.OrdLogReg import OrdLogReg
from models.OrdBasicNN import OrdBasicNN
from Preprocess import load_train_test
import pandas as pd
import numpy as np


def first_approach_cv(model, metrics, test_years):
    """
    Run leave-one-out cross validation over the test years and evaluate
    :param model: model for predicting teams points at the end of the season(regression)
    :param metrics: metrics to evaluate the predicted table
    :return: a dictionary -     <metric_name>: <avg_metric_result>
    """
    first_app_results = []  # list of metrics names
    for test_year in test_years:
        first_app_results.append([])
        x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=test_year,
                                                                            approach=1,
                                                                            prefix_path='')
        model.fit(x_train, y_train)
        first_app = FirstApproach(model=model)
        pred_table = first_app.predict_table(x_test, z_test)
        true_table = pd.read_csv(f'data/tables/table_{test_year}.csv')[['team_name', 'PTS']]
        for metric_name, metric in metrics:
            first_app_results[-1].append(metric(pred_table, true_table))

    mean_cv_values = np.array(first_app_results).mean(axis=0)
    std_cv_values = np.array(first_app_results).std(axis=0)
    return {metric_name: f'{mean:.3f} +- {std:.3f}' for mean, std, (metric_name, _) in
            zip(mean_cv_values, std_cv_values, metrics)}


def second_approach_cv(model, metrics, test_years):
    """
    Run leave-one-out cross validation over the test years and evaluate
    :param model: model for predicting matches outcome probabilities
    :param metrics: metrics to evaluate the predicted table
    :return: a dictionary of dictionaries -
             outer dictionary - <ranking_method>: <inner_dictionary>
             inner dictionary - <metric_name>: <avg_metric_result>
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
        std_cv_values = np.array(mid_res[ranking_method]).std(axis=0)
        res[ranking_method] = {metric_name: f'{mean:.3f} +- {std:.3f}' for mean, std, (metric_name, _) in
                               zip(mean_cv_values, std_cv_values, metrics)}
    return res


def second_approach_cv_advanced(model, metrics, test_years):
    """
    Run leave-one-out cross validation over the test years and evaluate
    :param model: model for predicting matches outcome probabilities
    :param metrics: metrics to evaluate the predicted table
    :return: a dictionary -     <metric_name>: <avg_metric_result>
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
        model.fit(x_train, y_train)
        second_app = SecondApproach(model)

        true_table = pd.read_csv(f'data/tables/table_{test_year}.csv')[['team_name', 'PTS']]
        pred_table, (acc, adj_acc) = second_app.predict_table(x_test, z_test, 'advanced_simulation', y_test)
        for metric_name, metric in metrics:
            mid_res[-1].append(metric(pred_table, true_table))
        mid_res[-1].append(acc)
        mid_res[-1].append(adj_acc)

    metric_names = [metric_name for metric_name, _ in metrics] + ['accuracy', 'adj_accuracy']
    mean_cv_values = np.array(mid_res).mean(axis=0)
    std_cv_values = np.array(mid_res).std(axis=0)
    return {metric_name: f'{mean:.3f} +- {std:.3f}' for mean, std, (metric_name, _) in
            zip(mean_cv_values, std_cv_values, metrics)}
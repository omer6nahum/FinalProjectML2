from models.LogReg import LogReg
from models.LinReg import LinReg
from models.BasicNN import BasicNN
from models.AdvancedNN import AdvancedNN
from models.OrdLogReg import OrdLogReg
from models.OrdBasicNN import OrdBasicNN
from EvaluationMetrics import adj_hamming_normalized, adj_MAP_normalized, spearman, points_error
from Preprocess import load_x_y_z_pickle
import pandas as pd
from CrossValidation import first_approach_cv, second_approach_cv, second_approach_cv_advanced


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    test_years = range(15, 22)
    metrics = [('adj_hamming', adj_hamming_normalized), ('adj_MAP', adj_MAP_normalized),
               ('Spearman', spearman), ('L1_distance', points_error)]

    # Basic Part
    # -- First Approach
    lin_reg = LinReg()
    first_app_results = first_approach_cv(lin_reg, metrics, test_years)
    final_df = pd.DataFrame(first_app_results, index=['LinReg'])

    # -- Second approach
    input_shape = load_x_y_z_pickle(15, 2)[0].shape[1]
    log_reg = LogReg()
    basic_nn = BasicNN(input_shape=input_shape, lr=1e-3, num_epochs=50, batch_size=128, num_units=4150)
    second_app_models = [('LogReg', log_reg), ('BasicNN', basic_nn)]

    for model_name, model in second_app_models:
        second_app_model_result = second_approach_cv(model, metrics, test_years)
        for ranking_method, results in sorted(second_app_model_result.items()):
            row_model_name = f'{model_name}_{ranking_method}'
            row_values = results
            final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    # Advanced Part
    advanced_nn = AdvancedNN(input_shape=input_shape, hidden_lstm_dim=3, hidden_first_fc_dim=input_shape // 2,
                              num_epochs=50, batch_size=64, lr=1e-3, optimizer=None, num_units=input_shape // 8)
    advanced_model_result = second_approach_cv_advanced(advanced_nn, metrics, test_years)
    row_model_name = 'AdvancedNN'
    row_values = advanced_model_result
    final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    # Creative Part
    ord_log_reg = OrdLogReg()
    ord_basic_nn = OrdBasicNN(input_shape=input_shape, lr=1e-3, num_epochs=50, batch_size=128, num_units=4150)
    second_app_models = [('OrdLogReg', ord_log_reg), ('OrdBasicNN', ord_basic_nn)]

    for model_name, model in second_app_models:
        second_app_model_result = second_approach_cv(model, metrics, test_years)
        for ranking_method, results in sorted(second_app_model_result.items()):
            row_model_name = f'{model_name}_{ranking_method}'
            row_values = results
            final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    print(final_df)
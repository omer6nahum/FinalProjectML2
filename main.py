from models.LogReg import LogReg
from models.LinReg import LinReg
from models.BasicNN import BasicNN
from models.AdvancedNN import AdvancedNN
from models.OrdLogReg import OrdLogReg
from models.OrdNN import OrdNN
from models.BinNN import BinNN
from EvaluationMetrics import adj_hamming_normalized, adj_MAP_normalized, spearman, points_error, correct_champion
from Preprocess import load_x_y_z_pickle
import pandas as pd
from CrossValidation import first_approach_cv, second_approach_cv, second_approach_cv_advanced


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    test_years = range(15, 22)
    metrics = [('adj_hamming', adj_hamming_normalized), ('adj_MAP', adj_MAP_normalized),
               ('Spearman', spearman), ('L1_distance', points_error), ('% correct champion', correct_champion)]

    # Basic Part
    # -- First Approach
    lin_reg = LinReg()
    first_app_results = first_approach_cv(lin_reg, metrics, test_years)
    final_df = pd.DataFrame(first_app_results, index=['LinReg'])

    # -- Second approach
    input_shape = load_x_y_z_pickle(15, 2)[0].shape[1]
    log_reg = LogReg()
    basic_nn = BasicNN(input_shape=input_shape, num_epochs=75, lr=1e-3, batch_size=64,
                       num_units=[100, 100], dropout=0.1, activations=['sigmoid']*2)

    second_app_models = [('LogReg', log_reg), ('BasicNN', basic_nn)]

    for model_name, model in second_app_models:
        second_app_model_result = second_approach_cv(model, metrics, test_years)
        for ranking_method, results in sorted(second_app_model_result.items()):
            row_model_name = f'{model_name}_{ranking_method}'
            row_values = results
            final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    # Advanced Part
    advanced_nn = AdvancedNN(input_shape=input_shape, num_epochs=50, dropout=0.1,
                             lr=1e-3, batch_size=16, num_units=100,
                             hidden_first_fc_dim=100, hidden_lstm_dim=15)
    advanced_model_result = second_approach_cv_advanced(advanced_nn, metrics, test_years)
    row_model_name = 'AdvancedNN'
    row_values = advanced_model_result
    final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    # Creative Part
    ord_log_reg = OrdLogReg()
    ord_nn = OrdNN(input_shape=input_shape, num_epochs=75, lr=1e-3, batch_size=64,
                   num_units=[100, 100], dropout=0.1, activations=['sigmoid'] * 2)
    bin_nn = BinNN(input_shape=input_shape, num_epochs=50, lr=1e-3, batch_size=32,
                   num_units=[250, 125, 50, 25], dropout=0.3, activations=['sigmoid']*4, loss='nll')
    bin_nn_mse = BinNN(input_shape=input_shape, num_epochs=75, lr=1e-3, batch_size=32,
                       num_units=[250, 50], dropout=0.3, activations=['sigmoid']*2, loss='mse')
    second_app_creative_models = [('OrdLogReg', ord_log_reg), ('OrdNN', ord_nn),
                                  ('BinNN', bin_nn), ('BinNN-MSE', bin_nn_mse)]

    for model_name, model in second_app_creative_models:
        second_app_model_result = second_approach_cv(model, metrics, test_years)
        for ranking_method, results in sorted(second_app_model_result.items()):
            row_model_name = f'{model_name}_{ranking_method}'
            row_values = results
            final_df = final_df.append(pd.DataFrame(row_values, index=[row_model_name]))

    print(final_df)
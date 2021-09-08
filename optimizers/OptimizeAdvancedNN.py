from models.AdvancedNN import AdvancedNN
from Preprocess import load_x_y_z_pickle
from CrossValidation import second_approach_cv_advanced
from itertools import product
import pickle

if __name__ == '__main__':
    test_years = range(15, 22)
    input_shape = load_x_y_z_pickle(15, 2, dir_path='../pickles/data/')[0].shape[1]
    metrics = []

    lr_list = [1e-3]
    num_epochs_list = [25, 50]
    batch_size_list = [16, 32]
    num_units_fc_list = [25, 50, 100]
    num_units_lstm_list = [3, 15, 50]
    dropout_list = [0.1, 0.3]
    adj_acc_dict = dict()
    all_params = list(product(lr_list, num_epochs_list, batch_size_list, num_units_fc_list,
                              num_units_fc_list, num_units_lstm_list, dropout_list))
    n = len(all_params)
    for i, (lr, num_epochs, batch_size, num_units_first_fc,
            num_units_fc, num_units_lstm, dropout) in enumerate(all_params):
        model = AdvancedNN(input_shape=input_shape, num_epochs=num_epochs, dropout=dropout,
                           lr=lr, batch_size=batch_size, num_units=num_units_fc,
                           hidden_first_fc_dim=num_units_first_fc, hidden_lstm_dim=num_units_lstm)
        second_app_model_result = second_approach_cv_advanced(model, metrics, test_years, prefix_path='../')
        adj_acc = second_app_model_result['adj_accuracy']
        adj_acc_dict[(lr, num_epochs, batch_size, num_units_first_fc, num_units_fc, num_units_lstm)] = adj_acc
        print(f'Param set {i}/{n}')
        print(f'Final adj_acc for {(lr, num_epochs, batch_size, num_units_first_fc, num_units_fc, num_units_lstm, dropout)} : {adj_acc}')

        # periodically pickle mid results:
        if i % 10 == 0:
            with open('optimized_advanced_nn_new.pkl', 'wb') as f:
                pickle.dump(adj_acc_dict, f)

    print(adj_acc_dict)
    with open('optimized_advanced_nn_new.pkl', 'wb') as f:
        pickle.dump(adj_acc_dict, f)

    print('\n\n\n-------------------------------------')
    for k, v in sorted(adj_acc_dict.items(), key=lambda x: x[1], reverse=True):
        print(f'{k} ::: {v}')

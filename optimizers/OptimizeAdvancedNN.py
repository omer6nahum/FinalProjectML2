from models.AdvancedNN import AdvancedNN
from Preprocess import load_x_y_z_pickle
from tqdm import tqdm
from CrossValidation import second_approach_cv_advanced
from itertools import product
import pickle

if __name__ == '__main__':
    test_years = range(15, 22)
    input_shape = load_x_y_z_pickle(15, 2)[0].shape[1]
    metrics = []

    lr_list = [1e-3, 1e-5]
    num_epochs_list = [20]
    batch_size_list = [64, 128]
    num_units_fc_list = [input_shape//2, input_shape//8, input_shape//64]
    num_units_lstm_list = [3, 15, 50]
    adj_acc_dict = dict()
    skip = True
    for (lr, num_epochs, batch_size, num_units_first_fc, num_units_fc, num_units_lstm) in \
            tqdm(product(lr_list, num_epochs_list, batch_size_list,
                         num_units_fc_list, num_units_fc_list, num_units_lstm_list)):
        if (lr, num_epochs, batch_size, num_units_first_fc, num_units_fc, num_units_lstm) \
                == (1e-05, 20, 64, 129, 4150, 50):
            skip = False
        if skip:
            continue
        model = AdvancedNN(input_shape=input_shape, num_epochs=num_epochs,
                           lr=lr, batch_size=batch_size, num_units=num_units_fc,
                           hidden_first_fc_dim=num_units_first_fc, hidden_lstm_dim=num_units_lstm)
        second_app_model_result = second_approach_cv_advanced(model, metrics, test_years)
        adj_acc = second_app_model_result['adj_accuracy']
        adj_acc_dict[(lr, num_epochs, batch_size, num_units_first_fc, num_units_fc, num_units_lstm)] = adj_acc
        print(f'Final adj_acc for {(lr, num_epochs, batch_size, num_units_first_fc, num_units_fc, num_units_lstm)} : {adj_acc}')

    # print(adj_acc_dict)
    # with open('optimized_advanced_nn.pkl', 'wb') as f:
    #     pickle.dump(adj_acc_dict, f)
    #
    # print('\n\n\n-------------------------------------')
    # for k, v in sorted(adj_acc_dict.items(), key=lambda x: x[1], reverse=True):
    #     print(f'{k} ::: {v:4f}')



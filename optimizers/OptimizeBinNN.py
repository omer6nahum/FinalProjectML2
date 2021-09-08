from models.BinNN import BinNN
from Preprocess import load_x_y_z_pickle
from CrossValidation import second_approach_cv
from itertools import product
import pickle

if __name__ == '__main__':
    test_years = range(15, 22)
    input_shape = load_x_y_z_pickle(15, 2, dir_path='../pickles/data/')[0].shape[1]
    metrics = []

    lr_list = [1e-3]
    num_epochs_list = [25, 50, 75]
    batch_size_list = [16, 32, 64]
    dropout_list = [0.1, 0.3, 0.5]
    num_units_list = [[100, 100], [250, 125, 50, 25], [250, 50], [100, 50, 20]]
    activation_list = [['sigmoid', 'sigmoid'],
                       ['sigmoid', 'relu', 'sigmoid'],
                       ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']]
    loss_list = ['mse', 'nll']
    adj_acc_dict = dict()
    all_params = list(product(lr_list, num_epochs_list, batch_size_list,
                              dropout_list, num_units_list, activation_list, loss_list))
    n = len(all_params)
    for i, (lr, num_epochs, batch_size, dropout, num_units, activations, loss) in enumerate(all_params):
        if len(num_units) != len(activations):
            continue
        model = BinNN(input_shape=input_shape, num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                      num_units=num_units, dropout=dropout, activations=activations, loss=loss)
        second_app_model_result = second_approach_cv(model, metrics, test_years, prefix_path='../')
        adj_acc = second_app_model_result['expectation']['adj_accuracy']
        adj_acc_dict[(lr, num_epochs, batch_size, dropout, tuple(num_units), tuple(activations), loss)] = adj_acc
        print(f'Param -set {i}/{n}')
        print(f'{(lr, num_epochs, batch_size, dropout, num_units, activations, loss)} :: {adj_acc}')

        # periodically pickle mid results:
        if i % 10 == 0:
            with open('optimized_bin_nn.pkl', 'wb') as f:
                pickle.dump(adj_acc_dict, f)

    print(adj_acc_dict)
    with open('optimized_bin_nn.pkl', 'wb') as f:
        pickle.dump(adj_acc_dict, f)

    print('\n\n\n-------------------------------------')
    for k, v in sorted(adj_acc_dict.items(), key=lambda x: x[1], reverse=True):
        print(f'{k} ::: {v}')



from models.BasicNN import BasicNN
from EvaluationMetrics import adj_hamming_normalized, adj_MAP_normalized, spearman, points_error
from Preprocess import load_x_y_z_pickle
from tqdm import tqdm
from main import second_approach_cv
from itertools import product
import pickle

if __name__ == '__main__':
    test_years = range(15, 22)
    metrics = [('adj_hamming', adj_hamming_normalized), ('adj_MAP', adj_MAP_normalized),
               ('Spearman', spearman), ('L1_distance', points_error)]
    input_shape = load_x_y_z_pickle(15, 2)[0].shape[1]
    metrics = []

    lr_list = [1e-3, 1e-4, 1e-5]
    num_epochs_list = [5, 10, 20, 50]
    batch_size_list = [32, 64, 128]
    num_units_list = [input_shape//2, input_shape//8, input_shape//64]
    adj_acc_dict = dict()
    for (lr, num_epochs, batch_size, num_units) in tqdm(product(lr_list, num_epochs_list, batch_size_list, num_units_list)):
        model = BasicNN(input_shape=input_shape, num_epochs=num_epochs,
                           lr=lr, batch_size=batch_size, num_units=num_units)
        second_app_model_result = second_approach_cv(model, metrics, test_years)
        adj_acc = second_app_model_result['expectation']['adj_accuracy']
        adj_acc_dict[(lr, num_epochs, batch_size, num_units)] = adj_acc

    print(adj_acc_dict)
    with open('optimized_basic_nn.pkl', 'wb') as f:
        pickle.dump(adj_acc_dict, f)

    print('\n\n\n-------------------------------------')
    for k, v in sorted(adj_acc_dict.items(), key=lambda x: x[1], reverse=True):
        print(f'{k} ::: {v:4f}')



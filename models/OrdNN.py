import numpy as np
from Preprocess import load_train_test
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from deps import LABELS, LABELS_REV
import time
from models.BasicNN import InnerBasicNN, MatchesDataset


class OrdNN:
    def __init__(self, input_shape, num_epochs=50, batch_size=32, lr=1e-3,
                 num_units=None, activations=None, dropout=0.3):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print(f'using {self.device}')
        if use_cuda:
            torch.cuda.empty_cache()
        self.model1 = InnerBasicNN(input_shape, self.device, num_labels=2, num_units=num_units,
                                   activations=activations, dropout=dropout).to(self.device)
        self.model2 = InnerBasicNN(input_shape, self.device, num_labels=2, num_units=num_units,
                                   activations=activations, dropout=dropout).to(self.device)
        self.batch_size = batch_size
        self.optimizer1 = optim.Adam(self.model1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr=lr)
        # self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.8)
        self.labels = LABELS
        self.is_fitted = False
        self.num_epochs = num_epochs

    def fit(self, X, y):
        """
        Fit the neural network models by running the training loop process for each model.
        :param X: explaining variables matrix (without ones column).
        :param y: explained variable vector (categorical label).
        :return: -.
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        y = np.array([self.labels[y_i] for y_i in y])
        y1 = y > 0
        y2 = y > 1

        self.fit_sub_model(X, y1, model_type=1)
        self.fit_sub_model(X, y2, model_type=2)

        self.is_fitted = True

    def fit_sub_model(self, X, y, model_type):
        """
        Fit the neural network model by running the training loop process.
        :param X: explaining variables matrix (without ones column).
        :param y: explained variable vector (categorical label).
        :param model_type: in [1, 2]
        :return: -.
        """
        assert model_type in [1, 2]
        if model_type == 1:
            model = self.model1
            optimizer = self.optimizer1
        else:
            model = self.model2
            optimizer = self.optimizer2
        print(f'Fitting model {model_type}')
        dataset = MatchesDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        n = len(trainloader)
        model.init_model_weights()
        model.train()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            t_start = time.time()
            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs).to(self.device)
                loss = model.loss_function(outputs, labels).to(self.device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # print loss per epoch
            print(f'Finished epoch {epoch + 1} in {(time.time() - t_start):.3f} sec : Loss={(running_loss / n):.3f}')

        if model_type == 1:
            self.model1 = model
            self.optimizer1 = optimizer
        else:
            self.model2 = model
            self.optimizer2 = optimizer

    def predict(self, X_new):
        """
        Predict probability distribution for the explained variable classes for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted probability distribution over the explained variable labels (for each new point) as tensor.
        """

        proba_outputs = []
        dataset = MatchesDataset(X_new)
        testloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            for inputs in testloader:
                inputs = inputs[0].to(self.device)
                pred1 = np.array(F.softmax(self.model1(inputs), dim=1).to('cpu'))  # ['H', 'D'+'A']
                pred2 = np.array(F.softmax(self.model2(inputs), dim=1).to('cpu'))  # ['H'+'D', 'A']

                res = np.zeros((pred1.shape[0], 3))
                res[:, 0] = pred1[:, 0]  # H
                res[:, 2] = pred2[:, 1]  # A
                res[:, 1] = 1 - res[:, 0] - res[:, 2]  # D
                proba_outputs.append(res)

        return np.concatenate(proba_outputs)

    def save_params(self, path):
        """
        Save model params as torch model format.
        :param path: path for the params file.
        :return: None
        """
        torch.save(self.model, path)

    def load_params(self, path):
        """
        Load model params as pickle.
        :param path: path of the loaded params.
        :return:
        """
        self.model = torch.load(path)
        self.is_fitted = True


if __name__ == '__main__':
    # OrdBasicNN, Second Approach:
    x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=21, approach=2, prefix_path='../')

    model = OrdNN(input_shape=x_train.shape[1], num_epochs=1, lr=1e-3)
    model.fit(x_train, y_train)
    y_proba_pred = model.predict(x_test)
    print(y_proba_pred)




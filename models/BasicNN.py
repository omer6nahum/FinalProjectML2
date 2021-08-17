import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from Preprocess import load_train_test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
from deps import LABELS, LABELS_REV
import time


class MatchesDataset(Dataset):
    def __init__(self, matches, labels=None):
        super().__init__()
        self.matches_dataset = self.convert_matches_to_dataset(matches, labels)

    def __len__(self):
        return len(self.matches_dataset)

    def __getitem__(self, index):
        return self.matches_dataset[index]

    def convert_matches_to_dataset(self, matches, labels):
        """
        converts matches(second approach representation) to tensor dataset
        :param matches: squads representation
        :param labels: match output
        :return:
        """
        matches_list = list()
        for match in matches:
            matches_list.append(torch.tensor(match, dtype=torch.float, requires_grad=False))

        all_matches = torch.stack(matches_list)
        if labels is None:
            return TensorDataset(all_matches)
        else:
            all_labels = torch.tensor(labels, dtype=torch.long, requires_grad=False)
            return TensorDataset(all_matches, all_labels)


class InnerBasicNN(nn.Module):
    def __init__(self, input_shape, device, num_labels=3, num_units=None):
        super().__init__()
        self.device = device
        self.num_units = input_shape // 2 if num_units is None else num_units
        self.num_labels = num_labels
        self.sequential = nn.Sequential(
            nn.Linear(input_shape, self.num_units),
            nn.Sigmoid(),
            nn.Linear(self.num_units, self.num_units),
            nn.Sigmoid(),
            nn.Linear(self.num_units, num_labels)
        ).to(self.device)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.sequential(x)


class BasicNN:
    def __init__(self, input_shape, num_epochs=100, batch_size=32, lr=1e-2, optimizer=None, num_units=None):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print(f'using {self.device}')
        if use_cuda:
            torch.cuda.empty_cache()
        self.model = InnerBasicNN(input_shape, self.device, num_labels=3, num_units=num_units).to(self.device)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) if optimizer is None else optimizer
        # self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.8)
        self.labels = {'H': 0, 'D': 1, 'A': 2}
        self.is_fitted = False
        self.num_epochs = num_epochs

    def fit(self, X, y):
        """
        Fit the neural network model by running the training loop process.
        :param X: explaining variables matrix (without ones column).
        :param y: explained variable vector (categorical label).
        :return: -.
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        y = np.array([self.labels[y_i] for y_i in y])

        dataset = MatchesDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        n = len(trainloader)
        self.model.train()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            t_start = time.time()
            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).to(self.device)
                loss = self.model.loss_function(outputs, labels).to(self.device)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # print loss per epoch
            print(f'Finished epoch {epoch + 1} in {(time.time() - t_start):.3f} sec : Loss={(running_loss / n):.3f}')

        self.is_fitted = True

    def predict(self, X_new):
        """
        Predict probability distribution for the explained variable classes for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted probability distribution over the explained variable labels (for each new point) as tensor.
        """

        # correct = 0
        # total = 0
        proba_outputs = []
        dataset = MatchesDataset(X_new)
        testloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for inputs in testloader:
                inputs = inputs[0].to(self.device)
                proba_output = F.softmax(self.model(inputs), dim=1).to('cpu')
                proba_outputs.append(proba_output)
        return np.array(torch.cat(proba_outputs, dim=0))

    def save_params(self, path):
        """
        Save model params as torch model format.
        :param path: path for the params file.
        :return: None
        """
        torch.save(model, path)

    def load_params(self, path):
        """
        Load model params as pickle.
        :param path: path of the loaded params.
        :return:
        """
        self.model = torch.load(path)
        self.is_fitted = True


if __name__ == '__main__':
    # BasicNN, Second Approach:
    x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=21, approach=2, prefix_path='../')

    model = BasicNN(input_shape=x_train.shape[1], num_epochs=0, lr=1e-3)
    model.fit(x_train, y_train)
    y_proba_pred = model.predict(x_test)
    print(y_proba_pred)




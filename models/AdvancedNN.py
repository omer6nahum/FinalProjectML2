import numpy as np
from Preprocess import load_train_test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
from deps import LABELS, LABELS_REV
import time

# todo: validate AdvancedNN fit function,
#       rewrite predict function,
#       rewrite InnerAdvancedNN


class MatchesSequencesDataset(Dataset):
    def __init__(self, matches, labels=None):
        super().__init__()
        self.matches_sequences_dataset = self.convert_matches_sequences_to_dataset(matches, labels)

    def __len__(self):
        return len(self.matches_sequences_dataset)

    def __getitem__(self, index):
        return self.matches_sequences_dataset[index]

    def convert_matches_sequences_to_dataset(self, matches, labels):
        """
        converts matches(second approach representation) to tensor dataset
        :param matches: each match is a list of size 3:
                        squads representation, home team sequence, away team sequence
        :param labels: match output
        :return:
        """
        squads_list = list()
        home_sequence_list = list()
        home_sequence_len_list = list()
        away_sequence_list = list()
        away_sequence_len_list = list()
        for match in matches:
            squads_list.append(torch.tensor(match[0], dtype=torch.float, requires_grad=False))
            home_sequence_list.append(torch.tensor(match[1], dtype=torch.float, requires_grad=False))
            home_sequence_len_list.append(self.sequence_len(match[1]))
            away_sequence_list.append(torch.tensor(match[2], dtype=torch.float, requires_grad=False))
            away_sequence_len_list.append(self.sequence_len(match[2]))

        all_squads = torch.stack(squads_list)
        all_home_sequences = torch.stack(home_sequence_list)
        all_home_sequences_len = torch.tensor(home_sequence_len_list, dtype=torch.long, requires_grad=False)
        all_away_sequences = torch.stack(away_sequence_list)
        all_away_sequences_len = torch.tensor(away_sequence_len_list, dtype=torch.long, requires_grad=False)
        if labels is None:
            return TensorDataset(all_squads, all_home_sequences, all_home_sequences_len,
                                 all_away_sequences, all_away_sequences_len)
        else:
            all_labels = torch.tensor(labels, dtype=torch.long, requires_grad=False)
            return TensorDataset(all_squads, all_home_sequences, all_home_sequences_len,
                                 all_away_sequences, all_away_sequences_len, all_labels)

    @staticmethod
    def sequence_len(sequence):
        counter = 0
        for i in range(4, -1, -1):
            if np.isnan(sequence[i][0]):
                break
            counter += 1
        return counter


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


class AdvancedNN:
    def __init__(self, input_shape, num_epochs=100, batch_size=32, lr=1e-2, optimizer=None, num_units=None):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print(f'using {self.device}')
        if use_cuda:
            torch.cuda.empty_cache()
        self.model = InnerAdvancedNN(input_shape, self.device, num_labels=3, num_units=num_units).to(self.device)
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

        dataset = MatchesSequencesDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
        n = len(trainloader)
        self.model.train()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            t_start = time.time()
            running_loss = 0.0
            i = 0
            self.optimizer.zero_grad()
            for data in trainloader:
                i += 1
                squads, home_sequence, home_sequence_len, away_sequence, away_sequence_len, labels = data
                home_sequence = home_sequence.to(self.device)
                home_sequence_len = home_sequence_len.to(self.device)
                away_sequence = away_sequence.to(self.device)
                away_sequence_len = away_sequence_len.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(squads, home_sequence, home_sequence_len, away_sequence, away_sequence_len)\
                    .to(self.device)
                loss = self.model.loss_function(outputs, labels).to(self.device)
                loss = loss / self.batch_size
                loss.backward()
                if i % self.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                running_loss += loss.item()

            running_loss = self.batch_size * (running_loss / len(trainloader))
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
    x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=21, approach=2,
                                                                        part='advanced', prefix_path='../')
    # y_train = np.array([LABELS[y_i] for y_i in y_train])
    # dataset = MatchesSequencesDataset(x_train, y_train)
    # trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print()

    model = AdvancedNN(input_shape=x_train.shape[1], num_epochs=0, lr=1e-3)
    # model.fit(x_train, y_train)
    # y_proba_pred = model.predict(x_test)
    # print(y_proba_pred)




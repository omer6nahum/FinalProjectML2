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

        default_empty_sequence = np.zeros((5, 3))
        default_empty_sequence[0:4, :] = np.nan
        for match in matches:
            squads_list.append(torch.tensor(match[0], dtype=torch.float, requires_grad=False))
            home_len = self.sequence_len(match[1])
            if home_len > 0:
                home_sequence_len_list.append(home_len)
                home_sequence_list.append(torch.tensor(match[1], dtype=torch.float, requires_grad=False))
            else:
                home_sequence_len_list.append(1)
                home_sequence_list.append(torch.tensor(default_empty_sequence, dtype=torch.float, requires_grad=False))

            away_len = self.sequence_len(match[2])
            if home_len > 0:
                away_sequence_len_list.append(away_len)
                away_sequence_list.append(torch.tensor(match[2], dtype=torch.float, requires_grad=False))
            else:
                away_sequence_len_list.append(1)
                away_sequence_list.append(torch.tensor(default_empty_sequence, dtype=torch.float, requires_grad=False))

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


class InnerAdvancedNN(nn.Module):
    def __init__(self, input_shape, device, num_labels=3, num_units=None,
                 hidden_lstm_dim=20, hidden_first_fc_dim=None):
        super().__init__()
        self.device = device
        self.hidden_lstm_dim = hidden_lstm_dim
        self.hidden_first_fc_dim = hidden_first_fc_dim
        self.num_units = input_shape // 2 if num_units is None else num_units
        self.hidden_lstm_dim = hidden_lstm_dim
        self.hidden_first_fc_dim = input_shape // 2 if hidden_first_fc_dim is None else hidden_first_fc_dim
        self.num_labels = num_labels
        self.firstFC = nn.Sequential(
            nn.Linear(input_shape, self.hidden_first_fc_dim),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_lstm_dim, batch_first=True)
        self.sequential = nn.Sequential(
            nn.Linear(self.hidden_first_fc_dim + 2 * self.hidden_lstm_dim, self.num_units),
            nn.Sigmoid(),
            nn.Linear(self.num_units, self.num_units),
            nn.Sigmoid(),
            nn.Linear(self.num_units, num_labels)
        ).to(self.device)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x):
        squads, home_sequence, home_sequence_len, away_sequence, away_sequence_len = x
        home_sequence = home_sequence[:, -home_sequence_len:, :]
        away_sequence = away_sequence[:, -away_sequence_len:, :]
        squads_embedding = self.firstFC(squads)
        _, (__, lstm_home) = self.lstm(home_sequence)
        _, (__, lstm_away) = self.lstm(away_sequence)
        x = torch.cat([squads_embedding.squeeze(), lstm_home.squeeze(), lstm_away.squeeze()]).unsqueeze(0)
        return self.sequential(x)


class AdvancedNN:
    def __init__(self, input_shape, hidden_lstm_dim=20, hidden_first_fc_dim=None, num_epochs=100,
                 batch_size=32, lr=1e-2, optimizer=None, num_units=None):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print(f'using {self.device}')
        if use_cuda:
            torch.cuda.empty_cache()
        self.model = InnerAdvancedNN(input_shape=input_shape, device=self.device, num_labels=3,
                                     num_units=num_units, hidden_lstm_dim=hidden_lstm_dim,
                                     hidden_first_fc_dim=hidden_first_fc_dim).to(self.device)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) if optimizer is None else optimizer
        # self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.8)
        self.labels = {'H': 0, 'D': 1, 'A': 2}
        self.is_fitted = False
        self.num_epochs = num_epochs

    def fit(self, X, y):
        """
        Fit the neural network model by running the training loop process.
        :param X: explaining variables tensor containing:
                  squads representation, home team sequence, away team sequence
        :param y: explained variable vector (categorical label).
        :return: -.
        """
        assert isinstance(X, list)
        assert isinstance(y, np.ndarray)
        assert len(X) == y.shape[0]
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
                squads = squads.to(self.device)
                home_sequence = home_sequence.to(self.device)
                home_sequence_len = home_sequence_len.to(self.device)
                away_sequence = away_sequence.to(self.device)
                away_sequence_len = away_sequence_len.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model((squads, home_sequence, home_sequence_len, away_sequence, away_sequence_len))\
                    .to(self.device)
                loss = self.model.loss_function(outputs, labels).to(self.device)
                loss = loss / self.batch_size
                loss.backward()
                if i % self.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                running_loss += loss.item()

            running_loss = self.batch_size * running_loss
            # print loss per epoch
            print(f'Finished epoch {epoch + 1} in {(time.time() - t_start):.3f} sec : Loss={(running_loss / n):.3f}')

        self.is_fitted = True

    def predict(self, X_new):
        """
        Predict probability distribution for the explained variable classes for new observations.
        :param X_new: explaining variables tensor containing:
                      squads representation, home team sequence, away team sequence
        :return: predicted probability distribution over the explained variable labels (for each new point) as tensor.
        """

        proba_outputs = []
        dataset = MatchesSequencesDataset(X_new)
        testloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs in testloader:
                squads, home_sequence, home_sequence_len, away_sequence, away_sequence_len, = inputs
                squads = squads.to(self.device)
                home_sequence = home_sequence.to(self.device)
                home_sequence_len = home_sequence_len.to(self.device)
                away_sequence = away_sequence.to(self.device)
                away_sequence_len = away_sequence_len.to(self.device)
                proba_output = F.softmax(self.model((squads, home_sequence, home_sequence_len,
                                                    away_sequence, away_sequence_len)), dim=1).to('cpu')
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
    # print()
    input_shape = x_train[0][0].shape[0]  # squad dim
    model = AdvancedNN(input_shape=input_shape, hidden_lstm_dim=20, hidden_first_fc_dim=100, num_epochs=3,
                       batch_size=32, lr=1e-3, optimizer=None, num_units=None)
    model.fit(x_train, y_train)
    y_proba_pred = model.predict(x_test)
    print(y_proba_pred)




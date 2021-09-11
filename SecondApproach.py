import numpy as np
import pandas as pd
from models.LogReg import LogReg
from models.OrdLogReg import OrdLogReg
from models.BasicNN import BasicNN
from models.AdvancedNN import AdvancedNN
from models.OrdNN import OrdNN
from models.BinNN import BinNN
from random import random
from deps import LABELS_REV
from collections import deque


model_options = [LogReg, BasicNN, AdvancedNN, OrdLogReg, OrdNN, BinNN]
ranking_method_options = ['expectation', 'simulation', 'advanced_simulation']
WIN_VEC = np.array([1, 0, 0])
DRAW_VEC = np.array([0, 1, 0])
LOSE_VEC = np.array([0, 0, 1])


class SecondApproach:

    def __init__(self, model):
        assert type(model) in model_options
        assert model.is_fitted
        self.model = model

    def predict_table(self, X_test, z_test, ranking_method, y_test=None):
        """
        Predict the final table based on the inner models.
        :param X_test: features of new observations (season matches).
        :param z_test: teams names in matches corresponding to observations in X_test.
        :param ranking_method: either 'expectation' or 'simulation'.
        :param y_test: true outcomes for test matches.
        :return: a DataFrame of the predicted table (teams and predicted number of points).
        """
        assert ranking_method in ranking_method_options
        teams = set(z_test.flatten())

        if ranking_method == 'advanced_simulation':
            pts_pred, (acc, adj_acc) = self.advanced_simulation(teams, X_test, z_test, y_test=y_test)
        else:
            probs = self.model.predict(X_test)
            if ranking_method == 'expectation':
                pts_pred = self.expectation(teams, probs, z_test)
            else:
                assert ranking_method == 'simulation'
                pts_pred = self.simulation(teams, probs, z_test)
            acc = self.accuracy(X_test, y_test)
            adj_acc = self.adjusted_accuracy(X_test, y_test)

        pred_table_df = pd.DataFrame(pts_pred, index=['PTS']).T
        pred_table_df = pred_table_df.reset_index(drop=False)
        pred_table_df = pred_table_df.rename({'index': 'team_name'}, axis=1)
        pred_table_df = pred_table_df.sort_values('PTS', ascending=False)
        pred_table_df = pred_table_df.reset_index(drop=True)
        return pred_table_df, (acc, adj_acc)

    def accuracy(self, X_test, y_test=None):
        if y_test is None:
            return 0
        assert X_test.shape[0] == y_test.shape[0]
        y_pred = self.model.predict(X_test)
        y_pred = np.array([LABELS_REV[np.argmax(y_pred[i])] for i in range(y_pred.shape[0])])
        num_true = 0
        num_total = 0

        for i in range(y_pred.shape[0]):
            num_total += 1
            num_true += 1 if y_test[i] == y_pred[i] else 0

        return num_true / num_total

    def adjusted_accuracy(self, X_test, y_test=None):
        if y_test is None:
            return 0
        assert X_test.shape[0] == y_test.shape[0]
        y_pred = self.model.predict(X_test)
        y_pred = np.array([LABELS_REV[np.argmax(y_pred[i])] for i in range(y_pred.shape[0])])
        num_true = 0
        num_total = 0
        num_draws = 0

        for i in range(y_pred.shape[0]):
            num_total += 1
            if y_test[i] == y_pred[i]:
                num_true += 1
            elif y_test[i] == 'D' or y_pred[i] == 'D':
                num_true += 0.5
            num_draws += 1 if y_test[i] == 'D' else 0

        adj_acc = (num_true - num_draws * 0.5) / (num_total - num_draws * 0.5)
        return adj_acc

    @staticmethod
    def expectation(teams, probs, z_test):
        """
        Calculate the expected number of points for each teams based on the given probabilities.
        :param teams: relevant team names.
        :param probs: probabilities for match outcomes for each match.
        :param z_test: teams names in matches corresponding to observations in X_test.
        :return: a dictionary mapping teams to expected number of points.
        """
        table = {team: 0 for team in teams}
        for pr, match_teams in zip(probs, z_test):
            hometeam, awayteam = match_teams[0], match_teams[1]
            pr_home, pr_draw, pr_away = pr
            table[hometeam] += 3 * pr_home + 1 * pr_draw
            table[awayteam] += 3 * pr_away + 1 * pr_draw
        return table

    @staticmethod
    def simulation(teams, probs, z_test, num_simulations=10):
        """
        Simulate season matches based on the given probabilities, and calculate number of points.
        :param teams: relevant team names.
        :param probs: probabilities for match outcomes for each match.
        :param z_test: teams names in matches corresponding to observations in X_test.
        :param num_simulations: how much simulations to run
        :return: a dictionary mapping teams to expected number of points (mean of all simulations).
        """
        table = {team: 0 for team in teams}
        for simulation in range(num_simulations):
            for pr, match_teams in zip(probs, z_test):
                hometeam, awayteam = match_teams[0], match_teams[1]
                pr_home, pr_draw, pr_away = pr
                result = random()
                if result < pr_home:  # home team won
                    table[hometeam] += 3
                elif result < pr_home + pr_draw:  # draw
                    table[hometeam] += 1
                    table[awayteam] += 1
                else:  # away team won
                    table[awayteam] += 3
        table = {k: v/num_simulations for k, v in table.items()}  # mean over all simulations
        return table

    def advanced_simulation(self, teams, x_test, z_test, num_simulations=100, y_test=None):
        """
        Dynamically simulate season matches, in the given order, based on self.model probabilities prediction,
        and calculate number of points.
        :param teams:  relevant team names.
        :param x_test: relevant squad representation for
        :param z_test: teams names in matches corresponding to observations in X_test (given matches order).
        :param num_simulations: how much simulations to run
        :param y_test: true outcomes of test matches
        :return: table, (accuracy, adj_accuracy)
        table is a dictionary mapping teams to expected number of points.
        accuracy and adj_accuracy are mean of all simulations
        """
        assert isinstance(self.model, AdvancedNN)

        table = {team: 0 for team in teams}
        adjusted_accuracies = []
        accuracies = []
        for simulation in range(num_simulations):
            accuracy = 0
            adjusted_accuracy = 0
            num_draws = 0
            # for each team save the last 5 matches results from the simulation
            teams_sequences = {team: deque([np.array([np.nan]*3)]*5, maxlen=5) for team in teams}
            for i, (squads_repr, match_teams) in enumerate(zip(x_test, z_test)):
                squads_repr = squads_repr[0]  # ignoring true last 5 matches results
                hometeam, awayteam = match_teams[0], match_teams[1]
                # a documentation from Preprocess for how an x_i should be like:
                #   x_i is a *list*:
                #   x_i[0] is 2-squads representation
                #   x_i[1] is a (5, 3) shaped array of 5 last matches of home team
                #   x_i[2] is a (5, 3) shaped array of 5 last matches of away team
                home_sequence = self.create_sequence_matrix(teams_sequences[hometeam])
                away_sequence = self.create_sequence_matrix(teams_sequences[awayteam])
                x = [squads_repr, home_sequence, away_sequence]
                pr = self.model.predict([x])[0]
                pr_home, pr_draw, pr_away = pr
                result = random()
                if result < pr_home:  # home team won
                    table[hometeam] += 3
                    teams_sequences[hometeam].append(WIN_VEC)
                    teams_sequences[awayteam].append(LOSE_VEC)
                elif result < pr_home+pr_draw:  # draw
                    table[hometeam] += 1
                    table[awayteam] += 1
                    teams_sequences[hometeam].append(DRAW_VEC)
                    teams_sequences[awayteam].append(DRAW_VEC)
                else:  # away team won
                    table[awayteam] += 3
                    teams_sequences[hometeam].append(LOSE_VEC)
                    teams_sequences[awayteam].append(WIN_VEC)

                if y_test is not None:
                    add_accuracy, add_adjusted_accuracy = self.update_accuracies(result, pr, y_test[i])
                    accuracy += add_accuracy
                    adjusted_accuracy += add_adjusted_accuracy
                    if y_test[i] == 'D':
                        num_draws += 1
            accuracies.append(accuracy / 380)
            adjusted_accuracies.append((adjusted_accuracy - num_draws * 0.5) / (380 - num_draws * 0.5))
        table = {k: v / num_simulations for k, v in table.items()}  # mean over all simulations
        return table, (np.mean(accuracies), np.mean(adjusted_accuracies))

    @staticmethod
    def create_sequence_matrix(team_sequence: deque):
        """
        Transform the deque team_sequence into a matrix of shape (5, 3).
        Each Row represents a match result (Win/Draw/Lose). Overall 5 last matches.
        The oreder of the rows in this matrix is:
        matrix[-1] is the last match, matrix[-2] is the second last match, and so on.
        :param team_sequence: deque of maxlen 5
        :return: the transformed matrix
        """
        return np.vstack(team_sequence).astype(np.float)

    @staticmethod
    def update_accuracies(result, pr, true_outcome):
        """
        :param result: a number in [0, 1] representing the predicted outcome of the match.
        :param pr: predicted probabilities for match outcome
        :param true_outcome: either {'H', 'D', 'A'}
        :return: numbers to add to accuracy and adjusted_accuracy
        """
        add_accuracy = 0
        add_adj_accuracy = 0
        pr_home, pr_draw, pr_away = pr

        if result < pr_home:  # home team won
            if true_outcome == 'H':
                add_accuracy = 1
                add_adj_accuracy = 1
            if true_outcome == 'D':
                add_adj_accuracy = 0.5

        elif result < pr_home + pr_draw:  # draw
            if true_outcome == 'D':
                add_adj_accuracy = 1
                add_accuracy = 1
            else:
                add_adj_accuracy = 0.5

        else:  # away team won
            if true_outcome == 'A':
                add_accuracy = 1
                add_adj_accuracy = 1
            if true_outcome == 'D':
                add_adj_accuracy = 0.5

        return add_accuracy, add_adj_accuracy

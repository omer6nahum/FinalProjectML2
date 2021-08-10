import pandas as pd
from models.LogReg import LogReg
from Preprocess import load_train_test
from random import random

model_options = [LogReg]
ranking_method_options = ['expectation', 'simulation']


class SecondApproach:

    def __init__(self, model):
        assert type(model) in model_options
        assert model.is_fitted
        self.model = model

    def predict_table(self, X_test, z_test, ranking_method):
        """
        Predict the final table based on the inner models.
        :param X_test: features of new observations (season matches).
        :param z_test: teams names in matches corresponding to observations in X_test.
        :param ranking_method: either 'expectation' or 'simulation'.
        :return: a DataFrame of the predicted table (teams and predicted number of points).
        """
        assert ranking_method in ranking_method_options
        probs = self.model.predict(X_test)
        teams = set(z_test.flatten())
        if ranking_method == 'expectation':
            pts_pred = self.expectation(teams, probs, z_test)
        else:
            assert ranking_method == 'simulation'
            pts_pred = self.simulation(teams, probs, z_test)
        pred_table_df = pd.DataFrame(pts_pred, index=['PTS']).T
        pred_table_df = pred_table_df.reset_index(drop=False)
        pred_table_df = pred_table_df.rename({'index': 'team_name'}, axis=1)
        pred_table_df = pred_table_df.sort_values('PTS', ascending=False)
        pred_table_df = pred_table_df.reset_index(drop=True)
        return pred_table_df

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
    def simulation(teams, probs, z_test):
        """
        Simulate season matches based on the given probabilities, and calculate number of points.
        # TODO: introduce a parameter for number of simulations for each match.
        :param teams: relevant team names.
        :param probs: probabilities for match outcomes for each match.
        :param z_test: teams names in matches corresponding to observations in X_test.
        :return: a dictionary mapping teams to expected number of points.
        """
        table = {team: 0 for team in teams}
        for pr, match_teams in zip(probs, z_test):
            hometeam, awayteam = match_teams[0], match_teams[1]
            pr_home, pr_draw, pr_away = pr
            result = random()
            if result < pr_home:
                table[hometeam] += 3
            elif result < pr_home+pr_draw:
                table[hometeam] += 1
                table[awayteam] += 1
            else:
                table[awayteam] += 3
        return table


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=21,
                                                                          approach=2,
                                                                          prefix_path='')

    log_reg = LogReg()
    log_reg.fit(x_train, y_train)
    second_approach = SecondApproach(log_reg)
    pred_table_expectation = second_approach.predict_table(x_test, z_test, ranking_method='expectation')
    print(pred_table_expectation)
    pred_table_simulation = second_approach.predict_table(x_test, z_test, ranking_method='simulation')
    print(pred_table_simulation)
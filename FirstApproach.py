import pandas as pd
from models.LinReg import LinReg

model_options = [LinReg]


class FirstApproach:

    def __init__(self, model):
        assert type(model) in model_options
        assert model.is_fitted
        self.model = model

    def predict_table(self, X_test, z_test):
        """
        Predict the final table based on the inner models.
        :param X_test: features of new observations (teams squads representations).
        :param z_test: teams names corresponding to observations in X_test.
        :return: a DataFrame of the predicted table (teams and predicted number of points).
        """
        y_pred = self.model.predict(X_test)
        pred_table = sorted(zip(z_test, y_pred), key=lambda x: x[1], reverse=True)
        pred_table_df = pd.DataFrame(data=pred_table, columns=['team_name', 'PTS'])
        pred_table_df['team_name'] = pred_table_df['team_name'].apply(lambda l: l[0])
        return pred_table_df

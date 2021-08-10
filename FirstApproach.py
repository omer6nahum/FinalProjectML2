import pandas as pd
from models.LinReg import LinReg
from Preprocess import load_train_test

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


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, z_train, z_test = load_train_test(test_year=21,
                                                                          approach=1,
                                                                          prefix_path='')

    # Validate is_fitted logic in instantiation:
    lin_reg_fitted_by_function = LinReg()
    lin_reg_fitted_by_load = LinReg()
    lin_reg_non_fitted = LinReg()
    lin_reg_fitted_by_function.fit(x_train, y_train)
    lin_reg_fitted_by_load.load_params('pickles/models/lin_reg_ver1.pkl')
    first_approach_model_fitted_by_function = FirstApproach(lin_reg_fitted_by_function)
    first_approach_model_fitted_by_load = FirstApproach(lin_reg_fitted_by_load)
    try:
        first_approach_model_not_fitted = FirstApproach(lin_reg_non_fitted)
    except AssertionError:
        print('Success.')

    # Validate predict table:
    pred_table = first_approach_model_fitted_by_function.predict_table(x_test, z_test)
    print(pred_table)
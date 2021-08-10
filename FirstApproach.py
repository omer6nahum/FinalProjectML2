from models.LinReg import LinReg
from Preprocess import create_train_test

model_options = [LinReg]


class FirstApproach:

    def __init__(self, model):
        assert type(model) in model_options
        assert model.is_fitted
        self.model = model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test(test_year=21,
                                                                          approach=1,
                                                                          prefix_path='')

    # Validate is_fitted logic in instantiation:
    lin_reg_fitted_by_function = LinReg()
    lin_reg_fitted_by_load = LinReg()
    lin_reg_non_fitted = LinReg()
    lin_reg_fitted_by_function.fit(x_train, y_train)
    lin_reg_fitted_by_load.load_params('data/pickles/models/lin_reg_ver1.pkl')
    first_approach_model_fitted_by_function = FirstApproach(lin_reg_fitted_by_function)
    first_approach_model_fitted_by_load = FirstApproach(lin_reg_fitted_by_load)
    try:
        first_approach_model_not_fitted = FirstApproach(lin_reg_non_fitted)
    except AssertionError:
        print('Success.')
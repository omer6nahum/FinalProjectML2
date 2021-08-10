from Preprocess import create_train_test
from models.LinReg import LinReg
import pandas as pd


if __name__ == '__main__':
    # LinReg, First Approach:
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test(test_year=21, approach=1)

    # Fit and predict:
    lin_reg = LinReg()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)

    # Create tables
    # TODO(omermadmon): this should be a function of FirstApproach class):
    clubs_2021_pred = sorted(zip(z_test, y_pred), key=lambda x: x[1], reverse=True)
    clubs_2021_true = sorted(zip(z_test, y_test), key=lambda x: x[1], reverse=True)  # TODO: load data of real table
    predicted_table = pd.DataFrame(data=clubs_2021_pred)
    true_table = pd.DataFrame(data=clubs_2021_true)
    print(predicted_table)
    print(true_table)

    # Verify ANOVA:
    print(lin_reg.ANOVA())  # TODO(omermadmon): resolve bug (negative residuals dof)

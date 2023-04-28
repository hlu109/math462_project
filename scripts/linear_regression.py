import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    area = "city of london"
    monthly_data, yearly_data = utils.load_dataset()

    def to_date(seconds):
        start_time = monthly_data["date"][0]
        return np.array(
            [start_time + timedelta(seconds=num) for num in seconds])

    area_data = utils.get_area(monthly_data, area)

    X = np.array(area_data["seconds"]).reshape(
        -1, 1)  # can also add other features like crimes and
    y = np.array(area_data["average_price"])

    # split train/test data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=False)

    # create model
    degree = 2
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = make_pipeline(polybig_features, std_scaler, lin_reg)

    polynomial_regression.fit(X_train, y_train)
    # y_pred = polynomial_regression.predict(X_test)

    plt.plot(X, y)
    plt.gca().set_xticks([])
    plt.plot(X_train.reshape(-1), polynomial_regression.predict(X_train))
    plt.plot(X_test.reshape(-1), polynomial_regression.predict(X_test))
    plt.show()
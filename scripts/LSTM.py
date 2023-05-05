import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

    ds = area_data.loc[:, ['seconds', 'average_price']]

    # split train/test data
    train_ds, val_ds = train_test_split(ds,
                                                        test_size=0.3,
                                                        shuffle=False)

    # copied from example code in the text 
    tf.random.set_seed(42)  # extra code – ensures reproducibility
    
    # create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
    ])

    # extra code – defines a utility function we'll reuse several time
    # TODO: move this to utils
    def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=500):
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(   
            monitor="val_mae", patience=50, restore_best_weights=True)
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
        history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                            callbacks=[early_stopping_cb])
        valid_loss, valid_mae = model.evaluate(valid_set)
        return history, valid_mae * 1e6

    fit_and_evaluate(model, train_ds, val_ds, learning_rate=0.02)

    # plt.plot(X, y)
    # plt.gca().set_xticks([])
    # # plt.plot(X_train.reshape(-1), polynomial_regression.predict(X_train))
    # # plt.plot(X_test.reshape(-1), polynomial_regression.predict(X_test))
    # plt.show()
# coding=utf-8
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from ml.svm.svm_exe import svc

from ml.tf_exe.o1_linear_regression import linear_regrission_fn

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("/Users/tong/Desktop/important/data/housing.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

california_housing_dataframe["median_house_value"] /= 1000.0

# 合成特征和离群值
"""

1,创建一个合成特征，即另外两个特征的比例
2,将此新特征用作线性回归模型的输入
3,通过识别和截取（移除）输入数据中的离群值来提高模型的有效性

"""


def train_model(learning_rate, steps, batch_size, input_feature):
    periods = 10  # every step will log some data
    steps_per_period = steps / periods

    my_feature = input_feature

    my_feature_data = california_housing_dataframe[[input_feature]].astype('float32')

    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label].astype('float32')

    # create inpput function
    train_input_fn = lambda: linear_regrission_fn(my_feature_data, targets, batch_size)

    # create test function
    predict_training_input_fn = lambda: linear_regrission_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # create input column
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # create optimater
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    regressor = tf.estimator.LinearRegressor(feature_columns, optimizer=optimizer)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []

    for period in range(0, periods):
        regressor.train(input_fn=train_input_fn, steps=steps_per_period)

        predictions = regressor.predict(predict_training_input_fn)

        predictions = np.array([item["predictions"][0] for item in predictions])

        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))

        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)

        y_extents = np.array([0, sample[my_label].max()])

        weight = regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Create a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

    return calibration_data


if __name__ == '__main__':

    # total_rooms 与 population 比
    california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]

    print(california_housing_dataframe.describe())

    calibration_data = train_model(0.5, 5000, 5, "rooms_per_person")

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(calibration_data["predictions"], calibration_data["targets"])

    plt.subplot(1, 2, 2)
    california_housing_dataframe["rooms_per_person"].hist()


    california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

    california_housing_dataframe["rooms_per_person"].hist()
    plt.show()

    calibration_data = train_model(0.5, 5000, 5, "rooms_per_person")

    plt.scatter(calibration_data["predictions"], calibration_data["targets"])
    plt.show()



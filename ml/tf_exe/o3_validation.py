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

from ml.tf_exe.o1_linear_regression import linear_regrission_fn
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def preprocess_features_fn(california_housing_dataframe):
    selected_feature = california_housing_dataframe[
        ["latitude", "longitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
         "median_income"]]

    processed_feature = selected_feature.copy()
    processed_feature["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"])

    return processed_feature


def preprocess_targets(california_housing_dataframe):
    targets = pd.DataFrame()
    targets["median_house_value"] = (california_housing_dataframe["median_house_value"] / 1000.0)
    return targets


def split_data(processed_features, targets):
    trainX, testX, trainY, testY = train_test_split(processed_features, targets, test_size=0.33, random_state=42)

    # trainX=processed_features.head(12000)
    # trainY=targets.head(12000)
    #
    # testX=processed_features.tail(5000)
    # testY=targets.tail(5000)

    return trainX, trainY, testX, testY


def plt_data(processed_features, targets):
    training_examples, training_targets, validation_examples, validation_targets = split_data(processed_features,
                                                                                              targets)

    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())

    # plt.show()


def construct_features(processed_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in processed_features])


def train_model(learning_rate,
                steps, batch_size,
                training_examples,
                training_targets,
                validation_examples,
                validation_targets):
    periods = 10  # every step will log some data
    steps_per_period = steps / periods

    # create optimater
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    regressor = tf.estimator.LinearRegressor(feature_columns=construct_features(training_examples), optimizer=optimizer)

    # create inpput function
    train_input_fn = lambda: linear_regrission_fn(training_examples, training_targets, batch_size)

    # create test function
    predict_training_input_fn = lambda: linear_regrission_fn(training_examples, training_targets, num_epochs=1,
                                                             shuffle=False)
    predict_testing_input_fn = lambda: linear_regrission_fn(validation_examples, validation_targets, num_epochs=1,
                                                            shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    train_root_mean_squared_errors = []
    test_root_mean_squared_errors = []

    for period in range(0, periods):
        regressor.train(input_fn=train_input_fn, steps=steps_per_period)

        train_predictions = regressor.predict(predict_training_input_fn)

        test_predictions = regressor.predict(predict_testing_input_fn)

        train_predictions = np.array([item["predictions"][0] for item in train_predictions])
        test_predictions = np.array([item["predictions"][0] for item in test_predictions])

        train_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(train_predictions, training_targets))
        test_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_predictions, validation_targets))

        print(" train period %02d : %0.2f" % (period, train_root_mean_squared_error))
        print(" trest period %02d : %0.2f" % (period, test_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        train_root_mean_squared_errors.append(train_root_mean_squared_error)
        test_root_mean_squared_errors.append(test_root_mean_squared_error)

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(train_root_mean_squared_errors, label="training")
    plt.plot(test_root_mean_squared_errors, label="validation")
    plt.legend()

    return regressor


if __name__ == '__main__':
    california_housing_dataframe = pd.read_csv("/Users/tong/Desktop/important/data/housing.csv", sep=",")
    california_housing_test_data = pd.read_csv("/Users/tong/Desktop/important/data/california_housing_test.csv", sep=",")

    processed_features = preprocess_features_fn(california_housing_dataframe)

    print(processed_features)

    targets = preprocess_targets(california_housing_dataframe)

    print(targets)

    # plt_data(processed_features, targets)

    training_examples, training_targets, validation_examples, validation_targets = split_data(processed_features,
                                                                                              targets)

    regressor = train_model(learning_rate=0.0001,
                            steps=100,
                            batch_size=1,
                            training_examples=training_examples,
                            training_targets=training_targets,
                            validation_examples=validation_examples,
                            validation_targets=validation_targets)

    # plt.show()

    processed_test_features = preprocess_features_fn(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    test_input_fn=lambda : linear_regrission_fn(processed_test_features,test_targets["median_house_value"],num_epochs=1,shuffle=False)

    test_predictions = regressor.predict(test_input_fn)

    test_predictions = np.array([item["predictions"][0] for item in test_predictions])

    test_rmse = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
    print("Final RMSE (on test data): %0.2f" % test_rmse)














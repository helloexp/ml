# coding=utf-8

# 使用 FTRL 优化算法进行模型训练
# 通过独热编码、分箱和特征组合创建新的合成特征


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
from ml.tf_exe.o3_validation import preprocess_features_fn, preprocess_targets, split_data
from sklearn.model_selection import train_test_split


def construct_feature_columns(training_examples):
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    # Divide households into 7 buckets.
    bucketized_households = tf.feature_column.bucketized_column(
        households, boundaries=get_quantile_based_boundaries(
            training_examples["households"], 7))

    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    # Divide latitude into 10 buckets.
    bucketized_latitude = tf.feature_column.bucketized_column(
        latitude, boundaries=get_quantile_based_boundaries(
            training_examples["latitude"], 10))

    # Divide housing_median_age into 7 buckets.
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        housing_median_age, boundaries=get_quantile_based_boundaries(
            training_examples["housing_median_age"], 7))

    # Divide median_income into 7 buckets.
    bucketized_median_income = tf.feature_column.bucketized_column(
        median_income, boundaries=get_quantile_based_boundaries(
            training_examples["median_income"], 7))

    # Divide rooms_per_person into 7 buckets.
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        rooms_per_person, boundaries=get_quantile_based_boundaries(
            training_examples["rooms_per_person"], 7))

    long_x_lat = tf.feature_column.crossed_column(
        set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

    feature_columns = {bucketized_longitude, bucketized_latitude, bucketized_housing_median_age, bucketized_households,
                       bucketized_median_income, bucketized_rooms_per_person, long_x_lat}

    return feature_columns


def train_model(learning_rate,
                steps, batch_size,
                training_examples,
                training_targets,
                validation_examples,
                validation_targets):
    periods = 10  # every step will log some data
    steps_per_period = steps / periods

    # create optimater
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)

    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples),
                                             optimizer=optimizer)

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


def get_quantile_based_boundaries(feature_values, num_backets):
    boundaries = np.arange(1, num_backets) / num_backets

    quantile = feature_values.quantile(boundaries)

    print(quantile)


if __name__ == '__main__':
    california_housing_dataframe = pd.read_csv("/Users/tong/Desktop/important/data/housing.csv", sep=",")
    california_housing_test_data = pd.read_csv("/Users/tong/Desktop/important/data/california_housing_test.csv",
                                               sep=",")

    processed_features = preprocess_features_fn(california_housing_dataframe)

    targets = preprocess_targets(california_housing_dataframe)

    training_examples, training_targets, validation_examples, validation_targets = split_data(processed_features,
                                                                                              targets)

    # minimal_features = ["latitude", "median_income"]
    #
    # assert minimal_features, "You must select at least one feature!"
    #
    # minimal_training_examples = training_examples[minimal_features]
    # minimal_validation_examples = validation_examples[minimal_features]

    regressor = train_model(
        learning_rate=1,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    processed_test_features = preprocess_features_fn(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    test_input_fn = lambda: linear_regrission_fn(processed_test_features, test_targets["median_house_value"],
                                                 num_epochs=1, shuffle=False)

    test_predictions = regressor.predict(test_input_fn)

    test_predictions = np.array([item["predictions"][0] for item in test_predictions])

    test_rmse = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
    print("Final RMSE (on test data): %0.2f" % test_rmse)

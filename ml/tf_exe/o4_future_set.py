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
from ml.tf_exe.o3_validation import preprocess_features_fn, preprocess_targets, split_data, train_model
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    california_housing_dataframe = pd.read_csv("/Users/tong/Desktop/important/data/housing.csv", sep=",")
    california_housing_test_data = pd.read_csv("/Users/tong/Desktop/important/data/california_housing_test.csv",
                                               sep=",")

    processed_features = preprocess_features_fn(california_housing_dataframe)

    targets = preprocess_targets(california_housing_dataframe)

    training_examples, training_targets, validation_examples, validation_targets = split_data(processed_features,
                                                                                              targets)

    correlation_excamples = training_examples.copy()
    # 通过皮尔逊相关矩阵选择最佳特征组合
    corr = correlation_excamples.corr()


    display.display(corr)

    corr.to_csv("corr.csv")

    minimal_features = ["latitude", "median_income"]

    assert minimal_features, "You must select at least one feature!"

    minimal_training_examples = training_examples[minimal_features]
    minimal_validation_examples = validation_examples[minimal_features]

    regressor = train_model(
        learning_rate=0.01,
        steps=500,
        batch_size=5,
        training_examples=minimal_training_examples,
        training_targets=training_targets,
        validation_examples=minimal_validation_examples,
        validation_targets=validation_targets)

    processed_test_features = preprocess_features_fn(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    test_input_fn = lambda: linear_regrission_fn(processed_test_features, test_targets["median_house_value"],
                                                 num_epochs=1, shuffle=False)

    test_predictions = regressor.predict(test_input_fn)

    test_predictions = np.array([item["predictions"][0] for item in test_predictions])

    test_rmse = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
    print("Final RMSE (on test data): %0.2f" % test_rmse)

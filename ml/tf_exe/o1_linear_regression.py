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

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("/Users/tong/Desktop/important/data/housing.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))


def linear_regrission_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels



def get_housing_data():
    california_housing_dataframe["median_house_value"] /= 1000.0
    # housing_median_age	total_rooms	total_bedrooms	population	households	median_income
    features = ["total_rooms", "housing_median_age", "total_bedrooms", "population", "households", "median_income"]
    my_feature = california_housing_dataframe[features]
    feature_columns = [tf.feature_column.numeric_column(f) for f in features]
    target = california_housing_dataframe["median_house_value"]
    return my_feature, target, feature_columns


def get_linear_regression():


    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    # TODO
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    model = tf.estimator.LinearRegressor(feature_columns, optimizer=optimizer)
    return model

def train_predict(model):
    model.train(input_fn=lambda: linear_regrission_fn(my_feature, target,batch_size=100,num_epochs=1000), steps=100)
    predict_regression_fn = lambda: linear_regrission_fn(my_feature, target, num_epochs=1, shuffle=False)
    predictions = model.predict(predict_regression_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    return predictions

if  __name__ == '__main__':


    my_feature, target, feature_columns = get_housing_data()
    model=get_linear_regression()
    predictions=train_predict(model)

    print("feature_columns",feature_columns)

    mean_squared_error = metrics.mean_squared_error(predictions, target)
    root_mean_squared_error = math.sqrt(mean_squared_error)

    median_house_value = california_housing_dataframe["median_house_value"]

    min_house_value = median_house_value.min()
    max_house_value = median_house_value.max()

    min_max_difference = max_house_value - min_house_value

    print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
    print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

    print("Min. Median House Value: %0.3f" % min_house_value)
    print("Max. Median House Value: %0.3f" % max_house_value)
    print("Difference between Min. and Max.: %0.3f" % min_max_difference)
    print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

    df = pd.DataFrame()
    df["predictions"] = pd.Series(predictions)
    df["target"] = pd.Series(target)

    print(df.describe())

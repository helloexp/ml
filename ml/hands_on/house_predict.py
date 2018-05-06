# -*- coding:utf-8 -*-

import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml.hands_on.transforms import CombinedAttributesAdder, DataFrameSelector

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test_func(data, test_ratio):
    permutation = np.random.permutation(len(data))

    test_data_size = int(len(data) * test_ratio)

    test_data_index = permutation[:test_data_size]
    train_index = permutation[test_data_size:]

    return data.iloc[train_index], data.iloc[test_data_index]


#
# def test_set_check(identify,test_ratio,hash_func):
#     digest = hash_func(np.int64(identify)).hexdigest()
#     print digest
#     return digest[-1] < 256 * test_ratio
#
#
# def split_train_data_by_id(data,test_ratio,id_column,hash=hashlib.sha1):
#
#     ids=data[id_column]
#     in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio, hash))
#     return data.loc[~in_test_set], data.loc[in_test_set]


def strat_sampling(housing):
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    # print housing["income_cat"]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    return strat_train_set, strat_test_set


def data_insight(data):
    housing = data.copy()
    population_ = housing["population"] / 100
    jet = plt.get_cmap("jet")
    housing.plot(kind="scatter", x="latitude", y="longitude", alpha=0.4, s=population_,
                 label="population", c="median_house_value", cmap=jet, colorbar=True)
    plt.legend()
    plt.show()


def dataframe_test():
    pd.DataFrame.plot()


def data_correlation(data):
    corr = data.corr()

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    # pd.scatter_matrix(housing[attributes], figsize=(12, 8))

    # plt.show()

    # print corr["median_house_value"].sort_values()


def attribuate_combation(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    data_correlation(housing)


def render_nan_value(data):
    data_num = data.drop(["ocean_proximity"], axis=1)
    imputer = Imputer(strategy="median")

    imputer.fit(data_num)

    # print housing.median().values
    #
    # print imputer.statistics_

    X = imputer.transform(data_num)

    housing_tr = pd.DataFrame(X, columns=data_num.columns)

    return housing_tr


def convert_text_to_num_value(data):
    encoder = LabelEncoder()
    return encoder.fit_transform(data)


def one_hot_encode(housing_ocean):
    encoder = OneHotEncoder()
    housing_one_hot = encoder.fit_transform(housing_ocean.reshape((-1, 1)))
    return housing_one_hot


def conver_text_to_onehot(data):
    binarizer = LabelBinarizer(sparse_output=True)
    return binarizer.fit_transform(data)


def housing_num_pipeline(num_attribs):
    num_pipline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ("imputer", Imputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler())])

    return num_pipline


def text_pipeline(cat_attribs):
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer())
    ])

    return cat_pipeline


def train(strat_train_set, model):
    housing_prepared, housing_labels = X_Y(strat_train_set)

    model.fit(housing_prepared, housing_labels)


def X_Y(strat_train_set):
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    num_pipeline = housing_num_pipeline(num_attribs)
    cat_pipeline = text_pipeline(["ocean_proximity"])
    feature_union = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline)])
    housing_prepared = feature_union.fit_transform(housing)
    return housing_prepared, housing_labels


def fit_predict(model,housing_prepared,housing_labels,test_x):
    model.fit(housing_prepared, housing_labels)
    predict_label = model.predict(test_x)

    return predict_label


if __name__ == '__main__':
    housing = load_housing_data()

    # print house.head()
    # print housing.info()
    # print housing["ocean_proximity"].value_counts()
    # print housing["ocean_proximity"].count()

    # print housing.describe()
    # housing.hist(bins=50,figsize=(20,15))
    # plt.show()

    # train_data,test_data = split_train_test(housing, 0.2)
    # print len(train_data)
    # print len(test_data)

    # house_with_index=housing.reset_index()

    # train_data, test_data=train_test_split(house_with_index, test_size=0.33, random_state=42)

    strat_train_set, strat_test_set = strat_sampling(housing)

    # housing = strat_train_set.drop("median_house_value", axis=1)
    # housing_labels = strat_train_set["median_house_value"].copy()

    # data_insight(strat_train_set)

    # data_correlation(housing)

    # attribuate_combation(housing)

    # housing_tr=render_nan_value(housing)

    # convert in one encoder
    # housing_ocean = convert_text_to_num_value(housing["ocean_proximity"])
    # housing_one_hot=one_hot_encode(housing_ocean)
    #
    # print housing_one_hot[1]
    # housing_one_hot = conver_text_to_onehot(housing["ocean_proximity"])
    # print housing_one_hot[1]

    # housing_num = housing.drop("ocean_proximity", axis=1)
    # print housing.head()

    # num_attribs = list(housing_num)

    # num_pipeline = housing_num_pipeline(num_attribs)

    # housing_num_tr = num_pipeline.fit_transform(housing_num)

    # cat_pipeline = text_pipeline(["ocean_proximity"])
    # housing_cat_tr = cat_pipeline.fit_transform(housing)

    # feature_union = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline)])


    # print "housing.shape",housing.shape
    # housing_prepared = feature_union.fit_transform(housing)
    # print "housing_prepared",housing_prepared.shape

    linear_regression = LinearRegression()
    decision_tree_regressor=DecisionTreeRegressor()
    random_forest_regressor=RandomForestRegressor()

    test_x, test_label = X_Y(strat_test_set)
    housing_prepared, housing_labels = X_Y(strat_train_set)


    # train(strat_train_set,linear_regression)
    linear_predict = fit_predict(linear_regression, housing_prepared, housing_labels, test_x)
    error = np.sqrt(mean_squared_error(test_label, linear_predict))
    # print "linear",zip(test_label,linear_predict)


    descision_predict = fit_predict(decision_tree_regressor, housing_prepared, housing_labels, test_x)
    # print "descision_predict",zip(test_label,descision_predict)
    error = np.sqrt(mean_squared_error(test_label, descision_predict))

    # random_forest_predict = fit_predict(random_forest_regressor, housing_prepared, housing_labels, test_x)
    # print "random_forest_predict",zip(test_label,random_forest_predict)

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search_cv = GridSearchCV(random_forest_regressor, param_grid, cv=5, scoring='neg_mean_squared_error',n_jobs=4)

    gred_cv_predict = fit_predict(grid_search_cv, housing_prepared, housing_labels, test_x)
    # print "gred_cv_predict",zip(test_label,gred_cv_predict)
    error = np.sqrt(mean_squared_error(test_label, gred_cv_predict))




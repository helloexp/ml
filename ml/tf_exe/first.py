# coding=utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = tf.constant([5, 2])
print(x)

y = tf.Variable([5, 6])
print(y)

y.assign([8, 9])
print(y)

g = tf.Graph()

with g.as_default():
    a = tf.constant([1], name="a")
    b = tf.constant([1], name="b")
    c = tf.constant([2], name="b")

    sum = tf.add(a, b, name="sum")
    sum2 = tf.add(sum, c, name="sum")

    with tf.Session() as ss:
        print(sum.eval())
        print(sum2.eval())

graph = tf.Graph()

with graph.as_default():
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    ones = tf.ones([6], dtype=tf.int32)

    just_beyond_primes = tf.add(primes, ones)

    twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
    beyond_doubled = twos * just_beyond_primes

    some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)

    one = tf.constant(1, dtype=tf.int32)
    brocast = tf.add(primes, one)

    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])

    a_reshaped = tf.reshape(a, [2, 3])
    b_reshaped = tf.reshape(b, [3, 1])

    a_b_matmul = tf.matmul(a_reshaped, b_reshaped)

    dice1 = tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
    dice2 = tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
    add_dices = tf.add(dice1, dice2)
    concat = tf.concat(values=[dice1, dice2, add_dices], axis=1)

    with tf.Session() as ss:
        print("primes", primes.eval())
        print("ones", ones.eval())
        print("just_beyond_primes", just_beyond_primes.eval())
        print("beyond_doubled", beyond_doubled.eval())
        print("some_matrix", some_matrix.eval())
        print("brocast", brocast.eval())
        print("a_b_matmul", a_b_matmul.eval())
        print("concat", concat.eval())

series1 = pd.Series([1, 2, 3, 4])

series2 = pd.Series([2, 3, 4, 5])

df = pd.DataFrame({"s1": series1, "s2": series2})

print(df)

california_housing_dataframe = pd.read_csv("/Users/tong/Desktop/important/data/housing.csv", sep=",")
print(california_housing_dataframe.describe())

hist = california_housing_dataframe.hist("housing_median_age")

# plt.show()

print(california_housing_dataframe["median_house_value"])
print(california_housing_dataframe["median_house_value"][0:10])


city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })

print("population / 10", (population / 10))
print("np.log(serise)",np.log(population))

population_apply = population.apply(lambda a: a > 1000000)
print("population_apply",population_apply)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']


cities["isSan"]=(cities["Area square miles"]>50) & cities["City name"].apply(lambda a:a.startswith("San"))

print(cities)

print("city_names.index",city_names.index)
print(cities.reindex([2, 0, 1,5]))

print(cities.reindex(np.random.permutation(cities.index)))




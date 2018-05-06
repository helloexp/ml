
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras

from nn.mytf import generate_data

graph=tf.Graph()

with graph.as_default():
    i=tf.placeholder(tf.float32,shape=(None,1),name="i")
    a=tf.Variable(tf.constant(10.0,shape=[1]),name="a")
    b=tf.Variable(tf.constant(20.0,shape=[1]),name="b")

    n1=tf.add(i,a,name="n1")
    n2=tf.add(i,b,name="n2")
    n3=tf.add(n1,n2,name="n3")


# with tf.Session(graph=graph) as sess:
#     sess.run(tf.global_variables_initializer())
#
#     input_Data = np.array([2]).reshape(-1, 1)
#
#     output = sess.run(n3, feed_dict={i: input_Data})
#
#     print output


#print graph.as_graph_def()

x_train, y_train = generate_data(1, 0)
# plt.scatter(x_train, y_train)
# plt.title("linear_regression")
#
# plt.show()
#
def linear_regression_model():
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(
        input_dim=1,
        units=1,
        name="dense"
    ))

    model.compile(loss="mse",optimizer="sgd")

    return model



module=linear_regression_model()
module.fit(x_train,y_train,batch_size=50,epochs=100,verbose=0)

x_test=np.linspace(0,1,100)
np.random.shuffle(x_test)
y_test_keras=module.predict(x_test)

plt.title("keras_exe")

plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test_keras)

plt.show()










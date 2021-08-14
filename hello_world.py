import tensorflow as tf
import numpy as np

print(tf.__version__)

print("Loading data...\n")
data = np.loadtxt("./data/mnist.csv", delimiter=",")
print("MNIST dataset loaded.\n")

# Train a feed forward neural network for image classification

x_train = data[:, 1:]
y_train = data[:, 0]
x_train = x_train/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

print(model.summary())

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Training model....\n")
model.fit(x_train, y_train, epochs=3, batch_size=32)

print("Model trained successfully...")
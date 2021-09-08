import tensorflow as tf
import pandas as pd
print(tf.__version__)
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D

model = Sequential([
    Conv1D(filters=16, kernel_size=3, input_shape=(128, 64), kernel_initializer="random_uniform", bias_initializer="zeros", activation="relu"),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(64, kernel_initializer="he_uniform", bias_initializer="ones", activation="relu")
])

# Add more layers

model.add(Dense(64,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),
                activation="relu"),)
model.add(Dense(8,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),
                activation="relu"))


# Custom weight and bias initializers
def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))

print(model.summary())


#visualising the initialized weights and biases

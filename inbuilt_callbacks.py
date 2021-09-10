import tensorflow as tf
print(tf.__version__)

# Load the diabetes dataset

from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

# Save the input and target variables

from sklearn.model_selection import train_test_split

data = diabetes_dataset['data']
targets = diabetes_dataset['target']

# Split the data set into training and test sets

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

# Build the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = tf.keras.Sequential([
    Dense(128, activation="relu", input_shape=(train_data.shape[1],)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)
])

model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])

# Larning rate scheduler

# Define the learning rate schedule function

def lr_function(epoch, lr):
    if epoch % 2 == 0:
        return lr
    else:
        return lr + epoch/1000

history = model.fit(train_data, train_targets, epochs=10,
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_function, verbose=1)], verbose=False)

# Train the model with a difference schedule

history = model.fit(train_data, train_targets, epochs=10,
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda x:1/(3+5*x), verbose=1)], verbose=False)

# CSV logger

# Train the model with a CSV logger

history = model.fit(train_data, train_targets, epochs=10, callbacks=[tf.keras.callbacks.CSVLogger("results.csv")],
                    verbose=False)

# Load the CSV

import pandas as pd

pd.read_csv("results.csv", index_col='epoch')

# Print the epoch number at the beginning of each epoch

epoch_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=lambda epoch,logs: print('Starting Epoch {}!'.format(epoch+1)))

# Print the loss at the end of each batch

batch_loss_callback = tf.keras.callbacks.LambdaCallback(
    on_batch_end=lambda batch,logs: print('\n After batch {}, the loss is {:7.2f}.'.format(batch, logs['loss'])))

# Inform that training is finished

train_finish_callback = tf.keras.callbacks.LambdaCallback(
    on_train_end=lambda logs: print('Training finished!'))

# Train the model with the lambda callbacks

history = model.fit(train_data, train_targets, epochs=5, batch_size=100,
                    callbacks=[epoch_callback, batch_loss_callback,train_finish_callback], verbose=False)

# Train the model with the ReduceLROnPlateau callback

history = model.fit(train_data, train_targets, epochs=100, batch_size=100,
                    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="loss",factor=0.2, verbose=1)], verbose=False)
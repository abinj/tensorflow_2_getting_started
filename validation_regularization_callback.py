import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


diabetes_dataset = load_diabetes()
print(diabetes_dataset["DESCR"])

diabetes_dataset.keys()
data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]

# Normalize the target data, this will make clearer training curves
targets = (targets - targets.mean(axis=0)) / targets.std()

# Split the data into training and test sets
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)
print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)


# Train a feedforward neural network model
def get_model():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(train_data.shape[1],)),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1)
    ])
    return model


model = get_model()

print(model.summary())

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(train_data, train_targets, epochs=100, validation_split=0.15, batch_size=64, verbose=False)

# Evaluate model on test set
model.evaluate(test_data, test_targets, verbose=2)

# plot learning curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs epcohs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# Model regularization
def get_regularized_model(wd, rate):
    model = Sequential([
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu", input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dense(1)
    ])
    return model


# Re-build the model with weight decay and dropout layers
model = get_regularized_model(1e-5, 0.3)

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model with validation set
history = model.fit(train_data, train_targets, epochs=100, validation_split=0.15, batch_size=64, verbose=False)

# Evaluate the model on test set
model.evaluate(test_data, test_targets, verbose=2)

# Plot the learning curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# Callbacks
class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Starting Training....")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training: Starting batch {batch}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: finished batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")

    def on_train_end(self, logs=None):
        print("Training End")


model = get_regularized_model(1e-5, 0.3)
model.compile(optimizer="adam", loss="mse")
model.fit(train_data, train_targets, epochs=3, batch_size=128, verbose=False, callbacks=[TrainingCallback()])
model.evaluate(test_data, test_targets, verbose=False)
prediction_out = model.predict(test_data, verbose=False)



# Early stopping and patience

# Re-train the unregularised model

unregularised_model = get_model()
unregularised_model.compile(optimizer="adam", loss="mse")
unreg_history = unregularised_model.fit(train_data, train_targets, epochs=100, validation_split=0.15,
                                        batch_size=64, verbose=False, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

# Evaluate the model on the test set

unregularised_model.evaluate(test_data, test_targets, verbose=2)

# Re-train the regularised model

regularised_model = get_regularized_model(1e-8, 0.2)
regularised_model.compile(optimizer="adam", loss="mse")
reg_history = regularised_model.fit(train_data, train_targets, epochs=100,
                                   validation_split=0.15, batch_size=64, verbose=False, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])


# Evaluate the model on the test set

regularised_model.evaluate(test_data, test_targets, verbose=2)

# Plot the training and validation loss


fig = plt.figure(figsize=(12, 5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title('Unregularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(122)

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()

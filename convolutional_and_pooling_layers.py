from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D

model = Sequential([
    Conv2D(16, kernel_size=3, padding='SAME', activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(64, activation='softmax')
])

print(model.summary())

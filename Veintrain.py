import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

X_train = np.load('C:/Users/yoges/Downloads/Processed_Vein_Data/X_train.npy')
y_train = np.load('C:/Users/yoges/Downloads/Processed_Vein_Data/y_train.npy')
X_val = np.load('C:/Users/yoges/Downloads/Processed_Vein_Data/X_val.npy')
y_val = np.load('C:/Users/yoges/Downloads/Processed_Vein_Data/y_val.npy')

vein_model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(y_train.shape[1], activation='softmax')
])

vein_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = vein_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

vein_model.save('vein_model.h5')

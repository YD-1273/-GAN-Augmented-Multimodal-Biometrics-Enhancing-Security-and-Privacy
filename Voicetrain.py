import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


X_train_voice = np.load('C:/Users/yoges/Downloads/Processed_Voice_Features/X_train_voice.npy')
y_train_voice = np.load('C:/Users/yoges/Downloads/Processed_Voice_Features/y_train_voice.npy')

X_train_voice = X_train_voice / np.max(np.abs(X_train_voice))

label_encoder = LabelEncoder()
y_train_voice = label_encoder.fit_transform(y_train_voice)

y_train_voice = to_categorical(y_train_voice)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_voice, y_train_voice, test_size=0.2, random_state=42)

voice_model = models.Sequential([
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

voice_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_voice = voice_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

voice_model.save('voice_model.h5')

print("Training complete.")

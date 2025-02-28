import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X_train_fused = np.load('C:/Users/yoges/Downloads/Processed_Fused_Features/X_train_fused.npy')
y_train = np.load('C:/Users/yoges/Downloads/Processed_Voice_Features/y_train_voice.npy')  # Example of using voice labels

num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)

X_train, X_val, y_train, y_val = train_test_split(X_train_fused, y_train, test_size=0.2, random_state=42)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

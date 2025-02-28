import os
import numpy as np
from sklearn.model_selection import train_test_split

X_train_voice = np.load('C:/Users/yoges/Downloads/Processed_Voice_Features/X_train_voice.npy')
y_train_voice = np.load('C:/Users/yoges/Downloads/Processed_Voice_Features/y_train_voice.npy')

X_train, X_val_voice, y_train, y_val_voice = train_test_split(X_train_voice, y_train_voice, test_size=0.2, random_state=42)

output_dir = 'C:/Users/yoges/Downloads/Processed_Voice_Features/'
np.save(os.path.join(output_dir, 'X_val_voice.npy'), X_val_voice)
np.save(os.path.join(output_dir, 'y_val_voice.npy'), y_val_voice)

print("Validation set for voice data created and saved.")

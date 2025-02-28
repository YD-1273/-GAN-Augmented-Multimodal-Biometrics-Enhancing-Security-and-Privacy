import numpy as np
import os

voice_features_path = 'C:/Users/yoges/Downloads/Processed_Voice_Features/X_train_voice.npy'
vein_features_path = 'C:/Users/yoges/Downloads/Processed_Vein_Data/X_train.npy'

X_train_voice = np.load(voice_features_path)
X_train_vein = np.load(vein_features_path)

num_samples = min(X_train_voice.shape[0], X_train_vein.shape[0])
X_train_voice = X_train_voice[:num_samples]
X_train_vein = X_train_vein[:num_samples]

X_train_voice = X_train_voice.reshape(num_samples, -1)
X_train_vein = X_train_vein.reshape(num_samples, -1)

X_train_fused = np.concatenate((X_train_voice, X_train_vein), axis=1)

output_dir = 'C:/Users/yoges/Downloads/Processed_Fused_Features'
os.makedirs(output_dir, exist_ok=True)
fused_features_path = os.path.join(output_dir, 'X_train_fused.npy')
np.save(fused_features_path, X_train_fused)

print(f"Feature fusion complete! The fused features are saved to {fused_features_path}.")
print(f"Shape of fused feature set: {X_train_fused.shape}")
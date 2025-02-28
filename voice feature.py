import librosa
import numpy as np
import os

def extract_mfcc(audio_path, sr=22050, n_mfcc=13, duration=3, target_shape=(224, 224)):
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    
    if mfcc.shape[1] < target_shape[1]:
        mfcc_resized = np.pad(mfcc, ((0, 0), (0, target_shape[1] - mfcc.shape[1])), mode='constant')
    else:
        mfcc_resized = mfcc[:, :target_shape[1]]
    
    mfcc_resized = np.resize(mfcc_resized, target_shape)

    return mfcc_resized

audio_dir = 'C:/Users/yoges/Downloads/Voice/'

def extract_features_from_directory(audio_dir, output_dir=None, target_shape=(224, 224)):
    X = []
    y = []
    
    for subdir, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(subdir, file)
                
                mfcc = extract_mfcc(audio_path, target_shape=target_shape)
                
                X.append(mfcc)
                
                label = os.path.basename(subdir)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    if output_dir:
        np.save(os.path.join(output_dir, 'X_train_voice.npy'), X)
        np.save(os.path.join(output_dir, 'y_train_voice.npy'), y)
    
    return X, y

output_dir = 'C:/Users/yoges/Downloads/Processed_Voice_Features/'
X_voice, y_voice = extract_features_from_directory(audio_dir, output_dir, target_shape=(224, 224))

print("Feature extraction completed and saved.")

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

base_dir = 'C:/Users/yoges/Downloads/Vein'
data = []
labels = []

for person_id in os.listdir(base_dir):
    person_path = os.path.join(base_dir, person_id)
    
    if not os.path.isdir(person_path):
        continue

    for hand in ['right', 'left']:
        hand_path = os.path.join(person_path, hand)
        
        if not os.path.isdir(hand_path):
            continue

        for img_file in os.listdir(hand_path):
            img_path = os.path.join(hand_path, img_file)
            
            if not img_file.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                continue
            
            finger_type = img_file.split('_')[0]
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            image = cv2.resize(image, (224, 224))
            
            image = image / 255.0
            
            data.append(image)
            labels.append(f"{person_id}_{hand}_{finger_type}")

data = np.array(data).reshape(-1, 224, 224, 1)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_val, y_train, y_val = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)

output_dir = 'C:/Users/yoges/Downloads/Processed_Vein_Data'
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

print("Data preparation complete!")
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
  
import numpy as np
import cv2
import os

cover_image_path = 'C:/Users/yoges/Downloads/cover_image.png'

if not os.path.exists(cover_image_path):
    raise FileNotFoundError(f"Cover image not found at {cover_image_path}")

cover_image = cv2.imread(cover_image_path)

if cover_image is None:
    raise ValueError(f"Failed to load image from {cover_image_path}. Please check the file integrity.")

fused_features_path = 'C:/Users/yoges/Downloads/Processed_Fused_Features/X_train_fused.npy'
fused_features = np.load(fused_features_path)

scaled_features = np.interp(fused_features.flatten(), (fused_features.min(), fused_features.max()), (0, 255))
scaled_features = scaled_features.astype(np.uint8)

image_size = cover_image.shape[0] * cover_image.shape[1] * 3
data_size = scaled_features.size

if image_size < data_size:
    required_pixels = scaled_features.size // 3
    print(f"Cover image size: {cover_image.shape[0]} x {cover_image.shape[1]} = {cover_image.shape[0] * cover_image.shape[1]} pixels")
    print(f"Required pixels: {required_pixels}")
    
    required_height = (required_pixels // cover_image.shape[1]) + 1
    cover_image_resized = cv2.resize(cover_image, (cover_image.shape[1], required_height))
    cv2.imwrite('C:/Users/yoges/Downloads/cover_image_resized.png', cover_image_resized)
    cover_image = cover_image_resized

image_flat = cover_image.flatten()

for i in range(data_size):
    pixel_value = image_flat[i]
    image_flat[i] = (pixel_value & 0xFE) | (scaled_features[i] >> 7)

stego_image = image_flat.reshape(cover_image.shape)

output_dir = 'C:/Users/yoges/Downloads/Processed_Stego_Image'
os.makedirs(output_dir, exist_ok=True)
stego_image_path = os.path.join(output_dir, 'stego_image.png')
cv2.imwrite(stego_image_path, stego_image)

print(f"Stego image saved to {stego_image_path}.")
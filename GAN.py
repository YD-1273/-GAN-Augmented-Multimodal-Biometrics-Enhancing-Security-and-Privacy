import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scipy.linalg import sqrtm
import cv2

fused_features_path = 'C:/Users/yoges/Downloads/Processed_Fused_Features/X_train_fused.npy'
fused_features = np.load(fused_features_path)

scaled_features = np.interp(fused_features.flatten(), (fused_features.min(), fused_features.max()), (0, 255))
scaled_features = scaled_features.astype(np.uint8)

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Dense(fused_features.shape[1], activation='tanh')
    ])
    return model

def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(512, input_dim=input_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator(fused_features.shape[1])
gan = build_gan(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

def train_gan(epochs, batch_size, latent_dim):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        idx = np.random.randint(0, fused_features.shape[0], half_batch)
        real_samples = fused_features[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"{epoch}/{epochs} [D loss: {0.5 * (d_loss_real[0] + d_loss_fake[0])}] [G loss: {g_loss}]")

train_gan(epochs=20, batch_size=64, latent_dim=latent_dim)

def evaluate_discriminator_accuracy():
    real_labels = np.ones((fused_features.shape[0], 1))
    fake_labels = np.zeros((fused_features.shape[0], 1))

    real_predictions = discriminator.predict(fused_features)
    real_accuracy = np.mean(np.round(real_predictions) == real_labels)

    noise = np.random.normal(0, 1, (fused_features.shape[0], latent_dim))
    generated_samples = generator.predict(noise)

    fake_predictions = discriminator.predict(generated_samples)
    fake_accuracy = np.mean(np.round(fake_predictions) == fake_labels)

    discriminator_accuracy = (real_accuracy + fake_accuracy) / 2
    print(f"Discriminator Accuracy: {discriminator_accuracy * 100:.2f}%")

evaluate_discriminator_accuracy()

def resize_images(images, target_size):
    """Resizes the images to the target size and ensures each image has 3 channels."""
    resized_images = np.zeros((images.shape[0], target_size[0], target_size[1], 3), dtype=np.uint8)
    
    for i in range(images.shape[0]):
        image = images[i].astype(np.uint8)

        resized_image = cv2.resize(image, (target_size[1], target_size[0]))
        
        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        
        resized_images[i] = resized_image
    
    return resized_images

def calculate_fid(real_images, generated_images):
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    real_images_resized = resize_images(real_images, (299, 299))
    generated_images_resized = resize_images(generated_images, (299, 299))

    real_features = inception_model.predict(real_images_resized)
    generated_features = inception_model.predict(generated_images_resized)

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid_score

real_images = fused_features
noise = np.random.normal(0, 1, (fused_features.shape[0], latent_dim))
generated_images = generator.predict(noise)

fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score}")

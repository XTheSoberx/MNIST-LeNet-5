import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

noise_factor = 0.5
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
print (x_train.shape)
print (x_test.shape)
X_train = x_train.reshape(-1,784)
X_test = x_test.reshape(-1,784)
print (X_train.shape) 
print (X_test.shape)

#input_img = tf.keras.layers.Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
input_img = tf.keras.layers.Input(shape=(784,))
#x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
#x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
#x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.Dense(784, activation='relu')(input_img)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

#encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = tf.keras.layers.Dense(16,activation='relu')(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = tf.keras.layers.Dense(32, activation='relu')(encoded)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(x)
#x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = tf.keras.layers.UpSampling2D((2, 2))(x)
#x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = tf.keras.layers.UpSampling2D((2, 2))(x)
#x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
#x = tf.keras.layers.UpSampling2D((2, 2))(x)
#decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tb = tf.keras.callbacks.TensorBoard('./logs/MNIST-Papillon')

history = autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[tb])

decoded_imgs = autoencoder.predict(X_test)


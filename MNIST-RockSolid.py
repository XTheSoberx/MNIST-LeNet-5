import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import MNIST Dataset
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Split Dataset reshaping Labels and Targets
x_train = (x_train.reshape(60000,28,28,1).astype('float32'))/255
x_test = (x_test.reshape(10000,28,28,1).astype('float32'))/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the ROCKSOLID Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, kernel_size=(5,5), activation=tf.nn.relu, input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.175))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

# Compile the model with ADAM optimizer of loss categorical crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TensorBoard Analisis
#tensorboard --logdir=./logs --port 6006
tb = tf.keras.callbacks.TensorBoard('./logs/MNIST-RockSolid')

# Fit "RockSolid" Model
history = model.fit(x_train, y_train, epochs=15, verbose=1, validation_data=(x_test,y_test), callbacks=[tb])

#Evalutate and Save the Model than analize RockSteady performance with PLT 
V_loss, V_acc = model.evaluate(x_test, y_test)
print('[This model  accuracy=[', V_acc*100, "%]   loss=[", V_loss,"]]")
Nomefile = input('Type your RockSolid Model Name ')
Nomefile = Nomefile + '.h5'
model.save (Nomefile)
tf.keras.models.load_model (Nomefile)
print(Nomefile,' Model saved')
V_loss, V_acc = model.evaluate(x_test, y_test)
print('RockSolid [',Nomefile, '] accuracy=[', V_acc*100, "%]   loss=[", V_loss,"]")

plt.figure(figsize=(40, 20))

plt.subplot(2,2,1)
plt.title('Accuracy ' + Nomefile)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

plt.subplot(2,2,2)
plt.title('Loss ' + Nomefile)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

plt.subplot(2,2,3)
plt.title('y_test Confusion Matrix')
label = np.argmax(model.predict(x_test), axis=1)
target = np.argmax(y_test, axis=1)
confmat = confusion_matrix(target, label)
sns.heatmap(confmat, annot=True, cmap='tab20c',fmt='d',linewidths=.5)

plt.show()
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


#Leggi e processa i dati del dataset MNIST
(X_train,y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')
X_train /= 255
X_test /= 255
n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

#Crea architettura rete neurale LeNet-5
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

#Compila modello
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Imposta i dati log per analisi visiva su TensorBoard
tensor_board = tf.keras.callbacks.TensorBoard('./logs/MNIST-LeNet-5')

#Apprendimento modello
history = model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1, validation_data=(X_test,y_test), callbacks=[tensor_board])

#Invia "tensorboard --logdir=./logs --port 6006" su terminale per analisi visiva su TensorBoard

#Salvataggio lettura e valutazione modello
V_loss, V_acc = model.evaluate(X_test, y_test)
print('[This model  accuracy=[', V_acc*100, "%]   loss=[", V_loss,"]]")
Nomefile = input('inserire il nome del file dove verrà salvato il modello ')
Nomefile = Nomefile + '.h5'
model.save (Nomefile)
tf.keras.models.load_model (Nomefile)
print(Nomefile,' salvato correttamente')
print('modello [',Nomefile, '] accuracy=[', V_acc*100, "%]   loss=[", V_loss,"]")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
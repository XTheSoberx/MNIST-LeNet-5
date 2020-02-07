import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train=x_train/255
x_test=x_test/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.LSTM(128),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(32, activation =tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

tb = tf.keras.callbacks.TensorBoard(".\logs\MNIST-RNN")

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), callbacks=[tb])

# Evalutate and Save the Model than analize RockSteady performance with PLT 
V_loss, V_acc = model.evaluate(x_test, y_test)
print('[This model  accuracy=[', V_acc*100, "%]   loss=[", V_loss,"]]")
Nomefile = input('Type your RockSolid Model Name ')
Nomefile = Nomefile + '.h5'
model.save (Nomefile)
print(Nomefile,' Model saved')
print('RockSolid [',Nomefile, '] accuracy=[', V_acc*100, "%]   loss=[", V_loss,"]")

# Plot grid unit
plt.figure(figsize=(16, 100))

# Plot Accuracy
plt.subplot2grid((10, 20),(0, 0), colspan=9, rowspan=4)
plt.title('Accuracy ' + Nomefile)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

# Plot Loss
plt.subplot2grid((10, 20), (0, 10), colspan=9, rowspan=4)
plt.title('Loss ' + Nomefile)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

# Plot Confusion Matrix
plt.subplot2grid((10, 20), (5, 0), colspan=9, rowspan=4)
plt.title('y-test Confusion Matrix')
label = np.argmax(model.predict(x_test), axis=1)
target = np.argmax(y_test, axis=1)
confmat = confusion_matrix(target, label)
sns.heatmap(confmat, annot=True, cmap='Blues',fmt='d',linewidths=.5,vmin=0,vmax=10)

# Plot 50 prediction errors
predicted_classes = model.predict_classes(x_test)
incorrect_indices = np.nonzero(label != target)[0]
j=1
r=1
for i, incorrect in enumerate(incorrect_indices[:49]):
    j=(i+1)//10
    r=(i+1)%10
    if i:
        plt.subplot2grid((10,20),(5,10))
        plt.imshow(x_test[incorrect].reshape(28,28), cmap='Greys', interpolation='none')
        plt.title("P:{},T:{}".format(predicted_classes[incorrect],target[incorrect]))
        plt.xticks([])
        plt.yticks([])
    plt.subplot2grid((10,20),(j+5,r+10))
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='Greys', interpolation='none')
    plt.title("P:{},T:{}".format(predicted_classes[incorrect],target[incorrect]))
    plt.xticks([])
    plt.yticks([])
plt.show()
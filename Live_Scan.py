import cv2
import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model('MRS_9954_0021.h5')
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = np.asarray(frame,dtype='uint8')
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    gray = cv2.resize(gray, (28,28))
    gray = np.array(gray)
    gray = gray.flatten()
    gray = (gray.reshape(1, 28,28,1).astype('float32'))/255
    ans = loaded_model.predict(gray)
    ans = np.argmax(ans)

    cv2.putText(frame, " RockSolid : " + str(ans), (10, 330),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Press q to exit',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()
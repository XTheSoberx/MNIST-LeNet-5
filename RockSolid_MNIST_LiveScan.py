import cv2
import tensorflow as tf
import numpy as np

# Load model from file and set video 
loaded_model = tf.keras.models.load_model('MRS_9956_0021.h5')
cap = cv2.VideoCapture(0)
cap.set(4, 5*128)

# Define function for label a section of image frame with its own label
def frame_label(frame, label, location = (20,30)):
    cv2.putText(frame, label, location, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color = (0, 0, 255),
                thickness =  1,
                lineType =  cv2.LINE_AA)

# Define function for select and resize a rectangle of frame image
def cut_digit(frame, rect, pad = 10):
    x, y, w, h = rect
    gray_digit = gray[y-pad:y+h+pad, x-pad:x+w+pad]
    gray_digit = gray_digit/255.0
    if gray_digit.shape[0] >= 32 and gray_digit.shape[1] >= 32:
       gray_digit = cv2.resize(gray_digit, (28, 28))
    else:
        return
    return gray_digit

# Capture frame-by-frame
for i in range(10000):
    ret, frame = cap.read()
    frame = np.asarray(frame,dtype='uint8')

# Convert and normalize frame in grayscale and extract x_test
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    aprx, contours, hierarchy = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]
    for rect in rects:
        x, y, w, h = rect
        if i >= 0:
            gray_frame = cut_digit(frame, rect, pad = 15)
            if gray_frame is not None:
               gray_frame = np.expand_dims(gray_frame, 0)
               gray_frame = np.expand_dims(gray_frame, 3)
               
# Predict y_test from model and x_test
               label = loaded_model.predict(gray_frame)
               label = str(np.argmax(label))
               cv2.rectangle(frame, (x - 15, y - 15), (x + 15 + w, y + 15 + h),
                             color = (0, 0, 255))
               frame_label(frame, label, location = (rect[0], rect[1]))
    cv2.imshow('RockSolid_MNIST_LiveScan -->Press q to quit<--', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
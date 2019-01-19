import cv2
import numpy as np
import tensorflow as tf

#font = cv2.FONT_HERSHEY_SIMPLEX
cp = cv2.VideoCapture(0)
cp.set(4, 5*128)
SIZE = 28
img_rows, img_cols = 28, 28
input_shape = (28, 28, 1)
first_dim = 0
second_dim = 3

def annotate(frame, label, location = (20,30)):
    #writes label on image#
    cv2.putText(frame, label, location, CHAIN_APPROX_SIMPLE,
                fontScale = 0.5,
                color = (0, 0, 255),
                thickness =  1,
                lineType =  cv2.LINE_AA)
def extract_digit(frame, rect, pad = 10):
    x, y, w, h = rect
    cropped_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad]
    cropped_digit = cropped_digit/255.0
    #only look at images that are somewhat big:
    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        cropped_digit = cv2.resize(cropped_digit, (28, 28))
    else:
        return
    return cropped_digit
#def img_to_mnist(frame, tresh = 90):
    #gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    ##adaptive here does better with variable lighting:
    #gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     #cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    #return gray_img
model= tf.keras.models.load_model('MRS_9954_0021.h5')
labelz = dict(enumerate(["0", "1", "2", "3", "4",
                         "5", "6", "7", "8", "9"]))
for i in range(1000):
    ret, frame = cp.read(0)
    #final_img = img_to_mnist(frame)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    final_img = gray_img
    image_shown = frame
    _, contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]
    #draw rectangles and predict:
    for rect in rects:
        x, y, w, h = rect
        if i >= 0:
            mnist_frame = extract_digit(frame, rect, pad = 15)
            if mnist_frame is not None: #and i % 25 == 0:
                mnist_frame = np.expand_dims(mnist_frame, first_dim) #needed for keras
                mnist_frame = np.expand_dims(mnist_frame, second_dim) #needed for keras
                class_prediction = model.predict_classes(mnist_frame, verbose = False)[0]
                prediction = np.around(np.max(model.predict(mnist_frame, verbose = False)), 2)
                label = str(prediction) # if you want probabilities
                cv2.rectangle(image_shown, (x - 15, y - 15), (x + 15 + w, y + 15 + h),
                              color = (0, 0, 255))
                label = labelz[class_prediction]
                annotate(image_shown, label, location = (rect[0], rect[1]))
    cv2.imshow('frame', image_shown)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
import tensorflow as tf
import cv2
import numpy as np
import math


def process(img_input):

    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)


    (thresh, img_binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


    h,w = img_binary.shape

  
    ratio = 100/h
    new_h = 100
    new_w = w * ratio

    img_empty = np.zeros((110,110), dtype=img_binary.dtype)
    img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    img_empty[:img_binary.shape[0], :img_binary.shape[1]] = img_binary

    img_binary = img_empty


    cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 컨투어의 무게중심 좌표를 구합니다. 
    M = cv2.moments(cnts[0][0])
    center_x = (M["m10"] / M["m00"])
    center_y = (M["m01"] / M["m00"])

    # 무게 중심이 이미지 중심으로 오도록 이동시킵니다. 
    height,width = img_binary.shape[:2]
    shiftx = width/2-center_x
    shifty = height/2-center_y

    Translation_Matrix = np.float32([[1, 0, shiftx],[0, 1, shifty]])
    img_binary = cv2.warpAffine(img_binary, Translation_Matrix, (width,height))


    img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
    flatten = img_binary.flatten() / 255.0

    return flatten



model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.load_weights('mnist_checkpoint')


cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while(True):

    ret, frame = cap.read()

    if ret == False:
        break;
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([150, 50, 50])
    upper_red = np.array([180,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    cv2.imshow("VideoFrame", frame)
    cv2.imshow("Red", mask)
    
    key=cv2.waitKey(1)
    if key == ord('w'):
        img_captured = cv2.imwrite("img_captured.png",mask)
        break
        
    if key == ord('q'):
        break

img = cv2.imread("img_captured.png")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(imthres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 30 and h > 30:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 1)
        x1 = x
        y1 = y
        w_t = w
        h_t = h
        
Y = y1 - 5  # 잘라진 이미지 y 시작 좌표
H = h_t + 10  # 잘라진 이미지 height 길이
X = x1 - 5  # 잘라진 이미지 x 시작 좌표
W = w_t + 10  # 잘라진 이미지 width 길이

# cv2.imshow("result",img)

cut_img = img[(Y):(Y+H), (X):(X+W)]
out = cut_img.copy()
out = 255 - out
cv2.imshow("result1",out)


flatten = process(out)

predictions = model.predict(flatten[np.newaxis,:])

with tf.compat.v1.Session() as sess:
    print(tf.argmax(predictions, 1).eval())

# cv2.imshow('img_roi', out)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
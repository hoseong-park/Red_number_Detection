import cv2
import numpy as np

capture = cv2.VideoCapture(cv2.CAP_DSHOW+0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    
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

cv2.imshow("result",img)

cut_img = img[(Y):(Y+H), (X):(X+W)]
out = cut_img.copy()
out = 255 - out
cv2.imshow("result1",out)

cv2.waitKey(0)

capture.release()
cv2.destroyAllWindows()
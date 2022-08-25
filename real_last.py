import tensorflow as tf
import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Firebase 연결

cred = credentials.Certificate('arec-522ed-firebase-adminsdk-g5fpo-edc5f51016.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://arec-522ed-default-rtdb.firebaseio.com/'
})

# MNIST 이용한 숫자인식
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

# 학습모델
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.load_weights('mnist_checkpoint')

cap = cv2.VideoCapture(0,cv2.CAP_V4L) # Jetson nano 에서 사용할 때
# cap = cv2.VideoCapture(cv2.CAP_DSHOW+0) # 컴퓨터에서 사용할 때
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lst = [] # 인식숫자 저장 리스트
recognize_num = 0 # 인식횟수
wrong_answer = 0 # 틀린횟수
initial = datetime.now() # 초기시간 정의


# 카메라 연결 및 이미지 식별

while(True):
    Number_det = 0 # 숫자라고 인식했는지 0: 숫자인식 X , 1: 숫자인식 O
    
    ret, frame = cap.read()

    if ret == False:
        break;
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([150, 50, 50])
    upper_red = np.array([180,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    #cv2.imshow("VideoFrame", frame) # 이미지 출력
    #cv2.imshow("Red", mask) # 이미지 출력
    cv2.waitKey(1)
    img_captured = cv2.imwrite("img_captured!.png",mask) # 빨간색 검출 이미지 저장

    img = cv2.imread("img_captured!.png")
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, imthres = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(imthres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 컨투어 이미지 생성
    os.remove("./img_captured!.png")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 30: # 가로 세로 30 픽셀 이상인 것이 인식 됐다면(숫자인식)
            Number_det += 1
            
            if Number_det == 1:
                x1 = x
                y1 = y
                w_t = w
                h_t = h
                Y = y1 - 5  # 잘라진 이미지 y 시작 좌표
                H = h_t + 10  # 잘라진 이미지 height 길이
                X = x1 - 5  # 잘라진 이미지 x 시작 좌표
                W = w_t + 10  # 잘라진 이미지 width 길이
                
                if X<0 or X>630 or Y<0 or Y>470: # 좌표가 사진크기에 벗어나면 다시 재정의
                    break;

                cut_img = img[(Y):(Y+H), (X):(X+W)]
                out = cut_img.copy()
                out = 255 - out # 자른 이미지 색 반전

                #cv2.imshow("result1",out) # 자른 이미지 출력

                flatten = process(out) 

                predictions = model.predict(flatten[np.newaxis,:]) # 예측한 숫자

                with tf.compat.v1.Session() as sess:

                    now = datetime.now() # 현재시간 기록
                    diff = now - initial # 초기시간과 현재시간의 차이
                    
                    if diff.seconds > 2: # 2초 마다 인식 숫자 저장
                        lst.extend(tf.argmax(predictions, 1).eval()) # 인식한 숫자 리스트에 이어 붙이기
                        print("저장!", tf.argmax(predictions, 1).eval())
                        print(lst)
                        dir = db.reference()
                        dir.update({'상황':'정답 판별 중'})
                        
                        solution = ' '.join(str(s) for s in lst)
                        dir = db.reference('Detecting')                        
                        dir.update({'인식숫자': solution})
                        initial = datetime.now() # 초기시간 재정의
                        recognize_num += 1
                        
                    # 5회 인식 되어 정답 판단
                    if recognize_num == 5: 
                        if lst[0]==lst[1]==lst[2]==lst[3]==lst[4]: 
                            print("정답", tf.argmax(predictions, 1).eval(),"미션종료")
                            Bingo = lst[0]
                            Bingo.tolist()
                            Answer = int(Bingo)
                            dir = db.reference('Answer')
                            dir.update({'정답': Answer}) # 정답 출력
                            lst = [] # 저장값 초기화
                            recognize_num = 0 # 미션수행 반복을 위해 인식횟수 초기화
                            break;
                        
                        # 5번 동안 인식한 숫자가 일치하지 않을 때
                        else: 
                            wrong_answer += 1 # 틀린횟수 + 1
                            print("인식한 숫자 오답")
                            dir = db.reference()
                            dir.update({'상황':'정답 못찾음'})
                            dir = db.reference()
                            dir.update({'틀린 횟수': wrong_answer})
                            if wrong_answer == 3:
                                dir = db.reference()
                                dir.update({'실패':'숫자 인식 중 이라면 육안식별 요망, 틀린횟수 초기화'})
                                wrong_answer = 0 # 틀린횟수 초기화
                            
                            recognize_num = 0
                            lst = []
                            initial = datetime.now()
                            break;
    
    
    # 숫자로 인식한 것이 없는 상황
    
    if Number_det == 0: 
        nnow = datetime.now()
        ndiff = nnow - initial
        if ndiff.seconds > 2:
            print("글씨로 인식한게 없음")
            dir = db.reference()
            dir.update({'상황':'인식된 글씨가 없음'})
            recognize_num = 0
            lst = [] # 저장 숫자 리스트 초기화
            initial = datetime.now()

cap.release()
cv2.destroyAllWindows()

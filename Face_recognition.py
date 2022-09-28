import cv2
import numpy as np
import math

# id_ 수정하기 (우리 dataset id랑 체계가 다름)

labels = ["s1", "s2", "s3", "Ko Taehyeon"]

base_dir = './IMG/train/'

face_classifier = cv2.CascadeClassifier('C:\HORUS\IMG_Preprocessing\FOR_OFFLINE\Cascades\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\HORUS\IMG_Preprocessing\FOR_OFFLINE\model_2022-09-27 01_36_33.098850.xml')

cap = cv2.VideoCapture(0)

if cap.isOpened() == False : # 카메라 생성 확인
    exit()
    
while True:
    ret, img = cap.read() # Realtime으로 촬영되고 있는 사진 get
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.5, 5)
    
    for (x, y, w, h) in faces :
        ROI_gray = gray[y:y+h, x:x+w] # 앞 과정에서 get한 img에서 얼굴 부분만 따로 get
        id_, conf = recognizer.predict(ROI_gray) # 얼굴 유사도 확인
        print(id_, conf)
        
        if conf>=50:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_-1]
            cv2.putText(img, f'{name}, coef:{math.ceil(conf)}%', (x,y), font, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        
    cv2.imshow('Sample', img)
    if cv2.waitKey(10) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
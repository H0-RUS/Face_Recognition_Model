import cv2
import math

# id_ 수정하기 (우리 dataset id랑 체계가 다름)

labels = ["kth", "ang"]

base_dir = './IMG/train/'

face_classifier = cv2.CascadeClassifier('C:\HORUS\Face_RM\Cascades\haarcascade_frontalface_default.xml')
original_recognizer = cv2.face.LBPHFaceRecognizer_create()
masked_recognizer = cv2.face.LBPHFaceRecognizer_create()
original_recognizer.read('C:\HORUS\Face_RM\Model\original_LBP_model.xml')
masked_recognizer.read('C:\HORUS\Face_RM\Model\masked_LBP_model.xml')

cap = cv2.VideoCapture(0)

if cap.isOpened() == False : # 카메라 생성 확인
    exit()
    
while True:
    ret, img = cap.read() # Realtime으로 촬영되고 있는 사진 get
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.5, 5)
    
    for (x, y, w, h) in faces :
        ROI_gray = gray[y:y+h, x:x+w] # 앞 과정에서 get한 img에서 얼굴 부분만 따로 get
        id_origin, conf_origin = original_recognizer.predict(ROI_gray) # 얼굴 유사도 확인
        id_masked, conf_masked = masked_recognizer.predict(ROI_gray)
        print(id_origin, conf_origin, conf_masked)
        
        if conf_origin>50 and conf_origin > conf_masked:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_origin-1]
            cv2.putText(img, f'{name}, coef:{math.ceil(conf_origin)}%', (x,y), font, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        elif conf_masked>50 and conf_masked > conf_origin:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_masked-1]
            cv2.putText(img, f'{name}, coef:{math.ceil(conf_masked)}%', (x,y), font, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow('Sample', img)
    if cv2.waitKey(10) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
# 무조건 Google Colabotory에서만 실행해야 함!

import cv2
import numpy as np
import os
import glob

base_dir = './IMG/train/'
train_data, train_labels = [], []
recognizer = cv2.face.LBPHFaceRecognizer_create()

dirs = [d for d in glob.glob(base_dir+'/*') if os.path.isdir(d)]

print('Collecting train data set...')

for dir in dirs:
    id = dir.split('_')[1]
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %dfiles'%(dir, len(files)))
    
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(img, dtype=np.uint8))
        train_labels.append(int(id))
        
train_data = np.asarray(train_data)
train_labels = np.int32(train_labels)

print('Starting LBP Model training...')
model = recognizer
model.train(train_data, train_labels)
model.write('./IMG/train/all_face.xml')
print("Model trained successfully!")
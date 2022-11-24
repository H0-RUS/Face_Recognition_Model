# 무조건 Google Colabotory에서만 실행해야 함!

import cv2
import numpy as np
import os
import glob

def RMmaker(base_dir, name):
    train_data, train_labels = [], []
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    dirs = [d for d in glob.glob(base_dir+'/*') if os.path.isdir(d)]
    print(dirs)
    print('Collecting train data set...')

    for dir in dirs:
        id = int(dir.split('/')[4].split('\\')[1].split('_')[1])
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
    model.write('C:/HORUS/Face_RM/Model/'+name+'_LBP_model.xml')
    print("Model trained successfully!")

RMmaker('C:/HORUS/Face_RM/IMG/masked', 'masked')
RMmaker('C:/HORUS/Face_RM/IMG/original', 'original')
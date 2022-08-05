import os
import cv2
import numpy as np
import random

DATADIR = './data/characters_train_set/'
CATEGORIES = os.listdir(DATADIR)


IMG_SIZE = 28

## PREPROCESSNIG NEEDS
shiftRight = np.float32([
	[1, 0, 1],
	[0, 1, 0]
])
shiftLeft = np.float32([
	[1, 0, -1],
	[0, 1, 0]
])
shiftUp = np.float32([
  [1, 0, 0],
  [0, 1, -1]
])
shiftDown = np.float32([
  [1, 0, 0],
  [0, 1, 1]
])

save_dir = './augmentedData'

training_data = []
def create_training_data():
    os.mkdir(save_dir)
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        os.mkdir(save_dir + '/' + category)
        save_path = os.path.join(save_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
              img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
              new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

              

              # ROTATING
              (h, w) = new_array.shape[:2]
              (cX, cY) = (w // 2, h // 2) # finding midpoints of image to rotate from 
              #rotate 5 clockwise
              rotateClock5 = cv2.getRotationMatrix2D((cX, cY), 5, 1.0) # getting the rotation matrix
              rotatedClock5 = cv2.warpAffine(new_array, rotateClock5, (w, h)) # applying the rotation matrix
              #rotate 10 clockwise
              rotateClock10 = cv2.getRotationMatrix2D((cX, cY), 10, 1.0) # getting the rotation matrix
              rotatedClock10 = cv2.warpAffine(new_array, rotateClock10, (w, h)) # applying the rotation matrix
              #rotate 5 counterClock
              rotateCounterClock5 = cv2.getRotationMatrix2D((cX, cY), -5, 1.0) # getting the rotation matrix
              rotatedCounterClock5 = cv2.warpAffine(new_array, rotateCounterClock5, (w, h)) # applying the rotation matrix
              #rotate 10 counterClock
              rotateCounterClock10 = cv2.getRotationMatrix2D((cX, cY), -10, 1.0) # getting the rotation matrix
              rotatedCounterClock10 = cv2.warpAffine(new_array, rotateCounterClock10, (w, h)) # applying the rotation matrix
              #applying shift matrices
              shiftedUp = cv2.warpAffine(new_array, shiftUp, (new_array.shape[1], new_array.shape[0]))
              shiftedDown = cv2.warpAffine(new_array, shiftDown, (new_array.shape[1], new_array.shape[0]))
              shiftedRight = cv2.warpAffine(new_array, shiftRight, (new_array.shape[1], new_array.shape[0]))
              shiftedLeft = cv2.warpAffine(new_array, shiftLeft, (new_array.shape[1], new_array.shape[0]))

              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'original.jpg'), new_array)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'rotatedClock5.jpg'), rotatedClock5)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'rotatedCounterClock5.jpg'), rotatedCounterClock5)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'rotatedClock10.jpg'), rotatedClock10)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'rotatedCounterClock10.jpg'), rotatedCounterClock10)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'shiftedUp.jpg'), shiftedUp)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'shiftedDown.jpg'), shiftedDown)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'shiftedRight.jpg'), shiftedRight)
              cv2.imwrite(os.path.join(save_path,img.split('.')[0]+'shiftedLeft.jpg'), shiftedLeft)
create_training_data()
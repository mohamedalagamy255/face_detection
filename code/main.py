
import os
import cv2
import numpy as np
import pandas as pd
import random
import psutil
import os
import time
import tensorflow as tf
from functions import *
from models import *
from training import train_onet ,train_pnet ,train_rnet
from tensorflow.keras.layers import Dense , Input ,Conv2D ,MaxPooling2D , Flatten



file_dir=os.path.join(r"C:\Users\HI5\Downloads\New_folder","wider_face_train_bbx_gt.txt")
wider_data_dir=r"C:\Users\HI5\Downloads\New_folder\WIDER_train\WIDER_train\images"
images_boxes = get_image_name_and_its_boxes(file_dir)

for images_bb in images_boxes:
    image=cv2.imread(os.path.join(wider_data_dir,images_bb[0]))
    for bb in images_bb[1]:
        cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(255,0,0),2)
    cv2.imshow("image",image)
    key= cv2.waitKey(1000) 
    if key == 27:
        break
    
    
cv2.destroyAllWindows()





# pnet
pnet_=PNet((12,12,3))
data_dir = r"D:\dff\12"
with open(os.path.join(data_dir,"negative.txt"),"r") as f:
    read_all_lines = f.readlines()
DATA_LENGTH = len(read_all_lines)
BATCH_SIZE = 64

train_pnet(10,BATCH_SIZE,base_lr = .001,model_store_path = "D:\p_net_model" , DATA_LENGTH = DATA_LENGTH)





# r_net 
Rnet_=RNet((24,24,3))
model_path = "D:\r_net_model"
Rnet_.model.save(r"D:\r_net_model\model_rnet.h5" )

data_dir = r"D:\dff\24"
BATCH_SIZE = 64
with open(os.path.join(data_dir,"negative.txt"),"r") as f:
    read_all_lines = f.readlines()
    
DATA_LENGTH = 253440 # the size of the part images 
iterations  = int(len(read_all_lines) / DATA_LENGTH)

train_rnet(10,BATCH_SIZE,base_lr = .001,model_store_path = r"D:\r_net_model" , DATA_LENGTH = DATA_LENGTH)





# build o_net
onet_model = ONet((48,48,3)).model

data_dir = r"D:\dff\48"
land_images_dir = r"D:\datasets\celeb_faces\img_align_celeba\img_align_celeba"
landmark_file = r"D:\datasets\celeb_faces\list_landmarks_align_celeba.csv"

all_images_path , all_images_landmark = get_landmarks_data(land_images_dir,landmark_file)
all_images_path , all_images_landmark = all_images_path *2 ,all_images_landmark *2
DATA_LENGTH = len(all_images_landmark)
BATCH_SIZE = 64

# datasets for onet
positive_data = Data_set(BATCH_SIZE,os.path.join(data_dir,"positive.txt"),1,DATA_LENGTH).dataset
negative_data = Data_set(BATCH_SIZE,os.path.join(data_dir,"negative.txt"),0,DATA_LENGTH).dataset
part_data = Data_set(BATCH_SIZE,os.path.join(data_dir,"part.txt"),-1,DATA_LENGTH).dataset



training_data = [positive_data , negative_data ,part_data]
train_onet(30,training_data,BATCH_SIZE,base_lr = .0008,model_store_path = "D:\o_net_model")




# test the all model
image_dir = os.path.join(r"C:\Users\HI5\Downloads\New_folder\WIDER_train\WIDER_train\images\0--Parade","0_Parade_marchingband_1_219.jpg")

model_pnet = tf.keras.models.load_model(os.path.join(r"D:\p_net_model","model_pnet.h5"))
model_rnet = tf.keras.models.load_model(os.path.join(r"D:\r_net_model","model_rnet.h5"))
model_onet = tf.keras.models.load_model(os.path.join(r"D:\o_net_model","model_onet.h5"))
detector = MtcnnDetector(model_pnet,model_rnet,model_onet,threshold=[0.6, 0.7,0.8])

def face_detect(image_dir, detector):
    image_=cv2.imread(image_dir).astype(np.float32)
    images_to_rgb = cv2.cvtColor(image_ , cv2.COLOR_BGR2RGB)
    faces = detector.detect_face(images_to_rgb)
    keep =nms(faces , 0.2)
    faces = faces[keep] 
    print(len(faces))
    image_ = image_.astype(np.uint8)
    for bb in faces:
        cv2.rectangle(image_ , (int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])) ,(255,0,0), 2)
    
    cv2.imshow("imag" ,image_)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

face_detect(image_dir, detector)


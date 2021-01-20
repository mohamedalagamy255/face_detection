


import tensorflow as tf
import numpy as np
import os
from models import MtcnnDetector
from functions import *
import cv2



file_dir=os.path.join(r"C:\Users\HI5\Downloads\New_folder","wider_face_train_bbx_gt.txt")
wider_data_dir=r"C:\Users\HI5\Downloads\New_folder\WIDER_train\WIDER_train\images"

images_boxes = get_image_name_and_its_boxes(file_dir) # first we get the image name and its gts boxes 
images_boxes = np.array(images_boxes)

def get_all_dir(x):
    return os.path.join(r"C:\Users\HI5\Downloads\New_folder\WIDER_train\WIDER_train\images",x)






data_dir = r"D:\dff"

neg_save_dir =  os.path.join(data_dir,"48/negative")
pos_save_dir =  os.path.join(data_dir,"48/positive")
part_save_dir = os.path.join(data_dir,"48/part")


for dir_path in [neg_save_dir,pos_save_dir,part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

post_save_file = os.path.join(data_dir,"48/positive.txt")
neg_save_file = os.path.join(data_dir,"48/negative.txt")
part_save_file = os.path.join(data_dir,"48/part.txt")





f1 = open(post_save_file, 'w')
f2 = open(neg_save_file, 'w')
f3 = open(part_save_file, 'w')





model_pnet =tf.keras.models.load_model(os.path.join(r"D:\p_net_model","model_pnet.h5"))
model_rnet = tf.keras.models.load_model(os.path.join(r"D:\r_net_model","model_rnet.h5"))





all_boxes = list()
image_idx = 0
n_idx = 0
p_idx = 0
d_idx = 0
image_size = 48

detector = MtcnnDetector(model_pnet,model_rnet)
for image ,offsets in zip(list(map(get_all_dir,images_boxes[:,0])),list(images_boxes[:,1])):
    image=cv2.imread(image).astype(np.float32)
    images_to_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    
    #gts = images_bb[1]
    gts = np.array(offsets, dtype=np.float32).reshape(-1, 4)
    
    # here we pass the images_to_rgb to the p_net_detector and get the boxes and aligned boxes
    p_boxes , p_boxes_align = detector.detect_pnet(images_to_rgb)
    boxes , boxes_align = detector.detect_rnet(images_to_rgb , p_boxes_align)
    if boxes_align is None:
        image_idx += 1
        continue
    dets = convert_to_square(boxes_align) # here we will convert the detected box to square 
    dets[:, 0:4] = np.round(dets[:, 0:4])
    
    
    
    
        
    
    images_to_rgb = images_to_rgb[...,::-1]
    for box in dets :
        
        x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
        
        width = x_right - x_left + 1
        height = y_bottom - y_top + 1
        
        
        # here we ignore some small boxes and some out of the image
        if width < 20 or x_left < 0 or y_top < 0 or x_right > images_to_rgb.shape[1] - 1 or y_bottom > images_to_rgb.shape[0] - 1:
            continue
            
        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gts)
        cropped_im = images_to_rgb[y_top:y_bottom + 1, x_left:x_right + 1, :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        
        # save negative images and write label
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            
        else:
            # find gt_box with the highest iou
            idx = np.argmax(Iou)
            assigned_gt = gts[idx]
            x1, y1, x2, y2 = assigned_gt

            # compute bbox reg label
            offset_x1 = (x1 - x_left) / float(width)
            offset_y1 = (y1 - y_top) / float(height)
            offset_x2 = (x2 - x_right) / float(width)
            offset_y2 = (y2 - y_bottom) / float(height)
            
            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
                
            elif np.max(Iou) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
                
    image_idx +=1
        
    print("%s images done, pos: %s part: %s neg: %s"%(image_idx, p_idx, d_idx, n_idx))
    if image_idx %100 ==0:
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss / (1024*1024))

f1.close()
f2.close()
f3.close()
    
    





from functions import *
import os
import numpy as np
import cv2




file_dir=os.path.join(r"C:\Users\HI5\Downloads\New_folder","wider_face_train_bbx_gt.txt")
wider_data_dir=r"C:\Users\HI5\Downloads\New_folder\WIDER_train\WIDER_train\images"

images_boxes = get_image_name_and_its_boxes(file_dir)
data_dir = r"D:\dff"

neg_save_dir =  os.path.join(data_dir,"12/negative")
pos_save_dir =  os.path.join(data_dir,"12/positive")
part_save_dir = os.path.join(data_dir,"12/part")

for dir_path in [neg_save_dir,pos_save_dir,part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


post_save_file = os.path.join(data_dir,"12/positive.txt")
neg_save_file = os.path.join(data_dir,"12/negative.txt")
part_save_file = os.path.join(data_dir,"12/part.txt")

f1 = open(post_save_file, 'w')
f2 = open(neg_save_file, 'w')
f3 = open(part_save_file, 'w')

p_idx = 0
n_idx = 0
d_idx = 0
idx = 0
box_idx = 0




for images_bb in images_boxes:
    image=cv2.imread(os.path.join(wider_data_dir,images_bb[0]))
    boxes = images_bb[1]
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")
    
    height, width, channel = image.shape
    
    neg_num = 0
    while neg_num < 50:
        size = np.random.randint(12, min(width, height) / 2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = image[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        if w < 0 or h < 0:
            continue
    
    
        # generate negative examples that have overlap with gt
        for i in range(5):
            size = np.random.randint(12,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)
            cropped_im = image[int(ny1) : int(ny1) + size, int(nx1) : int(nx1) + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                
        # generate positive examples and part faces
        for i in range(20):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size
            
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            
            cropped_im = image[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            
            box_ = np.asarray(box).reshape(1, -1)
            
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
        
                

    
f1.close()
f2.close()
f3.close()










def get_boxes(number,offset,file_):
    boxes = []
    offset+=1
    for i in range(number):
        bb = file_[offset].split(" ")[:4]
        bb = list(map(float, bb))
        bb[2]=bb[0]+bb[2]
        bb[3]=bb[1]+bb[3]
        boxes.append(bb)
        offset +=1
    
    return boxes ,offset


def get_image_name_and_its_boxes(file_name):
    
    with open(file_name,"r",encoding="utf-8") as f:
        file_name = f.readlines()
    
    file_name = [c.replace("\n","") for c in file_name]
    file_name = [c.replace("/","\\") for c in file_name]
    
    offset=0
    images_names_and_boxex = []
    while offset < len(file_name) :
        image_name = file_name[offset]
        offset +=1
        number_of_boxes = int(file_name[offset])
        
        boxes , offset =get_boxes(number_of_boxes,offset,file_name)
        
        images_names_and_boxex.append([image_name,boxes])
        
    return images_names_and_boxex




def IoU(box, boxes):
    box=np.asarray(box)
    boxes=np.asarray(boxes).reshape(-1,4)
    
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,(box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr




def nms(dets, thresh, mode="Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep



def convert_to_square(bbox):
    
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox



def compute_accuracy(x,b):
    x=np.squeeze(x)
    b=np.squeeze(b)
    acc_1=np.where(np.array(x) > .8,1,0).mean()
    acc_2 = np.where(np.array(b) < .8 ,1,0).mean()
    acc = (acc_1 +acc_2) /2
    return acc



def get_landmarks_data(images_dir , landmark_file):
    landmark_data = pd.read_csv(landmark_file)
    all_images_path = []
    all_images_landmark = []
    for _,row in landmark_data.iterrows():
        all_images_path.append(os.path.join(images_dir,row["image_id"]))
        all_images_landmark.append(row.values[1:])
        
    return all_images_path , all_images_landmark



def get_land_image(image_dir, land_marks):
    image = tf.io.read_file(image_dir)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image,(48,48))
    image = tf.cast(image ,tf.float32)
    land_marks = tf.cast(land_marks ,tf.float32)
    
    x_scale = 1 / 178
    y_scale = 1 / 218
    
    new_land_marks = []
    for x,y in zip([0,2,4,6,8],[1,3,5,7,9]):
        new_land_marks.append(land_marks[x]*x_scale)
        new_land_marks.append(land_marks[y]*y_scale)
    
    land_marks = new_land_marks
    image = image/255.0
    return image , land_marks
    
    

    
    
        





class LossFn:
    def __init__(self, cls_factor = 1 ,box_factor = 1 ,landmark_factor = 1):
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.landmark_factor = landmark_factor
        self.loss_cls = tf.keras.losses.binary_crossentropy
        self.loss_box = tf.keras.losses.mse
        self.loss_landmark = tf.keras.losses.mse
    
    def clf_loss(self,gt_label,pred_label):
        pred_label = tf.squeeze(pred_label)
        gt_label = tf.squeeze(gt_label)
        
        return self.loss_cls(gt_label,pred_label) * self.cls_factor
    
    def box_loss(self,gt_offset,pred_offset):
        
        pred_offset = tf.squeeze(pred_offset)
        gt_offset = tf.squeeze(gt_offset)
        
        return self.loss_box(gt_offset,pred_offset) * self.box_factor
    
    def landmark_loss(self,gt_landmark,pred_landmark):
        pred_landmark = tf.squeeze(pred_landmark)
        gt_landmark = tf.squeeze(gt_landmark)
        
        return self.loss_landmark(gt_landmark,pred_landmark) * self.landmark_factor





class Data_set(object):
    def __init__(self,batch_size , file_dir,label,data_length,pointer=0):
        self.label = label
        self.batch_size = batch_size
        self.file_dir = file_dir
        self.data_length = data_length
        self.pointer = pointer
        
        
        if self.label ==1:
            self.images___dirs ,self.offsets__ =self.get_images_dir_and_offsets()
            
            data_set_offsets = tf.data.Dataset.from_tensor_slices(self.offsets__)
            
            
            self.labels = [1] * self.data_length
            data_set_labels = tf.data.Dataset.from_tensor_slices(self.labels)
        
        
            data_set_images=tf.data.Dataset.from_tensor_slices(self.images___dirs)
            data_set_images = data_set_images.map(self.get_image)
            
            
        
            self.dataset = tf.data.Dataset.zip((data_set_images,data_set_offsets,data_set_labels))
            self.dataset = self.dataset.batch(self.batch_size)
            
            
        elif self.label ==-1:
            self.images___dirs ,self.offsets__ =self.get_images_dir_and_offsets()
            
            data_set_offsets = tf.data.Dataset.from_tensor_slices(self.offsets__)
        
        
            data_set_images=tf.data.Dataset.from_tensor_slices(self.images___dirs)
            data_set_images = data_set_images.map(self.get_image)
            
            
        
            self.dataset = tf.data.Dataset.zip((data_set_images,data_set_offsets))
            self.dataset = self.dataset.batch(self.batch_size)
        
        elif self.label == 0:
            self.images___dirs=self.get_images_dir_and_offsets()
            
            
            
            self.labels = [0] * self.data_length
            data_set_labels = tf.data.Dataset.from_tensor_slices(self.labels)
        
        
            data_set_images=tf.data.Dataset.from_tensor_slices(self.images___dirs)
            data_set_images = data_set_images.map(self.get_image)
            
            
        
            self.dataset = tf.data.Dataset.zip((data_set_images,data_set_labels))
            self.dataset = self.dataset.batch(self.batch_size)
            
        
    
    def get_images_dir_and_offsets(self):
        with open(self.file_dir,"r") as f:
            read_all_lines = f.readlines()
        
        if self.label == 1 or self.label ==-1:
            images_dirs = [c.replace("\n","").split(" ")[0] for c in read_all_lines]
            images_dirs = images_dirs * int(self.data_length / len(images_dirs))
            images_dirs =  images_dirs + images_dirs[:self.data_length - len(images_dirs)]
            offsets = [list(map(float,c.replace("\n","").split(" ")[2:])) for c in read_all_lines]
            offsets = offsets * int(self.data_length / len(offsets))
            offsets = offsets + offsets[:self.data_length - len(offsets)]
            
            return images_dirs , offsets
        elif self.label == 0:
            images_dirs = [c.replace("\n","").split(" ")[0] for 
                           c in read_all_lines[self.pointer*self.data_length : (self.pointer + 1)*self.data_length]]
            return images_dirs
            
    
    
    
    def get_image(self , image_dir):
        image = tf.io.read_file(image_dir)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image ,tf.float32)
        image = (image/128.0 ) -1
        return image
    
    
    
        
    


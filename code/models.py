


class PNet(object):
    def __init__(self ,input_shape):
        self.input_shape = input_shape
        self.model = self.build_network()
        
        
    def build_network(self):
        # base layers 
        inputs = Input(shape= self.input_shape)
        x = Conv2D(10,kernel_size = 3,strides=1,activation="relu")(inputs)
        x=MaxPooling2D(pool_size= (2,2),strides=2)(x)
        x=Conv2D(16,kernel_size = 3, strides=1 ,activation= "relu")(x)
        x=Conv2D(32,kernel_size = 3,strides=1 ,activation ="relu")(x)
        
        # detection 
        detection_layer = Conv2D(1 , kernel_size = 1, strides = 1 , activation= "sigmoid", name = "detection_layer")(x)
        #bounding box regression
        reg_layer = Conv2D (4 ,kernel_size = 1 ,strides = 1 , name = "reg_layer")(x)
        
        model = tf.keras.models.Model(inputs , [detection_layer,reg_layer] , name = "PNET")
        
        return model
        




class RNet(object):
    def __init__(self ,input_shape):
        self.input_shape = input_shape
        self.model = self.build_network()
        
        
    def build_network(self):
        # base layers 
        inputs = Input(shape= self.input_shape)
        x = Conv2D(28,kernel_size = 3,strides=1,activation="relu")(inputs)
        x = MaxPooling2D(pool_size= (3,3),strides=2)(x)
        x = Conv2D(48,kernel_size = 3, strides=1 ,activation= "relu")(x)
        x = MaxPooling2D(pool_size= (3,3),strides=2)(x)
        x = Conv2D(64,kernel_size = 2,strides=1 ,activation ="relu")(x)
        x = Flatten()(x)
        x = Dense(128 ,activation = "relu")(x)
        
        # detection 
        detection_layer = Dense(1 ,activation = "sigmoid")(x)
        #bounding box regression
        reg_layer = Dense(4)(x)
        
        
        model = tf.keras.models.Model(inputs , [detection_layer,reg_layer] , name = "RNET")
        
        return model




class ONet(object):
    def __init__(self ,input_shape):
        self.input_shape = input_shape
        self.model = self.build_network()
        
        
    def build_network(self):
        # base layers 
        inputs = Input(shape= self.input_shape)
        x = Conv2D(32,kernel_size = 3,strides=1,activation="relu")(inputs)
        x = MaxPooling2D(pool_size= (3,3),strides=2)(x)
        x = Conv2D(64,kernel_size = 3, strides=1 ,activation= "relu")(x)
        x = MaxPooling2D(pool_size= (3,3),strides=2)(x)
        x = Conv2D(64,kernel_size = 3,strides=1 ,activation ="relu")(x)
        x = MaxPooling2D(pool_size= (2,2),strides=2)(x)
        x = Conv2D(128,kernel_size = 2,strides=1 ,activation ="relu")(x)
        x = Flatten()(x)
        x = Dense(256 ,activation = "relu")(x)
        
        # detection 
        detection_layer = Dense(1 ,activation = "sigmoid")(x)
        #bounding box regression
        reg_layer = Dense(4)(x)
        
        
        
        model = tf.keras.models.Model(inputs , [detection_layer,reg_layer] , name = "ONET")
        
        return model




class MtcnnDetector(object):
    def __init__(self,pnet = None , rnet = None ,onet = None ,min_face_size = 12 ,stride = 2 ,threshold = [ 0.6 , 0.7 , 0.8] , scale_factor = 0.709):
        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride = stride
        self.threshold = threshold
        self.scale_factor = scale_factor
        
        
        
    def square_bbox(self , bbox):
        # convert bbox to square
        square_bbox = bbox.copy()
        
        
        h = bbox[:, 3] - bbox[:, 1] +1
        w = bbox[:, 2] - bbox[:, 0] +1
        l = np.maximum(h,w)
        
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
        square_bbox[:, 1] = bbox[:, 1] +h*0.5 -l*0.5
        
        square_bbox[:, 2] = square_bbox[:, 0] + l -1
        square_bbox[:, 3] = square_bbox[:, 1] + l -1
        
        return square_bbox
    
    
    def generate_bounding_box(self, map, reg, scale, threshold):
        # generate bbox from feature map
        
        stride = 2
        cell_size = 12
        
        t_index = np.where(map > threshold)
        
        if t_index[0].size == 0:
            return np.array([]) # which means i found nothing
        
        dx1 , dy1 , dx2 , dy2 = [reg[0,t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([ dx1, dy1, dx2, dy2])
        
        score = map[t_index[0] ,t_index[1] ,0]
        
        boundingbox = np.vstack([np.round((stride*t_index[1]) / scale),
                                np.round((stride*t_index[0]) / scale),
                                np.round((stride*t_index[1] + cell_size) / scale),
                                np.round((stride*t_index[0] + cell_size) / scale),
                                score,
                                reg,
                                ])
        return boundingbox.T
    
    
    def resize_image(self ,img ,scale):
        
        height , width ,channel = img.shape
        
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        new_dim = (new_width,new_height)
        
        img_resized = cv2.resize(img , new_dim ,interpolation=cv2.INTER_LINEAR)
        #img_resized = tf.image.resize(img,new_dim,method= "bilinear")
        
        
        return img_resized
    
    
    def pad(self, bboxes, w, h):
        
        tmpw = (bboxes[:, 2] - bboxes[:, 0] +1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] +1).astype(np.int32)
        numbox = bboxes.shape[0]
        
        dx = np.zeros((numbox, ))
        dy = np.zeros((numbox, ))
        edx , edy =tmpw.copy() -1 ,tmph.copy() -1
        
        x , y ,ex , ey =bboxes[:, 0] ,bboxes[:, 1] ,bboxes[:, 2] ,bboxes[:, 3]
        
        tmp_index = np.where(ex > w -1)
        edx[tmp_index] = tmpw[tmp_index] + w  -2 -ex[tmp_index]
        ex[tmp_index] = w -1
        
        
        tmp_index = np.where(ey > h -1)
        edy[tmp_index] = tmph[tmp_index] + h  -2 -ey[tmp_index]
        ey[tmp_index] = h-1
        
        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0
        
        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 -y[tmp_index]
        y[tmp_index] = 0
        
        return_list = [dy , edy ,dx ,edx ,y , ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]
        
        return return_list
    
    
    
    
    
    def detect_pnet(self , im):
        # get faces cadidates
        
        h , w, c = im.shape[:3]
        net_size = 12
        
        current_scale = float(net_size) / self.min_face_size 
        #print(im.numpy()[0][0])
        im_resized  = self.resize_image(im, current_scale)
        current_height , current_width , _ = im_resized.shape
        im_resized = tf.convert_to_tensor(im_resized)
        #print(im_resized.numpy()[0][0])
        
        
        all_boxes = list()
        
        while min(current_height ,current_width) > net_size :
            
            feeds_imgs = []
            feeds_imgs.append(im_resized)
            
            # notice here you want to cahnge the layer input shape
            cls_map_np , reg_np  =self.pnet_detector(tf.reshape(feeds_imgs,(1,current_height,current_width,3))/255.0)
            tf.keras.backend.clear_session()
            
            
            # get the cls ,reg as numpy array
            cls_map_np = cls_map_np.numpy()
            reg_np = reg_np.numpy()
            
            boxes = self.generate_bounding_box(cls_map_np[0,:,:] , reg_np , current_scale ,self.threshold[0])
            
            
            current_scale *=self.scale_factor
            im_resized  = self.resize_image(im, current_scale)
            current_height , current_width , _ = im_resized.shape
            im_resized = tf.convert_to_tensor(im_resized)
            
            if boxes.size == 0:
                continue
            
            keep = nms(boxes[:, :5] , 0.5,"Union")
            boxes = boxes[keep]
            all_boxes.append(boxes)
            
            
        if len(all_boxes) == 0:
            return None ,None
            
        all_boxes =np.vstack(all_boxes)
            
        keep = nms(all_boxes[:,0:5] ,0.7,"Union")
        all_boxes = all_boxes[keep]
            
        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        
            
        boxes = np.vstack([all_boxes[:, 0] ,all_boxes[:, 1] ,all_boxes[:, 2] ,all_boxes[:, 3] ,all_boxes[:, 4] ])
            
        boxes = boxes.T
            
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh
            
            
        # refine the boxes 
        boxes_align = np.vstack([ align_topx ,align_topy ,align_bottomx, align_bottomy, all_boxes[:, 4] ])
            
        boxes_align =boxes_align.T
            
        return boxes , boxes_align
        #return all_boxes
        
        
        
    def detect_rnet(self ,im ,dets):
        h ,w,c = im.shape
        
        if dets is None :
            return None ,None
        dets = self.square_bbox(dets)
        dets[:,0:4] = np.round(dets[:,0:4])
        
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        
        
        croped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i] , tmpw[i] ,3) ,dtype = np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            crop_im = cv2.resize(tmp,(24,24))
            croped_ims_tensors.append(crop_im)
        
        
        feed_images = tf.convert_to_tensor(np.array(croped_ims_tensors).reshape(-1,24,24,3) / 255.0)
        
        cls_map ,reg =self.rnet_detector(feed_images)
        cls_map = cls_map.numpy()
        reg = reg.numpy()
        
        keep_inds = np.where(cls_map > self.threshold[1])[0]
        
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg =reg[keep_inds]
            
        else :
            return None ,None
        
        keep =nms(boxes , 0.7)
        
        if len(keep) == 0:
            return None ,None
        
        
        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        
        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1
        
        boxes = np.vstack([ keep_boxes[:,0],keep_boxes[:,1],keep_boxes[:,2],keep_boxes[:,3],keep_cls[:,0]])
        
        
        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh
        
        
        
        boxes_align = np.vstack([align_topx,align_topy,align_bottomx,align_bottomy,keep_cls[:, 0]])
    
        
        
        boxes = boxes.T
        boxes_align = boxes_align.T
        
        
        return  boxes , boxes_align
            
    
    
    
    def detect_onet(self ,im ,dets):
        h, w, c = im.shape
        
        if dets is None:
            return None
        
        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        
        croped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i] , tmpw[i] ,3) ,dtype = np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            crop_im = cv2.resize(tmp,(48,48))
            croped_ims_tensors.append(crop_im)
            
        feed_images = tf.convert_to_tensor((np.array(croped_ims_tensors).reshape(-1,48,48,3) / 128.0) - 1)
        
        cls_map ,reg =self.onet_detector(feed_images)
        cls_map = cls_map.numpy()
        reg = reg.numpy()
        #land = land.numpy()
        
        
        keep_inds = np.where(cls_map > self.threshold[2])[0]
        
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg =reg[keep_inds]
            #land = land[keep_inds]
        else :
            return None 
        
        
        keep =nms(boxes , 0.7)
        
        if len(keep) == 0:
            return None 
        
        
        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        #keep_landmark = land[keep]
        
        
        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1
        
        
        boxes = np.vstack([ keep_boxes[:,0],keep_boxes[:,1],keep_boxes[:,2],keep_boxes[:,3],keep_cls[:,0]])
        
        
        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh
        
        #align_landmark_topx = keep_boxes[:, 0]
        #align_landmark_topy = keep_boxes[:, 1]
        
        boxes_align = np.vstack([align_topx,align_topy,align_bottomx,align_bottomy,keep_cls[:, 0]])
        
        
        
        boxes = boxes.T
        boxes_align = boxes_align.T
        #landmark_align = landmark.T
        
        return boxes_align
    
    
    def detect_face(self,img):
        boxes_align = np.array([])
        #landmark_align =np.array([])
        
        t = time.time()
        
        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()
            
            
        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()
        
        
        # onet
        if self.onet_detector:
            boxes_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([])
            
        
        t3 = time.time() - t
        t = time.time()
        print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))
        
        return boxes_align
    


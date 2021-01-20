


def train_pnet(end_epoch,batch_size,frequent = 50 ,base_lr = 0.01,model_store_path = "",DATA_LENGTH = 0):
    #if not os.path.exists(model_store_path):
        #os.makedirs(model_store_path)
    lossfn = LossFn()
    #net = PNet((12,12,3)).model
    path= os.path.join(model_store_path,"model_pnet.h5")
    net = tf.keras.models.load_model(path)
    optimizer = tf.keras.optimizers.Adam(lr=base_lr)
    
    positive_data = Data_set(BATCH_SIZE,os.path.join(data_dir,"positive.txt"),1,DATA_LENGTH).dataset
    negative_data = Data_set(BATCH_SIZE,os.path.join(data_dir,"negative.txt"),0,DATA_LENGTH).dataset
    part_data = Data_set(BATCH_SIZE,os.path.join(data_dir,"part.txt"),-1,DATA_LENGTH).dataset
    
    for cur_epoch in range(1,end_epoch + 1):
        accuracy_list = []
        cls_loss_list = []
        bbox_loss_list = []
        #landmark_loss_list = []
        i=0
        
        
        
        for positive_batch , negative_batch ,part_batch in zip(positive_data,negative_data,part_data) :
            positive_images_batch , positive_offset_batch ,poitive_labels = positive_batch
            
            negative_images_batch , negative_labels = negative_batch
            
            part_images_batch , part_offset_batch = part_batch
            with tf.GradientTape() as tape:
                cls_pred_pp ,offset_pred_pp = net(positive_images_batch)
                cls_pred_n ,_ = net(negative_images_batch)
                _,offset_pred_pa = net(part_images_batch)
                cls_all_pred = tf.concat([cls_pred_pp,cls_pred_n],axis = 0)
                labesl_all_true = tf.concat([poitive_labels,negative_labels],axis =0)
                
                cls_loss = lossfn.clf_loss(labesl_all_true,cls_all_pred)
                
                offset_all_pred = tf.concat([offset_pred_pp,offset_pred_pa],axis =0)
                offset_all_true =tf.concat([positive_offset_batch,part_offset_batch] , axis =0)
                offset_loss = lossfn.box_loss(offset_all_true,offset_all_pred)
                offset_loss = tf.reduce_mean(offset_loss)
                
                all_loss = cls_loss*1.0+offset_loss*0.5
                accuracy_list.append(compute_accuracy(cls_pred_pp,cls_pred_n))
                cls_loss_list.append(cls_loss.numpy())
                bbox_loss_list.append(offset_loss.numpy())
                if (i+1)%100 ==0:
                    
                    print("Epoch : ",cur_epoch,"cls_loss : ", np.mean(cls_loss_list)," regres_loss : ", np.mean(bbox_loss_list)
                          ," Accuracy : ",np.mean(accuracy_list) , "batch : ",i)
                
                i+=1
            gradients = tape.gradient(all_loss ,net.trainable_variables)
            optimizer.apply_gradients(zip(gradients,net.trainable_variables))
        
        print("end of the epoch ") 
        net.save(os.path.join(model_store_path,"model_pnet.h5"))




def train_rnet(end_epoch,batch_size,frequent = 50 ,base_lr = 0.01,model_store_path = "" , DATA_LENGTH = 0):
    
    lossfn = LossFn()
    path= os.path.join(model_store_path,"model_rnet.h5")
    #net = tf.keras.models.load_model(path)
    net=RNet((24,24,3)).model
    optimizer = tf.keras.optimizers.Adam(lr=base_lr)
    
    
    for cur_epoch in range(1,end_epoch + 1):
        
        for pointer in range(iterations):
            
            accuracy_list = []
            cls_loss_list = []
            bbox_loss_list = []
            #landmark_loss_list = []
            i=0
            
            positive_data = Data_set(batch_size,os.path.join(data_dir,"positive.txt"),1,DATA_LENGTH , pointer).dataset
            negative_data = Data_set(batch_size,os.path.join(data_dir,"negative.txt"),0,DATA_LENGTH, pointer).dataset
            part_data = Data_set(batch_size,os.path.join(data_dir,"part.txt"),-1,DATA_LENGTH, pointer).dataset
            
        
            for positive_batch , negative_batch ,part_batch in zip(positive_data,negative_data,part_data) :
            
                positive_images_batch , positive_offset_batch ,poitive_labels = positive_batch
                negative_images_batch , negative_labels = negative_batch
                part_images_batch , part_offset_batch = part_batch
            
                with tf.GradientTape() as tape:
                    cls_pred_pp ,offset_pred_pp = net(positive_images_batch)
                    cls_pred_n ,_ = net(negative_images_batch)
                    _,offset_pred_pa = net(part_images_batch)
                    cls_all_pred = tf.concat([cls_pred_pp,cls_pred_n],axis = 0)
                    labesl_all_true = tf.concat([poitive_labels,negative_labels],axis =0)
                    
                    cls_loss = lossfn.clf_loss(labesl_all_true,cls_all_pred)
                    
            
                    offset_all_pred = tf.concat([offset_pred_pp,offset_pred_pa],axis =0)
                    offset_all_true =tf.concat([positive_offset_batch,part_offset_batch] , axis =0)
            
                    offset_loss = lossfn.box_loss(offset_all_true,offset_all_pred)
                    offset_loss = tf.reduce_mean(offset_loss)
                    
                
            
                    all_loss = cls_loss*1.0+offset_loss*0.5
                
                    accuracy_list.append(compute_accuracy(cls_pred_pp,cls_pred_n))
                    cls_loss_list.append(cls_loss.numpy())
                    bbox_loss_list.append(offset_loss.numpy())
                    
                    
                gradients = tape.gradient(all_loss ,net.trainable_variables)
                optimizer.apply_gradients(zip(gradients,net.trainable_variables))
                
                
                
                
                if (i+1)%100 ==0:
                    print("Epoch : ",cur_epoch,"cls_loss : ", np.mean(cls_loss_list)," regres_loss : ", np.mean(bbox_loss_list)," Accuracy : ",np.mean(accuracy_list) , "batch : ",i)
                    process = psutil.Process(os.getpid())
                    print( "memory usage : " , process.memory_info().rss / (1024*1024))
                
                i+=1
        
            print("end of iteration : " , pointer) 
            net.save(os.path.join(model_store_path,"model_rnet.h5"))
            print("model is saved ")
        
    





def train_onet(end_epoch,training_data,batch_size,frequent = 50 ,base_lr = 0.01,model_store_path = ""):
    #if not os.path.exists(model_store_path):
        #os.makedirs(model_store_path)
    lossfn = LossFn()
    #net = ONet((48,48,3)).model
    path= os.path.join(model_store_path,"model_onet.h5")
    net = tf.keras.models.load_model(path)
    optimizer = tf.keras.optimizers.Adam(lr=base_lr)
    
    positive_data , negative_data ,part_data  = training_data
    
    for cur_epoch in range(1,end_epoch + 1):
        accuracy_list = []
        cls_loss_list = []
        bbox_loss_list = []
        #landmark_loss_list = []
        i=0
        
        
        for positive_batch , negative_batch ,part_batch ,   in zip(positive_data,negative_data,part_data ) :
            positive_images_batch , positive_offset_batch ,poitive_labels = positive_batch
            negative_images_batch , negative_labels = negative_batch
            part_images_batch , part_offset_batch = part_batch
            #images_land , land_points = land_batch
            
            with tf.GradientTape() as tape:
                cls_pred_pp ,offset_pred_pp = net(positive_images_batch)
                cls_pred_n ,_  = net(negative_images_batch)
                _,offset_pred_pa  = net(part_images_batch)
                #_,_,pred_land = net(images_land)
                cls_all_pred = tf.concat([cls_pred_pp,cls_pred_n],axis = 0)
                labesl_all_true = tf.concat([poitive_labels,negative_labels],axis =0)
                
                cls_loss = lossfn.clf_loss(labesl_all_true,cls_all_pred)
                
            
                offset_all_pred = tf.concat([offset_pred_pp,offset_pred_pa],axis =0)
                offset_all_true =tf.concat([positive_offset_batch,part_offset_batch] , axis =0)
                
                
            
                offset_loss = lossfn.box_loss(offset_all_true,offset_all_pred)
                offset_loss = tf.reduce_mean(offset_loss)
                
                
                
                
            
                all_loss = cls_loss*1.0+offset_loss*0.5
                
                accuracy_list.append(compute_accuracy(cls_pred_pp,cls_pred_n))
                cls_loss_list.append(cls_loss.numpy())
                bbox_loss_list.append(offset_loss.numpy())
                
                
                
                
                if (i+1)%100 ==0:
                    
                    print("Epoch : ",cur_epoch,"cls_loss : ", np.mean(cls_loss_list)," regres_loss : ",np.mean(bbox_loss_list) ," Accuracy : ",np.mean(accuracy_list) , "batch : ",i)
                    process = psutil.Process(os.getpid())
                    print( "memory usage : " , process.memory_info().rss / (1024*1024))
                    net.save(os.path.join(model_store_path,"model_onet.h5"))
                    print("model saved ")
                i+=1
            gradients = tape.gradient(all_loss ,net.trainable_variables)
            optimizer.apply_gradients(zip(gradients,net.trainable_variables))
        
        print("end of the epoch ") 
        net.save(os.path.join(model_store_path,"model_onet.h5"))
        print("model saved ") 
                


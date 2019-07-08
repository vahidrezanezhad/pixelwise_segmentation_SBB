import os
import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
from tqdm import tqdm




def bluring(img_in,kind):
    if kind=='guass':
        img_blur = cv2.GaussianBlur(img_in,(5,5),0)
    elif kind=="median":
        img_blur = cv2.medianBlur(img_in,5)
    elif kind=='blur':
        img_blur=cv2.blur(img_in,(5,5))
    return img_blur

def elastic_transform(image, alpha, sigma,seedj, random_state=None):
    
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(seedj)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def color_images(seg, n_classes):
    ann_u=range(n_classes)
    if len(np.shape(seg))==3:
        seg=seg[:,:,0]
        
    seg_img=np.zeros((np.shape(seg)[0],np.shape(seg)[1],3)).astype(float)
    colors=sns.color_palette("hls", n_classes)
    
    for c in ann_u:
        c=int(c)
        segl=(seg==c)
        seg_img[:,:,0]+=segl*(colors[c][0])
        seg_img[:,:,1]+=segl*(colors[c][1])
        seg_img[:,:,2]+=segl*(colors[c][2])
    return seg_img

   
def resize_image(seg_in,input_height,input_width):
    return cv2.resize(seg_in,(input_width,input_height),interpolation=cv2.INTER_NEAREST)
def get_one_hot(seg,input_height,input_width,n_classes):
    seg=seg[:,:,0]
    seg_f=np.zeros((input_height, input_width,n_classes))
    for j in range(n_classes):
        seg_f[:,:,j]=(seg==j).astype(int)
    return seg_f

    
def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    classes_true=np.unique(Yi)
    for c in classes_true:
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))
    return mIoU
def data_gen(img_folder, mask_folder, batch_size,input_height, input_width,n_classes):
    c = 0
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
    while True:
        img = np.zeros((batch_size, input_height, input_width, 3)).astype('float')
        mask = np.zeros((batch_size, input_height, input_width, n_classes)).astype('float')
    
        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            #print(img_folder+'/'+n[i])
            filename=n[i].split('.')[0]
            train_img = cv2.imread(img_folder+'/'+n[i])/255.
            #train_img =  cv2.resize(train_img, (input_width, input_height),interpolation=cv2.INTER_NEAREST)# Read an image from folder and resize
          
            img[i-c] = train_img #add to array - img[0], img[1], and so on.
            train_mask = cv2.imread(mask_folder+'/'+filename+'.png')
            #print(mask_folder+'/'+filename+'.png')
            #print(train_mask.shape)
            train_mask = get_one_hot( resize_image(train_mask,input_height,input_width),input_height,input_width,n_classes)
            #train_mask = train_mask.reshape(224, 224, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]
        
            mask[i-c] = train_mask
    
        c+=batch_size
        if(c+batch_size>=len(os.listdir(img_folder))):
            c=0
            random.shuffle(n)
        yield img, mask
        
def otsu_copy(img):
    img_r=np.zeros(img.shape)
    img1=img[:,:,0]
    img2=img[:,:,1]
    img3=img[:,:,2]
    _, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_r[:,:,0]=threshold1
    img_r[:,:,1]=threshold2
    img_r[:,:,2]=threshold3
    return img_r
def get_patches(dir_img_f,dir_seg_f,img,label,height,width,indexer):

    
    img_h=img.shape[0]
    img_w=img.shape[1]
    
    nxf=img_w/float(width)
    nyf=img_h/float(height)
    
    if nxf>int(nxf):
        nx=int(nxf)+1
    if nyf>int(nyf):
        ny=int(nyf)+1
        
    for i in range(nx):
        for j in range(ny):
            index_x_d=i*width
            index_x_u=(i+1)*width
            
            index_y_d=j*height
            index_y_u=(j+1)*height
            
            if index_x_u>img_w:
                index_x_u=img_w
                index_x_d=img_w-width
            if index_y_u>img_h:
                index_y_u=img_h
                index_y_d=img_h-height
                
            
            img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]
            label_patch=label[index_y_d:index_y_u,index_x_d:index_x_u,:]
            
            cv2.imwrite(dir_img_f+'/img_'+str(indexer)+'.png', img_patch )
            cv2.imwrite(dir_seg_f+'/img_'+str(indexer)+'.png' ,   label_patch )
            indexer+=1
    return indexer

def get_patches_num(dir_img_f,dir_seg_f,img,label,height,width,indexer,n_patches):
    img_h=img.shape[0]
    img_w=img.shape[1]
    
    for jjj in range(n_patches):
        nxf=random.randint(0, img_w-width)
        nyf=random.randint(0, img_h-height)
        
        index_x_d=nxf
        index_x_u=nxf+width
        
        index_y_d=nyf
        index_y_u=nyf+height
        
        if index_x_u>img_w:
            index_x_u=img_w
            index_x_d=img_w-width
        if index_y_u>img_h:
            index_y_u=img_h
            index_y_d=img_h-height
            
        
        img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]
        label_patch=label[index_y_d:index_y_u,index_x_d:index_x_u,:]
        
        cv2.imwrite(dir_img_f+'/img_'+str(indexer)+'.png', img_patch )
        cv2.imwrite(dir_seg_f+'/img_'+str(indexer)+'.png' ,   label_patch )
        indexer+=1
    return indexer
        
def get_patches_num_scale(dir_img_f,dir_seg_f,img,label,height,width,indexer,n_patches,scaler):

    
    img_h=img.shape[0]
    img_w=img.shape[1]
    
    height_scale=int(height*scaler)
    width_scale=int(width*scaler)
    
    for jjj in range(n_patches):
        nxf=random.randint(0, img_w-width_scale)
        nyf=random.randint(0, img_h-height_scale)
        
        index_x_d=nxf
        index_x_u=nxf+width_scale
        
        index_y_d=nyf
        index_y_u=nyf+height_scale
        
        if index_x_u>img_w:
            index_x_u=img_w
            index_x_d=img_w-width_scale
        if index_y_u>img_h:
            index_y_u=img_h
            index_y_d=img_h-height_scale
            
        
        img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]
        label_patch=label[index_y_d:index_y_u,index_x_d:index_x_u,:]
        
        img_patch=resize_image(img_patch,height,width)
        label_patch=resize_image(label_patch,height,width)
        
        cv2.imwrite(dir_img_f+'/img_'+str(indexer)+'.png', img_patch )
        cv2.imwrite(dir_seg_f+'/img_'+str(indexer)+'.png' ,   label_patch )
        indexer+=1
    return indexer

def provide_patches(dir_img,dir_seg,dir_flow_train_imgs,
                    dir_flow_train_labels,
                    input_height,input_width,blur_k,blur_aug,n_patches,
                    flip_aug,binarization,elastic_aug,scaling,scales,flip_index,
                    augmentation=False,patches=False):
    
    imgs_cv_train=np.array(os.listdir(dir_img))
    segs_cv_train=np.array(os.listdir(dir_seg))
    
    indexer=0
    for im, seg_i in tqdm(zip(imgs_cv_train,segs_cv_train)):
        img_name=im.split('.')[0]
    
        if not patches:
            cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png', resize_image(cv2.imread(dir_img+'/'+im),input_height,input_width ) )
            cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png' ,  resize_image(cv2.imread(dir_seg+'/'+img_name+'.png'),input_height,input_width ) )
            indexer+=1
            
            if augmentation:
                if flip_aug:
                    for f_i in flip_index:
                        cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                    resize_image(cv2.flip(cv2.imread(dir_img+'/'+im),f_i),input_height,input_width) )
                        
                        cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png' , 
                                    resize_image(cv2.flip(cv2.imread(dir_seg+'/'+img_name+'.png'),f_i),input_height,input_width) ) 
                        indexer+=1
                        
                if blur_aug:   
                    for blur_i in blur_k:
                        cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                    (resize_image(bluring(cv2.imread(dir_img+'/'+im),blur_i),input_height,input_width) ) )
                        
                        cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png' , 
                                    resize_image(cv2.imread(dir_seg+'/'+img_name+'.png'),input_height,input_width)  ) 
                        indexer+=1
                        
                if elastic_aug:
                    cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                elastic_transform( resize_image(cv2.imread(dir_img+'/'+im),
                                                                input_height,input_width),alpha=800, sigma=12,seedj=indexer))
                    
                    
                    cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png',
                                elastic_transform ( resize_image(cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                               input_height,input_width) ,alpha=800, sigma=12,seedj=indexer)  )
                    indexer+=1
                    
                if binarization:
                    cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                            resize_image(otsu_copy( cv2.imread(dir_img+'/'+im)),input_height,input_width ))
                    
                    cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png',
                                            resize_image( cv2.imread(dir_seg+'/'+img_name+'.png'),input_height,input_width ))
                    
                        
                    

            
            
        if patches:
            indexer=get_patches_num(dir_flow_train_imgs,dir_flow_train_labels,
                         cv2.imread(dir_img+'/'+im),cv2.imread(dir_seg+'/'+img_name+'.png'),
                          input_height,input_width,indexer=indexer,n_patches=n_patches)
            
            if augmentation:
                if flip_aug:
                    for f_i in flip_index:
                        indexer=get_patches_num(dir_flow_train_imgs,dir_flow_train_labels,
                                                cv2.flip( cv2.imread(dir_img+'/'+im) , f_i),
                                                cv2.flip( cv2.imread(dir_seg+'/'+img_name+'.png') ,f_i),
                                                input_height,input_width,indexer=indexer,n_patches=n_patches)
                if blur_aug:   
                    for blur_i in blur_k:
                        indexer=get_patches_num(dir_flow_train_imgs,dir_flow_train_labels,
                                                bluring( cv2.imread(dir_img+'/'+im) , blur_i),
                                                bluring( cv2.imread(dir_seg+'/'+img_name+'.png') ,blur_i),
                                                input_height,input_width,indexer=indexer,n_patches=n_patches)
                if elastic_aug:  
                    indexer=get_patches_num(dir_flow_train_imgs,dir_flow_train_labels,
                                            elastic_transform( cv2.imread(dir_img+'/'+im) ,alpha=800, sigma=12,seedj=indexer),
                                            elastic_transform( cv2.imread(dir_seg+'/'+img_name+'.png') ,alpha=800,sigma=12,seedj=indexer),
                                            input_height,input_width,indexer=indexer,n_patches=n_patches)             
        

                if scaling:  
                    for sc_ind in scales:
                        indexer=get_patches_num_scale(dir_flow_train_imgs,dir_flow_train_labels,
                                                 cv2.imread(dir_img+'/'+im) ,
                                                 cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                input_height,input_width,indexer=indexer,n_patches=n_patches,scaler=sc_ind)
                if binarization:
                    indexer=get_patches_num(dir_flow_train_imgs,dir_flow_train_labels,
                                            otsu_copy( cv2.imread(dir_img+'/'+im)),
                                            otsu_copy( cv2.imread(dir_seg+'/'+img_name+'.png')),
                                            input_height,input_width,indexer=indexer,n_patches=n_patches)
    
    
    
    
    
    
    
    


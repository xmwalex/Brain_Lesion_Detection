#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:39:04 2018
Python 2.7
@author: ximing wang

Train a U-net model for the phone datasets.

The initial weights are essential for successful training. 
In little chances, the optimization will be trapped to local minima, so I set the initial weights for repeatable results.If you start training for 1 epoch and see accuracy is 0.00, please stop the program and restart training. 

Usage: python train_phone_finder.py find_phone

"""
from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
import os
import sys
import glob
from keras.layers import Input, Conv2D, Concatenate,MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import ModelCheckpoint

##########################################################
# define the dice coefficient function and the loss function basd on dice coefficient
#########################################################    

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

##########################################################
# define the function for calculate the true positive rate
#########################################################    

def accuracy(infos):
    tp=0
    fp=0
    for i in infos:
        if(abs(float(i[1])-i[3])<0.05 and abs(float(i[2])-i[4])<0.05):
            tp+=1
        else:
            fp+=1
    return float(tp)/(tp+fp)

def calc_accuracy(images,infos,model,left,top,width,height):
    pred=model.predict(images,batch_size=5)
    #
    for i in range(0,len(pred)):
        predimg = label(pred[i,:,:,0].squeeze())
        regions = regionprops(predimg)
        if(not regions):
            infos[i][3:]=[.5,.5]
        else:
            area = [ele.area for ele in regions]
            largest_blob_ind = np.argmax(area)
            ind_pred=np.divide(np.add(list(regions[largest_blob_ind].centroid),[left,top]),[width,height])
            infos[i][3:]=[ind_pred[1],ind_pred[0]]
        
    return accuracy(infos)


##########################################################
# define the Unet with image size as 320*480. 
#########################################################    

img_rows = 320
img_cols = 480
from keras.initializers import RandomUniform
def get_unet():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu',kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1,seed=1),bias_initializer='Zeros')(inputs)    
    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation='relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation='relu')(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation='relu')(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation='relu')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), padding="same", activation='relu')(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation='relu')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), padding="same", activation='relu')(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation='relu')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), padding="same", activation='relu')(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation='relu')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), padding="same", activation='relu')(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)   

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef]) 

    return model


##########################################################
# define the data generator. Since there are only about 100 images, I use shifting,
#rotation, flip,zoom, to augment the data
#########################################################    


SEED=10
def phone_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.5,
            height_shift_range=0.5,
            rotation_range=90,
            fill_mode='wrap',
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.5,
            height_shift_range=0.5,
            rotation_range=90,
            fill_mode='wrap',
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def main():
    
    ##########################################################
    # load label information, create folders
    #########################################################    
    
    input_folder= sys.argv[1]
    label_path = input_folder+'/labels.txt'
    
    with open(label_path) as f: # load label information
        content=f.readlines()
    
    infos=[]
    for line in content:
        info = line.strip('\n').split(' ')
        infos.append(info)
    
    infos=sorted(infos)
    
    images=np.zeros([len(infos),img_rows,img_cols,3])
    masks=np.zeros([len(infos),img_rows,img_cols,1])
    ##########################################################
    # preprocess the images. Created masks based on the lable infomation.
    #########################################################    
    for i in range(0,len(infos)):
        img_path = input_folder+'/'+infos[i][0]
        bgr_img = cv2.imread(img_path)
        width,height = bgr_img.shape[:2]
        left = (width-img_rows)/2
        right = width-img_rows-left
        top = (height-img_cols)/2
        bottom = height-img_cols-top
        
        tmp_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        infos[i][1:]=map(float,infos[i][1:])
        
        expand = 0.05*max(tmp_img.shape)
        x_cor=[float(infos[i][1])*tmp_img.shape[1]-expand,float(infos[i][1])*tmp_img.shape[1]+expand]
        y_cor=[float(infos[i][2])*tmp_img.shape[0]-expand,float(infos[i][2])*tmp_img.shape[0]+expand]
        x_cor=map(int,map(round,x_cor))
        y_cor=map(int,map(round,y_cor))
        
        f=lambda x,minbound,maxbound:[max(minbound,x[0]),min(maxbound,x[1])]
        x_cor = f(x_cor,0,tmp_img.shape[1])
        y_cor = f(y_cor,0,tmp_img.shape[0])    
        
        phoneseed = (tmp_img[y_cor[0]:y_cor[1],x_cor[0]:x_cor[1]])
        thresh=np.bitwise_or(phoneseed>200,phoneseed<100)# use simple thresholding to roughly segment phone
        thresh = np.array(thresh, dtype=np.uint8)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        mask = np.zeros(tmp_img.shape)
        mask[y_cor[0]:y_cor[1],x_cor[0]:x_cor[1]] = closing*255
    ##########################################################
    #   save all images and masks in to folder
    #########################################################    
    
    
        mask_cropped=mask[left:-right,top:-bottom]
        bgr_cropped = bgr_img[left:-right,top:-bottom]
    
    ##########################################################
    #   save all images and masks in to folder
    #########################################################    
        images[i,:,:,:]=bgr_cropped
        masks[i,:,:,0]=mask_cropped
    
    
    ##########################################################
    # split the train data set and test data set by 90% vs 10%
    #########################################################    
    X_train, X_test, y_train,  y_test,infos_train,infos_test = train_test_split(images, masks,infos, test_size=0.1)
    print('Training input is', X_train.shape)
    print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
    print('Testing set is', X_test.shape)
    
    
    ##########################################################
    # parepare the model for training
    #########################################################    
    model = get_unet()
    print(model.summary())
    
    
    ##########################################################
    # Use binary crossentroy as loss function. train for 10 ephocs. 
    #You can also try to use the dice coeficient as loss function, which is commented below
    #########################################################    
    
    weight_saver = ModelCheckpoint('best.h5', monitor='val_dice_coef', 
                                                  save_best_only=True, save_weights_only=True)
    
    
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef]) #ecobill
    
    history = model.fit_generator(phone_generator(X_train, y_train, 5),
                               steps_per_epoch = 300,
                               validation_data = (X_test, y_test),
                               epochs=3, verbose=1,
                               callbacks = [weight_saver])
    
    model.compile(optimizer=Adam(3e-5), loss='binary_crossentropy', metrics=['accuracy',dice_coef]) #ecobill
    
    history = model.fit_generator(phone_generator(X_train, y_train, 5),
                               steps_per_epoch = 300,
                               validation_data = (X_test, y_test),
                               epochs=3, verbose=1,
                               callbacks = [weight_saver])
    
    
    #
    #model.compile(optimizer=Adam(lr=3.0e-5), loss=dice_coef_loss, metrics=['accuracy',dice_coef])
    #
    #history = model.fit_generator(phone_generator(X_train, y_train, 5),
    #                           steps_per_epoch = 300,
    #                           validation_data = (X_test, y_test),
    #                           epochs=5, verbose=1,
    #                           callbacks = [weight_saver])
    
    model.save_weights('final.h5')
    
    ##########################################################
    # evaluate the model
    #########################################################    
    
    print("Training set accuracy:")
    print(calc_accuracy(X_train,infos_train,model,left,top,width,height))
    
    print("Test set accuracy:")
    print(calc_accuracy(X_test,infos_test,model,left,top,width,height))
    
    print("Total set accuracy:")
    print(calc_accuracy(images,infos,model,left,top,width,height))

if __name__ == "__main__":
    main()

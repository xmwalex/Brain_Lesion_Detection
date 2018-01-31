#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:11:30 2018
Python 2.7
@author: ximing wang

Usage: python phone_finder.py find_phone/0.jpg
    
"""
from __future__ import print_function

import numpy as np
import cv2
from skimage.measure import label, regionprops
import os
import sys
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.optimizers import Adam


##########################################################
# define the Unet with image size as 320*480. 
#########################################################    

img_rows = 320
img_cols = 480

def get_unet():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)    
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

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)   #9

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy']) 

    return model



def main():
    ##########################################################
    # Load the weights for the model
    #########################################################    
    
    model = get_unet()
    
    if(os.path.isfile('final.h5')):
        model.load_weights('final.h5')
#    if(os.path.isfile('best.h5')):
#        model.load_weights('best.h5')
    else:
        print('weights not found! Please train the model first and place the weights file at the same directory')
        sys.exit()
    
    ##########################################################
    # Load and preprocess the image
    #########################################################    
    
    img_path = sys.argv[1]
    bgr_img = cv2.imread(img_path)
    width,height = bgr_img.shape[:2]
    left = (width-img_rows)/2
    right = width-img_rows-left
    top = (height-img_cols)/2
    bottom = height-img_cols-top
    
    bgr_clipped = bgr_img[left:-right,top:-bottom]
    image=np.expand_dims(bgr_clipped,axis=0)
    
    ##########################################################
    # Make predictions
    #########################################################    
    
    pred=model.predict(image)
    for i in range(0,len(pred)):
        predimg = label(pred[i,:,:,0].squeeze())
        regions = regionprops(predimg)
        if(not regions):# in case no phones are found, just output the center of the image
            print([.5,.5])
        else:
            area = [ele.area for ele in regions]
            largest_blob_ind = np.argmax(area)
            ind_pred=np.divide(np.add(list(regions[largest_blob_ind].centroid),[left,top]),[width,height])
            print([ind_pred[1],ind_pred[0]])

if __name__ == "__main__":
    main()            

# lesion_detectio
A simple U-net algorithm to segment lesion in CT images
Environment:
python 2.7
keras 2.1.2
tensorflow 1.4.1

The script requires installation of keras and tensorflow.
to install tensorflow, please follow the instructions at:
https://www.tensorflow.org/install/install_linux

to install keras, please follow instructions at:
https://keras.io/#installation

Usage:
1. prepare the training images and masks. 
Put training images, masks and a 'label.txt' file into one folder
Training images are images with lesions
Masks are segmented lesion mask which is used as gold standard
label.txt should list all images and corresponding masks file name
e.g. 
1.dicom 1_mask.dicom
2.dicom 2_mask.png

Here I assume your images are in dicom format and the mask is in png format. 
You are free the modify the code to handle nii files etc.

2. Train
use command like:
'python train_lesion__detecter.py folder_path'

the folder_path is the folder where you put all data in the first step

3. Predict
use command like:
'python lesion_finder.py img_path'

the img_path is the image you want to detect lesion

4. batch predict
use command like:
'python lesion_finder_batch.py folder_path'

the folder_path is the folder of all images you want to detect lesions






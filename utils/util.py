'''
Utilities file for PSIG-GAN

Uses Tensorflow 2.0 and tf.keras

Currently implemented: 
    - Creating databatches using tf.keras.preprocessing
'''

import tensorflow as tf 
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img

from PIL import Image 

def createDataBatch(img_dir,batch_size):

    '''
    Creates a tf.keras image dataset based off of the img_dir param
    Loads in all of the images and organizes into batches of shape [batch_size, img_wid, img_hgt, num_color_channels]
    Returns a tf.keras.Dataset object

    @param img_dir : <<string>> path to a directory of images
    @param batch_size: <<int>> number of images per batch

    Returns a 
    '''

    datagen = ImageDataGenerator(
                                                    featurewise_center=False,
                                                    samplewise_center=False,
                                                    featurewise_std_normalization=False,
                                                    samplewise_std_normalization=False,
                                                    zca_whitening=False,
                                                    zca_epsilon=1e-06,
                                                    rotation_range=0,
                                                    width_shift_range=0.0,
                                                    height_shift_range=0.0,
                                                    brightness_range=None,
                                                    shear_range=0.0,
                                                    zoom_range=0.0,
                                                    channel_shift_range=0.0,
                                                    fill_mode="nearest",
                                                    cval=0.0,
                                                    horizontal_flip=False,
                                                    vertical_flip=False,
                                                    rescale=None,
                                                    preprocessing_function=None,
                                                    data_format=None,
                                                    validation_split=0.0,
                                                    dtype=None,
                                                    )
    
    # Create image dataset using datagen.flow_from_directory()
    dataset = datagen.flow_from_directory(
                                        img_dir,
                                        target_size=(256, 256),
                                        color_mode="rgb",
                                        classes=None,
                                        class_mode="binary",
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=None,
                                        save_to_dir=None,
                                        save_prefix="",
                                        save_format="png",
                                        follow_links=False,
                                        subset=None,
                                        interpolation="nearest",
                                        )

    return dataset

def save_image_batch(image_tensor, write_dir):

    '''
    Saves generated TensorFlow images to .png files using OpenCV

    @param image_tensor: <<tf Tensor>> must be 4D for [num_images, img_wid, img_hgt, num_color_channels]
    @param write_dir : <<string>> directory to save images to 

    '''    

    for i in range(image_tensor.shape[0]):
        write_path = write_dir + "/generated_{}.png".format(i)
        save_img(write_path, image_tensor[i,:,:,:])

 
if __name__ == "__main__":

    '''
    scrap test code 
    ''' 

    path = '/home/data/dcgan-data/'
    batch_size = 64
    iter = createDataBatch(path, batch_size)

    for i in range(len(iter)):
        batch,_ = iter.next()
        print(i) 
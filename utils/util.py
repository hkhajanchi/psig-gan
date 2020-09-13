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


def createDataBatch(img_dir,batch_size,scaleDim):

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
                                        target_size=(2000,2000),
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


def normalize(image_tensor):
     '''
     Converts an RGB image into normalized greyscale 
     '''
     images = []

     for i in range(image_tensor.shape[0]):
        image = image_tensor[i,:,:,:]

        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.expand_dims(image[:,:,0], axis=2)
        images.append(image)

     images = tf.convert_to_tensor(images)
     return images 

def cropCentralImage(image_tensor,frac): 
    '''
    Crops the central fraction of the image tensor 
    '''
    return tf.image.central_crop(image_tensor, frac)

if __name__ == "__main__":

    '''
    scrap test code 
    ''' 

    path = '/home/data/GrassClover/'
    batch_size = 64
    iter = createDataBatch(path, batch_size, 512)

    images,_ = iter.next()
    print(images.shape)

    things = cropCentralImage(images, 0.25)
    print(things.shape)

    save_image_batch(things, os.getcwd()+'/test')


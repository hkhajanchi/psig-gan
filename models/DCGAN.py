import tensorflow as tf
import tf.keras as keras
from tensorflow.keras import layers

from base import GAN
"""
TF 2.0 Implementation of DCGAN to use for baseline testing of PSIG-GAN
Implements the abstract GAN class specified in base.py

"""

class DCGAN(GAN):


    def __init__(self,latent_shape, output_image_shape, num_gen_images):

        """
        Creates a DCGAN with Keras-based models for both Discriminator and Generator 
        Includes loss computation for both models and training functionality

        Constructor args:
            @param latent_shape: tuple of ints identifying the shape of the seed fed to the generator [num_imgs, seed_dim]
            @param output_image_shape: tuple of ints specifying the generator's output shape. Should be 512x512x3 for RGB based images
            @param num_gen_images: int specifying the batch size of images to be generated

        """

        self.latent_shape = latent_shape
        self.output_shape = [num_gen_images, output_image_shape]

        # Instantiate generator models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    
    def _build_discriminator(self): 

        '''
        Builds a discriminator network (binary classification using DCNN)
        Takes in an image, and predicts whether the image is fake or real
        Implemented using Tensorflow 2.0

        Architecture:
            - Convolution layers : 
            - Pooling layers :
            - Batch Norm :
            - Dense : 
            - Activations: 

        The input image size will be the same as the output image size that is generated by the generator,
        hence: discrim_input_shape = self.output_image_shape
        '''

        discrim_input_shape = self.output_shape

        model = keras.Sequential()

        return model

    def _build_generator(self):
        
        ''' 
        Builds a generator model that generates an image based on a random latent vector
        Upsamples the latent vector using transposed convolutions and reshaping
        Implemented with Tensorflow 2.0

        Architecture:
        - Convolution layers : 
        - Pooling layers :
        - Batch Norm :
        - Dense : 
        - Activations: 


        '''

        gen_input_shape = self.latent_shape

        model = keras.Sequential()

        return model 

    def train_step(self, num_epochs, fake_data_batch, real_data_batch):

        """
        Implements the training routine for the DCGAN framework 
        Implemented using Tensorflow 2.0

        @param num_epochs: number of epochs for training
        @param fake_data_batch: 4D tensor containing fake images - [batch_size, img_len, img_wid, num_channels]
        @param real_data_batch: 4D tensor containing real images sampled from GrassWeeds repo - [batch_size, img_len, img_wid, num_channels]
        
        """
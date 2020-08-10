'''
PSIG-GAN
Training loop for GAN runs 

'''
from comet_ml import Experiment 
import tensorflow as tf 
from utils import util 
from models.DCGAN_Rev3 import DCGAN
import tensorflow.keras as keras
import os 
import datetime
import time
import math 

def train (GAN, data_batch, epochs, run_dir, gen_lr, disc_lr, gen_train_freq, disc_train_freq, logger):

    """
    --- GAN Training Loop -- 

    For every epoch: 
        - go through each batch in dataset 
        - run DCGAN's train_step() function
    - Every 15 epochs: 
        - save checkpoint 

    @param GAN: <<GAN Object>> a GAN object that implements the base.py GAN model 
    @param data_batch: <<TensorFlow Data Iterator>> iterator containing real images for discriminator training
    @param epochs : <<int>> number of training epochs
    @param run_dir : <<str>> path to save all training run data 
    @param gen_lr: <<float>> learning rate for ADAM optimizer (generator)
    @param disc_lr: <<float>> learning rate for discriminator Adam optimizer 
    @param logger:?
    @param gen_train_freq: generator training frequency per epoch
    @param disc_train_freq: discriminator training frequency per epoch
    """

    for epoch in range(epochs):
        
        # Create directory to save run images 
        epoch_save_path = run_dir + '/epoch_{}'.format(epoch)
        os.mkdir(epoch_save_path)

        start = time.time()
        batch_ctr = 0

         # lr decay function
        initial_lr = 0.01
        def lr_decay(initial_lr, epoch):
            return initial_lr * math.pow(0.666, epoch)
        lr = lr_decay(initial_lr, epoch)
        # lr schedule
        gen_lr = lr
        disc_lr = lr

        for i in range(len(data_batch)):
            
            # Extract image tensor and disregard labels
            real_data,_ = data_batch.next()

            # Train GAN 
            generated = GAN.train_step(real_data,gen_lr,disc_lr,gen_train_freq, disc_train_freq, epoch, logger)

            # Save images to epoch dir 
            util.save_image_batch(generated, epoch_save_path)

            print("Batch {} completed".format(i))

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def trainLoop(num_epochs, batch_size, gen_lr, disc_lr, gen_train_freq, disc_train_freq, logger):
        # Create run directory for current training run 
        os.chdir(os.path.expanduser('~') + '/psig-gan/runs/')
        run_dir = str(datetime.datetime.now()).replace(' ','',) + "--" + "disc_lr"+ str(disc_lr) + " " + "gen_lr"+ str(gen_lr) + " " + "gen_train_freq" + str(gen_train_freq) + " " + "disc_train_freq" + str(disc_train_freq) + " "  + "num_epochs"+ str(num_epochs)

        os.mkdir(run_dir)
        # Instantiate GAN
        gan = DCGAN(latent_shape=100, output_image_shape=256, num_gen_images=batch_size, gen_filter_size=5, discrim_filter_size=5, gen_num_channels=128, discrim_num_channels=64)
        # Load real grassweeds image data 
        data_path = '/home/data/dcgan-data/'
        data_batch = util.createDataBatch(data_path,batch_size)
        # Define Comet-ML API key here for error logging
        comet_api_key = 'Gdy4QDrOmu0P01XuBI33rPuIS'
        #Define Comet Project Name
        logger = Experiment(comet_api_key, project_name="psig-gan")
        
        #Define experiment name
        logger.set_name("disc_lr"+ str(disc_lr) + " " + "gen_lr"+ str(gen_lr) + " " + "gen_train_freq"+ str(gen_train_freq) + " " + "disc_train_freq" + str(disc_train_freq) + " "  + "num_epochs"+ str(num_epochs))
        
        # Execute training loop
        train(gan, data_batch, num_epochs, run_dir, gen_lr, disc_lr, gen_train_freq, disc_train_freq, logger)
    
if __name__ == "__main__":
    # Set GPU here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define hyperparams for GAN training
    num_epochs = 200
    batch_size = 32
    gen_lr  = 1e-2
    disc_lr = 1e-2
    gen_train_freq = 1
    disc_train_freq = 1
    # Define Comet-ML API key here for error logging
    comet_api_key = 'Gdy4QDrOmu0P01XuBI33rPuIS'
    #Define Comet Project Name
    logger = Experiment(comet_api_key, project_name="psig-gan")
    # Execute training loop
    trainLoop(num_epochs, batch_size, gen_lr, disc_lr, gen_train_freq, disc_train_freq,logger)
    
    gen_lr  = 1e-1
    disc_lr = 1e-5
    # Execute training loop
    trainLoop(num_epochs, batch_size, gen_lr, disc_lr, gen_train_freq, disc_train_freq,logger)

    """ gen_lr  = 1e-4
    disc_lr = 1e-4
    # Execute training loop
    trainLoop(num_epochs, batch_size, gen_lr, disc_lr, gen_train_freq, disc_train_freq, logger)
    
    gen_lr  = 1e-4
    disc_lr = 1e-3
    # Execute training loop
    trainLoop(num_epochs, batch_size, gen_lr, disc_lr, gen_train_freq, disc_train_freq, logger) """

    #Terminal Summary
    # gan.generator.summary(line_length=None, positions=None, print_fn=None)
    # gan.discriminator.summary(line_length=None, positions=None, print_fn=None)

    #Graphic Summary
    # tf.keras.utils.plot_model(gan.generator, to_file="generator.png", show_shapes=True, show_layer_names=True)
    # tf.keras.utils.plot_model(gan.discriminator, to_file="discriminator.png", show_shapes=True, show_layer_names=True) """

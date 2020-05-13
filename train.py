'''
PSIG-GAN
Training loop for GAN runs 

'''
import tensorflow as tf 
from utils import util 
from models.DCGAN import DCGAN
import os 
import datetime
import time 

def train (GAN, data_batch, epochs, run_dir, gen_lr, disc_lr):

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

    """

    for epoch in range(epochs):
        
        # Create directory to save run images 
        epoch_save_path = run_dir + '/epoch_{}'.format(epoch)
        os.mkdir(epoch_save_path)

        start = time.time()
        batch_ctr = 0

        for i in range(len(data_batch)):
            
            # Extract image tensor and disregard labels
            real_data,_ = data_batch.next()

            # Train GAN 
            generated = GAN.train_step(real_data,gen_lr,disc_lr)

            # Save images to epoch dir 
            util.save_image_batch(generated, epoch_save_path)

            print("Batch {} completed".format(i))

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


if __name__ == "__main__":

    # Define hyperparams for GAN training
    num_epochs = 100
    batch_size = 64
    gen_lr  = 1e-4
    disc_lr = 1e-4

    # Create run directory for current training run 
    os.chdir(os.path.expanduser('~') + '/Research/psig-gan/runs/')
    run_dir = str(datetime.datetime.now()).replace(' ','')
    os.mkdir(run_dir)

    # Instantiate GAN
    gan = DCGAN(latent_shape=100, output_image_shape=256, num_gen_images=batch_size, gen_filter_size=10, discrim_filter_size=10, gen_num_channels=128, discrim_num_channels=64)

    # Load real grassweeds image data 
    data_path = '/home/data/dcgan-data/'
    data_batch = util.createDataBatch(data_path,batch_size)

    # Execute training loop
    train(gan, data_batch, num_epochs, run_dir, gen_lr, disc_lr)
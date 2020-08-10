from comet_ml import Experiment
import tensorflow as tf 
from utils import util 
from models.DCGAN_BW import DCGAN
import os 
import datetime
import time
import math

def loss(preds):

    loss_fcn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_fcn(tf.ones_like(preds),preds)

    return real_loss


def train_discrim(GAN, disc_lr, data_batch, epochs, logger,batch_size):

    start = time.time()

    for i in range(len(data_batch) - 5):

        imgs,_ = data_batch.next()
        imgs = util.normalize(imgs)

        # lr decay function
        initial_lr = 0.01
        def lr_decay(initial_lr, i):
            return initial_lr * math.pow(0.25, i)
        lr = lr_decay(initial_lr, i)
        # lr schedule
        disc_lr = lr
        logger.log_metric ('Disc Learning Rate', disc_lr,step=i) 

        for epoch in range(epochs):

            start = time.time()
            gan.discTrain(disc_lr, imgs, epoch, logger)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        print("Batch {} completed".format(i))

    # check discrim validation loss 
    for j in range(5):

        imgs,_ = data_batch.next()
        imgs = util.normalize(imgs)

        preds = gan.discriminator(imgs)
        pred_loss = loss(preds)
        print("valid loss at iteration {}".format(i))
        print(pred_loss) 
        logger.log_metric ('Validation Loss', pred_loss,step=i)

        
        


if __name__ == "__main__":

    # set GPU here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define Comet-ML API key here for error logging
    comet_api_key = 'Gdy4QDrOmu0P01XuBI33rPuIS'

    #Define Comet Project Name
    logger = Experiment(comet_api_key, project_name="psig-gan")
        
    #Define experiment name
    logger.set_name("discrim-only training") #+ str( datetime.datetime.now().replace(' ','',)  )) 

    #define epochs and lr here
    
    num_epochs = 15
    batch_size = 64
    disc_lr = 0.01

    # Load real grassweeds image data 
    data_path = '/home/data/dcgan-data/'
    data_batch = util.createDataBatch(data_path,batch_size)

    #make gan object here 
    gan = DCGAN(latent_shape=100, output_image_shape=256, num_gen_images=batch_size, gen_filter_size=5, discrim_filter_size=5, gen_num_channels=128, discrim_num_channels=64)

    # call discrim training loop
    train_discrim(gan,disc_lr, data_batch, num_epochs, logger, batch_size)
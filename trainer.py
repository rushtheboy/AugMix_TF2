import os
import sys
import time
import logging
import multiprocessing
from models.mymodel import resnet_v1_eembc_quantized
import numpy as np
import config
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import OrderedEnqueuer, Progbar
from models.resnet20 import resnet_v1
from data_generator import DataGenerator
from utils import CTLEarlyStopping, CTLHistory

seed=1234
np.random.seed(seed)
tf.random.set_seed(seed)

###################################################################################


def yaml_load(theconfig):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

# get parameters from yaml
input_shape = [32, 32, 3]
num_classes = 10
myconfig = yaml_load(os.path.join(sys.path[0], "tiny2.yml"), "r")
num_filters = myconfig['model']['filters']
kernel_sizes = myconfig['model']['kernels']
strides = myconfig['model']['strides']
l1p = float(myconfig['model']['l1'])
l2p = float(myconfig['model']['l2'])
skip = bool(myconfig['model']['skip'])
avg_pooling = bool(myconfig['model']['avg_pooling'])
batch_size = myconfig['fit']['batch_size']
num_epochs = myconfig['fit']['epochs']
#verbose = config['fit']['verbose']
#patience = config['fit']['patience']
#save_dir = config['save_dir']
model_name = myconfig['model']['name']
#loss = config['fit']['compile']['loss']
#model_file_path = os.path.join(save_dir, 'model_best.h5')

# quantization parameters
if 'quantized' in model_name:
    logit_total_bits = myconfig["quantization"]["logit_total_bits"]
    logit_int_bits = myconfig["quantization"]["logit_int_bits"]
    activation_total_bits = myconfig["quantization"]["activation_total_bits"]
    activation_int_bits = myconfig["quantization"]["activation_int_bits"]
    alpha = myconfig["quantization"]["alpha"]
    use_stochastic_rounding = myconfig["quantization"]["use_stochastic_rounding"]
    logit_quantizer = myconfig["quantization"]["logit_quantizer"]
    activation_quantizer = myconfig["quantization"]["activation_quantizer"]
    final_activation = bool(myconfig['model']['final_activation'])

#initial_lr = config['fit']['compile']['initial_lr']
#lr_decay = config['fit']['compile']['lr_decay']

# get the model instance
print("\nLoading model")
model = resnet_v1_eembc_quantized(input_shape=input_shape, num_classes=num_classes, l1p=l1p, l2p=l2p,
                              num_filters=num_filters,
                              kernel_sizes=kernel_sizes,
                              strides=strides,
                              logit_total_bits=logit_total_bits, logit_int_bits=logit_int_bits, activation_total_bits=activation_total_bits, activation_int_bits=activation_int_bits,
                              alpha=alpha, use_stochastic_rounding=use_stochastic_rounding,
                              logit_quantizer=logit_quantizer, activation_quantizer=activation_quantizer,
                              skip=skip,
                              avg_pooling=avg_pooling,
                              final_activation=final_activation)
model.summary()
print("")

# loss function to be used for training
kld = tf.keras.losses.KLDivergence()
entropy = tf.keras.losses.CategoricalCrossentropy()

# metric to keep track of 
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_accuracy = tf.keras.metrics.CategoricalAccuracy()
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

# initialize total steps to None and update it from config file
total_steps = None

lr_max = config.max_lr
lr_min = config.min_lr

# earlystopping for custom training loops
es = CTLEarlyStopping(monitor="val_loss", mode="min", patience=5)

# history object to plot and save progression in the end
history = CTLHistory(filename=config.plot_name)

###################################################################################


def jsd_loss_fn(y_true, y_pred_clean, y_pred_aug1, y_pred_aug2):
    # cross entropy loss that is used for clean images only
    loss = entropy(y_true, y_pred_clean)

    mixture = (y_pred_clean + y_pred_aug1 + y_pred_aug2) / 3.

    loss += 12. * (kld(y_pred_clean, mixture) + 
                   kld(y_pred_aug1, mixture) +
                   kld(y_pred_aug2, mixture)) / 3.
    return loss

###################################################################################

# learning rate scheduler
def get_lr(step):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

###################################################################################

@tf.function
def train_step(clean, aug1, aug2, labels, optim):
    with tf.GradientTape() as tape:
        # get predictions on clean images
        y_pred_clean = model(clean, training=True)
        
        # get predictions on augmented images
        y_pred_aug1 = model(aug1, training=True)
        y_pred_aug2 = model(aug2, training=True)

        # calculate loss
        loss_value = jsd_loss_fn(y_true = labels, 
                            y_pred_clean = y_pred_clean,
                            y_pred_aug1 = y_pred_aug1,
                            y_pred_aug2 = y_pred_aug2)
        
    grads = tape.gradient(loss_value, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value, y_pred_clean



@tf.function
def validate_step(images, labels):
    y_pred = model(images, training=False)
    loss = entropy(labels, y_pred)
    return loss, y_pred



###################################################################################


def train(training_data, 
            validation_data, 
            batch_size=32, 
            nb_epochs=100,
            min_lr=1e-5,
            max_lr=1.0,
            save_dir_path=""):
    

    x_train, y_train, y_train_cat = training_data
    x_test, y_test, y_test_cat = validation_data
    test_indices = np.arange(len(x_test))

    # get the training data generator. We are not using validation generator because the 
    # data is already loaded in memory and we don't have to perform any extra operation 
    # apart from loading the validation images and validation labels.
    ds = DataGenerator(x_train, y_train_cat, batch_size=batch_size)
    enqueuer = OrderedEnqueuer(ds, use_multiprocessing=True)
    enqueuer.start(workers=multiprocessing.cpu_count())
    train_ds = enqueuer.get()

    # get the total number of training and validation steps
    nb_train_steps = int(np.ceil(len(x_train) / batch_size))
    nb_test_steps = int(np.ceil(len(x_test) / batch_size))
    
    global total_steps
    total_steps = nb_train_steps * config.num_epochs


    # get the optimizer
    # SGD with cosine lr is causing NaNs. Need to investigate more
    optim = optimizers.Adam(learning_rate=0.0001) 
                           
    
    # checkpoint prefix
    checkpoint_prefix = os.path.join(save_dir_path, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optim, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, 
                                                    directory=save_dir_path,
                                                    max_to_keep=10)
    
    # check for previous checkpoints, if any
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Checkpoint restored from {}".format(checkpoint_manager.latest_checkpoint))
        starting_epoch = checkpoint.save_counter.numpy()       
    else:
        print("Initializing from scratch.")
        starting_epoch = 0
        
    # sanity check for epoch number. For example, if someone restored a checkpoint
    # from 15th epoch and want to train for two more epochs, then we need to explicitly
    # encode this logic in the for loop
    if nb_epochs <= starting_epoch:
        nb_epochs += starting_epoch
    
    
    for epoch in range(starting_epoch, nb_epochs):
        pbar = Progbar(target=nb_train_steps, interval=0.5, width=30)

        # Train for an epoch and keep track of 
        # loss and accracy for each batch.
        for bno, (images, labels) in enumerate(train_ds):
            if bno == nb_train_steps:
                break

            # Get the batch data 
            clean, aug1, aug2 = images
            loss_value, y_pred_clean = train_step(clean, aug1, aug2, labels, optim)

            # Record batch loss and batch accuracy
            train_loss(loss_value)
            train_accuracy(labels, y_pred_clean)
            pbar.update(bno+1)
    
        # Validate after each epoch
        for bno in range(nb_test_steps):
            # Get the indices for the current batch
            indices = test_indices[bno*batch_size:(bno + 1)*batch_size]
            
            # Get the data 
            images, labels = x_test[indices], y_test_cat[indices]

            # Get the predicitions and loss for this batch
            loss_value, y_pred = validate_step(images, labels)

            # Record batch loss and accuracy
            test_loss(loss_value)
            test_accuracy(labels, y_pred)

        
        # get training and validataion stats
        # after one epoch is completed 
        loss = train_loss.result()
        acc =  train_accuracy.result()
        val_loss = test_loss.result()
        val_acc = test_accuracy.result()

        # record values in the history object
        history.update([loss, acc], [val_loss, val_acc])

        # print loss values and accuracy values for each epoch 
        # for both training as well as validation sets
        print(f"""Epoch: {epoch+1} 
                train_loss: {loss:.6f}  train_acc: {acc*100:.2f}%  
                test_loss:  {val_loss:.6f}  test_acc:  {val_acc*100:.2f}%\n""")
        
        # get the model progress
        improved, stop_training = es.check_progress(val_loss)
        # check if performance of model has imporved or not
        if improved:
            print("Saving model checkpoint.")
            checkpoint.save(checkpoint_prefix)
        if stop_training:
            break
            
        # plot and save progression
        history.plot_and_save(initial_epoch=starting_epoch)
                
        
        # Reset the losses and accuracy
        train_loss.reset_states() 
        train_accuracy.reset_states()
        test_loss.reset_states() 
        test_accuracy.reset_states()
        print("")
        print("*"*78)
        print("")

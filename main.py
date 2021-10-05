import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers,optimizers

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_type",type=str,help="Model Type")
    parser.add_argument("-b","--batchsize",type=int,default=64,help="batch size")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="learning rate")
    parser.add_argument("-e","--epochs",type=int,default=300,help="Epochs")
    parser.add_argument("-o","--output_model_folder",type=str,default="./output_model_folder",help="Folder to save .h5 models")
    parser.add_argument("-oh","--output_training_hist_folder",type=str,default="./training_hists",help="Folder to save training loss and accuracy histories (train loss,train acc ,test loss ,test acc) of each model")
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    model_type = args.model_type
    batchsize = args.batchsize
    learning_rate = args.learning_rate
    epochs = args.epochs
    output_model_folder = args.output_model_folder
    output_training_hist_folder = args.output_training_hist_folder

    device = tf.test.gpu_device_name()
    with tf.device(device):
        if device:
            print("Device : ", device)
        else:
            print("Device : CPU ")
        #Get dataset
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()

        #one-hot coding
        y_train = to_categorical(y_train,num_classes=10)
        y_test = to_categorical(y_test,num_classes=10)

        #Get mean and std of training dataset for normalization
        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))

        #Get data augmentation
        train_aug = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing((32,32)),
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.04),
        ])

        test_aug =  layers.experimental.preprocessing.Resizing((32,32))

        #Get train and test dataset with batchsize and data augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        train_dataset = (
            train_dataset
            .batch(batchsize)
            .map(lambda x,y: (train_aug(x),y),num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x,y: ((x-mean)/(std+1e-7),y),num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        test_dataset = (
            test_dataset
            .batch(batchsize)
            .map(lambda x,y: (test_aug(x),y),num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x,y: ((x-mean)/(std+1e-7),y),num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
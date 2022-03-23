import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from keras import layers
from keras.layers import Rescaling
import pandas as pd
from keras import preprocessing
import tensorflow_datasets
import math
import os

# Import images and split into training and validation sets
from keras import callbacks
from keras.layers import BatchNormalization


path = '/Users/alexk/Documents/GitHub/LeNet-MobileNetV2-For-Binary-Classification-of-Infectious-Keratitis/Images'
os.chdir("/Users/alexk/Documents/GitHub/LeNet-MobileNetV2-For-Binary-Classification-of-Infectious-Keratitis/Images")

train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                   vertical_flip=True,
                                   horizontal_flip=True,  # horizontal flip
                                   validation_split=0.2)

train_ds = train_datagen.flow_from_directory(
    directory="/Users/alexk/Documents/GitHub/WSSEF-Project/data",
    subset='training',
    seed=42,
    batch_size=64,
    target_size=(256, 256),
    class_mode='binary',
    classes=['Bacterial', 'Fungal'])

val_ds = train_datagen.flow_from_directory(
    directory="/Users/alexk/Documents/GitHub/WSSEF-Project/data",
    subset='validation',
    seed=42,
    batch_size=64,
    shuffle=True,
    class_mode='binary',
    target_size=(256, 256),
    classes=['Bacterial', 'Fungal'])

train_dataset = keras.preprocessing.image_dataset_from_directory(
    directory=path,
    subset='training',
    seed=42,
    shuffle=True,
    batch_size=64,
    image_size=(256, 256),
    validation_split=0.2,
    class_names=['Bacterial', 'Fungal'])

# Visualize example images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

def LeNet_model(input_shape=(256, 256, 3), num_classes=2):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    #device = 'gpu:0' if not tf.test.is_gpu_available() else 'cpu'
    #with tf.device(device):
    #    model.add(tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=42))
    model.add(tf.keras.layers.Rescaling(1. / 255))

    # First set of CONV_RELU_POOL layers
    model.add(layers.Conv2D(filters=20, kernel_size=5, padding='same', input_shape=input_shape))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    # Second set of CONV_RELU_Pool layers
    model.add(layers.Conv2D(filters=50, kernel_size=5, padding='same'))
    model.add(layers.ReLU())

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    # Flatten
    model.add(layers.Flatten())

    # Dense layer
    model.add(layers.Dense(units=1, activation='sigmoid'))

    return model


LeNet_model = LeNet_model()
LeNet_model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
LeNet_model.summary()

history = LeNet_model.fit(
    train_ds, epochs=50, callbacks=callbacks.Callback(),
    validation_data=val_ds
)

df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc[['loss', 'val_loss']]
df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
df_auc = df_loss_acc[['auc', 'val_auc']]
df_auc.rename(columns={'auc': 'train', 'val_auc': 'validation'}, inplace=True)
df_loss.plot(title='Model loss', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Loss')
df_auc.plot(title='Model AUC', figsize=(12, 8)).set(xlabel='Epoch', ylabel='AUC')
df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')
plt.show()
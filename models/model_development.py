import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import preprocessing
from tensorflow import keras
from keras import layers
from keras import regularizers
import pandas as pd

hyperparameters = {'batch_size': 128,
                   'random_flip': 'horizontal_and_vertical',
                   'random_rotation': (0.2),
                   'dropout': 0.5,
                   'L2': 0.3,
                   'base_LR': 0.001,
                   'initial_epochs': 20,
                   'fine_tune_epochs': 50,
                   'frozen_layer': 72}

BATCH_SIZE = hyperparameters['batch_size']
IMG_SIZE = (224, 224)
path = "/Users/alexk/Documents/GitHub/LeNet-MobileNetV2-For-Binary-Classification-of-Infectious-Keratitis/Images"

train_dataset = keras.preprocessing.image_dataset_from_directory(path,
                                                                 shuffle=True,
                                                                 subset='training',
                                                                 seed=42,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 validation_split=0.2)
validation_dataset = keras.preprocessing.image_dataset_from_directory(path,
                                                                      shuffle=True,
                                                                      subset='validation',
                                                                      seed=42,
                                                                      batch_size=BATCH_SIZE,
                                                                      validation_split=0.2,
                                                                      image_size=IMG_SIZE)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
    def data_augmenter():
        """
            Create a sequential model composed of horizontal flips and random contrast adjustments
        """
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(hyperparameters['random_flip']),
            tf.keras.layers.RandomRotation(factor=hyperparameters['random_rotation'])])
        return data_augmentation

data_augmentation = data_augmenter()

IMG_SHAPE = IMG_SIZE + (3,)
conv_base = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')


def MobileNetUlcerModel(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    # freeze the convolutional base
    conv_base.trainable = False

    # create the input layer
    inputs = tf.keras.Input(shape=IMG_SHAPE)

    # apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # data preprocessing using the same weights as the original pre-trained model
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = conv_base(x, training=False)

    # Add the new binary classification layers
    # global average pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # include dropout for regularization effect
    x = tf.keras.layers.Dropout(hyperparameters['dropout'])(x)

    # Add binary prediction layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(hyperparameters['L2']))(
        x)

    model = tf.keras.Model(inputs, outputs)

    return model


model = MobileNetUlcerModel()

base_learning_rate = hyperparameters['base_LR']

# Check if base RMSProp has larger AUC than Adams
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
#              loss='binary_crossentropy',
#              metrics=['AUC', 'accuracy'])

# with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
#    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=hyperparameters['initial_epochs'])

conv_base = model.layers[4]

conv_base.trainable = True

for layer in conv_base.layers[:hyperparameters['frozen_layer']]:
    layer.trainable = False

# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1 * hyperparameters['base_LR']),
#              loss='binary_crossentropy',
#              metrics=['AUC', 'accuracy'])

# history_tuning = model.fit(train_dataset,
#                           epochs=hyperparameters['fine_tune_epochs'],
#                           validation_data=validation_dataset)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1 * hyperparameters['base_LR']),
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC'])
# RMSprop (gradient-based learning optimizer leads to significant overfitting
# Higher AUC compared to validation accuracy indicates an inefficient classifier for RMSprop optimizer
with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
    history = model.fit(train_dataset,
                        epochs=hyperparameters['fine_tune_epochs'],
                        validation_data=validation_dataset)

# Achieves a 0.9552 peak validation accuracy indicating that MobileNet can handle data augmentation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1 * hyperparameters['base_LR']),
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC'])

with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
    history_tuning = model.fit(train_dataset,
                               epochs=hyperparameters['fine_tune_epochs'],
                               validation_data=validation_dataset)

df_loss_acc = pd.DataFrame(history_tuning.history)
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

model.save('/Users/alexk/Documents/GitHub/LeNet-MobileNetV2-For-Binary-Classification-of-Infectious-Keratitis'
           '/Model_Data/MobileNet_whole_image_fine_tune')

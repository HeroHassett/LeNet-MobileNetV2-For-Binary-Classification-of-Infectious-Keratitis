import tensorflow as tf
import tensorflow_datasets as tfds
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print(tf.config.experimental.list_physical_devices('CPU'))

#train_data = keras.preprocessing.image_dataset_from_directory(
#    path,
#    validation_split=0.2,
#    subset='training',
#    seed=42,
#    batch_size=64,
#    image_size=(256, 256),
#    class_names=['Bacterial', 'Fungal'])

# Visualize example images
#plt.figure(figsize=(10, 10))
#for images, labels in train_data.take(1):
#    for i in range(9):
#        ax = plt.subplot(3, 3, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(int(labels[i]))
#        plt.axis("off")

#val_ds = val_datagen.flow_from_directory(
#    directory="/Users/alexk/Documents/GitHub/WSSEF-Project/data",
#    subset='validation',
#    seed=42,
#    color_mode="rgb",
#    batch_size=64,
#    shuffle=True,
#    class_mode='binary',
#    target_size=(256, 256),
#    classes=['Bacterial', 'Fungal'])

#train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
 #                                  vertical_flip=True,
 #                                  horizontal_flip=True,# horizontal flip
 #                                  validation_split=0.2)

#val_datagen = ImageDataGenerator(validation_split=0.2)

#train_ds = train_datagen.flow_from_directory(
#    directory="/Users/alexk/Documents/GitHub/WSSEF-Project/data",
#    subset='training',
#    seed=42,
#    batch_size=64,
#    color_mode="rgb",
#    target_size=(256, 256),
#    shuffle=True,
#    class_mode='binary',
#    classes=['Bacterial', 'Fungal'])

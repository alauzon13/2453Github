# This script will split the data into training, testing, and splitting 
# Since Chydoridae only has 2 images and Daphnia has 1 image, we will oversample
# (duplicate these classes) in our dataset to ensure these classes appear in the 
# training, testing, and validation sets.

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
import os 
import numpy as np
import tensorflow as tf 

# Set seed for reproducibility
seed = 1013
np.random.seed(seed)
tf.random.set_seed(seed)

# 0. Load particles csv and images 
particles = pd.read_csv("extracted_particles.csv")

# Function to load and resize images from file paths
def load_and_resize_images(file_paths, folder_path, target_size=(300, 300)):
    images = []
    for file_path in file_paths:
        img = load_img(os.path.join(folder_path, file_path), target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return images

vignettes_folder = "vignettes"


# 1. Oversampling
def oversample_class(df, class_name, target_count):
    class_df = df[df["Class"] == class_name]
    oversampled_class_df = class_df.sample(target_count, replace=True)
    return oversampled_class_df

target_count = 100 

# Oversample chydoridae and daphnia
chydoridae_oversampled = oversample_class(particles, 'Chydoridae', target_count)
daphnia_oversampled = oversample_class(particles, 'Daphnia', target_count)

particles_oversampled = pd.concat([particles, chydoridae_oversampled, daphnia_oversampled])


# 2. Split into train, test, validation 
train_val, test = train_test_split(particles_oversampled, test_size=0.2, stratify=particles_oversampled['Class'])
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['Class'])  



# 3. Data Augmentation in training set

# Object that will augment the training images
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Filter training data for Chydoridae and Daphnia
train_chydoridae = train[train['Class'] == 'Chydoridae']
train_daphnia = train[train['Class'] == 'Daphnia']

# Load images for Chydoridae and Daphnia
train_chydoridae_images = load_and_resize_images(train_chydoridae['Vignette'], vignettes_folder)
train_daphnia_images = load_and_resize_images(train_daphnia['Vignette'], vignettes_folder)

# Combine images and labels
train_images = np.concatenate((train_chydoridae_images, train_daphnia_images), axis=0)
train_labels = list(train_chydoridae['Class']) + list(train_daphnia['Class'])

# Fit the data generator on the training images
datagen.fit(train_images)

# Use the data generator to create augmented images and add them to the training set
augmented_images = []
augmented_labels = []

for i, (aug_images, labels) in enumerate(datagen.flow(train_images, train_labels, batch_size=20)):
    if i >= len(train_images) // 20:  # Stop after one epoch
        break
    augmented_images.extend(aug_images)
    augmented_labels.extend(labels)

# Convert augmented images and labels to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Combine original and augmented images and labels
train_images = np.concatenate((train_images, augmented_images), axis=0)
train_labels = np.concatenate((train_labels, augmented_labels), axis=0)

# Save the splits to CSV files if needed
train_df = pd.DataFrame({'Image': list(range(len(train_images))), 'Class': train_labels})
val_df = pd.DataFrame({'Vignette': val['Vignette'], 'Class': val['Class']})
test_df = pd.DataFrame({'Vignette': test['Vignette'], 'Class': test['Class']})

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
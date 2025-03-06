import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array, save_img
import numpy as np
import os
import tensorflow as tf
from collections import Counter

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# 0. Load particles csv and images 
particles = pd.read_csv("extracted_particles.csv")

# Function to load and resize images from file paths
def load_and_resize_images(file_paths, folder_path, target_size=(128, 128)):
    images = []
    for file_path in file_paths:
        img = load_img(os.path.join(folder_path, file_path), target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Function to save augmented images
def save_augmented_images(aug_img, original_file_name, folder_path, suffix="_aug"):
    augmented_image_paths = []
    aug_file_name = f"{original_file_name.replace('.jpg', '')}{suffix}.jpg"
    aug_img_path = os.path.join(folder_path, aug_file_name)
    save_img(aug_img_path, aug_img)  # Save the augmented image
    augmented_image_paths.append(aug_file_name)  # Store the augmented image file name
    return augmented_image_paths

# 1. Oversampling
def oversample_class(df, class_name, target_count):
    class_df = df[df["Class"] == class_name]
    oversampled_class_df = class_df.sample(target_count, replace=True, random_state=seed)
    return oversampled_class_df

target_count = 100 

# Oversample Chydoridae and Daphnia
chydoridae_oversampled = oversample_class(particles, 'Chydoridae', target_count)
daphnia_oversampled = oversample_class(particles, 'Daphnia', target_count)

particles_oversampled = pd.concat([particles, chydoridae_oversampled, daphnia_oversampled])

# 2. Split into train, test, validation 
train_val, test = train_test_split(particles_oversampled, test_size=0.2, stratify=particles_oversampled['Class'], random_state=seed)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['Class'], random_state=seed)  

# 3. Data Augmentation for Chydoridae and Daphnia in training set
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and resize images from the vignettes folder
vignettes_folder = "vignettes"

# Filter training data for Chydoridae and Daphnia
train_chydoridae = train[train['Class'] == 'Chydoridae']
train_daphnia = train[train['Class'] == 'Daphnia']

# Load and resize images for Chydoridae and Daphnia
train_chydoridae_images = load_and_resize_images(train_chydoridae['Vignette'], vignettes_folder, target_size=(128, 128))
train_daphnia_images = load_and_resize_images(train_daphnia['Vignette'], vignettes_folder, target_size=(128, 128))

# Combine images and labels for Chydoridae and Daphnia
train_images = np.concatenate((train_chydoridae_images, train_daphnia_images), axis=0)
train_labels = np.array(list(train_chydoridae['Class']) + list(train_daphnia['Class']))
train_file_names = list(train_chydoridae['Vignette']) + list(train_daphnia['Vignette'])

# Ensure the lengths of train_images and train_labels match
assert len(train_images) == len(train_labels), "Mismatch between number of images and labels"

# Fit the data generator on the training images
datagen.fit(train_images)

# Initialize lists to store the augmented images' file names and labels
augmented_file_names = []
augmented_labels = []

# Iterate through the augmented images, save them with unique file names, and update lists
for i, (aug_images, labels) in enumerate(datagen.flow(train_images, train_labels, batch_size=20, shuffle=False)):
    if i >= len(train_images) // 20:  # Stop after one epoch
        break

    for j, aug_img in enumerate(aug_images):
        if aug_img.shape == (128, 128, 3):  # Ensure the image has the correct shape
            original_file_name = train_file_names[i * 20 + j]  # Get the original file name
            augmented_images = save_augmented_images(aug_img, original_file_name, vignettes_folder)
            augmented_file_names.extend(augmented_images)
            augmented_labels.extend([labels[j]])

# Combine original and augmented file names and labels
train_file_names.extend(augmented_file_names)
train_labels = np.concatenate((train_labels, np.array(augmented_labels)), axis=0)

# Combine with the rest of the training data
rest_train = train[~train['Class'].isin(['Chydoridae', 'Daphnia'])]
rest_train_images = load_and_resize_images(rest_train['Vignette'], vignettes_folder, target_size=(128, 128))
rest_train_labels = np.array(rest_train['Class'])
rest_train_file_names = list(rest_train['Vignette'])

# Combine all training data
all_train_images = np.concatenate((rest_train_images, train_images), axis=0)
all_train_labels = np.concatenate((rest_train_labels, train_labels), axis=0)
all_train_file_names = rest_train_file_names + train_file_names

# Debug: Print lengths of all_train_file_names and all_train_labels
print("Length of all_train_file_names:", len(all_train_file_names))
print("Length of all_train_labels:", len(all_train_labels))

# Count the occurrences of each class in all_train_labels
class_counts = Counter(all_train_labels)
print("Class counts in all_train_labels:", class_counts)

# Save the splits to CSV files if needed
train_df = pd.DataFrame({'Vignette': all_train_file_names, 'Class': all_train_labels})
val_df = pd.DataFrame({'Vignette': val['Vignette'], 'Class': val['Class']})
test_df = pd.DataFrame({'Vignette': test['Vignette'], 'Class': test['Class']})

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
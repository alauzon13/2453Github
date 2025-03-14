import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load particles CSV
particles = pd.read_csv("extracted_particles.csv")

# Oversampling function
def oversample_class(df, class_name, target_count):
    class_df = df[df["Class"] == class_name]
    oversampled_class_df = class_df.sample(target_count, replace=True, random_state=seed)
    return oversampled_class_df

target_count = 100  # Adjust as needed

# Oversample Chydoridae and Daphnia
chydoridae_oversampled = oversample_class(particles, 'Chydoridae', target_count)
daphnia_oversampled = oversample_class(particles, 'Daphnia', target_count)
particles_oversampled = pd.concat([particles, chydoridae_oversampled, daphnia_oversampled])

# Train-test-validation split
train_val, test = train_test_split(particles_oversampled, test_size=0.2, stratify=particles_oversampled['Class'], random_state=seed)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['Class'], random_state=seed)

# Data Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ensure the vignettes folder exists
vignettes_folder = "vignettes"

import time

def generate_and_save_augmented_images(df, folder_path, target_size=(128, 128), augmentations=2, class_label=None):
    augmented_image_paths = []

    for index, row in df.iterrows():
        original_file_name = row["Vignette"]
        original_image_path = os.path.join(folder_path, original_file_name)

        if not os.path.exists(original_image_path):
            print(f"Warning: Image not found - {original_image_path}")
            continue

        original_image = load_img(original_image_path, target_size=target_size)
        original_image_array = img_to_array(original_image)
        original_image_array = np.expand_dims(original_image_array, axis=0)

        # Get list of existing files before augmentation
        before_files = set(os.listdir(folder_path))

        # Generate augmented images
        i = 0
        for _ in datagen.flow(original_image_array, batch_size=1, save_to_dir=folder_path, 
                              save_prefix=f"{original_file_name.replace('.png', '')}_aug", save_format='png'):
            i += 1
            if i >= augmentations:
                break

        time.sleep(1)  # Small delay to ensure files are saved

        # Get list of new files added
        after_files = set(os.listdir(folder_path))
        new_files = list(after_files - before_files)

        # Append only newly created files
        for new_file in new_files:
            augmented_image_paths.append((new_file, class_label))

        print(f"Generated {len(new_files)} augmented images for {original_file_name}")

    return augmented_image_paths


# Filter training data for Chydoridae and Daphnia
train_chydoridae = train[train['Class'] == 'Chydoridae']
train_daphnia = train[train['Class'] == 'Daphnia']

# Generate and save augmented images, track file paths
aug_chydoridae = generate_and_save_augmented_images(train_chydoridae, vignettes_folder, class_label="Chydoridae")
aug_daphnia = generate_and_save_augmented_images(train_daphnia, vignettes_folder, class_label="Daphnia")



# Create DataFrame with new augmented image paths
augmented_df = pd.DataFrame(aug_chydoridae + aug_daphnia, columns=["Vignette", "Class"])

# Add to training
train = pd.concat([train, augmented_df])

# Save the updated DataFrames to CSV files
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

# Print statement to indicate completion
print("Data augmentation and saving completed.")








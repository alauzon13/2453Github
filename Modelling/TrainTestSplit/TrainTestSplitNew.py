import os
import numpy as np
import time
import pandas as pd
import json
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load particles CSV
particles = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/extracted_particles.csv")
# Load in text data
text_data = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/data_cleaned.csv")

# Process/Standardize data_clean
text_target = text_data["Class"]
text_tifffile = text_data["tifffile"]
text_particleID = text_data["ParticleID"]


## Remove unimportant columns
columns_to_drop = ['Class.Particle.ID', 'Rep', 'Date', 'Key', 'Image.File', 'Original.Reference.ID', 'Source.Image', 'Time', 'Timestamp', 'csvfile_x', 'csvfile_y', 'Year', 'Month', 'Day', 'Class', 'tifffile', 'ParticleID']

# Check if each column exists before dropping
columns_to_drop = [col for col in columns_to_drop if col in text_data.columns]

text_features_cleaned = text_data.drop(columns=columns_to_drop)


## One-hot encoding for features
categorical_cols = text_features_cleaned.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(text_features_cleaned[categorical_cols])

one_hot_features_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_cols))

## Label encoding for class
label_encoder = LabelEncoder()
encoded_class = label_encoder.fit_transform(text_target)

# One-hot encode the target variable
one_hot_encoded_class = to_categorical(encoded_class)

# Convert the one-hot encoded matrix to a DataFrame
one_hot_class_df = pd.DataFrame(one_hot_encoded_class, columns=[f'class_{i}' for i in range(one_hot_encoded_class.shape[1])])
decoder_dict = {i: label for i, label in enumerate(label_encoder.classes_)}

# Save the decoder dictionary to a JSON file
with open("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/decoder_dict.json", "w") as f:
    json.dump(decoder_dict, f)


## Concatenate the one-hot encoded DataFrame with text_features_cleaned
encoded_all = pd.concat([one_hot_features_df.reset_index(drop=True), one_hot_class_df.reset_index(drop=True)], axis=1)

## Data standardization 
scaler = StandardScaler()
numerical_columns = list(set(text_features_cleaned.columns) - set(categorical_cols))
numeric_standardized = pd.DataFrame(scaler.fit_transform(text_features_cleaned[numerical_columns]), columns=numerical_columns)

## Concatenate encoded and standardized data
text_all_cleaned = pd.concat([numeric_standardized.reset_index(drop=True), encoded_all.reset_index(drop=True)], axis=1)
text_all_cleaned['tifffile'] = text_tifffile
text_all_cleaned["ParticleID"] = text_particleID
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

# Train-test-validation split based on ParticleID
train_val, test = train_test_split(particles_oversampled, test_size=0.2, stratify=particles_oversampled['Class'], random_state=seed)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['Class'], random_state=seed)

# Filter text data based on the same splits
train_particle_ids = train['ParticleID']
val_particle_ids = val['ParticleID']
test_particle_ids = test['ParticleID']

train_text = text_all_cleaned[text_all_cleaned['ParticleID'].isin(train_particle_ids)]
val_text = text_all_cleaned[text_all_cleaned['ParticleID'].isin(val_particle_ids)]
test_text = text_all_cleaned[text_all_cleaned['ParticleID'].isin(test_particle_ids)]

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
vignettes_folder = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes"


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
train.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv", index=False)
val.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv", index=False)
test.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv", index=False)

train_text.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv", index=False)
val_text.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_val.csv", index=False)
test_text.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv", index=False)

# Print statement to indicate completion
print("Data augmentation and saving completed.")








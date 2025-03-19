import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import keras_hub
import json
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Softmax

# Load data
## Images
image_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv")
image_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv")
image_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv")
## Text
text_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv")
text_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_val.csv")
text_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv")
# Define image size and batch size
image_size = (300, 300)
batch_size = 64

## Process text data
def remove_cols(df):
    """
    Removes unecessary cols. 
    """
    new_df = df.drop(columns=["tifffile", "ParticleID"])
    return(new_df)

text_train, text_val, text_test = [remove_cols(df) for df in [text_train, text_val, text_test]]

# Extract class columns
class_columns = [col for col in text_train.columns if col.startswith('class_')]
# Extract feature columns (excluding class columns)
feature_columns = [col for col in text_train.columns if not col.startswith('class_')]
# Convert data to numpy arrays for MLP
X_train_text = text_train[feature_columns].to_numpy().astype(np.float32)
y_train_text = text_train[class_columns].to_numpy().astype(np.float32)
X_val_text = text_val[feature_columns].to_numpy().astype(np.float32)
y_val_text = text_val[class_columns].to_numpy().astype(np.float32)

# Convert text data into TensorFlow Dataset
text_train_tf = tf.data.Dataset.from_tensor_slices((X_train_text, y_train_text)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
text_val_tf = tf.data.Dataset.from_tensor_slices((X_val_text, y_val_text)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


## Process Image Data 

num_classes = len(image_train['Class'].unique())

# Remove augmented images
image_train = image_train[~image_train["Vignette"].str.contains("aug")]

# Add correct filepath in front of image paths and ensure they are strings
image_train["Vignette"] = image_train["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/{x}").astype(str)
image_val["Vignette"] = image_val["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/{x}").astype(str)
image_test["Vignette"] = image_test["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/{x}").astype(str)


# Map string labels to integer indices
# Load the decoder dictionary
with open("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/decoder_dict.json", "r") as f:
    decoder_dict = json.load(f)

# Create a mapping from string labels to integer indices
label_to_index = {label: index for index, label in decoder_dict.items()}
index_to_label = {index: label for label, index in label_to_index.items()}

# Map string labels to integer indices
image_train['Class'] = image_train['Class'].map(label_to_index)
image_val['Class'] = image_val['Class'].map(label_to_index)
image_test['Class'] = image_test['Class'].map(label_to_index)


# Resize images 
@tf.autograph.experimental.do_not_convert
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Adjust for PNG if needed
    image = tf.image.resize(image, image_size)
    return image, label

# One hot encoding 
@tf.autograph.experimental.do_not_convert
def input_preprocess(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

@tf.autograph.experimental.do_not_convert
def df_to_dataset(df, label_col, shuffle=True):
    df = df.copy()
    labels = df.pop(label_col).astype(int)
    
    # Create TensorFlow dataset from the features and labels
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))  # Shuffle the dataset if required

    # Apply the image loading and preprocessing
    ds = ds.map(lambda x, y: (load_and_preprocess_image(x['Vignette'], y)))  # Ensure correct image shape
    ds = ds.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # Further preprocessing

    return ds


# Make sure classes are encoded as integers
image_train['Class'] = image_train['Class'].astype(int)
image_val['Class'] = image_val['Class'].astype(int)
image_test['Class'] = image_test['Class'].astype(int)

# Convert DataFrame to TensorFlow Dataset (without batching here)
image_train_tf = df_to_dataset(image_train, 'Class')
image_val_tf = df_to_dataset(image_val, 'Class')
image_test_tf = df_to_dataset(image_test, 'Class')

# Apply the map function to keep only features (no batching yet)
image_train_features = image_train_tf.map(lambda x, y: x)  # Keep only images
text_train_features = text_train_tf.map(lambda x, y: x)  # Keep only text features
train_labels = text_train_tf.map(lambda x, y: y)  # Keep only labels (same in both datasets)

# Zip the features and labels first
train_ds = tf.data.Dataset.zip((image_train_features, text_train_features, train_labels))

# Batch the dataset after zipping, and the batch size will be automatically handled here
train_ds = train_ds.batch(batch_size)

# Same process for validation dataset
image_val_features = image_val_tf.map(lambda x, y: x)  # Keep only images
text_val_features = text_val_tf.map(lambda x, y: x)  # Keep only text features
val_labels = text_val_tf.map(lambda x, y: y)  # Keep only labels (same in both datasets)

# Zip the features and labels for validation dataset
val_ds = tf.data.Dataset.zip((image_val_features, text_val_features, val_labels))

# Batch the validation dataset after zipping
val_ds = val_ds.batch(batch_size)

# Prefetch to improve performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Load pre-trained MLP model
mlp_input = tf.keras.Input(shape=(64, 98), name='mlp_input')
mlp_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras")
mlp_model = Model(inputs=mlp_model.inputs, outputs=mlp_model.layers[-2].output)
mlp_model.trainable = False

# Load pre-trained CNN model
cnn_input = tf.keras.Input(shape=(64, 300, 300, 3), name='cnn_input')
cnn_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras")
cnn_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)  # Remove SoftMax
cnn_model.trainable = False  # Freeze CNN

# Remove the SoftMax layer of the individual models (access the penultimate layer)
cnn_output = cnn_model.output  # Output before SoftMax
mlp_output = mlp_model.output

# Concatenate the outputs of the CNN and MLP
combined = layers.concatenate([cnn_output, mlp_output])

# Add a fully connected layer with 512 neurons
combined = layers.Dense(512, activation='relu')(combined)

# Add the final SoftMax layer
final_output = layers.Dense(num_classes, activation='softmax')(combined)

# Define the collaborative model
collaborative_model = Model(inputs={"cnn_input": cnn_input, "mlp_input": mlp_input}, outputs=final_output)

# Compile the model
collaborative_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit the collaborative model
collaborative_model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)
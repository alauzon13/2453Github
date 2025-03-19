'''
Building a collaborative neural network. 
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten

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

# Convert DataFrame to TensorFlow Dataset
@tf.autograph.experimental.do_not_convert
def df_to_dataset(df, label_col, batch_size, shuffle=True):
    df = df.copy()
    labels = df.pop(label_col).astype(int)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.map(lambda x, y: (load_and_preprocess_image(x['Vignette'], y)))
    ds = ds.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# Make sure classes are encoded as integers
image_train['Class'] = image_train['Class'].astype(int)
image_val['Class'] = image_val['Class'].astype(int)
image_test['Class'] = image_test['Class'].astype(int)

# Convert DataFrame to TensorFlow Dataset
image_train_tf = df_to_dataset(image_train, 'Class', batch_size=batch_size)
image_val_tf = df_to_dataset(image_val, 'Class', batch_size=batch_size)
image_test_tf = df_to_dataset(image_test, 'Class', batch_size=batch_size)

# Load pre-trained CNN model
cnn_input = Input(shape=(300, 300, 3), name="cnn_input")
cnn_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras")
cnn_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)  # Remove SoftMax
cnn_model.trainable = False  # Freeze CNN

# Load pre-trained MLP model
mlp_input = Input(shape=(X_train_text.shape[1],), name="mlp_input")
mlp_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras")
mlp_output = mlp_model(mlp_input)  # Pass the input through the model to get the output
mlp_model = Model(inputs=mlp_input, outputs=mlp_output)
mlp_model.trainable = False

# Ensure compatible shapes for concatenation
cnn_output = Flatten()(cnn_model.output)
mlp_output = Flatten()(mlp_model.output)

combined_input = Concatenate()([cnn_output, mlp_output])
fc_layer = Dense(512, activation='relu')(combined_input)
output_layer = Dense(num_classes, activation='softmax')(fc_layer)  # Final classification layer

# Build final collaborative model
collaborative_model = Model(inputs={"cnn_input": cnn_input, "mlp_input": mlp_input}, outputs=output_layer)

# Compile the model
collaborative_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Combine datasets
@tf.autograph.experimental.do_not_convert
def combine_datasets(image_ds, text_ds):
    return tf.data.Dataset.zip((image_ds, text_ds)).map(
        lambda img, txt: ({"cnn_input": img[0], "mlp_input": txt[0]}, txt[1])
    )
train_dataset = combine_datasets(image_train_tf, text_train_tf)
val_dataset = combine_datasets(image_val_tf, text_val_tf)

# Train the collaborative model
collaborative_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25
)

# Save the model
collaborative_model.save("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/Collaborative/collaborative_model.keras")

print("Training complete! Model saved successfully.")


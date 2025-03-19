'''
Building a collaborative neural network. 
'''

import tensorflow as tf
import pandas as pd
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

# Load data
## Images
image_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv")
image_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv")
image_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv")
## Text
text_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv")
text_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_val.csv")
text_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv")


## Process text data
def remove_cols(df):
    """
    Removes unecessary cols. 
    """
    new_df = df.drop(columns=["tifffile", "ParticleID"])
    return(new_df)

text_train, text_val, text_test = [remove_cols(df) for df in [text_train, text_val, text_test]]



## Process Image Data 

# Define image size and batch size
image_size = (300, 300)
batch_size = 64
num_classes = len(image_train['Class'].unique())

# Remove augmented images
image_train = image_train[~image_train["Vignette"].str.contains("aug")]

# Add correct filepath in front of image paths and ensure they are strings
image_train["Vignette"] = image_train["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/{x}").astype(str)
image_val["Vignette"] = image_val["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes//{x}").astype(str)
image_test["Vignette"] = image_test["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes//{x}").astype(str)


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

# For each train, test, val:
# 1. Convert DataFrame to TensorFlow Dataset
# 2. One-hot-encode labels 
# 3. batch 

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

# Make sure classes are encoded as integers
image_train['Class'] = image_train['Class'].astype(int)
image_val['Class'] = image_val['Class'].astype(int)
image_test['Class'] = image_test['Class'].astype(int)

# Convert DataFrame to TensorFlow Dataset
image_train_tf = tf.data.Dataset.from_tensor_slices((image_train["Vignette"].values, image_train["Class"].values))
image_train_tf = image_train_tf.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

image_val_tf = tf.data.Dataset.from_tensor_slices((image_val["Vignette"].values, image_val["Class"].values))
image_val_tf = image_val_tf.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

image_test_tf = tf.data.Dataset.from_tensor_slices((image_test["Vignette"].values, image_test["Class"].values))
image_test_tf = image_test_tf.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Text data

class_columns = [col for col in text_train.columns if col.startswith('class_')]
# Extract feature columns (excluding class columns)
feature_columns = [col for col in text_train.columns if not col.startswith('class_')]
# Extract class columns
class_columns = [col for col in text_train.columns if col.startswith('class_')]
# Convert data to numpy arrays for MLP
X_train_text = text_train[feature_columns].to_numpy()
y_train_text = text_train[class_columns].to_numpy()
X_val_text = text_val[feature_columns].to_numpy()
y_val_text = text_val[class_columns].to_numpy()

# Convert text data into TensorFlow Dataset
text_train_tf = tf.data.Dataset.from_tensor_slices(X_train_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)
text_val_tf = tf.data.Dataset.from_tensor_slices(X_val_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)
y_train_tf = tf.data.Dataset.from_tensor_slices(y_train_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)
y_val_tf = tf.data.Dataset.from_tensor_slices(y_val_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Combine datasets
train_dataset = tf.data.Dataset.zip((image_train_tf, text_train_tf)).map(lambda img, txt: ({"cnn_input": img[0], "mlp_input": txt}, img[1]))
val_dataset = tf.data.Dataset.zip((image_val_tf, text_val_tf)).map(lambda img, txt: ({"cnn_input": img[0], "mlp_input": txt}, img[1]))


# Load pre-trained CNN model (remove SoftMax layer)
cnn_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras")
cnn_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)  # Remove SoftMax
cnn_model.trainable = False  # Freeze CNN

# Load pre-trained MLP model
mlp_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras")
mlp_model = Model(inputs=mlp_model.inputs, outputs=mlp_model.layers[-2].output)
mlp_model.trainable = False

combined_input = Concatenate()([cnn_model.output, mlp_model.output])
fc_layer = Dense(512, activation='relu')(combined_input)
output_layer = Dense(6, activation='softmax')(fc_layer)  # Final classification layer

# Build final collaborative model
collaborative_model = Model(inputs={"cnn_input": cnn_model.input, "mlp_input": mlp_model.input}, outputs=output_layer)

# Compile the model
collaborative_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the collaborative model
collaborative_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25
)

# Save the model
collaborative_model.save("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/Collaborative/collaborative_model.keras")

print("Training complete! Model saved successfully.")

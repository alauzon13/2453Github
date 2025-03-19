import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Input, concatenate, Dense
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt


# 0. Load in data 
## Images
image_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv")
image_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv")
image_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv")

## Text
text_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv")
text_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_val.csv")
text_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv")

def match_particle_counts(image_df, text_df):
    """
    Ensures image and text datasets have the same particle counts by downsampling
    to the minimum count per ParticleID.
    
    Parameters:
        image_df (pd.DataFrame): DataFrame containing image data.
        text_df (pd.DataFrame): DataFrame containing text data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Downsampled image and text datasets.
    """

    # Find the count of each ParticleID in both datasets
    image_counts = image_df['ParticleID'].value_counts()
    text_counts = text_df['ParticleID'].value_counts()

    # Get the minimum count for each ParticleID across both datasets
    min_counts = pd.concat([image_counts, text_counts], axis=1).min(axis=1)

    def downsample(df, min_counts):
        return (
            df.groupby('ParticleID')
            .apply(lambda x: x.sample(n=int(min_counts[x.name]), random_state=42), include_groups=False)
            .reset_index(drop=True)
        )

    # Apply downsampling to match counts in both datasets
    image_df_final = downsample(image_df, min_counts)
    text_df_final = downsample(text_df, min_counts)

    return image_df_final, text_df_final

# Example usage for train, test, and val
image_train_final, text_train_final = match_particle_counts(image_train, text_train)
image_test_final, text_test_final = match_particle_counts(image_test, text_test)
image_val_final, text_val_final = match_particle_counts(image_val, text_val)

# 1. Process Image Data
image_size = (300, 300)
num_classes = 6
batch_size = 64

# Add correct filepath in front of image paths and ensure they are strings
image_train_final["Vignette"] = image_train_final["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/{x}").astype(str)
image_val_final["Vignette"] = image_val_final["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes//{x}").astype(str)
image_test_final["Vignette"] = image_test_final["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes//{x}").astype(str)

# Map string labels to integer indices
label_to_index = {label: index for index, label in enumerate(image_train_final['Class'].unique())}
index_to_label = {index: label for label, index in label_to_index.items()}

image_train_final['Class'] = image_train_final['Class'].map(label_to_index)
image_val_final['Class'] = image_val_final['Class'].map(label_to_index)
image_test_final['Class'] = image_test_final['Class'].map(label_to_index)

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
train_ds = tf.data.Dataset.from_tensor_slices((image_train_final["Vignette"].values, image_train_final["Class"].values))
train_ds = train_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((image_val_final["Vignette"].values, image_val_final["Class"].values))
val_ds = val_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((image_test_final["Vignette"].values, image_test_final["Class"].values))
test_ds = test_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def extract_image_data(df):
    ImagesX = []
    LabelsY = []

    for images, labels in df:
        ImagesX.append(images.numpy())
        LabelsY.append(labels.numpy()) 
    
    ImagesX = np.concatenate(ImagesX, axis=0)
    LabelsY = np.concatenate(LabelsY, axis=0)
    
    return ImagesX, LabelsY


trainImagesX, trainLabelsY = extract_image_data(train_ds)
valImagesX, valLabelsY = extract_image_data(val_ds)
testImagesX, testLabelsY = extract_image_data(test_ds)

# 2. Process Text Data

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

trainAttrX = text_train[feature_columns].to_numpy()
testAttrX = text_test[feature_columns].to_numpy()
valAttrX = text_val[feature_columns].to_numpy()

trainY = text_train[class_columns].to_numpy()
valY = text_val[class_columns].to_numpy()
testY = text_test[class_columns].to_numpy()


# Load the original models
mlp_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras", custom_objects=None, compile=True, safe_mode=True)
cnn_model = tf.keras.models.load_model("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras", custom_objects=None, compile=True, safe_mode=True)

# Remove the last layer (softmax) from both models
mlp_model = keras.Model(inputs=mlp_model.inputs, outputs=mlp_model.layers[-2].output)
cnn_model = keras.Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)

# Define new inputs with the correct shapes
mlp_input_shape = mlp_model.input_shape[1:]  # Exclude the batch size dimension
cnn_input_shape = cnn_model.input_shape[1:]  # Exclude the batch size dimension

mlp_input = Input(shape=mlp_input_shape)
cnn_input = Input(shape=cnn_input_shape)

# Get the outputs from the modified models
mlp_output = mlp_model(mlp_input)
cnn_output = cnn_model(cnn_input)

# Concatenate the outputs
combinedInput = concatenate([mlp_output, cnn_output])

# Add a fully connected layer with 512 neurons
fc_layer = Dense(512, activation='relu')(combinedInput)

# Add a softmax layer to form the final output
output_layer = Dense(6, activation='softmax')(fc_layer)  # Adjust the number of classes as needed

# Create the collaborative model
collaborative_model = Model(inputs=[mlp_input, cnn_input], outputs=output_layer)

# Compile the model
collaborative_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(trainAttrX.shape)  # Expected: (num_samples, mlp_feature_dim)
print(trainImagesX.shape)  # Expected: (num_samples, cnn_feature_dim)
print(trainY.shape)  # Expected: (num_samples, num_classes)


print(trainY.shape)  # Should be (num_samples, 6)
print(valY.shape) 


def plot_hist(hist):
    """
    Plots the training and validation accuracy and saves the plot as an image.
    """
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CollaborativeModel/model_accuracy.png')  # Save the plot as an image
    plt.close()  # Close the plot to avoid displaying it

# Train the model
collaborative_model_history = collaborative_model.fit(
    [trainAttrX, trainImagesX], 
    trainY, 
    validation_data=([valAttrX, valImagesX], valY), 
    epochs=25, 
    batch_size=64
)

# Plot the training history
plot_hist(collaborative_model_history)

# Save the trained model
collaborative_model.save('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CollaborativeModel/final_collaborative_model.keras')
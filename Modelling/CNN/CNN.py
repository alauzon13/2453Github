import numpy as np
import pandas as pd
import tensorflow as tf  # For tf.data
import matplotlib.pyplot as plt
import keras
from keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, preprocess_input, decode_predictions
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the datasets
train_df = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/TrainTestSplit/train.csv")
val_df = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/TrainTestSplit/val.csv")
test_df = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/TrainTestSplit/test.csv")

# Define image size and batch size
image_size = (300, 300)
batch_size = 64
num_classes = len(train_df['Class'].unique())

# Add correct filepath in front of image paths and ensure they are strings
train_df["Vignette"] = train_df["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/{x}").astype(str)
val_df["Vignette"] = val_df["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes//{x}").astype(str)
test_df["Vignette"] = test_df["Vignette"].apply(lambda x: f"/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes//{x}").astype(str)


# Map string labels to integer indices
label_to_index = {label: index for index, label in enumerate(train_df['Class'].unique())}
index_to_label = {index: label for label, index in label_to_index.items()}

train_df['Class'] = train_df['Class'].map(label_to_index)
val_df['Class'] = val_df['Class'].map(label_to_index)
test_df['Class'] = test_df['Class'].map(label_to_index)

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


# Convert DataFrame to TensorFlow Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_df["Vignette"].values, train_df["Class"].values))
train_ds = train_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_df["Vignette"].values, val_df["Class"].values))
val_ds = val_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_df["Vignette"].values, test_df["Class"].values))
test_ds = test_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build model from pretrained weights 
def build_model(num_classes):
    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))
    model = EfficientNetV2B3(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=num_classes)

epochs = 25  # @param {type: "slider", min:8, max:80}

hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/model_accuracy.png')  # Save the plot as an image
    plt.close()  # Close the plot to avoid displaying it

plot_hist(hist)

# Save the final model in the recommended Keras format
model.save('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras')


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test accuracy: {test_accuracy:.2f}')

# Predict the classes for the test set
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_df['Class'].values  # Assuming test_df contains the true labels

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Save the performance metrics to a text file
with open('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/performance_metrics.txt', 'w') as f:
    f.write(f'Test accuracy: {test_accuracy:.2f}\n')
    f.write(f'Precision: {precision:.2f}\n')
    f.write(f'Recall: {recall:.2f}\n')
    f.write(f'F1 Score: {f1:.2f}\n')

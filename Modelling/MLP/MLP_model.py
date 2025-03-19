'''
This script will build the Multi-Layer Perceptron from the text data in data_cleaned.
'''

from keras import layers
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from itertools import product
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import json





# 0. Read in/Process data
text_train = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv")
text_val = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_val.csv")
text_test = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv")

def remove_cols(df):
    """
    Removes unecessary cols. 
    """
    new_df = df.drop(columns=["tifffile", "ParticleID"])
    return(new_df)

text_train, text_val, text_test = [remove_cols(df) for df in [text_train, text_val, text_test]]

# 2. Build MLP


def build_mlp(input_shape, num_classes, hidden_layers=3, neurons_per_layer=512, dropout_rate=0.5):
    """
    Builds an MLP model based on given parameters.
    
    Args:
    input_shape (int): The shape of the input feature vector.
    num_classes (int): Number of output classes.
    hidden_layers (int): Number of hidden layers (1 to 5).
    neurons_per_layer (int): Neurons per hidden layer (256, 512, 1024, or 2048).
    dropout_rate (float): Dropout rate to prevent overfitting.
    
    Returns:
    keras.Model: Compiled MLP model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # Adding hidden layers (1 to 5, configurable)
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons_per_layer, activation='relu'))
    
    # Dropout layer before output
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer with softmax activation
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Extract feature columns (excluding class columns)
feature_columns = [col for col in text_train.columns if not col.startswith('class_')]

# Extract class columns
class_columns = [col for col in text_train.columns if col.startswith('class_')]

# Convert data to numpy arrays for MLP
X_train = text_train[feature_columns].to_numpy()
y_train = text_train[class_columns].to_numpy()



X_val = text_val[feature_columns].to_numpy()
y_val = text_val[class_columns].to_numpy()


X_test = text_test[feature_columns].to_numpy()
y_test = text_test[class_columns].to_numpy()



input_shape = X_train.shape[1]
num_classes = len(class_columns)

# Grid search parameters
tested_hidden_layers = [1, 2, 3, 4, 5]
tested_neurons_per_layer = [256, 512, 1024, 2048]


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/model_accuracy.png')  # Save the plot as an image
    plt.close()  # Close the plot to avoid displaying it

def perform_grid_search():
    best_accuracy = 0
    best_model = None
    best_params = None
    
    for hidden_layers, neurons_per_layer in product(tested_hidden_layers,tested_neurons_per_layer):
        print(f"Testing configuration: {hidden_layers} layers, {neurons_per_layer} neurons, 0.5 dropout")
        model = build_mlp(input_shape, num_classes, hidden_layers, neurons_per_layer, dropout_rate=0.5)
        
        # Dummy training: Replace with actual training & validation
        history = model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val), verbose=0)
        plot_hist(history)  # Plot the training history

        val_acc = max(history.history['val_accuracy'])
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_params = (hidden_layers, neurons_per_layer)
    
    print(f"Best model: {best_params} with accuracy {best_accuracy}")
    return best_model

# Replace X_train, y_train, X_val, y_val with actual datasets
best_model = perform_grid_search()

best_model.save('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras')


# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# Predict the classes for the test set
y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')



# Save the performance metrics to a text file
with open('/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/performance_metrics.txt', 'w') as f:
    f.write(f'Test accuracy: {test_accuracy:.2f}\n')
    f.write(f'Precision: {precision:.2f}\n')
    f.write(f'Recall: {recall:.2f}\n')
    f.write(f'F1 Score: {f1:.2f}\n')
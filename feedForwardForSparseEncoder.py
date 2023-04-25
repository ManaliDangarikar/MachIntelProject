import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

with open('encoded_data_train.pkl', 'rb') as file:
    encoded_data_train = pickle.load(file)

with open('encoded_data_test.pkl', 'rb') as file:
    encoded_data_test = pickle.load(file)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

x_train /= 255.0

# Flatten the input data
x_train = tf.reshape(x_train, shape=(-1, 784))

x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)

x_test /= 255.0

# Flatten the input data
x_test = tf.reshape(x_test, shape=(-1, 784))

# Assuming x_train is your training data and encoded_data_train is the encoded data for x_train
# Shapes: x_train - (num_samples, input_dim), encoded_data - (num_samples, encoded_dim)

# Define the autoencoder architecture
input_dim = 784 # example input dimension
hidden_dim = 256 # example hidden dimension
encoded_dim = 256 # example encoding dimension

# Create input layers for x_train and encoded_data
# Create input tensors from encoded_data and x_train
encoded_data_input = Input(shape=(encoded_dim,))
x_train_input = Input(shape=(input_dim,))

# Concatenate or stack the input layers
# Concatenate example:
merged_inputs = concatenate([x_train_input, encoded_data_input], axis=-1)

# test data
merged_test_inputs = concatenate([x_test, encoded_data_test], axis=-1)

# Stack example:
# merged_inputs = tf.stack([input_x_train, input_encoded_data], axis=-1)

# Add additional layers to the neural network
# Example:
num_classes = 10
x = Dense(units=256, activation='relu')(merged_inputs)
x = Dense(units=128, activation='relu')(x)
output = Dense(units=num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=[x_train_input, encoded_data_input], outputs=output)

learning_rate = 0.001
sgd_optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

# Compile the model and specify the optimizer, loss function, and metrics
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Convert ground truth labels to one-hot encoded format
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Train the model with x_train and encoded_data as inputs
history = model.fit([x_train, encoded_data_train],
					y_train_one_hot, 
					epochs=50, 
					batch_size=32, 
					validation_data=([x_test, encoded_data_test], y_test_one_hot))

# Extract training loss and validation loss from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Extract training accuracy and validation accuracy from history (if applicable)
train_acc = None
val_acc = None
if 'accuracy' in history.history:
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

# Plot the training loss and validation loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# If accuracy is available, plot the training accuracy and validation accuracy
if train_acc is not None and val_acc is not None:
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt

# Define the autoencoder architecture
input_dim = 784 # example input dimension
hidden_dim = 256 # example hidden dimension
encoding_dim = 256 # example encoding dimension

# Define the encoder model
input_img = Input(shape=(input_dim,))
encoded = Dense(hidden_dim, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.01))(encoded)

# Define the decoder model
decoded = Dense(hidden_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

learning_rate = 0.001
sgd_optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

# Compile the autoencoder model
autoencoder.compile(optimizer=sgd_optimizer, loss='mse')

# Load and preprocess your data
# ...
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

# Concatenate the train and test data into a single array
x_data = tf.concat([x_train, x_test], axis=0)

# Train the sparse encoder
history = autoencoder.fit(x_train, x_train,
                batch_size=32,
                epochs=50,
                shuffle=True,
                validation_data=(x_test, x_test))

# Extract the trained sparse representations
encoder = Model(input_img, encoded)
encoded_data_train = encoder.predict(x_train)

encoded_data_test = encoder.predict(x_test)

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
	
# save things for future
autoencoder.save('encoder_model.h5')

# Assuming encoded_data is a Python object
with open('encoded_data_train.pkl', 'wb') as file:
    pickle.dump(encoded_data_train, file)

with open('encoded_data_test.pkl', 'wb') as file:
    pickle.dump(encoded_data_test, file)



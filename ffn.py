import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical

class Ffn(object):
    """
    Class for a feed forward network
    """
    
    def __init__(self, input_shape, h1_dim, h2_dim, y_dim):
        """
        constructor
        :param input_shape: input shape
        :param h1_dim: dimension of first hidden layer
        :param h2_dim: dimension of second hidden layer
        :param y_dim: dimension of output layer
        """
        #super(Ffn, self).__init__()
        self.input = Input(shape=(input_shape,))
        self.layer1 = Dense(units=h1_dim, activation='relu')
        self.layer2 = Dense(units=h2_dim, activation='relu')
        self.output = Dense(units=y_dim, activation='softmax')
        learning_rate = 0.001
        sgd_optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        self.model = Sequential([
                        self.input,
                        self.layer1,
                        self.layer2,
                        self.output])
        self.model.compile(optimizer=sgd_optimizer,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self.history = None
        self.model.summary()
        
    def train(self, x_train, y_train, x_test, y_test):
        """
        """
        print("y_train: ", tf.shape(y_train))
        print("y_test: ", tf.shape(y_test))
        self.history = self.model.fit(x_train, 
                                        y_train,
                                        epochs=50,
                                        batch_size=32,
                                        verbose=1,
                                        validation_data=(x_test, y_test))

    
    def plot_loss(self, data_path):
        # Extract training loss and validation loss from history
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        # Plot the training loss and validation loss
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(data_path)

    def plot_accuracy(self, data_path):
        # Extract training accuracy and validation accuracy from history (if applicable)
        train_acc = None
        val_acc = None
        if 'accuracy' in self.history.history:
            train_acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']

        # If accuracy is available, plot the training accuracy and validation accuracy
        if train_acc is not None and val_acc is not None:
            plt.plot(train_acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.show()
            plt.savefig(data_path)
            
    
def dataloader():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.cast(x_train, tf.float32)
    x_train /= 255.0

    # Flatten the input data
    x_train = tf.reshape(x_train, shape=(-1, 784))

    x_test = tf.cast(x_test, tf.float32)
    
    x_test /= 255.0

    # Flatten the input data
    x_test = tf.reshape(x_test, shape=(-1, 784))

    # Convert ground truth labels to one-hot encoded format
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    y_train = tf.cast(y_train, tf.float32)
    y_test = tf.cast(y_test, tf.float32)

	# print("y_test: ", tf.shape(y_test))
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = dataloader()
    ffn = Ffn(784, 256, 128, 10)
    ffn.train(x_train, y_train, x_test, y_test)
    ffn.plot_loss("E:\Sem1\MachineIntelligence\Project\project")
    ffn.plot_accuracy("E:\Sem1\MachineIntelligence\Project\project")
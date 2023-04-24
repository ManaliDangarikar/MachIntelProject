import time, os

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

class NeuralLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu'):
        super(NeuralLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.layers.Activation(activation)
        self.W = self.add_weight(shape=(784, self.units), initializer=tf.random_normal_initializer(stddev=1e-1), name='W')
        self.b = self.add_weight(shape=(self.units,), initializer=tf.constant_initializer(0.1), name='b')

    def call(self, inputs):
        z = tf.matmul(inputs, self.W) + self.b
        a = self.activation(z)

        # We graph the average density of neurons activation
        average_density = tf.reduce_mean(tf.reduce_sum(tf.cast((a > 0), tf.float32), axis=[1]))
        self.add_metric(average_density, name='AverageDensity')

        return a
	
class SoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(SoftmaxLayer, self).__init__()
        self.num_classes = num_classes
        self.W_s = self.add_weight(shape=(784, num_classes), initializer=tf.random_normal_initializer(stddev=1e-1), name='W_s')
        self.b_s = self.add_weight(shape=(num_classes,), initializer=tf.constant_initializer(0.1), name='b_s')

    def call(self, inputs):
        out = tf.matmul(inputs, self.W_s) + self.b_s
        y_pred = tf.nn.softmax(out)

        return y_pred

class Loss(tf.keras.losses.Loss):
    def __init__(self, sparsity_constraint):
        super(Loss, self).__init__()
        self.sparsity_constraint = sparsity_constraint
        self.total_loss = 0.0  # Initialize total_loss variable

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        y_true = tf.cast(y_true, tf.float32)  # Convert y_true to float32
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred + epsilon), axis=[1]))
        sparsity_loss = self.sparsity_constraint * tf.reduce_sum(y_pred, axis=[1])
        loss = cross_entropy + sparsity_loss

        # Add loss to the model's losses collection/ Add current loss to total_loss
        self.total_loss += loss

        return loss
	
def compute_accuracy(y_true, y_pred):
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y_true, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# Example code
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.neural_layer = NeuralLayer(units=784, activation='relu')
        self.softmax_layer = SoftmaxLayer(num_classes=10)

    def call(self, inputs):
        x = self.neural_layer(inputs)
        x = self.softmax_layer(x)
        return x

# -------------------------------------------------------------------------------------
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

# Create a tf.data.Dataset from the training data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Shuffle the dataset
train_dataset = train_dataset.shuffle(buffer_size=len(x_train))

# Batch the dataset
batch_size = 100
train_dataset = train_dataset.batch(batch_size=batch_size)

# Get an iterator from the dataset
train_iterator = iter(train_dataset)

# -------------------------------------------------------------------------------------

# Create a NeuralLayer instance
# neural_layer = NeuralLayer(units=784, activation='relu')

# a = neural_layer(x_train)

# Create a SoftmaxLayer instance
# softmax_layer = SoftmaxLayer(num_classes=10)

# y_pred = softmax_layer(a)

# Create a Loss instance
sparsity_constraint = 0
loss_fn = Loss(sparsity_constraint)

adam = Adam(learning_rate=1e-3)



model = MyModel()
model.compile(optimizer=adam, loss=loss_fn)
# Create a function for the training loop
@tf.function
def train_step(batch_x, batch_y, sc):

    with tf.GradientTape() as tape:
	    # a = neural_layer(batch_x)
	    # y_pred = softmax_layer(a)
        y_pred = model(batch_x, training=True)  # Replace with your model's forward pass
        # Convert batch_y to integer data type
        batch_y = tf.cast(batch_y, dtype=tf.int32)
        # Convert batch_y to one-hot encoding with the same number of classes as y_pred
        batch_y_onehot = tf.one_hot(batch_y, depth=y_pred.shape[-1])
        batch_y_onehot = tf.cast(batch_y_onehot, dtype=y_pred.dtype)
        # Convert y_pred to class labels and cast to the same data type as y_true
        y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), dtype=batch_y.dtype)
        
        # Compute loss
        loss_fn = Loss(sc)
        loss = loss_fn(batch_y_onehot, y_pred)  # Call your custom loss function

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update model weights
        adam.apply_gradients(zip(gradients, model.trainable_variables))
        #current_loss = tf.reduce_mean(loss)  # Or tf.reduce_mean(loss)
        #current_loss = tf.constant(current_loss, dtype=tf.float32)
        #current_loss_var = tf.Variable(loss, dtype=tf.float32, trainable=False)
        return loss

sparsity_constraints = [0, 1e-4, 5e-4, 1e-3, 2.7e-3]
# Iterate over different sparsity constraints
for sc in sparsity_constraints:
    result_folder = 'E:/Sem1/MachineIntelligence/Project/project' + '/results/' + str(int(time.time())) + '-fc-sc' + str(sc)
    # with tf.compat.v1.Session() as sess:  # Note the change from tf.Session() to tf.compat.v1.Session()
        # sess.run(tf.compat.v1.global_variables_initializer())  # Note the change from tf.global_variables_initializer() to tf.compat.v1.global_variables_initializer()
        # sw = tf.summary.create_file_writer(result_folder)  # Note the change from tf.summary.FileWriter() to tf.summary.create_file_writer()
    i=0
    for batch in train_iterator:
        # Extract batch_x and batch_y from the batch
        batch_x, batch_y = batch
        i = i + 1
        current_loss = train_step(batch_x, batch_y, sc)	
    #with sw.as_default():
     #   tf.summary.scalar('loss', current_loss, step=i + 1)

    #if (i + 1) % 100 == 0:
        # Evaluate accuracy on test set
        y_pred_test = model(x_test)
        acc = compute_accuracy(y_test, y_pred_test)  # Replace with your accuracy calculation function
        #tf.summary.scalar('accuracy', acc, step=i + 1)
        print('batch: %d, loss: %f, accuracy: %f' % (i , current_loss, acc))
 
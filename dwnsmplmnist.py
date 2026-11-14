from Matrix import Matrix
from Vector import Vector
from Layer import Layer
from Network import Network

# For the mnist dataset:
import tensorflow as tf
import random as rand

# 1. Download and load the data as raw arrays
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Convert training images to pure lists of floats (Downsampled to 14x14)
train_vectors = []
for image in x_train:
    # Downsample by picking every second pixel horizontally and vertically (28x28 -> 14x14)
    downsampled = image[::2, ::2]
    # Flatten 14x14 into 196, normalize to 0.0-1.0, and convert to native Python list
    flat_float_list = (downsampled.flatten() / 255.0).tolist()
    train_vectors.append(flat_float_list)

# 3. Convert test images to pure lists of floats (Downsampled to 14x14)
test_vectors = []
for image in x_test:
    downsampled = image[::2, ::2]
    flat_float_list = (downsampled.flatten() / 255.0).tolist()
    test_vectors.append(flat_float_list)

# 4. Convert the labels into standard Python lists of integers
train_labels = y_train.tolist()
test_labels = y_test.tolist()

# ---- Verification ----
print("\n\n Succesfully imported and downsampled mnist data")
print(f"Number of training vectors: {len(train_vectors)}")
print(f"Size of the first vector: {len(train_vectors[0])} (Expected: 196)")
print(f"Data type of a pixel: {type(train_vectors[0][0])}")


def to_one_hot(label: int, classes: int = 10) -> Vector:
    hot_list = [0.0] * classes
    hot_list[label] = 1.0
    return Vector(hot_list)

# Transform the Training Data
mnist_train_pairs = []
for i in range(len(train_vectors)):    
    input_vector = Vector(train_vectors[i])
    target_vector = to_one_hot(train_labels[i])
    mnist_train_pairs.append((input_vector, target_vector))

# Transform the Test Data
mnist_test_pairs = []
for i in range(len(test_vectors)):
    input_vector = Vector(test_vectors[i])
    target_vector = to_one_hot(test_labels[i])
    mnist_test_pairs.append((input_vector, target_vector))

# ---- Shuffling and truncating data to keep it fast ----
rand.shuffle(mnist_train_pairs)

# ---- This line makes the data smaller!
# mnist_train_pairs = mnist_train_pairs[:10000]

# ---- Initialize the Network ----
# Input layer must match the new downsampled size: 196 nodes.
# Hidden layer reduced to 64 for a massive speed boost.
mnist_network = Network([196, 64, 10],
                        hidden_activation=Layer.ReLU,
                        hidden_d_activation=Layer.dReLU)

# ---- Train the Network ----
mnist_network.batch_train(alpha=0.05, 
                    iterations=1, 
                    train_pairs=mnist_train_pairs,
                    batch_size=32,
                    update_frequency=1)

# ---- Save the Trained Network ----
mnist_network.save("blurry_mnist_network.pkl")

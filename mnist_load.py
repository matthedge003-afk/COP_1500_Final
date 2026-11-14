from Matrix import Matrix
from Vector import Vector
from Layer import Layer
from Network import Network

#For the mnist dataset:
import tensorflow as tf

# 1. Download and load the data as raw arrays
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Convert training images to pure lists of floats
train_vectors = []
for image in x_train:
    # Flatten 28x28 into 784, normalize to 0.0-1.0, and convert to native Python list
    flat_float_list = (image.flatten() / 255.0).tolist()
    train_vectors.append(flat_float_list)

# 3. Convert test images to pure lists of floats
test_vectors = []
for image in x_test:
    flat_float_list = (image.flatten() / 255.0).tolist()
    test_vectors.append(flat_float_list)

# 4. Convert the labels into standard Python lists of integers
train_labels = y_train.tolist()
test_labels = y_test.tolist()

# ---- Verification ----
print("\n\n Succesfully imported mnist data")
print(f"Number of training vectors: {len(train_vectors)}")
print(f"Size of the first vector: {len(train_vectors[0])}")
print(f"Data type of a pixel: {type(train_vectors[0][0])}")
print(f"Label for the first vector: {train_labels[0]}")


def to_one_hot(label: int, classes: int = 10) -> Vector:
    # Start with a list of zeros
    hot_list = [0.0] * classes
    # Set the target index to 1.0
    hot_list[label] = 1.0
    return Vector(hot_list)

# 2. Transform the Training Data
# Expects: List[Tuple[Vector, Vector]]
mnist_train_pairs = []
for i in range(len(train_vectors[:1000])):    # Only training on a subset 
    input_vector = Vector(train_vectors[i])
    target_vector = to_one_hot(train_labels[i])
    
    mnist_train_pairs.append((input_vector, target_vector))

# 3. Transform the Test Data (for evaluation later)
mnist_test_pairs = []
for i in range(len(test_vectors)):
    input_vector = Vector(test_vectors[i])
    target_vector = to_one_hot(test_labels[i])
    
    mnist_test_pairs.append((input_vector, target_vector))

# ---- More Verification ----
print(f"Transformed training pairs: {len(mnist_train_pairs)}")
print(f"Input shape: {mnist_train_pairs[0][0].shape}")    # Should be (784, 1)
print(f"Target shape: {mnist_train_pairs[0][1].shape}")  # Should be (10, 1)
print(f"Target vector for label {train_labels[0]}:")
print(mnist_train_pairs[0][1])

# ---- Load Pretrained MNIST network
mnist_network = Network.load("dumb_mnist_network.pkl")

for pair in mnist_test_pairs[0:20]:
    # print("Input:", pair[0])
    print("Output:", mnist_network.feedforward(pair[0]))
    print("Expected:",  pair[1])
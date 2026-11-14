# COP_1500_Final
Codebase for my COP 1500 Introduction to computer science final project


NOTE: Do not assume all functions in the matrix file or in the vetor file work. I am yet to have tested all of them and simply debugged the ones used in the layer aned network files. THIS ISN'T A FUNCTIONAL MATRIX LIBRARY. The point of this project is the write a NEURAL NETWORK library, NOT a matrix library. please dont be upset is some of the matrix functions a janky :)

Files and descriptions:

Matrix: this is the most math-intense file. It contains the matrix class that has all of the data structures and matrix operations algorithms in it. 

Vector: this file contains the vector class which inherits the matrix class. dynamic type conversions for vector operations were included in the matrix class where necessary. Not all the functions there have dynamic type checking.

Layer: One layer of a neural network

network: just a stack of layers with feedforward and backpropagation between them.

dwnsmplmnist.py: the training file for a network reading downsampled MNIST handwritten digits

dwnsmplload.py: loading and testing the downsampled mnist data from a serialized network

drawing.py: uses tkinter to allow the user to handwrite digits. If you're grading and cant find the dictionary, here it is!

all -.pkl files: serializations of trained networks. They might work for you.

xor.py trains a network on xor data. How original!

other files: theyre for something but ive forgotten at this point

images directory shows the project working.

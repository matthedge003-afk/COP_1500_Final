from Matrix import Matrix
from Vector import Vector
from Layer import Layer
from Network import Network

xor_data = [
    (Vector([0.0, 0.0]), Vector([0,1])),
    (Vector([0.0, 1.0]), Vector([1,0])),
    (Vector([1.0, 0.0]), Vector([1,0])),
    (Vector([1.0, 1.0]), Vector([0,1]))
]

pretrianed_network = Network.load("xor_network.pkl")

for item in xor_data:
    print("Input:" ,item[0], "\nOutput:", pretrianed_network.feedforward(item[0]))
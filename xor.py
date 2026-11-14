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

network = Network([2,4,5,4,2], hidden_activation=Layer.ReLU,
                               hidden_d_activation=Layer.dReLU)

network.train(0.01,3000,xor_data)

for i in range(len(network.body)):
    print(f"Layer: {i}\n")
    print(network.body[i],"\n\n")

network.save("xor_network.pkl")

for item in xor_data:
    print("Input:" ,item[0], "\nOutput:", 
        network.feedforward(item[0]))

# TODO: Separate files for every matrix. This is the only way not to make it horrible for myself.
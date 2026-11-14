from Layer import Layer
from Vector import Vector
from Matrix import Matrix
import math
from typing import List, Tuple, Union, Callable

class Network:
    def __init__(self, shape: list[int], 
                hidden_activation: Callable[[float], float] = Layer.sigmoid,
                hidden_d_activation: Callable[[float], float] = Layer.dsigmoid):

        self.hidden_activation  = hidden_activation
        self.hidden_d_activation = hidden_d_activation
        
        self.shape = shape
        self.body = [Layer(shape[i], shape[i+1]) for i in range(len(shape) - 1)]

    def feedforward(self, funcinput: 'Vector',) -> 'Vector':
        """
        input a vector and the network is applied to it.
        """

        if funcinput.length != self.body[0].inlen:
            raise ValueError(f"Network input dimension mismatch. Expected {self.body[0].inlen}, got {funcinput.length}.") 

        hold = funcinput

        for i in range(len(self.body) - 1):
            hold = self.body[i].feedforward(hold, self.hidden_activation, usesoftmax=False)

        #last layer uses softmax always.
        hold = self.body[-1].feedforward(hold, None, usesoftmax= True)
        
        # Return the final output.
        return hold

    def backpropagate(self, target: 'Vector'):
        """
        Executes a full backward calculus pass across the entire layer stack.
        
        Parameters:
        - target: The true labeled output classification Vector (e.g. one-hot encoded).
        - loss_derivative: A function tracking output error (like MSE derivative).
        - hidden_d_activation: The derivative of the hidden layers (e.g. Layer.dsigmoid).
        """

        output_layer = self.body[-1]
        
        # Local output error calculation (e.g., Output Activations - Targets)
        # Note: If pairing Softmax + Cross-Entropy, delta simplifies perfectly to: output_layer.act - target
        delta = output_layer.act - target
        
        # Separate output layer gradient update
        output_layer.d_weight = delta @ output_layer.input.T()
        output_layer.d_bias = delta

        # 2. Ripple Backwards Through Hidden States
        # Loop dynamically down through hidden layers in reverse execution order
        for i in range(len(self.body) - 2, -1, -1):
            current_layer = self.body[i]
            next_layer = self.body[i + 1]
            
            delta = current_layer.backpropagate(delta, next_layer, self.hidden_d_activation)

    def update(self, learning_rate: float):
        """
        "nudges" the weights and biases the gradients scaled by step size alpha
        """
        for layer in self.body:
            layer.update(learning_rate)

    def train(self, alpha: float, iterations: int, train_pairs: List[Tuple[Vector, Vector]],
              update_frequency: int = 100):

        for i in range(iterations):
            for x_input, y_target in train_pairs:
                self.feedforward(x_input)
                self.backpropagate(y_target)
                self.update(alpha)
            
            if i % update_frequency == 0:
                print(f"iteration: {i}")
                print(f"Last Activation: {self.body[-1].act}")

    def batch_train(self, alpha: float, iterations: int, train_pairs: List[Tuple[Vector, Vector]],
              batch_size: int = 32, update_frequency: int = 100):
        
        import random

        for i in range(iterations):
            # Shuffle data every iteration for standard mini-batch training
            random.shuffle(train_pairs)
            batchnum = 0
            # Process data in chunks of batch_size
            for batch_start in range(0, len(train_pairs), batch_size):
                batch = train_pairs[batch_start:batch_start + batch_size]
                current_batch_size = len(batch)
                
                # 1. Reset batch gradients to zero
                for layer in self.body:
                    layer.clear_gradients()

                batch_loss = 0
                
                # 2. Accumulate gradients over the batch samples
                for x_input, y_target in batch:
                    self.feedforward(x_input)
                    
                    # Backpropagate output layer
                    output_layer = self.body[-1]
                    delta = output_layer.act - y_target

                    for idx in range(output_layer.act.length):
                        act_val = output_layer.act[idx]
                        target_val = y_target[idx]
                        if target_val > 0:  # Since target is one-hot, only truth label contributes
                            batch_loss -= target_val * math.log(max(act_val, 1e-15))
                    
                    # Accumulate output layer gradients
                    if output_layer.d_weight is None:
                        output_layer.d_weight = delta @ output_layer.input.T()
                        output_layer.d_bias = delta
                    else:
                        output_layer.d_weight += delta @ output_layer.input.T()
                        output_layer.d_bias += delta
                    
                    # Ripple backwards through hidden states
                    for j in range(len(self.body) - 2, -1, -1):
                        current_layer = self.body[j]
                        next_layer = self.body[j + 1]
                        delta = current_layer.backpropagate(delta, next_layer, self.hidden_d_activation)
                
                # 3. Average the gradients across the batch size
                for layer in self.body:
                    layer.d_weight *= (1.0 / current_batch_size)
                    layer.d_bias *= (1.0 / current_batch_size)
                
                # 4. Apply the averaged step
                self.update(alpha)

                # 5. Update batchnum
                print("Batch #:", batchnum)
                batchnum += 1
                average_batch_loss = batch_loss/current_batch_size
                print("Loss: ", average_batch_loss)
                print("\n")
            
            if i % update_frequency == 0:
                print(f"iteration: {i}")
                print(f"Last Activation: {self.body[-1].act}")

    def save(self, filename: str):
        """
        Serializes the entire live network state into a single compressed binary file.
        Zero dependencies on CSV text formatting.
        """
        import pickle
        
        if not filename.endswith('.pkl'):
            filename += '.pkl'
            
        with open(filename, mode='wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Network saved to '{filename}'")

    @classmethod
    def load(cls, filename: str) -> 'Network':
        """
        Loads a Network instance from a .pkl file
        """
        import pickle
        
        if not filename.endswith('.pkl'):
            filename += '.pkl'
            
        # Read-Binary mode
        with open(filename, mode='rb') as f:
            network_instance = pickle.load(f)
            
        print(f"Network state successfully restored from '{filename}'")
        return network_instance
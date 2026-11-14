from typing import Callable
import math
from Vector import Vector
from Matrix import Matrix


class Layer:
    
    @classmethod
    def ReLU(cls, x: float) -> float:
        """
        Rectified Linear Unit
        """
        if x >= 0: return x
        else: return 0

    @classmethod
    def dReLU(cls, x: float) -> float:
        """
        The derivative of the Rectified Linear Unit"""
        if x >= 0: return 1
        else: return 0

    @classmethod
    def leaky_ReLU(cls, x: float) -> float:
        """
        Rectified Linear Unit
        """
        if x >= 0: return x
        else: return 0.01*x

    @classmethod
    def dleaky_ReLU(cls, x: float) -> float:
        """
        The derivative of the Rectified Linear Unit"""
        if x >= 0: return 1
        else: return 0.01

    @classmethod
    def sigmoid(cls, x: float) -> float:
        """
        The s-shaped, Logistic Sigmoid Function
        """
        return 1.0/(1+math.exp(-x))

    @classmethod
    def dsigmoid(cls, x: float) -> float:
        """
        The derivative of the Logistic Sigmoid Function
        """
        s = Layer.sigmoid(x)

        return s*(1-s)  

    def __init__(self, inlen, outlen):
        self.inlen = inlen
        self.outlen = outlen

        self.input =   Vector.zero(inlen)
        self.weight =  Matrix.random_xavier(outlen, inlen)
        self.bias =    Vector.zero(outlen)
        self.out =     Vector.zero(outlen)
        self.act =     Vector.zero(outlen) #activation

        self.d_weight = None
        self.d_bias = None

    def feedforward(self, input: 'Vector', activation: Callable[[float], float], usesoftmax: bool = False) -> 'Vector':
        self.input = input
        self.out = self.weight @ self.input + self.bias

        if usesoftmax:
            self.act = self.out.softmax()
        else:
            self.act = self.out.map(activation)

        return self.act

    def backpropagate(self, next_delta: 'Vector', next_layer: 'Layer', d_activation: Callable[[float], float]) -> 'Vector':
        """
        Calculates gradients to be used in the update function.
        The gradients found here are based on the incoming gradient. 
        For the case of the first one in the training loop, you have to 
        first chose an activation-loss pair. Cross entropy and softmax 
        exhibit beautiful concellation, but really are only useful for 
        classification-type problems
        """
        error_propagation = next_layer.weight.T() @ next_delta
        
        activation_derivatives = self.out.map(d_activation)
        self.delta = error_propagation * activation_derivatives
        self.d_weight = self.delta @ self.input.T()
        self.d_bias = self.delta

        return self.delta

    def update(self, alpha: float):
        """
        This APPLIES the step AFTER the gradients are found using backpropagate.
        """
        self.weight -= self.d_weight * alpha
        self.bias   -= self.d_bias   * alpha

    def __repr__(self) -> str:
        """
        Returns a detailed text representation displaying the actual inner matrix datasets.
        """
        def format_matrix_data(matrix_obj) -> str:
            """Helper to safely format nested matrix or vector data arrays to 4 decimals."""
            if not matrix_obj or not hasattr(matrix_obj, 'data'):
                return "  [Empty/None]"
            
            # If it's a Vector/Column Matrix containing numeric items directly: [x, y, z]
            if len(matrix_obj.data) > 0 and not isinstance(matrix_obj.data[0], list):
                return "\n".join(f"    [ {val: .4f} ]" for val in matrix_obj.data)
                
            # If it's a standard 2D Matrix list-of-lists: [[w1, w2], [w3, w4]]
            lines = []
            for row in matrix_obj.data:
                row_str = " ".join(f"{val: .4f}" for val in row)
                lines.append(f"    [ {row_str} ]")
            return "\n".join(lines)

        return (
            f"==================================================\n"
            f"Layer Object Status (Shape: {self.inlen} -> {self.outlen})\n"
            f"==================================================\n"
            f"▶ INPUT VECTOR Data:\n{format_matrix_data(self.input)}\n\n"
            f"▶ WEIGHTS MATRIX Data:\n{format_matrix_data(self.weight)}\n\n"
            f"▶ BIAS VECTOR Data:\n{format_matrix_data(self.bias)}\n\n"
            f"▶ RAW OUTPUT (self.out) Data:\n{format_matrix_data(getattr(self, 'out', None))}\n\n"
            f"▶ ACTIVATED OUTPUT (self.act) Data:\n{format_matrix_data(self.act)}\n"
            f"=================================================="
        )

    def clear_gradients(self):
        """Resets gradient accumulators to zero for a new batch."""
        self.d_weight = Matrix.zero(self.outlen, self.inlen)
        self.d_bias = Vector.zero(self.outlen)

    def batch_backpropagate(self, next_delta: 'Vector', next_layer: 'Layer', d_activation: Callable[[float], float]) -> 'Vector':

        error_propagation = next_layer.weight.T() @ next_delta
        activation_derivatives = self.out.map(d_activation)
        self.delta = error_propagation * activation_derivatives
        
        # CHANGED: Accumulate gradients using += instead of resetting them
        if self.d_weight is None or self.d_bias is None:
            self.d_weight = self.delta @ self.input.T()
            self.d_bias = self.delta
        else:
            self.d_weight += self.delta @ self.input.T()
            self.d_bias += self.delta

        return self.delta

import math
from Matrix import Matrix
from typing import List #, Union

class Vector(Matrix):
    def __init__(self, components: List[float]):
        if not components:
            raise ValueError("Vector components cannot be empty.")
        
        column_matrix_data = [[float(x)] for x in components]
         #inherited initialization
        super().__init__(column_matrix_data)

    @property
    def length(self):
        return self.rows

    def __getitem__(self, index: int) -> float:
        """Allows direct indexing like v[0] instead of v.data[0][0]."""
        return self.data[index][0]

    def __setitem__(self, index: int, value: float):
        """Allows direct value assignment like v[0] = 5.0."""
        self.data[index][0] = float(value)

    @classmethod
    def zero(cls, length: int) -> 'Vector':
        """
        Returns a zero vector
        """
        if length <= 0:
            raise TypeError("Negative Vector Length")

        zeroes = [0 for _ in range(length)]

        return cls(zeroes)
    
    def __mod__(self, other: 'Vector') -> 'Vector':
        """
        Performs a dot product using %
        """
        if self.length != other.length:
            raise ValueError("Dot Product: Vector dimension Mismatch")

        return sum(self[i] * other[i] for i in range(self.length))

    # def __add__(self, other):
    #     if self.length != other.length:
    #         raise ValueError(f"Vector Addition Mismatch: ({self.length}, {other.length})")

    #     result = Vector([0 for i in range(self.length)])
    #     for i in range(self.length):
    #         result.data[i] = self.data[i] + other.data[i]

    #     return result

    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def softmax(self) -> 'Vector':
        """
        Applies the numerically stable Softmax activation function.
        Transforms the vector components into a valid probability distribution.
        Returns a fresh Vector instance (Safe for backpropagation history).
        """
        raw_vals = [row[0] for row in self.data]
        
        max_val = max(raw_vals)
        
        exp_vals = [math.exp(x - max_val) for x in raw_vals]
        
        total_sum = sum(exp_vals)
        
        softmax_vals = [ev / total_sum for ev in exp_vals]
        
        return type(self)(softmax_vals)

    
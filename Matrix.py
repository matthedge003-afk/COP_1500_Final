import math
import random
from typing import List, Tuple, Union, Callable

class Matrix:
    def __init__(self, data: List[List[float]]):
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty.")
        
        self.rows = len(data)
        self.cols = len(data[0])
        if not all(len(row) == self.cols for row in data):
            raise ValueError("All matrix rows must have the same length.")
        
        self.data = [[float(val) for val in row] for row in data]

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the dimensions of the matrix as (rows, columns)."""
        return (self.rows, self.cols) 
        #here is the tuple. This is used for validation

    def __repr__(self) -> str:
        """Clean string representation for mathematical scanning."""
        lines = [f"  [{', '.join(f'{val:7.3f}' for val in row)}]" for row in self.data]
        return "[\n" + "\n".join(lines) + "\n]"

    @classmethod
    def zero(cls, rows: int, cols: int) -> 'Matrix':
        """Generates a matrix filled entirely with zeros."""
        return cls([[0.0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        """Generates an identity matrix (I) where diagonal elements equal 1.0."""
        return cls([[1.0 if r == c else 0.0 for c in range(size)] for r in range(size)])

    @classmethod
    def random_xavier(cls, rows: int, cols: int) -> 'Matrix':
        """
        Generates a Matrix initialized with Xavier random scaling.
        Crucial for breaking symmetry in Sigmoid layers.
        'rows' matches outlen, 'cols' matches inlen.
        """
        stddev = math.sqrt(1.0 / cols)
        
        data = [
            [random.gauss(0.0, stddev) for _ in range(cols)]
            for _ in range(rows)
        ]
        return cls(data)
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} vs {other.shape}")
        
        result = [
            [self.data[r][c] + other.data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        
        if self.cols == 1 and other.cols == 1:
            from Vector import Vector
            return Vector([row[0] for row in result])
            
        return Matrix(result)

    def __iadd__(self, other: 'Matrix') -> 'Matrix':
        """
        Performs in-place element-wise matrix addition (A += B).
        Requires identical shapes. Modifies the data matrix directly.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} vs {other.shape}")

        # Loop through rows then collumns
        for r in range(self.rows):
            self_row = self.data[r]
            other_row = other.data[r]
            for c in range(self.cols):
                self_row[c] += other_row[c]
                
        return self
        

    def __sub__(self, other):
        """
        Performs element-wise matrix subtraction.
        Requires identical shapes: A(i,j) + B(i,j)
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for subtraction: {self.shape} vs {other.shape}")
        
        result = [
            [self.data[r][c] - other.data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        return Matrix(result)

    def __neg__(self) -> 'Matrix':
        """
        Overloads the unary negative operator (-Matrix).
        Inverts the sign of every element inside the structure.
        """
        # Calculate the inverted numeric cell grid
        result_data = [
            [-self.data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        
        # Preserve Vector class type if the input was a Vector
        if type(self).__name__ == "Vector":
            flat_vector_data = [row[0] for row in result_data]
            return type(self)(flat_vector_data)
            
        return Matrix(result_data)
    
    def __isub__(self, other: 'Matrix') -> 'Matrix':
        """
        Performs in-place element-wise matrix addition (A += B).
        Requires identical shapes. Modifies the data matrix directly.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} vs {other.shape}")

        # Loop through rows on the outer loop to match standard memory layout
        for r in range(self.rows):
            self_row = self.data[r]
            other_row = other.data[r]
            for c in range(self.cols):
                self_row[c] -= other_row[c]
                
        # CRITICAL: Return self so Python doesn't overwrite your variable with None
        return self

    def __mul__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        """
        Overloaded multiplication operator (*).
        If 'other' is a Scalar -> Performs scalar multiplication (Matrix * Scalar).
        If 'other' is a Matrix -> Performs the Hadamard Product (Matrix ⊙ Matrix).
        """
        # Hadamard Product (Matrix * Matrix)
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shape mismatch for Hadamard Product: "
                    f"Cannot multiply {self.shape} by {other.shape} element-wise."
                )

            # Check if we are multiplying two Vector instances up front
            is_vector_mult = type(self).__name__ == "Vector" and type(other).__name__ == "Vector"

            result_data = []
            for r in range(self.rows):
                self_row = self.data[r]
                other_row = other.data[r]
                
                # Element-wise cell multiplication across columns
                new_row = [self_row[c] * other_row[c] for c in range(self.cols)]
                result_data.append(new_row)

            # If both inputs are vectors, flatten the 2D layout back to 1D for Vector.__init__
            if is_vector_mult:
                # Extracts values from [[x], [y], [z]] into a clean flat 1D list: [x, y, z]
                flat_vector_data = [row[0] for row in result_data]
                return type(self)(flat_vector_data)
            
            return Matrix(result_data)
            
        # Scalar Multiplication (Matrix * Scalar)
        if isinstance(other, (int, float)):
            result_data = [
                [self.data[r][c] * other for c in range(self.cols)]
                for r in range(self.rows)
            ]
            
            # If a single Vector is multiplied by a scalar, preserve Vector type
            if type(self).__name__ == "Vector":
                flat_vector_data = [row[0] for row in result_data]
                return type(self)(flat_vector_data)
                
            return Matrix(result_data)

        raise TypeError("Multiplication operand must be an int, float, or Matrix.")


    def __rmul__(self, other: Union[float, int]) -> 'Matrix':
        """
        Performs right-side matrix scalar multiplication (Scalar * Matrix).
        Uses the commutative property of scalar multiplication.
        """
        return self.__mul__(other)

    def __imul__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        """
        Overloaded in-place multiplication operator (*=).
        If 'other' is a Scalar -> Performs in-place scalar multiplication (Matrix *= Scalar).
        If 'other' is a Matrix -> Performs in-place Hadamard Product (Matrix *= Matrix).
        """
        # Track 1: In-Place Hadamard Product (Matrix *= Matrix)
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shape mismatch for in-place Hadamard Product: "
                    f"Cannot multiply {self.shape} by {other.shape} element-wise."
                )

            # Loop through the rows and mutate cells directly in place
            for r in range(self.rows):
                self_row = self.data[r]
                other_row = other.data[r]
                for c in range(self.cols):
                    self_row[c] *= other_row[c]
                    
            return self

        # Track 2: In-Place Scalar Multiplication (Matrix *= Scalar)
        if isinstance(other, (int, float)):
            for r in range(self.rows):
                self_row = self.data[r]
                for c in range(self.cols):
                    self_row[c] *= other
                    
            return self

        raise TypeError("In-place multiplication operand must be an int, float, or Matrix.")

    def map(self, func: Callable[[float], float]) -> 'Matrix':
        """
        Applies a scalar mathematical function element-wise across the matrix.
        Crucial for activating neural network nodes (e.g., Sigmoid, ReLU).
        """
        result = [[func(val) for val in row] for row in self.data]
        return Matrix(result)
    
    def __matmul__(self, other) -> 'Matrix':
        """
        Performs standard matirx multiplication using the "@" operator
        """
        if self.cols != other.rows:
            raise ValueError(
                f"Inner dimensions must match for multiplication. "
                f"Cannot multiply {self.shape} by {other.shape}"
            )

        result_data = [[0.0 for _ in range(other.cols)] for _ in range(self.rows)]

        other_cols = [list(col) for col in zip(*other.data)]

        for r in range(self.rows):
            row_vec = self.data[r]
            for c in range(other.cols):
                col_vec = other_cols[c]
                # Dot Product: Summation of A[i]*B[i]
                result_data[r][c] = sum(row_vec[i] * col_vec[i] for i in range(self.cols))

        if other.cols == 1:
            from Vector import Vector
            flat_results = [row[0] for row in result_data]
            return Vector(flat_results)

        return Matrix(result_data)

    def T(self) -> 'Matrix':
        """
        Returns the tranpose of the matrix
        """
        data = [
            [self.data[r][c] for r in range(self.rows)]
            for c in range(self.cols)
        ]
        #No special cases for vectors since the transpoe of a vector will always be a matrix.
        return Matrix(data)

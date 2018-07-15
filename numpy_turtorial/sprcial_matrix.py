import numpy as np
import matplotlib.pyplot as plt


# Diagonal matrices
diagonal = np.array([1, 2, 3, 4])

diag = np.diag(diagonal)

print(diag)

# The mutliplication between a diagonal matrix and a vector is thus just a ponderation of each element of the vector by v
v=np.array([4,3,2,1])

r=diag.dot(v)

print(r)

inv = np.linalg.inv(diag)
print(inv)

# Symmetric matrices if A=A.transpose()
A = np.array([[2, 4, -1], [4, -8, 0], [-1, 0, 3]])
print(A)
print(A.T)
print(A.transpose())


# Unit vectors
u=np.array([1,0])

# Orthogonal matrix
a=np.array([2,2])
b=np.array([2,-2])
print(a.dot(b))

# Orthogonal matrices
# A matrix is orthogonal if columns are mutually orthogonal and have a unit norm (orthonormal) and rows are mutually orthonormal and have unit norm.
# Property 1: A.T A=I
# Property 2: A.T =A.inv











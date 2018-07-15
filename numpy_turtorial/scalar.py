import numpy as np

# scalar is a single number
# vector is an array of numbers
# matrix is 2-D array
# tensor is n-dimensional array

# create a vector
x = np.array([1, 2, 3, 4, 5])

# create 3*2matrix
y = np.array([[1, 2], [3, 4], [5, 6]])

print(y)

matrix = np.matrix(y)

print(matrix)

# transsport

print(y.T)

# addition
z = np.array([[2, 3], [4, 5], [6, 7]])

print(y + z)

# add a scalar to a matrix

# Numpy can handle operations on arrays of different shapes. The smaller array will be extended to match the shape of the bigger one,
# The scalar was converted in an array of same shape as A.

m = y + 4
print(m)

# matrix product is also called dot product
A = np.array([[1, 2], [3, 4], [5, 6]])
B=np.array([[2],[4]])

C=np.dot(A,B)
print(C)

print(A.dot(B))

# Identity matrices
I=np.eye(3)
print(I)

# When ‘apply’ the identity matrix to a vector the result is this same vector:
# The space doesn’t change when we apply the identity matrix to it
v=np.array([1,2,3])

v=np.array([[1],[2],[3]])

r=I.dot(v)
print(v)
print(r)


# inverse
A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
inv = np.linalg.inv(A)

print(inv)


w=np.array([[2,-1],[1,1]])
x = np.linalg.inv(w).dot(np.array([0, 3]))

print(x)




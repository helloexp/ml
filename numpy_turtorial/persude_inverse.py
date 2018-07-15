import numpy as np

#method 1:svd
A = np.array([[7, 2], [3, 4], [5, 3]])
U, D, V = np.linalg.svd(A)

D_plus = np.zeros((A.shape[0], A.shape[1])).T
print(D_plus)
D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))
print(D_plus)

A_plus = V.T.dot(D_plus).dot(U.T)
print(A_plus)

A_p = np.linalg.pinv(A)
print(A_p)

print(A_plus.dot(A))
print(A.dot(A_plus))

# method 2:(A.T A)−1 A.T
A_plus_1 = np.linalg.inv(A.T.dot(A)).dot(A.T)

A = np.hstack((x, np.ones(np.shape(x))))





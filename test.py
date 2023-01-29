import numpy as np
from matplotlib import pyplot as plt

X = np.arange(0, 10, 0.1)
A = np.array(X + 10)
N = 0.5 * np.random.normal(0, 1, A.shape)
A = A + N
A = np.vstack((X, A))

# plot the data
plt.plot(A[0], A[1], 'o')
plt.show()
miu = [np.average(A[0]), np.average(A[1])]
print(miu)
A[0] = A[0] - miu[0]
A[1] = A[1] - miu[1]

[U, S, V] = np.linalg.svd(A, full_matrices=False)
print(U)
print(S)
print(V)
S[1] = 0
A = np.dot(U, np.dot(np.diag(S), V))
A[0] += miu[0]
A[1] += miu[1]
plt.plot(A[0], A[1], 'o')
plt.show()

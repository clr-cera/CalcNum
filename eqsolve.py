import numpy as np
np.set_printoptions(suppress=True)

def lu_decomp(A: np.ndarray):
    n = A.shape[0]

    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        L[i,i] = 1

    for k in range(n):
        for j in range(k,n):
            summ = L[k,:k] @ U[:k,j]
            U[k,j] = A[k,j] - summ
        for i in range(k+1,n):
            summ = L[i,:k] @ U[:k,k]
            L[i,k] = (A[i,k] - summ)/U[k,k]
    return L,U

A = np.matrix([[1,2,0],[1,3,1],[-2,0,1]])
S = np.matrix([[1,2,0],
              [2,15,3],
              [0,3,30]])
"""
L,U = lu_decomp(A)
"""

def chol_decomp(A: np.ndarray):
    n = A.shape[0]
    H = np.tril(A)

    for k in range(n-1):
        H[k,k] = np.sqrt(H[k,k])
        H[k+1:n,k] = H[k+1:n,k]/H[k,k]
        for j in range(k+1,n):
            H[j:n,j] = H[j:n,j] - H[j:n,k]*H[j,k]
    H[n-1,n-1] = np.sqrt(H[n-1,n-1])

    return H

#H = chol_decomp(S)
#print(H)

#print(H @ H.T)


def eliminacao_gauss_pivot(A,b):
    n = A.shape[0]
    for k in range(n - 1):
        # Find the pivot (row with max absolute value in column k from row k to n)
        p = np.argmax(np.abs(A[k:n, k])) + k  # np.argmax returns the index, so we adjust with +k
        # Swap rows k and p in A and b
        A[[k, p], k:n] = A[[p, k], k:n]
        b[[k, p]] = b[[p, k]]
        
        for i in range(k + 1, n):
            # Calculate multiplier for elimination
            m = -A[i, k] / A[k, k]
            # Perform the row operation
            A[i, k:n] = A[i, k:n] + m * A[k, k:n]
            b[i] = b[i] + m * b[k]

def sub_progressiva(L, b):
    """ Resolve um sitema triangular inferior por substituição progressiva """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]

    return y


def sub_regressiva(U, y):
    """ Resolve um sitema triangular superior por substituição regressiva """
    n = len(y)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]

    return x


def solve_lu(A, b):
    """ Resolve Ax = b utilizando decomposição LU. """
    L, U = lu_decomp(A)  # Decomposição LU
    y = sub_progressiva(L, b)  # Resolve Ly = b
    x = sub_regressiva(U, y)  # Resolve Ux = y
    return x

A = np.matrix([[500,3,3],[1,300,1],[3,5,100]], dtype=np.float64)
b = np.array([1,2,0])

#print("Solução utilizando nossa implementação do LU:")
#print(solve_lu(A, b))

#print("Solução correta:")
#print(np.linalg.solve(A,b))

#print("Gauss solution")
#eliminacao_gauss_pivot(A,b)
#print(A, b)
#print(np.linalg.solve(A,b))


def gauss_jacobi(A: np.ndarray, b: np.ndarray):
    tol = 0.00005
    n =A.shape[0]
    x0 = np.array([1]*n)
    D = np.diag(np.diag(A))
    C = np.eye(n) - np.linalg.inv(D).dot(A)
    g = np.linalg.inv(D).dot(b)
    for _ in range(20000):
        x0 = x0@C + g
        if np.linalg.norm(b-x0@A) < tol:
            return x0

    print("O método não converge")
    return x0

def gauss_seidel(A: np.ndarray, b: np.ndarray):
    tol = 0.05
    n = A.shape[0]
    x0 = b.copy()
    L = np.tril(A)
    R = A-L
    C = -1 * np.linalg.solve(L, R)
    g = np.linalg.solve(L, b)

    for _ in range(20000):
        x0 = x0@C +g

        if np.linalg.norm(x0@A -b) < tol:
            return x0
    print("Não converge")
    return x0


def gradient_conj(A: np.ndarray, b: np.ndarray):
    tol = 0.05
    alpha = 0.005
    x0 = b.copy()


    for _ in range(50000):
        grad_rev = b - x0@A
        x0 = x0 + alpha * grad_rev
        if np.linalg.norm(x0@A - b) < tol:
            return x0 
    print("O método não converge")
    return x0


print(gradient_conj(A, b))
print(np.linalg.solve(A, b))
#print(gauss_jacobi(A,b))
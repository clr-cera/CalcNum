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


#print(gradient_conj(A, b))
#print(np.linalg.solve(A, b))
#print(gauss_jacobi(A,b))

def bissecao(f, a, b):
    x = (a + b) / 2
    tol = 0.005

    for i in range(50000):
        if f(a) * f(x) < 0:
            b = x

        else:
            a = x

        print(a, b, x)
        if i == 100:
            exit()
        

        x = (a + b) / 2
        erro = abs(f(x))

        if erro < tol:
            break
    return x

f = lambda x: x**2 + x - 6
d = lambda x: 2*x + 1

# print(bissecao(f, -12, 24))

def newton(f, d):
    x = 5.5
    tol = 0.0005
    while True:
        dx = f(x)/d(x) if d(x) else 0
        x = x - dx
        if abs(dx) < tol:
            return x

def secantes(f):
    x0 = 5
    x1 = 5.5
    tol = 0.0005
    while True:
        f0 = f(x0)
        f1 = f(x1)
        ds = f1*(x1-x0) / (f1-f0)
        x0 = x1
        x1 = x1-ds
        if abs(ds) < tol:
            return x1

def ponto_fixo(f):
    x = 0
    xa = x
    xaa = xa
    tol = 0.0005
    while True:
        xaa = xa
        xa = x
        x = (6-x)**(1/2)

        if abs(f(x)) < tol:
            break
    p = np.log(abs(x - 2) / abs(xa - 2)) / np.log(abs(xa - 2) / abs(xaa - 2))
    print(p)
    return x

#print(newton(f,d))
#print(secantes(f))
#print(ponto_fixo(f))




def lagrange_interp(xi, yi, x):
    n = np.size(xi);
    m = np.size(x);

    L = np.ones((n,m));

    for i in np.arrange(n):
        for j in np.arrange(n):
            if(i != j):
                L[i,:] = (L[i,:]*(x-xi[j]))/(xi[i]-xi[j]);

    y = yi.dot(L);
    return y;

def newton_interp(xi,yi,x):
    n = np.size(xi); ni = np.size(x); N = np.ones((n,ni));
    D = np.zeros((n,n)); D[:,0] = yi;

    for j in np.arange(n-1): # matriz de diferenças divididas
        for i in np.arange(n-j-1):
            D[i,j+1] = (D[i+1,j]-D[i,j])/(xi[i+j+1]-xi[i]);

    for i in np.arange(1,n): # loop do produtório da forma de Newton
        N[i,:] = N[i-1,:]*(x-xi[i-1]);

    y = D[0,:].dot(N)

    return y

#Nunca foi testada, pode estar toda quebrada
def mmq(x, y, k):
    n = x.len()
    X = np.vander(x)
    X = X[:,n-k:n]
    a = np.linalg.solve((X.T @ X),(X.T @ y.T))
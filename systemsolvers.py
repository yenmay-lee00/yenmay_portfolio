"""
MATH2050 CW2 systemsolvers module

@author: Lee Yen May

"""

import numpy as np
# import matplotlib.pyplot as plt
import backward as bw

def no_pivoting(A,b,n,c):
    
    """
    Returns the augmented matrix M arrived at by starting from the augmented
    matrix [A b] and performing forward elimination without row interchanges
    until all of the entries below the main diagonal in the first c columns
    are 0.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        positive integer.
    c : integer
        positive integer that is at most n-1.
    
    Returns
    -------
    M : numpy.ndarray of shape (n,n+1)
        2-D array representing the matrix M.
    """
    
    # Create the initial augmented matrix
    M = np.hstack((A,b))
    
    # Continue here:...
    for i in range(c):
        
        for j in range(i+1, n):
            g = M[j,i] / M[i,i]
            M[j,i] = 0
            
            for k in range(i+1, n+1):
                M[j,k] = M[j,k] - g * M[i,k]
                
    return M

def no_pivoting_solve(A,b,n):
    """
    Returns the solution x arrived at by starting from the augmented
    matrix [A b] and performing forward elimination without row interchanges
    until all of the entries below the main diagonal in the first n-1 columns
    are 0, followed by backward substitution.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        positive integer.
    
    Returns
    -------
    x : numpy.ndarray of shape (n,1)
        2-D array representing the matrix x.
    """
    assert n >= 2 and isinstance(n, int), "The input n must be an integer such that n ≥ 2."
        
    x = bw.backward_substitution(no_pivoting(A,b,n,n-1), n)
    return x


def find_max(M, n, i):
    m = 0
    p = 1
    for j in range(i, n):
        
        if abs(M[j,i]) > m:
            m = abs(M[j,i])
            p = j
            
    return p+1
      
def partial_pivoting(A ,b ,n , c ):
    """
    The input A is of type numpy.ndarray and has shape (n, n) and represents the square matrix A. 
    The input b is of type numpy.ndarray and has shape (n, 1) and represents the column vector b. 
    The input n is an integer such that n ≥ 2.
    The input c is an integer such that 1 ≤ c ≤ n − 1, and it is used to (prematurely) stop the forward elimination with partial pivoting algorithm.
    """
    M = np.hstack((A,b))
    
    for i in range(c):
        p = find_max(M, n, i) - 1
        M[[i,p],:] = M[[p,i],:]
        
        for j in range(i+1, n):
            g = M[j,i] / M[i,i]
            M[j,i] = 0
            
            for k in range(i+1, n+1):
                M[j,k] = M[j,k] - g * M[i,k]
                
    return M

def partial_pivoting_solve(A ,b , n ):
    """
    The input A is of type numpy.ndarray and has shape (n, n) and represents the matrix A. 
    The input b is of type numpy.ndarray and has shape (n, 1) and represents the vector b. 
    The input n is an integer such that n ≥ 2.
    """
    assert n >= 2 and isinstance(n, int), "The input n must be an integer such that n ≥ 2."
        
    x = bw.backward_substitution(partial_pivoting(A,b,n,n-1), n)
    return x
    
def Doolittle(A , n ):
    
    U = A.copy()
    L = np.identity(n)
    
    for i in range(n):
            
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
        
    return L, U
    
def Gauss_Seidel(A ,b ,n , x0 , tol , maxits ):
    
    x = np.zeros((n,1))
    k = 1
    x_old = x0
    
    while(True):
        if k > maxits:
            x = "Desired tolerance not reached after maxits iterations have been performed ."
            break
        else:
            for i in range(n):
                sum1 = 0
                sum2 = 0
                for j in range(i):
                    sum1 += (-A[i][j] * x[j])
                for l in range(i+1, n):
                    sum2 += (-A[i][l] * x_old[l])
                    
                x[i] = (1.0 / A[i,i]) * (sum1 + sum2) + (b[i] / A[i][i])
                
            if np.max(np.abs(b-np.matmul(A,x))) < tol:
                break
            else:
                    x_old = x
                    k += 1
    return x    





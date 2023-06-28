"""
MATH2019 CW4 polynomial_interpolation module
"""

### Load other modules ###
import numpy as np
import matplotlib.pyplot as plt
### No further imports should be required ###

### Comment out incomplete functions before testing ###

#%%
def richardson(f,x0,h,k):
    """

    Parameters
    ----------
    f : function of one variable
        function to be approximated
    x0 : float
        initial point
    h : integer
        width
    k : integer
        level

    Returns
    -------
    deriv_approx : float
        extrapolation of f at k

    """
    N1 = lambda h: (f(x0+h)-f(x0-h)) / (2*h)
    arr = np.zeros((k,k))
    # generate first row
    for i in range(k):
        arr[0][i] = N1(h/(2**i))
    # generate remaining rows
    for j in range(1,k):
        ak = 1 / (1-4**(j))
        bk = -4**(j) / (1-4**(j))
        
        for m in range(k-j):
            arr[j][m] = ak*arr[j-1][m] + bk*arr[j-1][m+1]    
        
    deriv_approx = arr[-1][0]
    return deriv_approx

#%%
def richardson_errors(f,f_deriv,x0,n,h_vals,k_max):
    """
    h_vals = np.logspace(-5,1,20), k_max = 4, x0 = 0
    a) Error terms converge for all N. The higher the N, the faster the error converges.
    
    b) Error terms are parallel to each other.The error terms for larger N are greater.

    """
    error_matrix = np.zeros((k_max,n))
    
    for k in range(1,k_max+1):
        for h in range(len(h_vals)):
            error_matrix[k-1][h] = abs(f_deriv(x0) - richardson(f,x0,h_vals[h],k))
        
    fig = plt.figure() 
    for k in range(k_max):
        plt.loglog(h_vals, error_matrix[k], label = "N" + str(k+1))
        
    plt.ylabel('error term $\{E_{{k-1}{i}}\}_{i=0}^{n-1}$')
    plt.xlabel('widths $\{h_i\}_{i=0}^{n-1}$')
    plt.title('Error of Richardson Extrapolation')
    plt.legend()

    return error_matrix, fig

#### Your submission should have no code after this point ####




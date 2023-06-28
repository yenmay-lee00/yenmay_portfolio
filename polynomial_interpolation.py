"""
MATH2019 CW3 polynomial_interpolation module
"""

### Load other modules ###
import numpy as np
import matplotlib.pyplot as plt
### No further imports should be required ###

### Comment out incomplete functions before testing ###

#%%
def lagrange_poly(p,xhat,n,x,tol):
    '''
    Parameters
    ----------
    p : int
        order of polynomial
    xhat : numpy.narray
        nodal points
    n : int
        the number of evaluation points
    x : numpy.narray
        the evaluation points
    tol : float
        error small enough to identify the difference between two numbers as insignificant

    Returns
    -------
    lagrange_matrix : numpy.narray
        lagrange polynomials
    error_flag : int
        0 if nodal points xhat are distinct and 1 otherwise
    '''
    lagrange_matrix = np.ones((p+1, n))
    error_flag = 0
    
    for i in range(p+1):
        for j in range(p+1):
            if i != j:
                if np.abs(xhat[i] - xhat[j]) < tol:
                    error_flag = 1
                    return lagrange_matrix, error_flag   
                lagrange_matrix[i,:] = lagrange_matrix[i,:] * ((x-xhat[j])/(xhat[i]-xhat[j]))

    return lagrange_matrix, error_flag

#%%
def uniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    '''

    Parameters
    ----------
    a : float
        initial point of the nodal interpolation points
    b : float
        end point of the nodal interpolation points
    p : int
        order of polynomial
    n : int
        the number of evaluation points
    x : numpy.narray
        the evaluation points
    f : function
        original function
    produce_fig : bool
        produce figure when True and None when False

    Returns
    -------
    interpolant : numpy.narray
        the interpolating points to original function
    fig : plot
        figure

    '''
    xhat = np.linspace(a,b,p+1)
    tol = 1.0e-10
    L = lagrange_poly(p, xhat, n, x, tol)[0]
    interpolant = np.ones(n)
    ppx = 0
    
    for i in range(n):
        for j in range(p+1):
            ppx += f(xhat[j])*(L[j][i]) 
        
        interpolant[i] *= ppx
        ppx = 0
        
    ### Example plot - replace with your own
    # fig = plt.figure() ##This line is required before any other plot commands
    # plt.plot([0,1,2,3],[0,1,2,3]) 
    
    if produce_fig:
        fig = plt.figure() 
        plt.plot(x, interpolant, label='interpolant') 
        plt.plot(x, f(x), label='f(x)')
        plt.legend()
        plt.xlabel('x values')
        plt.ylabel('Interpolation Points f(x) and p(x)')
        plt.title('Uniform Polynomial Interpolation')
    else:
        return interpolant, None 
    return interpolant, fig

#%%
def nonuniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    '''

    Parameters
    ----------
    a : float
        initial point of the nodal interpolation points
    b : float
        end point of the nodal interpolation points
    p : int
        order of polynomial
    n : int
        the number of evaluation points
    x : numpy.narray
        the evaluation points
    f : function
        original function
    produce_fig : bool
        produce figure when True and None when False

    Returns
    -------
    nu_interpolant : numpy.narray
        the interpolating points to original function
    fig : plot
        figure

    '''
    A =[[-1, 1], [1, 1]]
    Y = [a, b]
    res = np.linalg.inv(A).dot(Y)
    m = res[0]
    c = res[1]
    
    xhat = []
    for i in range(p+1):
        y = m * np.cos((2*i+1) / (2*(p+1)) * np.pi) + c
        xhat.append(y)
        
    tol = 1.0e-10
    L = lagrange_poly(p, xhat, n, x, tol)[0]
    nu_interpolant = np.ones(n)
    ppx = 0
    
    for i in range(n):
        for j in range(p+1):
            ppx += f(xhat[j])*(L[j][i]) 
        
        nu_interpolant[i] *= ppx
        ppx = 0
        
    if produce_fig:
        fig = plt.figure() 
        plt.plot(x, nu_interpolant, label='interpolant') 
        plt.plot(x, f(x), label='f(x)')
        plt.legend()
        plt.xlabel('x values')
        plt.ylabel('Interpolation Points f(x) and p(x)')
        plt.title('Non-uniform Polynomial Interpolation')
    else:
        return nu_interpolant, None 
    
    return nu_interpolant, fig

#%%
def compute_errors(a,b,n,P,f):
    '''

    Parameters
    ----------
    a : float
        initial point of the nodal interpolation points
    b : float
        end point of the nodal interpolation points
    n : int
        the number of evaluation points
    P : numpy.ndarray
        all orders of polynomial until n
    f : function
        original function

    Returns
    -------
    error_matrix : numpy.ndarray
        the error for a range of polynomial degrees P
    fig : plot
        figure
    
    ANSWER:  
    Non-uniform errors series converges whereas uniform errors series diverges.
    Non-uniform errors series is decreasing whereas uniform series is decreasing then increasing.
    a) f(x) = cos(2pix) stabilizes.

    '''

    error_matrix = np.ones((2,n))
    k = 2000
    x = np.linspace(a,b,k)

    for p in P:
        uni_arr = uniform_poly_interpolation(a,b,p,k,x,f,False)[0]
        nonuni_arr = nonuniform_poly_interpolation(a,b,p,k,x,f,False)[0]
        for i in range(k):
            error_matrix[0][p-1] = max(abs(np.subtract(uni_arr, f(x))))
            error_matrix[1][p-1] = max(abs(np.subtract(nonuni_arr, f(x))))
                                    
    fig = plt.figure() 
    plt.semilogy(P, error_matrix[0], label='Uniform Errors') 
    plt.semilogy(P, error_matrix[1], label='Non-uniform Errors')
    plt.legend()
    plt.xlabel('Polynomial Degrees')
    plt.ylabel('Absolute Error')
    plt.title('Error Term')
    
    return error_matrix, fig

#%%
def piecewise_interpolation(a,b,p,m,n,x,f,produce_fig):
    '''

    Parameters
    ----------
    a : float
        initial point of the nodal interpolation points
    b : float
        end point of the nodal interpolation points
    p : int
        order of polynomial
    m : int
        number of subintervals
    n : int
        the number of evaluation points
    x : numpy.narray
        the evaluation points
    f : function
        original function
    produce_fig : bool
        produce figure when True and None when False

    Returns
    -------
    p_interpolant : numpy.narray
        the interpolating points to original function
    fig : plot
        figure

    '''
    p_interpolant = []
    xhat = np.linspace(a,b,m+1)
    
    for i in range(1,m+1):
        nonuni_arr = nonuniform_poly_interpolation(xhat[i-1],xhat[i],p,n,x,f,False)[0]
        p_interpolant.append(nonuni_arr[i])

    p_interpolant.insert(0,nonuniform_poly_interpolation(xhat[0],xhat[1],p,n,x,f,False)[0][0])
    p_interpolant.append(nonuniform_poly_interpolation(xhat[-1],xhat[-2],p,n,x,f,False)[0][-1])
    np.array(p_interpolant)
    
    if produce_fig:
        fig = plt.figure() 
        plt.plot(x, p_interpolant, label='interpolant') 
        plt.plot(x, f(x), label='f(x)')
        plt.legend()
        plt.xlabel('x values')
        plt.ylabel('Interpolation Points f(x) and p(x)')
        plt.title('Piecewise Polynomial Interpolation')
    else:
        return p_interpolant, None
    
    return p_interpolant, fig

#### Your submission should have no code after this point ####


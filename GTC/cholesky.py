"""
"""
from __future__ import division

import math 
import cmath 
import numpy as np

from GTC import la 

__all__ = ('cholesky_decomp',)

#----------------------------------------------------------------------------
def cholesky_decomp(a):
    """
    Evaluate the Cholesky decomposition of matrix ``a``
    
    .. versionadded:: 1.4.x
    
    :arg a: - positive definite matrix of ``float`` or ``complex`` 
    
    The matrix returned is lower triangular. 
            
    **example** ::
    
        >>> A = np.array( [[ 1.,  .2],[ .2,  5.]] ) 
        >>> cholesky_decomp(A) 
        array([[1.        , 0.        ],
              [0.2       , 2.22710575]])
        
    """
    N,M = a.shape
    assert N == M, "a square matrix is needed!"
    
    # Don't want `a` to change, but the algorithm is  
    # iterative and so matrix entries are modified.
    # Results will have an upper triangle of zeros 
    L = np.zeros( (N,M), dtype=a.dtype ) 
    
    for i in range(N):
        for j in range(i,N):
        
            s = a[j,i] + sum(
                -L[i,k]*L[j,k].conjugate()
                    for k in range(i-1,-1,-1) 
                    # Read the lower triangle
            )
                        
            if i == j: 
                if isinstance(s,(int,float)): 
                    if abs(s)>0:
                        p_i = math.sqrt( s )
                    else:
                        raise RuntimeError(
                            "cholesky_decomp: matrix `a` "
                            "is not positive definite"
                        )
                elif isinstance(s, complex):
                    if abs(s.imag) > 1E-13:
                        raise RuntimeError(
                            "cholesky_decomp: matrix `a` "
                            "is not positive definite"
                        )                        
                    p_i = cmath.sqrt( s.real )
                else:
                    assert False, "unexpected: !r".format(s)
                    
            L[j,i] = s / p_i
                 
    return L
    
#----------------------------------------------------------------------------
def cholesky_inv(L):
    """
    Return the inverse of matrix ``L``
    
    .. versionadded:: 1.4.x
    
    :arg L: a matrix obtained by Cholesky decomposition
    
    ``L`` is a lower triangular matrix in the format generated 
    by :func:`cholesky_decomp`. 
        
    **example** ::
    
        >>> A = np.array( [[ 1.,  .2],[ .2,  5.]] ) 
        >>> L = cholesky_decomp(A) 
        >>> Linv = cholesky_inv(L)
        >>> Linv
        array([[ 1.        ,  0.        ],
                [-0.08980265,  0.44901326]])
        >>> np.matmul(Linv,L)
        array([[1.00000000e+00, 0.00000000e+00],
               [4.98504854e-18, 1.00000000e+00]])
        
    """
    N,M = L.shape
    assert N == M, "a square matrix is needed!"
    
    b = L.copy()

    for i in range(N):
        b[i,i] = 1.0/L[i,i] 

        for j in range(i+1,N):
            b[j,i] = sum( 
                -b[j,k]*b[k,i] 
                    for k in range(i,j) 
            )/b[j,j]

    return b
    
#============================================================================
if __name__ == '__main__': 
    
    import doctest
    from GTC import *
    
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )


    
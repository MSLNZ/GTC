"""
  
Module contents
---------------
  
.. versionadded:: 2.0

"""
import math
import numpy as np

from GTC.lib import value
from GTC.type_a_linear_models import ModelFit
  
__all__ = (
    'ols',
    'wls',
    'gls'
)
#----------------------------------------------------------------------------
def lsfit(x,y,sig=None,fn=None):
    """
    Solve `x @ beta = y` for `beta` 
    
    :arg x: an array of stimulus data
    :arg y: a 1-D array of response data
    :arg sig: a 1-D array of standard deviations associated with response data
    :arg fn: returns a sequence of basis function values at a stimulus value
    :returns: `coef`, `res`, `ssr`
 
    A Singular Value Decomposition (SVD) algorithm is used to obtain the solution.
    
    The returned `coef` is an uncertain-number sequence of parameter estimates for `beta`.
        
    The returned `ssr` is the squared sum of residuals of predicted values to the `y` data.
    
    If `x` or `y` are arrays of uncertain numbers, only the values are used in calculations.

    When `sig` is not provided, the standard deviations are taken as unity
    
    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.
 
    """    
    # M - number of data points 
    # P - number of parameters to fit 
    M = len(x) 
    if sig is None: 
        sig = np.ones( (M,), dtype=np.float64 )
    else:
        sig = np.array( sig, dtype=np.float64 )
        
    # Allow uncertain-number inputs, but
    # construct arrays `a` and `b` from values 
    if fn is None:
        P = len( x[0] )   
        def row(row_i,s_i):
            return [ value(x_i/s_i) for x_i in row_i ]
            
        a = np.array( [ row(row_i,s_i) for row_i,s_i in zip(x,sig) ],
            dtype=np.float64 
        )
        b = np.array( [ y_i/s_i for y_i,s_i in zip(y,sig) ], 
            dtype=object 
        ) 
    else:       
        # fn(x_i) returns an P-sized array of values for
        # each basis function at the stimulus point `x_i`
        P = len( fn(x[0])  )   

        a = np.empty( (M,P), dtype=np.float64 )
        b = np.empty( (M,), dtype=object )    
        
        for i in range(M):
            afunc_i = fn(x[i])
            tmp = 1.0/sig[i]
            for j in range(P):
                a[i,j] = value( tmp*afunc_i[j] )
                
            b[i] = tmp*y[i] 
            
    if M <= P:
        raise RuntimeError( f"M = {M} but should be > {P}" )     
              
    u,w,vh = np.linalg.svd(a, full_matrices=False )
    v = vh.T    
    
    # Select almost singular values
    wmax = max(w)
    # wmin = min(w)
    # logC = math.log10(wmax/wmin)
    # # The base-b logarithm of C is an estimate of how many 
    # # base-b digits will be lost in solving a linear system 
    # # with the matrix. In other words, it estimates worst-case 
    # # loss of precision. 
    # # C is the condition number: the ratio of the largest to smallest 
    # # singular value in the SVD
    
    # `TOL` is used to adjust the small singular values set to zero
    # This avoids numerical precision problems, but will make the 
    # solution slightly less accurate. 
    TOL = 1E-5
    
    thresh = TOL*wmax 
    w = np.array([ 
        w_i if w_i >= thresh else 0. 
            for w_i in w 
    ])
    
    coef = v @ np.diag(1 / w) @ u.T @ b
    
    # Residuals and sum of squared residuals
    res = b - np.dot(a, coef) 
    ssr = math.fsum( value(res_i)**2 for res_i in res )

    return coef, res, ssr  
  
#----------------------------------------------------------------------------
def ols(x,y,fn=None):
    """Ordinary least squares fit of ``y`` to ``x``
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg fn: returns a sequence of basis function values at a stimulus value
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`OLSModel`

    The argument `x` may be an array of numbers or uncertain numbers. 
    
    The argument `y` must be an array of uncertain numbers.

    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.

    """
    M = len(y)
    
    if M != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),M)
        )
        
    sig = np.ones( (M,) )
    coef, res, ssr = lsfit(x,y,sig,fn)
   
    return OLSModel( coef,res,ssr,M,fn )

#----------------------------------------------------------------------------
def wls(x,y,u_y=None,fn=None):    
    """Weighted least squares fit of ``y`` to ``x``
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg u_y: sequence of standard uncertainties for response data    
    :arg fn: returns a sequence of basis function values at a stimulus value
    :returns:   an object containing regression results
    :rtype:     :class:`WLSModel`
 
    The argument `x` may be an array of numbers or uncertain numbers. 
    
    The argument `y` must be an array of uncertain numbers.

    When the optional argument `u_y` is not given, the uncertainties of the
    `y` elements are used for provide weights for the calculation.
    
    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.
 
    """
    M = len(y)
    
    if M != len(x):
        raise RuntimeError( f"len(x) != len(y)" )
 
    if u_y is None:
        u_y = [ y_i.u for y_i in y ]
    elif M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
        
    coef, res, ssr = lsfit(x,y,u_y,fn)
    
    return WLSModel( coef,res,ssr,M,fn )
    
#----------------------------------------------------------------------------
def gls(x,y,cv=None,fn=None,label=None):
    """Generalised least squares fit of ``y`` to ``x``
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg cv: an ``M`` by ``M`` real-valued covariance matrix for the responses
    :arg fn: a user-defined function relating ``x`` the response
    :returns:   an object containing regression results
    :rtype:     :class:`GLSModel``
 
    The argument `x` may be an array of numbers or uncertain numbers. 
    
    The argument `y` must be an array of uncertain numbers.

    When the optional argument `cv` is not given, a variance-covariance 
    matrix is obtained from the `y` elements.

    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.
 
    """
    M = len(y)        
    if fn is None:
        P = len(x[0])
    else:
        P = len( fn(x[0]) ) 

    if cv is None:
        from GTC import (variance, get_covariance)
        cv = np.empty( (M,M), dtype=np.float64 )
        for i in range(M):
            cv[i,i] = variance(y[i])
            for j in range(i+1,M):
                cv[i,j] = cv[j,i] = get_covariance(y[i],y[j])
    else:
        if cv.shape != (M,M):
            raise RuntimeError( f"{cv.shape} should be {({M},{M})}" )     
    
    def row_values(row_i): 
        return [ value(x_i) for x_i in row_i ]
        
    if fn is None:
        X = np.array( [ row_values(row_i) for row_i in x ],
            dtype=np.float64 
        )
    else:
        # fn expands the P basis function values at x_i
        X = np.array( [ row_values(fn(x_i)) for x_i in x ],
            dtype=np.float64
        )  
        
    K = np.linalg.cholesky(cv)
    Kinv = np.linalg.inv(K)

    Y = np.array( y, dtype=object ).T   

    a = Kinv @ X 
    b = Kinv @ Y 

    coef = ols(a,b).beta

    res = np.array([ value(Y[i] - np.dot(coef,X[i])) for i in range(M) ]
    , dtype=float
    )

    tmp = np.dot(Kinv, res)
    ssr = np.dot(tmp.T, tmp)

    return GLSModel( coef,res,ssr,M,fn )
  
#----------------------------------------------------------------------------
class OLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-B Ordinary Least-Squares:
'''
        return header + ModelFit.__str__(self)
 
class WLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-B Weighted Least-Squares:
'''
        return header + ModelFit.__str__(self)
        
class GLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-B Generalised Least-Squares:
'''
        return header + ModelFit.__str__(self)   
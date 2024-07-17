"""
               
"""
import math
import numpy as np

from GTC.lib import value
from GTC.type_a_linear_models import ModelFit
from GTC import SVD
from GTC import cholesky 

#----------------------------------------------------------------------------
def lsfit(x,y,sig=None,fn=None):
    """
    Solve `x @ beta = y` for `beta` 

    .. versionadded:: 1.4.x
    
    :arg x: an array of stimulus data
    :arg y: a 1-D array of response data
    :arg sig: a 1-D array of standard deviations associated with response data
    :arg fn: evaluate a sequence of basis function values at a stimulus value
    :returns: `coef`, `res`, `ssr`
    
    """    
  
    coef, res, ssr, _, _ = SVD.svdfit(x,y,sig,fn)
    
    return coef, res, ssr  
    
#----------------------------------------------------------------------------
def ols(x,y,fn=None):
    """Ordinary least squares fit of ``y`` to ``x``
    
    :arg x: sequence of stimulus values (independent-variable)
    :arg y: sequence of response data (dependent-variable)   
    :arg fn: a user-defined function relating the stimulus to the response
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`OLSFit``
    
    """
    M = len(y)
    
    if M != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),M)
        )
        
    sig = np.ones( (M,) )
    coef, res, ssr = lsfit(x,y,sig,fn)
   
    return ModelFit( coef,res,ssr,fn,M )

#----------------------------------------------------------------------------
def wls(x,y,u_y,fn=None):    
    """Weighted least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``M`` stimulus values (independent-variables)
    :arg y: a sequence of ``M`` responses (dependent-variable)  
    :arg u_y: a sequence of ``M`` standard uncertainties in the responses
    :arg fn: a user-defined function relating ``x`` the response
    :returns:   an object containing regression results
    :rtype:     :class:`WLSFit``
    
    """
    M = len(y)
    
    if M != len(x):
        raise RuntimeError( f"len(x) != len(y)" )
    if M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
    
    coef, res, ssr = lsfit(x,y,u_y,fn)
    
    return ModelFit( coef,res,ssr,fn,M )
    
#----------------------------------------------------------------------------
def gls(x,y,cv,fn=None,label=None):
    """Generalised least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``M`` stimulus values (independent-variables)
    :arg y: a sequence of ``M`` responses (dependent-variable)  
    :arg cv: an ``M`` by ``M`` real-valued covariance matrix for the responses
    :arg fn: a user-defined function relating ``x`` the response
    :returns:   an object containing regression results
    :rtype:     :class:`GLSFit``
    
    """
    M = len(y)        
    if fn is None:
        P = len(x[0])
    else:
        P = len( fn(x[0]) ) 

    if cv.shape != (M,M):
        raise RuntimeError( f"{cv.shape} should be {({M},{M})}" )     
    
    K = cholesky.cholesky_decomp(cv)
    Kinv = cholesky.cholesky_inv(K)
        
    if fn is None:
        X = np.array( x,dtype=object )
    else:
        # fn expands the P basis function values at x_i
        X = np.array( [ fn(x_i) for x_i in x ],
            dtype=object
        )  
        
    Y = np.array( y, dtype=object ).T   

    a = Kinv @ X 
    b = Kinv @ Y 
         
    coef = ols(a,b,fn=fn).beta

    res = np.array([ value(Y[i] - np.dot(coef,X[i])) for i in range(M) ]
    , dtype=float
    )

    tmp = np.dot(Kinv, res)
    ssr = np.dot(tmp.T, tmp)

    return ModelFit( coef,res,ssr,fn,M )
  
    
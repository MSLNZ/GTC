"""
  
Module contents
---------------
  
"""
import math
import numpy as np

from GTC.lib import value
from GTC.type_a_linear_models import ModelFit
# from GTC import SVD
# from GTC import cholesky 

from GTC.misc import _dtype_float

# #----------------------------------------------------------------------------
# def lsfit(x,y,sig=None,fn=None):
    # """
    # Solve `x @ beta = y` for `beta` 

    # .. versionadded:: 1.4.x
    
    # :arg x: an array of stimulus data
    # :arg y: a 1-D array of response data
    # :arg sig: a 1-D array of standard deviations associated with response data
    # :arg fn: evaluate a sequence of basis function values at a stimulus value
    # :returns: `coef`, `res`, `ssr`
    
    # """    
    # coef, res, ssr, _, _ = SVD.svdfit(x,y,sig,fn)
    
    # return coef, res, ssr  
  
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
    # M - number of data points 
    # P - number of parameters to fit 
    M = len(x) 
    if sig is None: 
        sig = np.ones( (len(x),) )
    else:
        sig = np.array( sig, dtype=np.float64 )
        
    # Allow uncertain-number inputs:
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

        a = np.empty( (M,P), dtype=_dtype_float(x) )
        b = np.empty( (M,), dtype=object )    
        
        for i in range(M):
            afunc_i = fn(x[i])
            tmp = 1.0/sig[i]
            for j in range(P):
                a[i,j] = tmp*afunc_i[j]
                
            b[i] = tmp*y[i] 
            
    if M <= P:
        raise RuntimeError( f"M = {M} but should be > {P}" )     
              
    u,w,vh = np.linalg.svd(a, full_matrices=False )
    v = vh.T    
    
    # `TOL` is used to set relatively small singular values to zero
    # Doing so avoids numerical precision problems, but will make the 
    # solution slightly less accurate. The value can be varied.
    TOL = 1E-5
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
    
    thresh = TOL*wmax 
    w = np.array([ 
        w_i if w_i >= thresh else 0. 
            for w_i in w 
    ])
    
    # coef = svbksb(u,w,v,b)
    coef = v @ np.diag(1 / w) @ u.T @ b
    
    # cv_coef = svdvar(v,w) 
    # wti = [
        # 1.0/(w_i*w_i) if w_i != 0 else 0.0
            # for w_i in w 
    # ]
    # cv_coef = v @ np.diag(wti) @ vh
    # print(cv_coef)
    # print([c.v for c in coef])

# Residuals and sum of squared residuals
    res = b - np.dot(a, coef) 
    ssr = math.fsum( value(res_i)**2 for res_i in res )
          
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
   
    return OLSModel( coef,res,ssr,M,fn )

#----------------------------------------------------------------------------
def wls(x,y,u_y=None,fn=None):    
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
 
    if u_y is None:
        u_y = [ y_i.u for y_i in y ]
    elif M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
        
    coef, res, ssr = lsfit(x,y,u_y,fn)
    
    return WLSModel( coef,res,ssr,M,fn )
    
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
    
    # K = cholesky.cholesky_decomp(cv)
    # Kinv = cholesky.cholesky_inv(K)
    K = np.linalg.cholesky(cv)
    Kinv = np.linalg.inv(K)
        
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
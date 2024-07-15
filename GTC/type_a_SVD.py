"""


Module contents
---------------

"""
import math
import numpy as np
from GTC.linear_algebra import _dtype_float

from GTC.lib import (
    UncertainReal,
    real_ensemble,
    append_real_ensemble,
    value,
)
from GTC import cholesky 
from GTC import magnitude, sqrt     # Polymorphic functions
from GTC import result              
from GTC import inf

_ureal = UncertainReal._elementary
_const = UncertainReal._constant

# We mostly use numpy routines in this module because they will 
# execute more quickly than the Numerical Recipe equivalents.
# #----------------------------------------------------------------------------
# # TODO solve and svbksb don't belong in the type-A module
# # They should be available in type-B and take generic arguments
# def svbksb(u,w,v,b):
    # """
    # Solve A*X = B
    
    # .. versionadded:: 1.4.x
    
    # :arg u: an ``M`` by ``P`` matrix
    # :arg w: an ``P`` element sequence
    # :arg v: an ``P`` by ``P`` matrix
    # :arg b: an ``P`` element sequence 
    
    # Returns a list containing the solution ``X`` 
    
    # """
    # # M,P = u.shape 
    # # tmp = np.zeros( (P,), dtype=float  ) 
    # # for j in range(P):
        # # if w[j] != 0:
            # # tmp[j] = math.fsum(
                # # u[i,j]*b[i] for i in range(M)
            # # ) / w[j]
 
    # # return np.matmul(v,tmp)

    # return np.matmul( v,1.0/w*np.matmul(u.T,b) )
# #------------------------------------------------
# def solve(a,b,TOL=1E-5):
    # """
    # Solve a.x = b
    
    # .. versionadded:: 1.4.x

    # """
    # u,w,vh = np.linalg.svd(a, full_matrices=False )
    # v = vh.T    

    # wmax = max(w)
    # thresh = TOL*wmax 
    # # wmin = min(w)
    # # logC = math.log10(wmax/wmin)
    # # # The base-b logarithm of C is an estimate of how many 
    # # # base-b digits will be lost in solving a linear system 
    # # # with the matrix. In other words, it estimates worst-case 
    # # # loss of precision. 

    # # # C is the condition number: the ratio of the largest to smallest 
    # # # singular value in the SVD
    
    # # `TOL` is used to set relatively small singular values to zero
    # # Doing so avoids numerical precision problems, but will make the 
    # # solution slightly less accurate. The value can be varied.
    # w = np.array([ 
        # w_i if w_i >= thresh else 0. 
            # for w_i in w 
    # ])
    
    # return svbksb(u,w,v,b)

#----------------------------------------------------------------------------
def svdfit(x,y,sig=None,fn=None):
    """
    Solve `x @ beta = y` for `beta` 

    .. versionadded:: 1.4.x
    
    :arg x: an array of stimulus data
    :arg y: a 1-D array of response data
    :arg sig: a 1-D array of standard deviations associated with response data
    :arg fn: evaluate a sequence of basis function values at a stimulus value
    :returns: `coef`, `cv_coef`, `ssr`
 
    The returned `coef` is a real-valued sequence of parameter estimates for `beta`.
    `cv_coef` is a real-valued covariance matrix associated with the estimates.
    `ssr` is the squared sum of residuals of the fit to the `y` values.
    
    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.
    
    """
    # M - number of data points 
    # P - number of parameters to fit 
    M = len(x) 
    if sig is None: sig = np.ones( (len(x),) )
        
    # Allow uncertain-number inputs:
    # construct arrays `a` and `b` from values 
    if fn is None:
        P = len( x[0]  )   
        def row(row_i,s_i):
            return [ value(x_i/s_i) for x_i in row_i ]
            
        a = np.array( [ row(row_i,s_i) for row_i,s_i in zip(x,sig) ],
            dtype=_dtype_float(x) 
        )
        b = np.array( [ value(y_i/s_i) for y_i,s_i in zip(y,sig) ], 
            dtype=_dtype_float(x) 
        )   
    else:       
        # fn(x_i) returns an P-sized array of values for
        # each basis function at the stimulus point `x_i`
        P = len( fn(x[0])  )   

        a = np.empty( (M,P), dtype=_dtype_float(x) )
        b = np.empty( (M,), dtype=_dtype_float(x) )    
        
        for i in range(M):
            afunc_i = fn(x[i])
            tmp = 1.0/sig[i]
            for j in range(P):
                a[i,j] = value( tmp*afunc_i[j] )
                
            b[i] = value( tmp*y[i] ) 
            
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
    w_inv = np.diag(1 / w)
    coef = v @ w_inv @ u.T @ b
    
    # cv_coef = svdvar(v,w) 
    wti = [
        1.0/(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    cv_coef = v @ np.diag(wti) @ vh

    # sum of squared residuals
    # `y` values may be uncertain numbers
    # so we take the value only
    ssr = 0.0 
    if fn is None:
        for i,row_i in enumerate(x):
            f_i = math.fsum(
                coef[j]*value(row_i[j])
                    for j in range(P)
            )
            tmp = (value(y[i]) - f_i)/sig[i]
            ssr += tmp*tmp            
    else:
        for i in range(M):
            afunc_i = fn(x[i])
            f_i = math.fsum(
                    coef[j]*afunc_i[j]
                        for j in range(P)
                )
            tmp = (value(y[i]) - f_i)/sig[i] 
            ssr += tmp*tmp 
          
    return coef, cv_coef, ssr 
 
# #----------------------------------------------------------------------------
# # This function is used internally in this module, and called by some 
# # unit tests in both test_type_a_SVD.py and test_type_b_SVD.py modules.
# def svdvar(v,w):
    # """
    # Calculate the variance-covariance matrix after ``svdfit``
    
    # .. versionadded:: 1.4.x
    
    # :arg v: an ``P`` by ``P`` matrix of float
    # :arg w: an ``P`` element sequence of float 
    
    # """
    # # Based on Numerical Recipes 'svdvar'   
    # wti = [
        # 1.0/(w_i*w_i) if w_i != 0 else 0.0
            # for w_i in w 
    # ]
    
    # P = len(w)  
    # cv = np.empty( (P,P), dtype=float )
    # for i in range(P):
        # for j in range(i+1):
            # cv[i,j] = cv[j,i] = math.fsum(
                # v[i,k]*v[j,k]*wti[k]
                    # for k in range(P)
            # )
    
    # return cv  
 
#----------------------------------------------------------------------------
def _coef_as_uncertain_numbers(coef,cv,df=inf,label='beta'):
    """
    Return a sequence of uncertain numbers representing parameter estimates 

    When the degrees of freedom are finite, the uncertain-number parameters  
    are added to an ensemble, so that effective degrees of freedom calculations
    can be performed downstream.
    
    """
    beta = []
    ensemble = []
    for i in range( len(coef) ):
        u_i = math.sqrt(cv[i,i])
        if abs(u_i) <= 1E-13:    
            u_i = 0.0
            b_i = _const(
                coef[i],
                label=f'{label}_{i}'
            )
        else:    
            b_i = _ureal(
                coef[i],
                u_i,
                df,
                label=f'{label}_{i}',
                independent=False
            )
            ensemble.append( b_i )
            
        beta.append( b_i )
    
    # An ensemble is only useful for finite DoF
    if not math.isinf(df): real_ensemble( ensemble, df )

    for i,b_i in enumerate(beta): 
        if b_i.u == 0.0: continue
        for j,b_j in enumerate(beta[:i]):
            if b_j.u == 0.0: continue                
            if abs(cv[i,j]) > 1E-13:
                b_i.set_correlation(cv[i,j]/(b_i.u*b_j.u),b_j)
            
    return beta
     
#----------------------------------------------------------------------------
def ols(x,y,fn=None,label='beta'):
    """Ordinary least squares fit of response data to stimulus values
    
    :arg x: sequence of stimulus values (independent-variable)
    :arg y: sequence of response data (dependent-variable)   
    :arg fn: evaluates a sequence of basis function values at the stimulus
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`OLSFit``

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    """
    # x can be MxP, alternatively x can be 1-D and `fn` expands the basis functions
    M = len(y)
    
    if M != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),M)
        )
            
    coef, cv, ssr = svdfit(x,y,fn=fn)

    df = M - len(coef)
    cv = ssr/df * cv

    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)

    return OLSFit( coef,ssr,M,fn )  
    
#----------------------------------------------------------------------------
def wls(x,y,u_y,fn=None,dof=None,label='beta'):
    """Weighted least squares fit of response data to stimulus values
    
    :arg x: sequence of stimulus values (independent-variable)
    :arg y: sequence of response data (dependent-variable)   
    :arg u_y: sequence of standard uncertainties for response data    
    :arg fn: evaluates a sequence of basis function values at the stimulus
    :arg dof: degrees of freedom    
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`WLSFit``

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    """
    M = len(y)   
    if M != len(x):
        raise RuntimeError( "len(x) != len(y)" )
    if M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
        
    coef, cv, ssr = svdfit(x,y,u_y,fn)   
    
    if dof is None:
        df = inf
    else:
        df = M - len(coef)
        cv = ssr/df * cv
        
    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)

    return WLSFit( coef,ssr,M,fn )  

#----------------------------------------------------------------------------
def rwls(x,y,s_y,fn=None,dof=None,label='beta'):
    """Relative least squares fit of response data to stimulus values
    
    :arg x: sequence of stimulus values (independent-variable)
    :arg y: sequence of response data (dependent-variable)   
    :arg s_y: sequence of scale factors for response data    
    :arg dof: degrees of freedom    
    :arg fn: evaluates a sequence of basis function values at the stimulus
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`WLSFit``

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    """
    M = len(y)   
    if M != len(x):
        raise RuntimeError( "len(x) != len(y)" )
    if M != len(s_y):
        raise RuntimeError( "len(x) != len(s_y)")
        
    coef, cv, ssr = svdfit(x,y,s_y,fn)   
    
    df = M - len(coef) if dof is None else dof        
    cv = ssr/df * cv
        
    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)
    
    return RWLSFit( coef,ssr,M,fn )  

#----------------------------------------------------------------------------
def gls(x,y,cv,fn=None,label='beta'):
    """Generalised least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``M`` stimulus values (independent-variables)
    :arg y: a sequence of ``M`` responses (dependent-variable)  
    :arg cv: an ``M`` by ``M`` real-valued covariance matrix for the responses
    :arg fn: evaluates a sequence of basis function values at the stimulus
    :returns:   an object containing regression results
    :rtype:     :class:`GLSFit``

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    """
    M = len(x) 
    if fn is None:
        P = len(x[0])
    else:
        P = len( fn(x[0]) ) 

    if cv.shape != (M,M):
        raise RuntimeError( f"{cv.shape} should be {({M},{M})}" )     
    
    K = np.linalg.cholesky(cv)
    Kinv = np.linalg.inv(K)

    # We may have uncertain-number input data
    # So create X and Y using the values 
    def row_values(row_i): 
        return [ value(x_i) for x_i in row_i ]
        
    if fn is None:
        X = np.array( [ row_values(row_i) for row_i in x ],
            dtype=_dtype_float(x) 
        )
    else:
        # fn expands the P basis function values at x_i
        X = np.array( [ row_values(fn(x_i)) for x_i in x ],
            dtype=_dtype_float(x)
        )  
        
    Y = np.array( [ value(y_i) for y_i in y ], 
        dtype=_dtype_float(x) 
    ).T   
    
    # The GLS is solved by transforming the input data 
    a = Kinv @ X 
    b = Kinv @ Y  
   
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
    
    TOL = 1E-5
    thresh = TOL*wmax 
    w = np.array([ 
        w_i if w_i >= thresh else 0. 
            for w_i in w 
    ]) 
   
    # coef = svbksb(u,w,v,b)
    
    w_inv = np.diag(1 / w)
    coef = v @ w_inv @ u.T @ b
    
    # cv_coef = svdvar(v,w) 
    wti = [
        1.0/(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    cv_coef = v @ np.diag(wti) @ vh

    ssr = 0.0 
    for i in range(M):
        afunc_i = X[i] if fn is None else fn(X[i]) 
        s = math.fsum(
                coef[j]*afunc_i[j]
                    for j in range(P)
            )
        tmp = value( Y[i] - s )
        ssr += tmp*tmp 
        
    df = inf   
    coef = _coef_as_uncertain_numbers(coef,cv_coef,df,label=label)    

    return GLSFit( coef,ssr,M,fn )
# #----------------------------------------------------------------------------
# def _gls(x,y,cv,fn,fn_inv=None,label=None):
    # """Generalised least squares fit of ``y`` to ``x``
    
    # :arg x: a sequence of ``M`` stimulus values (independent-variables)
    # :arg y: a sequence of ``M`` responses (dependent-variable)  
    # :arg cv: an ``M`` by ``M`` real-valued covariance matrix for the responses
    # :arg fn: a user-defined function relating ``x`` the response
    # :arg fn_inv: a user-defined function relating the response to the stimulus 
    # :returns:   an object containing regression results
    # :rtype:     :class:`GLSFit``
    
    # """
    # M = len(x) 

    # # P - number of parameters to fit 
    # afunc_i = fn(x[0])  
    # P = len( afunc_i )   

    # if cv.shape != (M,M):
        # raise RuntimeError( f"{cv.shape} should be {({M},{M})}" )     
    
    # K = cholesky.cholesky_decomp(cv)
    # Kinv = cholesky.cholesky_inv(K)
    
    # X = np.array( x, dtype=object )
    # Y = np.array( y, dtype=object ).T
    
    # Q = np.matmul(Kinv,X) 
    # Z = np.matmul(Kinv,Y) 
   
    # x = []
    # y = []
    # for i in range(M):
        # x.append( [ Q[i,j] for j in range(P) ] )    # x is M by P 
        # y.append( Z[i] )
         
    # coef, chisq, w, v = svdfit(x,y,np.ones( (M,) ),fn)
    # coef = _coef_as_uncertain_numbers(coef,chisq,w,v,M,label=label)
    
    # return GLSFit( coef,chisq,fn,fn_inv,M )
    

#-----------------------------------------------------------------------------------------
class LSFit(object):
 
    """
    Base class for regression results
    
    .. versionadded:: 2.0
    """

    def __init__(self,beta,ssr,N,fn):
        self._beta = beta
        self._ssr = ssr
        self._N = N
        self._fn = fn
        self._P = len(beta)
        
    def __repr__(self):
        return f"""{self.__class__.__name__}(
  beta={self._beta!r},
  ssr={self._ssr},
  N={self._N:G},
  P={self._P}
)"""

    def __str__(self):
        return f'''
  Number of observations: {self._N}
  Number of parameters: {self._P}
  Parameters: {self._beta!r}
  Sum of the squared residuals: {self._ssr:G}
'''

    # #------------------------------------------------------------------------
    # def y_from_x(self,x,s_label=None,y_label=None):
        # """
        # Return an uncertain number ``y`` that predicts the response to ``x``
        
        # :arg x: a real number array, or an uncertain real number array
        # :arg s_label: a label for an elementary uncertain number associated with observation variability  
        # :arg y_label: a label for the return uncertain number `y` 

        # This is a prediction of a single future response ``y`` to a stimulus ``x``
        
        # The variability in observations is based on residuals obtained during regression.
        
        # If an uncertain real number is used for ``x``, 
        # the uncertainty associated with ``x`` will be propagated into ``y``.

        # .. note::
            # When ``y_label`` is defined, the uncertain number returned will be 
            # declared an intermediate result (using :func:`~.result`)
        
        # """    
        # # The elements of `beta` form an ensemble of uncertain numbers
        # df = self.beta[0].df
        # noise = _ureal(
            # 0,
            # math.sqrt( self._ssr/df ),
            # df,
            # label=s_label,
            # independent=False
        # )
        # append_real_ensemble(self.beta[0],noise)
        
        # # `_y` estimates the mean response to `x`.
        # _y = np.dot( self.beta,np.array( self._fn(x) ) ) + noise
        
        # if y_label is not None: _y = result(_y,y_label)
            
        # return _y

    # #------------------------------------------------------------------------
    # def x_from_y(self,yseq,x_label=None,y_label=None):
        # """Estimate the stimulus ``x`` corresponding to the responses in ``yseq``
        
        # :arg yseq: a sequence of ``y`` observations 
        # :arg x_label: a label for the return uncertain number ``x`` 
        # :arg y_label: a label for the estimate of `y` based on ``yseq`` 
         
        # .. note::
            # When ``x_label`` is defined, the uncertain number returned will be 
            # declared an intermediate result (using :func:`~.result`)

        # """
        # # The function must be supplied by the client
        # if self._fn_inv is None:
            # raise RuntimeError( "An inverse function has not been defined" )
          
        # df = self.beta[0].df    # All beta are the same 
        
        # p = len(yseq)
        # y = math.fsum( yseq ) / p

        # _y = _ureal(
            # y,
            # math.sqrt( self._ssr/df/p ),
            # df,
            # label = x_label,
            # independent = False
        # )

        # append_real_ensemble(self.beta[0],_y)

        # _x = self._fn_inv(_y,self.beta)  
        
        # if y_label is not None: _x = result(_x,y_label)
        
        # return _x 

    @property
    def ssr(self):
        """Sum of the squared residuals
        
        The sum of the squared deviations between  
        predicted values and the actual data.
        
        If weights are used during the fit, the squares of 
        weighted deviations are summed.
        
        """
        return self._ssr  

    @property
    def beta(self):
        """Fitted parameters"""
        return self._beta 

    @property
    def N(self):
        """Number of observations in the sample"""
        return self._N

    @property
    def P(self):
        """Number of parameters"""
        return self._P

#----------------------------------------------------------------------------
class OLSFit(LSFit):

    """
    Results of an ordinary least squares regression
    """

    def __init__(self,beta,ssr,N,fn):
        LSFit.__init__(self,beta,ssr,N,fn)
 
    def __str__(self):
        header = '''
Ordinary Least-Squares Results:
'''
        return header + str(LSFit)
        
#----------------------------------------------------------------------------
class WLSFit(LSFit):

    """
    Results of a weighted least squares regression
    """

    def __init__(self,beta,ssr,N,fn):
        LSFit.__init__(self,beta,ssr,N,fn)
 
    def __str__(self):
        header = '''
Weighted Least-Squares Results:
'''
        return header + str(LSFit)
 
#----------------------------------------------------------------------------
class RWLSFit(LSFit):

    """
    Results of a weighted least squares regression
    """

    def __init__(self,beta,ssr,N,fn):
        LSFit.__init__(self,beta,ssr,N,fn)
 
    def __str__(self):
        header = '''
Relative weighted Least-Squares Results:
'''
        return header + str(LSFit)
 
#----------------------------------------------------------------------------
class GLSFit(LSFit):

    """
    Results of a general least squares regression
    """

    def __init__(self,beta,ssr,N,fn):
        LSFit.__init__(self,beta,ssr,N,fn)
 
    def __str__(self):
        header = '''
General Least-Squares Results:
'''
        return header + str(LSFit)
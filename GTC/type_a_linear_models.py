"""


Module contents
---------------

"""
import math
import numpy as np

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

#----------------------------------------------------------------------------
def _dtype_float(a):
    """Promote integer arrays to float 
    
    Use this to avoid creating an array that might truncate values when 
    you do not know the dtype.
    
    """
    try:
        if np.issubdtype(a.dtype, np.integer):
            return np.float64
        else:
            return a.dtype
    except AttributeError:  
            return np.float64
            
#----------------------------------------------------------------------------
def lsfit(x,y,sig=None,fn=None):
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
    if sig is None: 
        sig = np.ones( (len(x),) )
    else:
        sig = np.array( sig )
        
    # Allow uncertain-number inputs:
    # construct arrays `a` and `b` from values 
    if fn is None:
        P = len( x[0] )   
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

    # Residuals and sum of squared residuals
    res = b - np.dot(a, coef) 
    ssr = np.dot(res.T, res)
          
    return coef, cv_coef, res, ssr 
  
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
    M = len(y)
    
    if M != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),M)
        )
            
    coef, cv, res, ssr = lsfit(x,y,fn=fn)

    df = M - len(coef)
    cv = ssr/df * cv

    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)

    return OLSModel( coef,res,ssr,M,fn )  
    
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
        
    coef, cv, res, ssr = lsfit(x,y,u_y,fn)   
    
    if dof is None:
        df = inf
    else:
        df = M - len(coef)
        cv = ssr/df * cv
        
    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)

    return WLSModel( coef,res,ssr,M,fn )  

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
        
    coef, cv, res, ssr = lsfit(x,y,s_y,fn)   
    
    df = M - len(coef) if dof is None else dof        
    cv = ssr/df * cv
        
    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)
    
    return RWLSModel( coef,res,ssr,M,fn )  

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

    res = Y - np.dot(X, coef) 
    tmp = np.dot(Kinv, res)
    ssr = np.dot(tmp.T, tmp)
    
    df = inf   
    coef = _coef_as_uncertain_numbers(coef,cv_coef,df,label=label)    

    return GLSModel( coef,res,ssr,M,fn )

#-----------------------------------------------------------------------------------------
class ModelFit(object):
 
    """
    Base class for regression results
    
    .. versionadded:: 2.0
    """

    def __init__(self,beta,res,ssr,M,fn):
        self._beta = beta
        self._res = res
        self._ssr = ssr
        self._M = M
        self._fn = fn
        self._P = len(beta)
        
    def __repr__(self):
        return f"""{self.__class__.__name__}(
  beta={self._beta},
  residuals={self._res}
  ssr={self._ssr},
  M={self._M},
  P={self._P}
)"""

    def __str__(self):
        # Format coefficients in str display mode 
        beta = "["+", ".join(f'{b_i!s}' for b_i in self._beta)+" ]"
        return f'''
  Parameters: {beta}
  Number of observations: {self._M}
  Number of parameters: {self._P}
  Sum of the squared residuals: {self._ssr}
'''

    def predict(self,x_i,label=None):
        """Predict the response to `x_i` as an uncertain number
        
        :arg x_i: a stimulus
        
        The stimulus `x_i` should match the format used for fitting. So,
        if an M x P matrix of stimuli was used in the regression, then 
        `x_i` should be a P-element sequence. Alternatively, if an M x 1
        matrix of stimuli was used then `x_i` should be a single argument.

        The calculation uses the uncertain-number coefficients obtained 
        by the fit. The argument `x_i` may also be an uncertain number,
        or sequence of uncertain numbers.
        
        """
        if self._fn is not None:
            x_i = self._fn(x_i)
    
        y_i = sum(
            self._beta[j]*x_i[j]
                for j in range(self._P)
        )
        if label is None:
            return y_i 
        else:
            return result(y_i,label=label) 

    @property
    def residuals(self):
        """An array of differences between the actual data 
        and predicted values.
        
        """
        return self._res 
        
    @property
    def ssr(self):
        """Sum of squared residuals
        
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
    def M(self):
        """Number of observations in the sample"""
        return self._M

    @property
    def P(self):
        """Number of parameters"""
        return self._P

class OLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-A Ordinary Least-Squares:
'''
        return header + ModelFit.__str__(self)
 
class WLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-A Weighted Least-Squares:
'''
        return header + ModelFit.__str__(self)
        
class RWLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-A Relative Weighted Least-Squares:
'''
        return header + ModelFit.__str__(self)
        
class GLSModel(ModelFit):
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-A Generalised Least-Squares:
'''
        return header + ModelFit.__str__(self)
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

_ureal = UncertainReal._elementary
_const = UncertainReal._constant

# We mostly use numpy routines in this module because they will 
# execute more quickly than the Numerical Recipe equivalents.
# A consequence is that arguments cannot be uncertain numbers,
# which is possible in the type_a module. Perhaps we need to
# strip off the values.
#----------------------------------------------------------------------------
def svbksb(u,w,v,b):
    """
    Solve A*X = B
    
    .. versionadded:: 1.4.x
    
    :arg u: an ``M`` by ``P`` matrix
    :arg w: an ``P`` element sequence
    :arg v: an ``P`` by ``P`` matrix
    :arg b: an ``P`` element sequence 
    
    Returns a list containing the solution ``X`` 
    
    """
    # M,P = u.shape 
    # tmp = np.zeros( (P,), dtype=float  ) 
    # for j in range(P):
        # if w[j] != 0:
            # tmp[j] = math.fsum(
                # u[i,j]*b[i] for i in range(M)
            # ) / w[j]
 
    # return np.matmul(v,tmp)

    return np.matmul( v,1.0/w*np.matmul(u.T,b) )
#------------------------------------------------
def solve(a,b,TOL=1E-5):
    """
    Solve a.x = b
    
    .. versionadded:: 1.4.x

    """
    u,w,vh = np.linalg.svd(a, full_matrices=False )
    v = vh.T    

    wmax = max(w)
    thresh = TOL*wmax 
    # wmin = min(w)
    # logC = math.log10(wmax/wmin)
    # # The base-b logarithm of C is an estimate of how many 
    # # base-b digits will be lost in solving a linear system 
    # # with the matrix. In other words, it estimates worst-case 
    # # loss of precision. 

    # # C is the condition number: the ratio of the largest to smallest 
    # # singular value in the SVD
    
    # `TOL` is used to set relatively small singular values to zero
    # Doing so avoids numerical precision problems, but will make the 
    # solution slightly less accurate. The value can be varied.
    w = np.array([ 
        w_i if w_i >= thresh else 0. 
            for w_i in w 
    ])
    
    return svbksb(u,w,v,b)

#----------------------------------------------------------------------------
def svdfit(x,y,sig,fn):
    """
    Return estimates of the best-fit parameters for ``fn`` to the data

    .. versionadded:: 1.4.x
    
    :arg x: an ``M`` element array
    :arg y: an ``M`` element array
    :arg sig: an ``M`` element array of standard deviations in ``y``
    :arg fn: user-defined function to evaluate basis functions at a stimulus value
    
    """
    # `TOL` is used to set relatively small singular values to zero
    # Doing so avoids numerical precision problems, but will make the 
    # solution slightly less accurate. The value can be varied.
    TOL = 1E-5
    
    # fn(x_i) returns an P-sized array of values for
    # each basis function at the stimulus point `x_i`
    afunc_i = fn(x[0])  
    
    # M - number of data points 
    # P - number of parameters to fit 
    M = len(x) 
    P = len( afunc_i )   

    if M <= P:
        raise RuntimeError( f"M = {M} but should be > {P}" )     
        
    a = np.empty( (M,P), dtype=float )
    b = np.empty( (M,), dtype=float )    
    
    for i in range(M):
        tmp = 1.0/sig[i]
        for j in range(P):
            a[i,j] = value( tmp*afunc_i[j] )
            
        b[i] = value( tmp*y[i] ) 
        
        if i < M-1:
            afunc_i = fn(x[i+1])

    # TODO: these two lines and the use of `value` are the only difference between this routine 
    # and the type-B one! Perhaps define an svd_decomp function reference as argument?
    # In this module, svd_decomp would wrap around these two lines
    u,w,vh = np.linalg.svd(a, full_matrices=False )
    v = vh.T    # NR routines work for V not V.T
    
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
    
    coef = svbksb(u,w,v,b)

    # Residuals -> chisq
    chisq = 0.0 
    for i in range(M):
        afunc_i = fn(x[i])
        s = math.fsum(
                coef[j]*afunc_i[j]
                    for j in range(P)
            )
        tmp = value( (y[i] - s)/sig[i] )
        chisq += tmp*tmp 
          
    # w and v are needed to evaluate parameter covariance 
    return coef, chisq, w, v
 
#----------------------------------------------------------------------------
# This function is used internally in this module, and called by some 
# unit tests in both test_type_a_SVD.py and test_type_b_SVD.py modules.
def svdvar(v,w):
    """
    Calculate the variance-covariance matrix after ``svdfit``
    
    .. versionadded:: 1.4.x
    
    :arg v: an ``P`` by ``P`` matrix of float
    :arg w: an ``P`` element sequence of float 
    
    """
    # Based on Numerical Recipes 'svdvar'
    P = len(w)  
    cv = np.empty( (P,P), dtype=float )
    
    wti = [
        1.0/(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    
    for i in range(P):
        for j in range(i+1):
            cv[i,j] = cv[j,i] = math.fsum(
                v[i,k]*v[j,k]*wti[k]
                    for k in range(P)
            )
    
    return cv  
 
#----------------------------------------------------------------------------
def coef_as_uncertain_numbers(coef,chisq,w,v,M,label=None):
    """
    Create uncertain numbers for the fitted parameters 

    """
    P = len(coef) 
    df = M - P
           
    s2 = chisq/df
    cv = s2*svdvar(v,w)
    
    beta = []
    ensemble = []
    for i in range(P):
        if label is None:
            label_i = f'b_{i}' 
        else:
            label_i = f'{label}_{i}'
            
        u_i = math.sqrt(cv[i,i])
        
        if abs(u_i) <= 1E-13:    # Negligible uncertainty 
            u_i = 0.0
            b_i = _const(
                coef[i],
                label=label_i
            )
        else:           
            b_i = _ureal(
                coef[i],
                u_i,
                df,
                label=label_i,
                independent=False
            )
            ensemble.append( b_i )
            
        beta.append( b_i )
            
    real_ensemble( ensemble, df )

    for i,b_i in enumerate(beta): 
        if b_i.u == 0.0: continue
        for j,b_j in enumerate(beta[:i]):
            if b_j.u == 0.0: continue                
            if abs(cv[i,j]) > 1E-13:
                b_i.set_correlation(cv[i,j]/(b_i.u*b_j.u),b_j)
            
    return beta
     
#----------------------------------------------------------------------------
def ols(x,y,fn,fn_inv=None,label=None):
    """Ordinary least squares fit of response data to stimulus values
    
    :arg x: sequence of stimulus values (independent-variable)
    :arg y: sequence of response data (dependent-variable)   
    :arg fn: a user-defined function relating the stimulus to the response
    :arg fn_inv: a user-defined function relating the response to the stimulus 
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
    coef, chisq, w, v = svdfit(x,y,sig,fn)
    coef = coef_as_uncertain_numbers(coef,chisq,w,v,M,label=label)

    return OLSFit( coef,chisq,fn,fn_inv,M )  
    
#----------------------------------------------------------------------------
def wls(x,y,u_y,fn,fn_inv=None,label=None):
    """Ordinary least squares fit of response data to stimulus values
    
    :arg x: sequence of stimulus values (independent-variable)
    :arg y: sequence of response data (dependent-variable)   
    :arg y: sequence of standard uncertainties for response data    
    :arg fn: a user-defined function relating the stimulus to the response
    :arg fn_inv: a user-defined function relating the response to the stimulus 
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`WLSFit``
    
    """
    M = len(y)   
    if M != len(x):
        raise RuntimeError( "len(x) != len(y)" )
    if M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
        
    coef, chisq, w, v = svdfit(x,y,u_y,fn)
    coef = coef_as_uncertain_numbers(coef,chisq,w,v,M,label=label)

    return WLSFit( coef,chisq,fn,fn_inv,M )  

#----------------------------------------------------------------------------
def gls(x,y,cv,fn,fn_inv=None,label=None):
    """Generalised least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``M`` stimulus values (independent-variables)
    :arg y: a sequence of ``M`` responses (dependent-variable)  
    :arg cv: an ``M`` by ``M`` real-valued covariance matrix for the responses
    :arg fn: a user-defined function relating ``x`` the response
    :arg fn_inv: a user-defined function relating the response to the stimulus 
    :returns:   an object containing regression results
    :rtype:     :class:`GLSFit``
    
    """
    M = len(x) 

    # P - number of parameters to fit 
    afunc_i = fn(x[0])  
    P = len( afunc_i )   

    if cv.shape != (M,M):
        raise RuntimeError( f"{cv.shape} should be {({M},{M})}" )     
    
    K = cholesky.cholesky_decomp(cv)
    Kinv = cholesky.cholesky_inv(K)
    
    X = np.array( x, dtype=object )
    Y = np.array( y, dtype=object ).T
    
    Q = np.matmul(Kinv,X) 
    Z = np.matmul(Kinv,Y) 
   
    x = []
    y = []
    for i in range(M):
        x.append( [ Q[i,j] for j in range(P) ] )    # x is M by P 
        y.append( Z[i] )
         
    coef, chisq, w, v = svdfit(x,y,np.ones( (M,) ),fn)
    coef = coef_as_uncertain_numbers(coef,chisq,w,v,M,label=label)

    return GLSFit( coef,chisq,fn,fn_inv,M )
    

#-----------------------------------------------------------------------------------------
class LSFit(object):
 
    """
    Base class for regression results
    
    .. versionadded:: 2.0
    """

    def __init__(self,beta,ssr,fn,fn_inv,N):
        self._beta = beta
        self._ssr = ssr
        self._N = N
        self._fn = fn
        self._fn_inv = fn_inv
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

    #------------------------------------------------------------------------
    def y_from_x(self,x,s_label=None,y_label=None):
        """
        Return an uncertain number ``y`` that predicts the response to ``x``
        
        :arg x: a real number array, or an uncertain real number array
        :arg s_label: a label for an elementary uncertain number associated with observation variability  
        :arg y_label: a label for the return uncertain number `y` 

        This is a prediction of a single future response ``y`` to a stimulus ``x``
        
        The variability in observations is based on residuals obtained during regression.
        
        If an uncertain real number is used for ``x``, 
        the uncertainty associated with ``x`` will be propagated into ``y``.

        .. note::
            When ``y_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)
        
        """    
        # The elements of `beta` form an ensemble of uncertain numbers
        df = self.beta[0].df
        noise = _ureal(
            0,
            math.sqrt( self._ssr/df ),
            df,
            label=s_label,
            independent=False
        )
        append_real_ensemble(self.beta[0],noise)
        
        # `_y` estimates the mean response to `x`.
        _y = np.dot( self.beta,np.array( self._fn(x) ) ) + noise
        
        if y_label is not None: _y = result(_y,y_label)
            
        return _y

    #------------------------------------------------------------------------
    def x_from_y(self,yseq,x_label=None,y_label=None):
        """Estimate the stimulus ``x`` corresponding to the responses in ``yseq``
        
        :arg yseq: a sequence of ``y`` observations 
        :arg x_label: a label for the return uncertain number ``x`` 
        :arg y_label: a label for the estimate of `y` based on ``yseq`` 
         
        .. note::
            When ``x_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        """
        # The function must be supplied by the client
        if self._fn_inv is None:
            raise RuntimeError( "An inverse function has not been defined" )
          
        df = self.beta[0].df    # All beta are the same 
        
        p = len(yseq)
        y = math.fsum( yseq ) / p

        _y = _ureal(
            y,
            math.sqrt( self._ssr/df/p ),
            df,
            label = x_label,
            independent = False
        )

        append_real_ensemble(self.beta[0],_y)

        _x = self._fn_inv(_y,self.beta)  
        
        if y_label is not None: _x = result(_x,y_label)
        
        return _x 

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

    def __init__(self,beta,ssr,fn,fn_inv,N):
        LSFit.__init__(self,beta,ssr,fn,fn_inv,N)
 
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

    def __init__(self,beta,ssr,fn,fn_inv,N):
        LSFit.__init__(self,beta,ssr,fn,fn_inv,N)
 
    def __str__(self):
        header = '''
Weighted Least-Squares Results:
'''
        return header + str(LSFit)
        
#----------------------------------------------------------------------------
class GLSFit(LSFit):

    """
    Results of a general least squares regression
    """

    def __init__(self,beta,ssr,fn,fn_inv,N):
        LSFit.__init__(self,beta,ssr,fn,fn_inv,N)
 
    def __str__(self):
        header = '''
General Least-Squares Results:
'''
        return header + str(LSFit)
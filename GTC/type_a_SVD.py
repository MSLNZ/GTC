"""


Module contents
---------------

"""
from __future__ import division

import math
import numpy as np

try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

from GTC.lib import (
    UncertainReal,
    real_ensemble,
    append_real_ensemble,
    value,
)
from GTC import magnitude, sqrt    # Polymorphic functions

_ureal = UncertainReal._elementary
_const = UncertainReal._constant

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
    Return the LS coefficients of the ``fn`` parameters 

    .. versionadded:: 1.4.x
    
    :arg x: an ``M`` element array
    :arg y: an ``M`` element array
    :arg sig: an ``M`` element array of float
    :arg fn: user-defined function to evaluate basis functions 
    
    """
    # `TOL` is used to set relatively small singular values to zero
    # Doing so avoids numerical precision problems, but will make the 
    # solution slightly less accurate. The value can be varied.
    TOL = 1E-5
    
    afunc_i = fn(x[0])  
    
    M = len(x) 
    P = len( afunc_i )  # fn() returns a sequence of length M 
     
    # M - number of data points 
    # P - number of parameters to fit 
   
    a = np.empty( (M,P), dtype=float )
    b = np.empty( (M,), dtype=float )    
    
    for i in range(M):
        tmp = 1.0/sig[i]
        for j in range(P):
            a[i,j] = value( tmp*afunc_i[j] )
            
        b[i] = value( tmp*y[i] ) 
        
        if i < M-1:
            afunc_i = fn(x[i+1])
    
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
def svdvar(v,w):
    """
    Calculate the variance-covariance matrix after ``svdfit``
    
    .. versionadded:: 1.4.x
    
    :arg v: an ``P`` by ``P`` matrix of float
    :arg w: an ``P`` element sequence of float 
    
    """
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
def coef_as_uncertain_numbers(coef,chisq,w,v,sig,label=None):
    """
    Create uncertain numbers for the fitted parameters 

    """
    M = len(sig)
    P = len(coef) 
    df = M - P
           
    s2 = chisq/df
    cv = s2*svdvar(v,w)
    
    beta = []
    ensemble = set()
    for i in range(P):
        if label is None:
            label_i = 'b_{}'.format(i)   
        else:
            label_i = '{}_{}'.format(label,i)
            
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
            ensemble.add( b_i )
            
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
def ols(x,y,fn,P,label=None):
    """Ordinary least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``M`` stimulus values (independent-variables)
    :arg y: a sequence of ``M`` responses (dependent-variable)  
    :arg fn: a user-defined function relating ``x`` the response
    :arg P: the number of parameters to be fitted 
    :arg label: a label for the fitted parameters
    
    Return a :class:`OLSFit` object containing the results 
    
    """
    M = len(y)
    
    if M != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),M)
        )
        
    if M <= P:
        raise RuntimeError(
            "M = {} but should be > {}".format(M,P)
        )     
    
    sig = np.ones( (M,) )
    coef, chisq, w, v = svdfit(x,y,sig,fn)
    coef = coef_as_uncertain_numbers(coef,chisq,w,v,sig,label=label)

    return OLSFit( coef,chisq,M,P )  

#-----------------------------------------------------------------------------------------
class LSFit(object):
 
    """
    Base class for regression results
    """

    def __init__(self,beta,ssr,N,P):
        self._beta = beta
        self._ssr = ssr
        self._N = N
        self._P = P
        
    def __repr__(self):
        return """{}(
  beta={!r},
  ssr={},
  N={},
  P={}
)""".format(
            self.__class__.__name__,
            self._beta,
            self._ssr,
            self._N,
            self._P
        )

    def __str__(self):
        return '''
  Number of points: {}
  Number of parameters: {}
  Parameters: {!r}
  Sum of the squared residuals: %G
'''.format(
    self._N,
    self._P,
    self._beta,
    self._ssr,
)

    # #------------------------------------------------------------------------
    # def mean_y_from_x(self,x,label=None):
    # """
    # Return the uncertain number ``y`` that is the response to ``x`` 
    
    # :arg x: a sequence of real numbers 
    # :arg label: a label for the uncertain number 
     
    # Returns an estimate of the mean response that would be  
    # observed from the stimulus vector ``x``. 
    
    # """
    # cxt = default.context 
    # ureal = cxt.elementary_real
    
    # df = self.N - self.P 
    # assert df >= 1, "Too few observations in the sample"
    
    # # The elements of `beta` are uncertain numbers
    # # This `y0` represents the mean response to `x` 
    # # with associated uncertainty.
    # y = sum(
        # x_i*b_i for x_i, b_i in itertools.izip(x,self.beta)
    # )
    
    # if label is not None: y.label = label

    # return y 
    
    @property
    def ssr(self):
        """Sum of the squared residuals
        
        The sum of the squared deviations between values 
        predicted by the model and the actual data.
        
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

    def __init__(self,beta,ssr,N,P):
        LSFit.__init__(self,beta,ssr,N,P)
 
    def __str__(self):
        header = '''
Ordinary Least-Squares Results:
'''
        return header + str(LSFit)

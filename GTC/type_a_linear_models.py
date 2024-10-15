"""

Module contents
---------------

.. versionadded:: 2.0

"""
import math
import numpy as np

from GTC import inf

from GTC.lib import (
    UncertainReal,
    real_ensemble,
    value,
)

_ureal = UncertainReal._elementary
_const = UncertainReal._constant

__all__ = (
    'ols',
    'rwls',
    'wls',
    'gls',
    'ModelFit', 'OLSModel', 'WLSModel', 'GLSModel', 'RWLSModel'
)
#----------------------------------------------------------------------------
def lsfit(x,y,sig=None,fn=None):
    """
    Solve `x @ beta = y` for `beta` 
    
    :arg x: an array of stimulus data
    :arg y: a 1-D array of response data
    :arg sig: a 1-D array of standard deviations associated with response data
    :arg fn: returns a sequence of basis function values at a stimulus value
    :returns: `coef`, `cv_coef`, `res`, and `ssr`
 
    A Singular Value Decomposition (SVD) algorithm is used to obtain the solution.
    
    The returned `coef` is a real-valued sequence of parameter estimates for `beta`.
    
    The returned `cv_coef` is a real-valued covariance matrix for the parameter estimates.
    
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
            dtype=np.float64 
        )
        b = np.array( [ value(y_i/s_i) for y_i,s_i in zip(y,sig) ], 
            dtype=np.float64 
        )   
    else:       
        # fn(x_i) returns an P-sized array of values for
        # each basis function at the stimulus point `x_i`
        P = len( fn(x[0])  )   

        a = np.empty( (M,P), dtype=np.float64 )
        b = np.empty( (M,), dtype=np.float64 )    
        
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
    
    # `TOL` is used to adjust the small singular values set to zero
    # This avoids numerical precision problems, but will make the 
    # solution slightly less accurate. 
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
    
    coef = v @ np.diag(1 / w) @ u.T @ b
    
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
    Return a sequence of uncertain numbers for parameter estimates 

    When the degrees of freedom are finite, the uncertain-number parameters  
    form an ensemble, allowing effective degrees of freedom calculations
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
                r_ij = cv[i,j]/(b_i.u*b_j.u)
                b_i.set_correlation(r_ij,b_j)
            
    return beta
     
#----------------------------------------------------------------------------
def ols(x,y,fn=None,label='beta'):
    """Ordinary least squares fit of response data to stimulus values
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg fn: returns a sequence of basis function values at a stimulus value
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`OLSModel`

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.

    **Example**::
    
        >>> x = [4.,8.,12.5,16.,20.,25.,31.,36.,40.,40.]
        >>> y = [3.7,7.8,12.1,15.6,19.8,24.5,30.7,35.5,39.4,39.5]
        >>> def fn(x_i): return [x_i,1]
        >>> b, a = lma.ols(x,y,fn).beta
        >>> print("Slope:",b)
        Slope:  0.9928(26)
        >>> print("Intercept:",a)
        Intercept: -0.222(70)
        
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
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg u_y: sequence of standard uncertainties for response data    
    :arg fn: returns a sequence of basis function values at a stimulus value
    :arg dof: degrees of freedom    
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`WLSModel`

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    By default the degrees of freedom in the parameter estimates are set 
    to infinity, because the standard uncertainties in `u_y` are known.
    The optional argument `dof` may be used to assign finite degrees of freedom.
    
    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.

    **Example**::

        >>> x = [1,2,3,4,5,6]
        >>> y = [3.3,5.6,7.1,9.3,10.7,12.1] 
        >>> u_y = [ 0.5 ]*len(x)       
        >>> b, a = lma.wls(x,y,u_y,lambda x_i: [x_i, 1] ).beta  
        >>> print("Slope:",b)
        Slope:  1.76(12)
        >>> print("Intercept:",a)
        Intercept: 1.87(47)

    """
    M = len(y)   
    if M != len(x):
        raise RuntimeError( "len(x) != len(y)" )
    if M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
        
    coef, cv, res, ssr = lsfit(x,y,u_y,fn)   
    
    df = inf if dof is None else dof
        
    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)

    return WLSModel( coef,res,ssr,M,fn )  

#----------------------------------------------------------------------------
def rwls(x,y,s_y,fn=None,dof=None,label='beta'):
    """Relative least squares fit of response data to stimulus values
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg s_y: sequence of relative scale factors for response data    
    :arg dof: degrees of freedom    
    :arg fn: evaluates a sequence of basis function values at the stimulus
    :arg label: suffix to label the fitted parameters
    :returns:   an object containing regression results
    :rtype:     :class:`RWLSModel`

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    By default the degrees of freedom in the parameter estimates is set 
    to the number of response data points minus 2. However, the optional 
    argument `dof` may be used to assign a different number of degrees of freedom.
    
    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.

    **Example**::

        >>> x = [1,2,3,4,5,6]
        >>> y = [3.014,5.225,7.004,9.061,11.201,12.762] 
        >>> u_y = [0.2,0.2,0.2,0.4,0.4,0.4]       
        >>> b,a = lma.rwls(x,y,u_y,lambda x_i: [x_i, 1] ).beta   
        >>> b
        ureal(1.972639...,0.0411776...,4, label='beta_0')
        >>> a
        ureal(1.137537...,0.1226144...,4, label='beta_1')

    """
    M = len(y)   
    if M != len(x):
        raise RuntimeError( "len(x) != len(y)" )
    if M != len(s_y):
        raise RuntimeError( "len(x) != len(s_y)")
        
    coef, cv, res, ssr = lsfit(x,y,s_y,fn)   
    
    cv = ssr/(M - len(coef)) * cv
    
    # It is possible to assign different degrees of freedom
    df = M - len(coef) if dof is None else dof        
        
    coef = _coef_as_uncertain_numbers(coef,cv,df,label=label)
    
    return RWLSModel( coef,res,ssr,M,fn )  

#----------------------------------------------------------------------------
def gls(x,y,cv,fn=None,dof=None,label='beta'):
    """Generalised least squares fit of ``y`` to ``x``
    
    :arg x: an array of stimulus data (independent-variable)
    :arg y: a 1-D array of response data (dependent-variable)
    :arg cv: an ``M`` by ``M`` real-valued covariance matrix for response data
    :arg fn: evaluates a sequence of basis function values at the stimulus
    :returns:   an object containing regression results
    :rtype:     :class:`GLSModel`

    If the arguments `x` and `y` are arrays of uncertain numbers, only the
    values are used in calculations.

    By default the degrees of freedom in the parameter estimates is set 
    to infinity, because the covariance matrix values are known.
    The optional argument `dof` may be used to assign finite degrees of freedom.
    
    When `fn` is provided, the routine will apply `fn` to each element 
    in `x`, which must be a 1-D array, to obtain the rows of the objective matrix. 
    When `fn` is `None`, `x` is taken as the objective matrix.

    **Example**::
    
        >>> import numpy
        >>> x = [ [x_i,1] for x_i in range(1,11) ]
        >>> y = [1.3, 4.1, 6.9, 7.5, 10.2, 12.0, 14.5, 17.1, 19.5, 21.0]
        >>> cv = numpy.diag([2, 2, 2, 2, 2, 5, 5, 5, 5, 5])
        >>> for i in range(5):
        ...     for j in range(i+1,5):
        ...         cv[i][j] = cv[j][i] = 1
        ...
        >>> for i in range(5,10):
        ...    for j in range(i+1,10):
        ...     cv[i][j] = cv[j][i] = 4
        ...
        >>> fit = lma.gls(x,y,cv)
        >>> print(fit)
        <BLANKLINE>
        Type-A Generalised Least-Squares:
        <BLANKLINE>
          Parameters: [ 2.20(20), -0.6(1.3) ]
          Number of observations: 10
          Number of parameters: 2
          Sum of the squared residuals: 2.07395...
        <BLANKLINE>
        
    """
    M = len(x) 
    if fn is None:
        P = len(x[0])
    else:
        P = len( fn(x[0]) ) 

    if cv.shape != (M,M):
        raise RuntimeError( f"{cv.shape} should be {({M},{M})}" )     
    
    # We may have uncertain-number input data
    # So create X and Y using the values 
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
        
    Y = np.array( [ value(y_i) for y_i in y ], 
        dtype=np.float64 
    ).T   
 
    # Transform the input data 
    K = np.linalg.cholesky(cv)
    Kinv = np.linalg.inv(K)

    a = Kinv @ X 
    b = Kinv @ Y  

    u,w,vh = np.linalg.svd(a, full_matrices=False )
    v = vh.T    

    # Identify almost singular values
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
   
    coef = v @ np.diag(1 / w) @ u.T @ b
    
    wti = [
        1.0/(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    cv_coef = v @ np.diag(wti) @ vh

    res = Y - np.dot(X, coef) 

    tmp = np.dot(Kinv, res)
    ssr = np.dot(tmp.T, tmp)
    
    df = inf if dof is None else dof
    coef = _coef_as_uncertain_numbers(coef,cv_coef,df=inf,label=label)    

    return GLSModel( coef,res,ssr,M,fn )
    
#-----------------------------------------------------------------------------------------
class ModelFit(object):
 
    """
    Base class for regression results
    
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
        """Predict the uncertain-number response to `x_i` 
        
        :arg x_i: a stimulus
        
        The stimulus `x_i` must be in the same format as one row of `x`, 
        the argument supplied for fitting. So, if an M x P matrix of stimuli 
        was used in the regression, then `x_i` should be a P-element sequence. 
        Alternatively, if an M x 1 matrix of stimuli was used, then `x_i` 
        should be a single argument.

        The calculation uses the uncertain-number coefficients from 
        the fit. The stimulus supplied for `x_i` may be pure numbers
        or uncertain numbers. If uncertain numbers are used, the predicted
        response will include uncertainty associated with `x_i`.
        
        """
        from GTC import result 
        
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
    """
    Type-A Ordinary Least-Squares
    """
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-A Ordinary Least-Squares:
'''
        return header + ModelFit.__str__(self)
 
class WLSModel(ModelFit):
    """
    Type-A Weighted Least-Squares
    """
    def __init__(self,beta,res,ssr,M,fn):
        ModelFit.__init__(self,beta,res,ssr,M,fn)
 
    def __str__(self):
        header = '''
Type-A Weighted Least-Squares:
'''
        return header + ModelFit.__str__(self)
        
class RWLSModel(ModelFit):
    """
    Type-A Relative Weighted Least-Squares
    """
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
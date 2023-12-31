"""
Still TODO:

    There is more to do to make the results of regression useful
    (NB: Look at Draper & Smith Ch 2 for guidance + printed notes from wikipedia on GLS )
        
        Given some vector x_0, we can calculate the value of y_0 using the fitted parameters. 
        (Need to implement)
            What is the uncertainty in y_0? 
            The variance calculations seem quite similar to a Mahalanobis distance, except that the CV is not inverted. 
            So, does it make sense to store the Cholesky decomp of the CV to speed up the calculation? 
            What about weighted cases? 
            It is not possible to go from y_0 to the vector x_0, of course!
            
    For type-A analysis, use the numpy svd routine. The routine here is needed for type-B work.
    Testing this routine with real-valued data is a useful check that it is correctly implemented.
    
    It should be possible to test this routine on uncertain-number datasets for linear fitting problems.
   
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
    complex_ensemble,
    append_real_ensemble,
    value,
)
from GTC import cholesky 
from GTC import magnitude, sqrt    # Polymorphic functions

_ureal = UncertainReal._elementary

# __all__ = (
    # 'ols',
    # 'wls',
    # 'gls',
    # 'OLSFit',
    # 'WLSFit',
    # 'GLSFit',
# )

#----------------------------------------------------------------------------
def _pythag(a,b):
    """
    Return sqrt(a*a + b*b) 
    
    Avoids numerical problems
    
    """
    ab_a = magnitude(a)
    ab_b = magnitude(b)
    if ab_a > ab_b:
        return ab_a*sqrt( 1.0 + (ab_b/ab_a)**2 )
    elif ab_b == 0.0:
        return 0.0  
    else:
        return ab_b*sqrt( 1.0 + (ab_a/ab_b)**2 )
                        
#----------------------------------------------------------------------------
def svd_decomp(a):
    """
    Find the singular value decomposition of ``a`` 
    
    .. versionadded:: 1.4.x
    
    :arg a: an ``m`` by ``n`` matrix
    
    The return value is a triplet ``U, w, V`` where
    ``U`` is  an ``m`` by ``n`` matrix 
    ``w`` is a sequence containing the ``n`` elements of a diagonal matrix ``W`` 
    ``V`` is an ``n`` by ``n`` matrix
    
    The decomposition of ``a = U * W * V.T``
        
    """
    u = a.copy()    # a copy avoids side effects
    M,N = u.shape
    
    w = np.empty( (N,), dtype=a.dtype ) 
    v = np.empty( (N,N), dtype=a.dtype )
    rv1 = np.empty( (N,), dtype=a.dtype )
    
    g = 0.0                 # May be uncertain    
    scale = anorm = 0.0     # floats
    
    for i in range(N):
        
        l = i + 1
        rv1[i] = scale * g 
        
        g = s = scale = 0.0        
        if i < M:
            scale += sum(
                abs( u[k,i] ) for k in range(i,M)
            )
            if scale != 0.0:
                for k in range(i,M):
                    u[k,i] /= scale 
                    s += u[k,i]*u[k,i]
                    
                f = u[i,i]
                if f<0:
                    g = sqrt(s)
                else:
                    g = -sqrt(s)
                    
                h = f*g - s 
                u[i,i] = f - g 
                
                for j in range(l,N):
                    s = sum( 
                        u[k,i]*u[k,j] for k in range(i,M) 
                    )
                    f = s/h 
                    for k in range(i,M):
                        u[k,j] += f*u[k,i]  
                        
                for k in range(i,M):
                    u[k,i] *= scale 
                    
        w[i] = scale*g
        
        g = s = scale = 0.0     
        if i < M and i != N - 1:
            scale += sum(
                abs( u[i,k] ) for k in range(l,N)
            )
            if scale != 0.0:
                for k in range(l,N):
                    u[i,k] /= scale 
                    s += u[i,k]*u[i,k]
                    
                f = u[i,l]
                
                if f<0:
                    g = sqrt(s)
                else:
                    g = -sqrt(s)
                h = f*g - s 
                u[i,l] = f - g 
                
                for k in range(l,N):
                    rv1[k] = u[i,k]/h 
                    
                for j in range(l,M):
                    s = sum(
                        u[j,k]*u[i,k] for k in range(l,N)
                    )
                    for k in range(l,N):
                        u[j,k] += s*rv1[k] 
                    
                for k in range(l,N):
                    u[i,k] *= scale 
            
        # ASSUME `anorm` is real-valued (`abs` uses value only)
        temp = abs(w[i]) + abs(rv1[i])  
        if temp > anorm:   
            anorm = temp
        
    for i in range(N-1,-1,-1):
        
        if i < N-1:
            if g != 0.0:
                for j in range(l,N):
                    v[j,i] = ( u[i,j]/u[i,l] )/g 

                for j in range(l,N):
                    s = sum(
                        u[i,k]*v[k,j] for k in range(l,N)
                    )
                    for k in range(l,N):
                        v[k,j] += s*v[k,i]
                        
            for j in range(l,N):
                v[i,j] = v[j,i] = 0.0 
                
        v[i,i] = 1.0 
        g = rv1[i] 
        l = i
    
    for i in range( min(M,N)-1, -1, -1 ):
        l = i + 1 
        g = w[i] 
        for j in range(l,N): 
            u[i,j] = 0.0 
            
        if g != 0.0:
            g = 1.0/g 
            for j in range(l,N):
                s = sum(
                    u[k,i]*u[k,j] for k in range(l,M)
                )
                f = (s/u[i,i])*g
                for k in range(i,M):
                    u[k,j] += f*u[k,i] 
                    
            for j in range(i,M): 
                u[j,i] *= g 
        else: 
            for j in range(i,M): 
                u[j,i] = 0.0 
                
        u[i,i] += 1 
      
    for k in range(N-1,-1,-1):
        for its in range(30):
            flag = True
            for l in range(k,-1,-1):
                nm = l - 1
                
                temp = abs(rv1[l]) + anorm
                if temp == anorm:
                    flag = False
                    break 
                    
                temp = abs(w[nm]) + anorm
                if temp == anorm: 
                    break 
                
            if flag is True:
                c = 0.0 
                s = 1.0 
                for i in range(l,k+1):
                    f = s*rv1[i]
                    rv1[i] *= c 
                    
                    temp = abs(f) + anorm
                    if temp == anorm: 
                        break
                        
                    g = w[i] 
                    h = _pythag(f,g)
                    w[i] = h 

                    h = 1.0/h 
                    c = g*h 
                    s = -f*h 
                    for j in range(M):
                        y = u[j,nm] 
                        z = u[j,i] 
                        u[j,nm] = y*c + z*s 
                        u[j,i] = z*c - y*s
                        
            z = w[k]
            
            if l == k:  # Convergence
                if z < 0.0:
                    w[k] = -z 
                    for j in range(N):
                        v[j,k] = -v[j,k]                        
                break
            
            if its == 29:
                raise RuntimeError(
                    "No convergence after 30 SVD iterations"
                )
                
            x = w[l] 
            nm = k - 1
            y = w[nm] 
            g = rv1[nm]
            h = rv1[k] 
            
            f = ((y - z)*(y + z) + (g - h)*(g + h))/(2.0*h*y)
            g = _pythag(f,1.0)
            
            temp = f + (g if f>=0 else -g) 
            f = ( (x - z)*(x + z) + h*(y/temp - h) )/x
            
            c = s = 1.0 
            for j in range(l,nm+1):
                i = j + 1
                g = rv1[i] 
                y = w[i] 
                h = s*g 
                g *= c 

                z = _pythag(f,h)
                rv1[j] = z 

                c = f/z 
                s = h/z 
                f = x*c + g*s 
                g = g*c - x*s 
                h = y*s 
                y *= c 

                for jj in range(N):
                    x = v[jj,j]
                    z = v[jj,i] 
                    v[jj,j] = x*c + z*s 
                    v[jj,i] = z*c - x*s 
                                        
                z = _pythag(f,h)      
                w[j] = z 
                
                if z != 0.0:
                    z = 1.0/z 
                    c = f*z 
                    s = h*z 
                    
                f = c*g + s*y 
                x = c*y - s*g 
                
                for jj in range(M):
                    y = u[jj,j]
                    z = u[jj,i]
                    u[jj,j] = y*c + z*s 
                    u[jj,i] = z*c - y*s   
  
            rv1[l] = 0.0
            rv1[k] = f
            w[k] = x 
            
    return u,w,v 
   
#----------------------------------------------------------------------------
def svbksb(u,w,v,b):
    """
    Solve A*X = B
    
    .. versionadded:: 1.4.x
    
    :arg u: an ``m`` by ``n`` matrix
    :arg w: an ``n`` element sequence
    :arg v: an ``n`` by ``n`` matrix
    :arg b: an ``m`` element sequence 
    
    Returns a list containing the solution ``X`` 
    
    """
    M,N = u.shape 
    tmp = np.empty( (N,1), dtype=u.dtype  ) 

    for j in range(N):
        if w[j] != 0:
            s = sum(
                u[i,j]*b[i] for i in range(M)
            ) / w[j]
        else:
            s = 0
            
        tmp[j,0] = s 
       
    return list( np.matmul(v,tmp).flat ) 
 
#----------------------------------------------------------------------------
def svdfit(x,y,sig,fn):
    """
    Return the LS coefficients of the ``fn`` parameters 

    .. versionadded:: 1.4.x
    
    :arg x: an ``N`` element array
    :arg y: an ``N`` element array
    :arg sig: an ``N`` element array of float
    :arg fn: user-defined function to evaluate basis functions 
    
    """
    # `TOL` is used to set relatively small singular values to zero
    # Doing so avoids numerical precision problems, but will make the 
    # solution slightly less accurate. The value can be varied.
    TOL = 1E-5
    
    i = 0
    afunc_i = fn(x[i])
    
    # N - number of data points 
    # M - number of parameters to fit 
    
    N = len(x) 
    M = len( afunc_i )
    
    # TODO: need to determine dtypes
    a = np.empty( (N,M) )
    b = np.zeros( (N,) )    
    
    for i in range(N):
        tmp = 1.0/sig[i]
        for j in range(M):
            a[i,j] = tmp*afunc_i[j]
            
        b[i] = tmp*y[i] 
        
        if i < N-1:
            afunc_i = fn(x[i+1])
             
    u,w,v = svd_decomp(a)
    
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
    w = [ 
        w_i if w_i >= thresh else 0. 
            for w_i in w 
    ]
    
    a = svbksb(u,w,v,b)
    
    # TODO: this is a value calculation
    chisq = 0.0 
    for i in range(N):
        afunc = fn(x[i])
        
        s = math.fsum(
            value( a[j]*afunc[j] )
                for j in range(M)
        )
        
        tmp = ( value(y[i]) - s)/sig[i]
        chisq += tmp*tmp 
          
    # w and v are needed to evaluate parameter covariance 
    return a, chisq, w, v
   
#----------------------------------------------------------------------------
def svdvar(v,w):
    """
    Calculate the variance-covariance matrix after ``svdfit``
    
    .. versionadded:: 1.4.x
    
    :arg v: an ``N`` by ``N`` matrix of float
    :arg w: an ``N`` element sequence of float 
    
    """
    N = len(w)  
    cv = np.empty( (N,N), dtype=float )
    
    wti = [
        1.0/(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    
    for i in range(N):
        for j in range(i+1):
            cv[i,j] = cv[j,i] = math.fsum(
                v[i,k]*v[j,k]*wti[k]
                    for k in range(N)
            )
    
    return cv  

#----------------------------------------------------------------------------
def wls(x,y,u_y,fn,P,label=None):    
    """Weighted least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``N`` stimulus values (independent-variables)
    :arg y: a sequence of ``N`` responses (dependent-variable)  
    :arg u_y: a sequence of ``N`` standard uncertainties in the responses
    :arg fn: a user-defined function relating ``x`` the response
    :arg P: the number of parameters to be fitted 
    :arg label: a label for the fitted parameters
    
    Return a :class:`WLSFit` object containing the results 
    
    """
    N = len(y)
    
    if N != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),N)
        )
        
    if N <= P:
        raise RuntimeError(
            "N {} should be > P {}".format(N,P)
        )     
        
    return WLSFit( _ls(x,y,u_y,fn,P,label=label) )

    
#----------------------------------------------------------------------------
def ols(x,y,fn,P,label=None):
    """Ordinary least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``N`` stimulus values (independent-variables)
    :arg y: a sequence of ``N`` responses (dependent-variable)  
    :arg fn: a user-defined function relating ``x`` the response
    :arg P: the number of parameters to be fitted 
    :arg label: a label for the fitted parameters
    
    Return a :class:`OLSFit` object containing the results 
    
    """
    N = len(y)
    
    if N != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),N)
        )
        
    if N <= P:
        raise RuntimeError(
            "N {} should be > P {}".format(N,P)
        )     
    
    sig = N*[1]

    return OLSFit( _ls(x,y,sig,fn,P,label=label) )
    
#----------------------------------------------------------------------------
def gls(x,y,cv,fn,P,label=None):
    """Generalised least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``N`` stimulus values (independent-variables)
    :arg y: a sequence of ``N`` responses (dependent-variable)  
    :arg cv: an ``N`` by ``N`` covariance matrix for the responses
    :arg fn: a user-defined function relating ``x`` the response
    :arg P: the number of parameters to be fitted 
    :arg label: a label for the fitted parameters
    
    Return a :class:`GLSFit` object containing the results 
    
    """
    N = len(y)
    
    if N != len(x):
        raise RuntimeError(
            "len(x) {} != len(y) {}".format(len(x),N)
        )
        
    if N <= P:
        raise RuntimeError(
            "N {} should be > P {}".format(N,P)
        )     

    if cv.shape != (N,N):
        raise RuntimeError(
            "cv.shape {0:!s} should be {({1},{1})}".format(cv.shape,N)
        )     
    
    K = cholesky.cholesky_decomp(cv)
    Kinv = cholesky.cholesky_inv(K)
    
    X = np.array( x )
    Y = np.array( y ).T
    
    Q = np.matmul(Kinv,X) 
    Z = np.matmul(Kinv,Y) 

    # X is N by M 
    x = []
    y = []
    for i in range(N):
        x.append( [ Q[i,j] for j in range(M) ] )
        y.append( Z[i,0] )
         
    sig = N*[1]

    return GLSFit( _ls(x,y,sig,fn,P,label=label) )
        
#----------------------------------------------------------------------------
def _ls(x,y,sig,fn,P,label=None):
    """
    
    """
    b, chisq, w, v = svdfit(x,y,sig,fn,P)
    
    df = N - P
    
    s2 = chisq/df
    cv = s2*svd.svdvar(v,w)
    
    u = []
    beta = []
    for i in range(P):
        u.append( math.sqrt(cv[i,i]) )
        
        if label is None:
            label_i = 'b_{}'.format(i)   
        else:
            label_i = '{}_{}'.format(label,i)
            
        b_i = _ureal(
            b[i],
            u[i],
            df,
            label=label_i,
            independent=False
        )        
            
        beta.append(b_i)
      
    real_ensemble( beta, df )

    for i in range(P):
        for j in range(i):
            den = u[i]*u[j]
            assert abs(den) > 1E-13, "unexpected: {!r}".format(den) 
            r = cv[i,j]/den
            if r != 0:
                beta[i].set_correlation(r,beta[j])
            
    return beta,chisq,N,P
    
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
 
#----------------------------------------------------------------------------
class WLSFit(LSFit):

    """
    Results of a weighted least squares regression
    """

    def __init__(self,beta,ssr,N,P):
        LSFit.__init__(self,beta,ssr,N,P)

    def __str__(self):
        header = '''
Weighted Least-Squares Results:
'''
        return header + str(LSFit)
 
    
#----------------------------------------------------------------------------
class GLSFit(LSFit):

    """
    Results of a generalised least squares regression
    """

    def __init__(self,beta,ssr,N,P):
        LSFit.__init__(self,beta,ssr,N,P)
  
    def __str__(self):
        header = '''
Generalised Least-Squares Results:
'''
        return header + str(LSFit)
  
#============================================================================
if __name__ == '__main__': 

    from GTC import *
    
    # import doctest    
    # doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

    A = np.array(
        [[ureal(1,.2), ureal(.5,.4)],
        [ureal(.2,.3), ureal(-1,.5)]]
    )
    # A = np.array(
        # [1, .5,.2, -1]
    # )
    A.shape = 2,2
    print(A)
    U,w,V = svd_decomp(A)
    print()
    # print(U)
    # print(V)
    print( np.matmul(A,np.matmul(V,np.matmul(np.diag(1.0/w),U.T))) )
    
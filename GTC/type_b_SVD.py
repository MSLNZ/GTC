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
               
"""
import math
import numpy as np

from GTC.lib import value
from GTC.type_b import mean

from GTC import cholesky 
from GTC import magnitude, sqrt    # Polymorphic functions

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
    
    :arg a: an ``M`` by ``P`` matrix
    
    The return value is a triplet ``U, w, V`` where
    ``U`` is  an ``M`` by ``P`` matrix 
    ``w`` is a sequence containing the ``P`` elements of a diagonal matrix ``W`` 
    ``V`` is an ``P`` by ``P`` matrix
    
    The decomposition of ``a = U * W * V.T``
        
    """
    u = a.copy()    # a copy avoids side effects
    M,P = u.shape
    
    w = np.empty( (P,), dtype=object ) 
    v = np.empty( (P,P), dtype=object )
    rv1 = np.empty( (P,), dtype=object )
    
    g = 0.0                 # May be uncertain    
    scale = anorm = 0.0     # floats
    
    for i in range(P):
        
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
                
                for j in range(l,P):
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
        if i < M and i != P - 1:
            scale += sum(
                abs( u[i,k] ) for k in range(l,P)
            )
            if scale != 0.0:
                for k in range(l,P):
                    u[i,k] /= scale 
                    s += u[i,k]*u[i,k]
                    
                f = u[i,l]
                
                if f<0:
                    g = sqrt(s)
                else:
                    g = -sqrt(s)
                h = f*g - s 
                u[i,l] = f - g 
                
                for k in range(l,P):
                    rv1[k] = u[i,k]/h 
                    
                for j in range(l,M):
                    s = sum(
                        u[j,k]*u[i,k] for k in range(l,P)
                    )
                    for k in range(l,P):
                        u[j,k] += s*rv1[k] 
                    
                for k in range(l,P):
                    u[i,k] *= scale 
            
        # ASSUME `anorm` is real-valued (`abs` uses value only)
        temp = abs(w[i]) + abs(rv1[i])  
        if temp > anorm:   
            anorm = temp
        
    for i in range(P-1,-1,-1):
        
        if i < P-1:
            if g != 0.0:
                for j in range(l,P):
                    v[j,i] = ( u[i,j]/u[i,l] )/g 

                for j in range(l,P):
                    s = sum(
                        u[i,k]*v[k,j] for k in range(l,P)
                    )
                    for k in range(l,P):
                        v[k,j] += s*v[k,i]
                        
            for j in range(l,P):
                v[i,j] = v[j,i] = 0.0 
                
        v[i,i] = 1.0 
        g = rv1[i] 
        l = i
    
    for i in range( min(M,P)-1, -1, -1 ):
        l = i + 1 
        g = w[i] 
        for j in range(l,P): 
            u[i,j] = 0.0 
            
        if g != 0.0:
            g = 1.0/g 
            for j in range(l,P):
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
      
    for k in range(P-1,-1,-1):
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
                    for j in range(P):
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

                for jj in range(P):
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
    
    :arg u: an ``M`` by ``P`` matrix
    :arg w: an ``P`` element sequence
    :arg v: an ``P`` by ``P`` matrix
    :arg b: an ``P`` element sequence 
    
    Returns a list containing the solution ``X`` 
    
    """
    # M,P = u.shape 
    # tmp = np.empty( (P,), dtype=object  ) 

    # for j in range(P):
        # if w[j] != 0:
            # s = sum(
                # u[i,j]*b[i] for i in range(M)
            # ) / w[j]
        # else:
            # s = 0
            
        # tmp[j] = s 
       
    # return np.matmul(v,tmp)
    
    return np.matmul( v,1.0/w*np.matmul(u.T,b) )

#------------------------------------------------
def solve(a,b,TOL=1E-5):
    """
    Solve a.x = b
    
    .. versionadded:: 1.4.x

    """
    u,w,v = svd_decomp(a)

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
    # Based on Numerical Recipes 'svdfit'
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
     
    a = np.empty( (M,P), dtype=object )
    b = np.empty( (M,), dtype=object )    
    
    for i in range(M):
        tmp = 1.0/sig[i]
        for j in range(P):
            a[i,j] = tmp*afunc_i[j]
            
        b[i] = tmp*y[i] 
        
        if i < M-1:
            afunc_i = fn(x[i+1])
             
    u,w,v = svd_decomp(a)
    
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
 
    coef = svbksb(u,w,v,b)

    # Residuals -> chisq
    chisq = 0.0 
    for i in range(M):
        afunc_i = fn(x[i])
        s = math.fsum(
                value( coef[j]*afunc_i[j] )
                    for j in range(P)
            )
        tmp = value( (y[i] - s)/sig[i] )
        chisq += tmp*tmp 
          
    # w and v are used to evaluate parameter covariance 
    return coef, chisq, w, v
   
# #----------------------------------------------------------------------------
# # This function is not needed to evaluate variance-covariance in this module,
# # but it completes the NR implementation and is called by some unit tests for 
# # this module.
# def svdvar(v,w):
    # """
    # Calculate the variance-covariance matrix after ``svdfit``
    
    # .. versionadded:: 1.4.x
    
    # :arg v: an ``P`` by ``P`` matrix of float
    # :arg w: an ``P`` element sequence of float 
    
    # """
    # P = len(w)  
    # cv = np.empty( (P,P), dtype=float )
    
    # wti = [
        # 1.0/value(w_i*w_i) if w_i != 0 else 0.0
            # for w_i in w 
    # ]
    
    # for i in range(P):
        # for j in range(i+1):
            # cv[i,j] = cv[j,i] = math.fsum(
                # value( v[i,k]*v[j,k]*wti[k] )
                    # for k in range(P)
            # )
    
    # return cv  

#----------------------------------------------------------------------------
def ols(x,y,fn,fn_inv=None):
    """Ordinary least squares fit of ``y`` to ``x``
    
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
   
    return OLSFit( coef,chisq,fn,fn_inv,M )

#----------------------------------------------------------------------------
def wls(x,y,u_y,fn,fn_inv=None):    
    """Weighted least squares fit of ``y`` to ``x``
    
    :arg x: a sequence of ``M`` stimulus values (independent-variables)
    :arg y: a sequence of ``M`` responses (dependent-variable)  
    :arg u_y: a sequence of ``M`` standard uncertainties in the responses
    :arg fn: a user-defined function relating ``x`` the response
    :arg fn_inv: a user-defined function relating the response to the stimulus 
    :returns:   an object containing regression results
    :rtype:     :class:`WLSFit``
    
    """
    M = len(y)
    
    if M != len(x):
        raise RuntimeError( f"len(x) != len(y)" )
    if M != len(u_y):
        raise RuntimeError( "len(x) != len(u_y)")
    
    coef, chisq, w, v = svdfit(x,y,u_y,fn)
    
    return WLSFit( coef,chisq,fn,fn_inv,M )


## _ls is used in the type-A calc to set up the elementary uncertain numbers.
# No need for that in type-B. However, do we want to allow the parameters 
# to be declared results, with labels?
# #----------------------------------------------------------------------------
# def _ls(x,y,sig,fn,label=None):
    # """
    
    # """
    # M = len(x) 
    # if M != len(y):
        # raise RuntimeError( "len(x) != len(y)" )

    # b, chisq, w, v = svdfit(x,y,sig,fn)
    
    # P = len(w)
    # if M <= P:
        # raise RuntimeError( f"{M} should be > {P}" )     
    
    # df = N - P
    
    # s2 = chisq/df
    # cv = s2*svdvar(v,w)
    
    # u = []
    # beta = []
    # for i in range(P):
        # u.append( math.sqrt(cv[i,i]) )
        
        # if label is None:
            # label_i = f'b_{i}'   
        # else:
            # label_i = f'{label}_{i}'
            
        # b_i = _ureal(
            # b[i],
            # u[i],
            # df,
            # label=label_i,
            # independent=False
        # )        
            
        # beta.append(b_i)
      
    # real_ensemble( beta, df )

    # for i in range(P):
        # for j in range(i):
            # den = u[i]*u[j]
            # assert abs(den) > 1E-13, "unexpected: {!r}".format(den) 
            # r = cv[i,j]/den
            # if r != 0:
                # beta[i].set_correlation(r,beta[j])
            
    # return beta,chisq,M,P
    
    
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
    M = len(y)        
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
        y.append( Z[i,0] )
         
    coef, chisq, w, v = svdfit(x,y,np.ones( (M,) ),fn)

    return GLSFit( coef,chisq,fn,fn_inv,M )
    
#-----------------------------------------------------------------------------------------
# TODO: don't define this class twice!
class LSFit(object):
 
    """
    Base class for regression results
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
  Number of points: {self._N}
  Number of parameters: {self._P}
  Parameters: {self._beta!r}
  Sum of the squared residuals: {self._ssr:G}
'''

    #------------------------------------------------------------------------
    def y_from_x(self,x,label=None):
        """
        Return ``y``, the response to ``x`` 
        
        :arg x: an uncertain real number 
        :arg label: a label for the uncertain number ``y``
         
        .. note::
            When ``label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)
        
        """
        # The elements of `beta` are uncertain numbers
        _y = np.dot( self.beta,np.array( self._fn(x) ) )
        
        if label is not None: _y = result(_y,label)
        
        return y 

    #------------------------------------------------------------------------
    def x_from_y(self,yseq,x_label=None):
        """Estimate the stimulus ``x`` corresponding to the responses in ``yseq``

        :arg yseq: a sequence of further observations of ``y``
        :arg x_label: a label for the return uncertain number `x` 

        The items in ``yseq`` must be uncertain real numbers.
        
        .. note::
            When ``x_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)
        
        """
        if self._fn_inv is None:
            raise RuntimeError( "An inverse function has not been defined" )

        y = mean( yseq ) 
        
        _x = self._fn_inv(y,self.beta)  
        
        if x_label is not None: _x = result(_x,x_label)
        
        return _x 

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

    def __init__(self,beta,ssr,fn,fn_inv,N):
        LSFit.__init__(self,beta,ssr,fn,fn_inv,N)
 
    def __str__(self):
        header = '''
Ordinary Least-Squares Results:
'''
        return header + str(LSFit)
 
# #----------------------------------------------------------------------------
# class WLSFit(LSFit):

    # """
    # Results of a weighted least squares regression
    # """

    # def __init__(self,beta,ssr,N,P):
        # LSFit.__init__(self,beta,ssr,N,P)

    # def __str__(self):
        # header = '''
# Weighted Least-Squares Results:
# '''
        # return header + str(LSFit)
 
    
# #----------------------------------------------------------------------------
# class GLSFit(LSFit):

    # """
    # Results of a generalised least squares regression
    # """

    # def __init__(self,beta,ssr,N,P):
        # LSFit.__init__(self,beta,ssr,N,P)
  
    # def __str__(self):
        # header = '''
# Generalised Least-Squares Results:
# '''
        # return header + str(LSFit)
  
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
    
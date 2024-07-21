"""
"""
import math
import numpy as np

from GTC.lib import value

#----------------------------------------------------------------------------
def _pythag(a,b):
    """
    Return sqrt(a*a + b*b) 
        
    This function avoids numerical problems with direct 
    evaluation and handles different argument types
    (`a` and `b` can be uncertain numbers)

    """
    # imports here to avoid circular reference problems
    from GTC.core import magnitude      # Polymorphic
    from GTC.core import sqrt           # Polymorphic

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
    # import here to avoid circular reference problems
    from GTC.core import sqrt           # Polymorphic

    # avoid side effects to input array
    u = a.copy()    

    M,P = u.shape    
    w = np.empty( (P,), dtype=object ) 
    v = np.empty( (P,P), dtype=object )
    rv1 = np.empty( (P,), dtype=object )
    
    g = 0.0                    
    scale = anorm = 0.0     
    
    for i in range(P):
        
        l = i + 1
        rv1[i] = scale * g 
        
        g = s = scale = 0.0        
        if i < M:
            scale = sum(
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
    return v @ np.diag(1.0/w) @ u.T @ b

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
def svdfit(x,y,sig=None,fn=None):
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
    if sig is None: sig = np.ones( (len(x),) )
    
    # Allow for uncertain-number inputs:
    # construct arrays `a` and `b` from values 
    if fn is None:
        P = len( x[0] )   
        def row(row_i,s_i):
            return [ x_i/s_i for x_i in row_i ]
            
        a = np.array( [ row(row_i,s_i) for row_i,s_i in zip(x,sig) ],
            dtype=object 
        )
        b = np.array( [ y_i/s_i for y_i,s_i in zip(y,sig) ], 
            dtype=object  
        )   
    else:       
        # fn(x_i) returns an P-sized array of values for
        # each basis function at the stimulus point `x_i`
        P = len( fn(x[0])  )   

        a = np.empty( (M,P), dtype=object )
        b = np.empty( (M,), dtype=object )    
        
        for i in range(M):
            afunc_i = fn(x[i])
            for j in range(P):
                a[i,j] = afunc_i[j]/sig[i]
                
            b[i] = y[i]/sig[i]

    u,w,v = svd_decomp(a)
    
    # `TOL` is used to set relatively small singular values to zero
    # Doing so avoids numerical precision problems, but will make the 
    # solution slightly less accurate. The value can be varied.
    TOL = 1E-5
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
 
    # coef = svbksb(u,w,v,b)
    coef = v @ np.diag(1 / w) @ u.T @ b

    # Residuals and sum of squared residuals
    res = b - np.dot(a, coef) 
    ssr = math.fsum( value(res_i)**2 for res_i in res )
 
    wti = [
        1.0/(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    cv_coef = v @ np.diag(wti) @ v.T
 
    return coef, res, ssr, w, v  

#----------------------------------------------------------------------------
# 
def svdvar(v,w):
    """
    Calculate the variance-covariance matrix after a SVD regression
    
    .. versionadded:: 1.4.x
    
    :arg v: an ``P`` by ``P`` matrix of float
    :arg w: an ``P`` element sequence of float 
    
    """
    # `v` and `w` may contain uncertain numbers
    P = len(w)  
    
    wti = [
        1.0/value(w_i*w_i) if w_i != 0 else 0.0
            for w_i in w 
    ]
    
    cv = np.empty( (P,P), dtype=np.float64 )
    for i in range(P):
        for j in range(i+1):
            cv[i,j] = cv[j,i] = math.fsum(
                value( v[i,k]*v[j,k]*wti[k] )
                    for k in range(P)
            )
    
    return cv  

#============================================================================
if __name__ == '__main__': 

    from GTC import *
    
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


    
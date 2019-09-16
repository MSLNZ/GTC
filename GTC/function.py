"""
Utility functions
-----------------
Functions :func:`complex_to_seq` and :func:`seq_to_complex` 
are useful to convert between the matrix representation of 
complex numbers and Python :obj:`complex`.

The function :func:`mean` evaluates the mean of a sequence.

Least-squares regression
------------------------

:func:`line_fit` implements an ordinary least-squares 
straight-line regression calculation that accepts uncertain 
real numbers for the independent and dependent variables.

:func:`line_fit_wls` implements a weighted least-squares 
straight-line regression calculation. It accepts uncertain 
real numbers for the independent and dependent variables.
It is also possible to specify weights for the regression.

:func:`line_fit_wtls` implements a total least-squares 
algorithm for a straight-line fitting that can perform a 
weighted least-squares regression when both `y` and `x` data  
are uncertain real numbers, it also handles correlation 
between (x,y) data pairs.

Module contents
---------------

"""
from __future__ import division

import sys
import math
import numpy as np 

try:  # Python 2
    import __builtin__ as builtins
    from collections import Iterable
    from itertools import izip
except ImportError:
    import builtins
    from collections.abc import Iterable
    izip = zip
    xrange = range

from GTC import (
    is_sequence,
    inf 
)

from GTC.named_tuples import InterceptSlope
from GTC.lib import UncertainReal
from GTC.vector import scale_vector

def value(x):
    try:
        return x.x
    except AttributeError:
        return x
        
__all__ = (
    'complex_to_seq',
    'seq_to_complex',
    'mean',
    'line_fit','line_fit_wls','line_fit_wtls',
    'LineFitOLS', 'LineFitWLS', 'LineFitWTLS'
)

HALF_PI = math.pi / 2.0
MAX = sys.float_info.max 
EPSILON = sys.float_info.epsilon 

#-----------------------------------------------------------------------------------------
class LineFit(object):
    
    """
    Base class for the results of regression to a line.
    """
    
    def __init__(self,a,b,ssr,N):
        self._a_b = InterceptSlope(a,b)
        self._ssr = ssr
        self._N = N
        
    def __repr__(self):
        return """{}(
  a={!r},
  b={!r},
  ssr={},
  N={}
)""".format(
            self.__class__.__name__,
            self._a_b[0],
            self._a_b[1],
            self._ssr,
            self._N
        )

    @property
    def a_b(self):
        """Return the intercept and slope as uncertain numbers
        """
        return self._a_b

    @property
    def intercept(self):
        """Return the intercept as an uncertain number.
        """
        return self._a_b[0]

    @property
    def slope(self):
        """Return the slope as an uncertain number.
        """
        return self._a_b[1]

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
    def N(self):
        """The number of points in the sample"""
        return self._N

    def __str__(self):
        a, b = self.a_b
        return '''
  Intercept: {!s}
  Slope: {!s}
  Correlation: {:.2G}
  Sum of the squared residuals: {}
  Number of points: {}
'''.format(
    a,
    b,
    a.get_correlation(b),
    self._ssr,
    self.N
)
 
#-----------------------------------------------------------------------------------------
class LineFitOLS(LineFit):
    
    """
    Class to hold results from an ordinary linear regression to data.
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Ordinary Least-Squares Results:
'''
        return header + LineFit.__str__(self)

#-----------------------------------------------------------------------------------------

class LineFitWLS(LineFit):
    
    """
    This object holds results from a weighted LS linear regression to data.
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Weighted Least-Squares Results:
'''
        return header + LineFit.__str__(self)

#-----------------------------------------------------------------------------------------
class LineFitWTLS(LineFit):
    
    """
    This object holds results from a TLS linear regression to data.
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Weighted Total Least-Squares Results:
'''
        return header + LineFit.__str__(self)


#--------------------------------------------------------------------
#
def line_fit(x,y):
    """Least-squares fit intercept and slope 
    
    :arg x:     sequence of independent variable data 
    :arg y:     sequence of dependent variable data

    :rtype:     a :class:`~function.LineFitOLS`
    
    ``y`` must be a sequence of uncertain real numbers.

    Performs an ordinary least-squares regression. 
    
    .. note::

        Uncertainty in the parameter estimates is found
        by propagation *through* the regression
        formulae. This does **not** take residuals into account.
        
        The function :func:`type_a.line_fit` performs a regression 
        analysis that evaluates uncertainty in 
        the parameter estimates using the residuals.
        
        If appropriate, the results from both type-A and type-B 
        analyses can be merged (see :func:`type_a.merge`).
        
    **Example**::

        >>> a0 =10
        >>> b0 = -3
        >>> u0 = .2

        >>> x = [ float(x_i) for x_i in xrange(10) ]
        >>> y = [ ureal(b0*x_i + a0,u0) for x_i in x ]

        >>> a,b = fn.line_fit(x,y).a_b
        >>> a
        ureal(10.0,0.1175507627290...,inf)
        >>> b
        ureal(-3.0,0.02201927530252...,inf)
        
    """  
    S = len(x) 
    
    S_x = sum( x ) 
    S_y = sum( y )

    k = S_x / S
    t = [ x_i - k for x_i in x ]

    S_tt = sum( t_i*t_i for t_i in t )
    
    b = sum( t_i*y_i/S_tt for t_i,y_i in izip(t,y) )
    a = (S_y - b*S_x)/S

    if not isinstance(a, UncertainReal):
        raise ValueError('"y" must be a sequence of uncertain real numbers. '
                         'You may want to use type_a.line_fit instead.')

    # The sum of squared residuals is now calculated but not used
    float_a = value(a)
    float_b = value(b)
    
    f2 = lambda x_i,y_i: (
        (y_i - float_a - float_b*x_i)
    )**2 
    
    ssr =  math.fsum( 
        f2( value(x_i), value(y_i) ) 
            for x_i,y_i in izip(x,y) 
    )

    return LineFitOLS(a,b,ssr,S)
 
#--------------------------------------------------------------------
#
def line_fit_wls(x,y,u_y=None):
    """-> Weighted least-squares linear regression
    
    :arg x:     sequence of independent variable data 
    :arg y:     sequence of dependent variable data
    :arg u_y:   sequence of uncertainties in ``y``

    :rtype:    :class:`~function.LineFitWLS` 
    
    ``y`` must be a sequence of uncertain real numbers.

    Performs a weighted least-squares regression. 

    Weights are calculated from the uncertainty of 
    the ``y`` elements unless the sequence ``u_y`` 
    is provided. 
    
    .. note::

        The uncertainty in the parameter estimates is found
        by propagation of uncertainty *through* the regression
        formulae. This does **not** take account of the residuals.
        
        The function :func:`type_a.line_fit_wls` can be used to 
        carry out a regression analysis that obtains uncertainty in 
        the parameter estimates due to the residuals.
        
        If necessary, the results of both type-A and type-B 
        analyses can be merged (see :func:`type_a.merge`).
        
    **Example**::

        >>> x = [1,2,3,4,5,6]
        >>> y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
        >>> u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        >>> y = [ ureal(y_i,u_y_i) for y_i, u_y_i in zip(y,u_y) ]
        
        >>> fit = function.line_fit_wls(x,y)
        >>> a, b = fit.a_b
        >>> a
        ureal(0.8852320675105...,0.5297081435088...,inf)
        >>> b
        ureal(2.0569620253164...,0.1778920167412...,inf)
        
    """
    if u_y is None:
        v = [ y_i.v for y_i in y ]
        u = [ math.sqrt(v_i) for v_i in v ]
    else:
        v = [ u_y_i**2 for u_y_i in u_y ]
        u = u_y     
        
    S = sum( 1.0/v_i for v_i in v)
    S_x = sum( x_i/v_i for x_i,v_i in izip(x,v) )    
    S_y = sum( y_i/v_i for y_i,v_i in izip(y,v) )

    k = S_x / S
    t = [ (x_i - k)/u_i for x_i,u_i in izip(x,u) ]

    S_tt = sum( t_i*t_i for t_i in t )

    b = sum( t_i*y_i/u_i/S_tt for t_i,y_i,u_i in izip(t,y,u) )
    a = (S_y - b*S_x)/S

    if not isinstance(a, UncertainReal):
        raise ValueError('"y" must be a sequence of uncertain real numbers. '
                         'You may want to use type_a.line_fit_wls instead.')

    # The sum of squared residuals is now calculated but not used
    float_a = value(a)
    float_b = value(b)
    
    f2 = lambda x_i,y_i,u_i: (
        (y_i - float_a - float_b*x_i)/u_i
    )**2 
    
    ssr =  math.fsum( 
        f2( value(x_i), value(y_i), u_i ) 
            for x_i,y_i,u_i in izip(x,y,u) 
    )

    return LineFitWLS(a,b,ssr,len(x))
    
#---------------------------------------------------------------------------
def sum(seq,*args,**kwargs):
    """Return the sum of elements in `seq`
    
    :arg seq: a sequence, :class:`~numpy.ndarray`, or iterable, of numbers or uncertain numbers
    :arg args: optional arguments when ``seq`` is an :class:`~numpy.ndarray`
    :arg kwargs: optional keyword arguments when ``seq`` is an :class:`~numpy.ndarray`
    
    .. versionadded:: 1.1

    """
    if isinstance(seq,np.ndarray):
        return np.asarray(seq).sum(*args, **kwargs)
        
    elif is_sequence(seq) or isinstance(seq,Iterable):
        return builtins.sum(seq)
        
    else:
        raise RuntimeError(
            "{!r} is not iterable".format(seq)
        )    
 
#---------------------------------------------------------------------------
def mean(seq,*args,**kwargs):
    """Return the arithmetic mean of the elements in `seq`
    
    :arg seq: a sequence, :class:`~numpy.ndarray`, or iterable, of numbers or uncertain numbers
    :arg args: optional arguments when ``seq`` is an :class:`~numpy.ndarray`
    :arg kwargs: optional keyword arguments when ``seq`` is an :class:`~numpy.ndarray`
    
    If the elements of ``seq`` are uncertain numbers, 
    an uncertain number is returned.
    
    **Example** ::
    
        >>> seq = [ ureal(1,1), ureal(2,1), ureal(3,1) ]
        >>> function.mean(seq)
        ureal(2.0,0.5773502691896257,inf)
        
    """
    if is_sequence(seq):
        return sum(seq)/len(seq)
        
    elif isinstance(seq,np.ndarray):
        return np.asarray(seq).mean(*args, **kwargs)
        
    elif isinstance(seq,Iterable):
        count = 0
        total = 0
        for i in seq:
            total += i
            count += 1
        return total/count
        
    else:
        raise RuntimeError(
            "{!r} is not iterable".format(seq)
        )
#---------------------------------------------------------------------------
def complex_to_seq(z):
    """Transform a complex number into a 4-element sequence

    :arg z: a number

    If ``z = x + yj``, then an array of the form ``[[x,-y],[y,x]]`` 
    can be used to represent ``z`` in matrix computations. 

    **Examples**::
        >>> import numpy
        >>> z = 1 + 2j
        >>> function.complex_to_seq(z)
        (1.0, -2.0, 2.0, 1.0)
        
        >>> m = numpy.array( function.complex_to_seq(z) )
        >>> m.shape = (2,2)
        >>> print( m )
        [[ 1. -2.]
         [ 2.  1.]]
        
    """
    z = complex(z)
    return (z.real,-z.imag,z.imag,z.real)

#---------------------------------------------------------------------------
def seq_to_complex(seq):
    """Transform a 4-element sequence into a complex number 

    :arg seq:   a 4-element sequence
    :raises RuntimeError: if ``seq`` is ill-conditioned

    **Examples**::

        >>> import numpy
        >>> seq = (1,-2,2,1)
        >>> z = function.seq_to_complex( seq )
        >>> z 
        (1+2j)
        >>> a = numpy.array((1,-2,2,1))
        >>> a.shape = 2,2
        >>> a
        array([[ 1, -2],
               [ 2,  1]])
        >>> z = function.seq_to_complex(a)
        >>> z 
        (1+2j)

    """
    if hasattr(seq,'shape'):
        if seq.shape != (2,2):
            raise RuntimeError("array shape illegal: {}".format(seq))
        elif (
            math.fabs( seq[0,0] - seq[1,1] ) > EPSILON
        or  math.fabs( seq[1,0] + seq[0,1] ) > EPSILON ):
            raise RuntimeError("ill-conditioned sequence: {}".format(seq))
        else:
            seq = list( seq.flat )
            
    elif is_sequence(seq):
        if len(seq) != 4:
            raise RuntimeError("sequence must have 4 elements: {}".format(seq))
        elif (
            math.fabs( seq[0] - seq[3] ) > EPSILON
        or  math.fabs( seq[1] + seq[2] ) > EPSILON ):
            raise RuntimeError("ill-conditioned sequence: {}".format(seq))
    
    else:
        raise RuntimeError("illegal argument: {}".format(seq))

    return complex(seq[0],seq[2])
    
#--------------------------------------------------------------------
ZEPS = 1E-10
def _dbrent(ax,bx,cx,fn,tol=math.sqrt(EPSILON)):
    """
    Minimise fn() and return x, fn(x) and df_dx(x), all floats
    
    `fn` must be a univariate function of an uncertain real that returns
    an uncertain real number.

    `ax`, `bx` and `cx` must be floats. `bx` must be between `ax` and `cx`
    and fn(bx) must be less than both fn(ax) and fn(cx).

    `context` - a GTC context
    
    `tol` - the fractional precision

    See also Numerical Recipes in C, 2nd ed, Section 10.3
    
    """
    ITMAX = 100

    deriv = lambda y,x: y.sensitivity(x)
    ureal = lambda x,u: UncertainReal._elementary(
            x,u,inf,None,True
    )    
    e = 0.0 # The distance moved on the step before last
    
    a = ax if ax < cx else cx
    b = ax if ax > cx else cx
    
    assert a <= bx and bx <= b, "Invalid initial values in _dbrent"
        
    # if fn( ureal(bx,1)) > fn( ureal(a,1)) or fn(ureal(bx,1)) > fn(ureal(b,1)):
        # assert False

    x = w = v = bx

    _u_ = ureal(x,1.0)    
    fn_u = fn( _u_ )
    
    fw = fv = fx = value(fn_u)
    dw = dv = dx = deriv(fn_u,_u_)

    # The routine keeps track of `a` and `b`, which bracket the minimum,
    # `x` is the point with the least function value found so far,
    # `w` is the point with the second least value, `v` is the previous
    # value of `w`, `u` is the point at which the function was most 
    # recently evaluated.
    
    for i in xrange(ITMAX):
        
        xm = 0.5*(a + b)
        tol1 = tol*abs(x) + ZEPS
        tol2 = 2.0*tol1

        if abs(x - xm) <= ( tol2 - 0.5*(b - a) ):
            return x, fx, dx
        
        if abs(e) > tol1:
            # initialise the d's to be out of bracket 
            d1 = 2.0*(b - a)
            d2 = d1
            
            # Secant method
            if dw != dx: d1 = (w - x)*dx/(dx - dw)  
            if dv != dx: d2 = (v - x)*dx/(dx - dv)  
            
            # Choose one estimate.
            # Insist that it be within the bracket 
            # and on the side pointed to by the derivative at `x` 
            u1 = x + d1
            u2 = x + d2
            OK1 = (a - u1)*(u1 - b) > 0.0 and dx*d1 <= 0.0
            OK2 = (a - u2)*(u2 - b) > 0.0 and dx*d2 <= 0.0
            
            olde, e = e, d

            if OK1 or OK2:
                if OK1 and OK2:
                    d = d1 if abs(d1) < abs(d2) else d2
                elif OK1:
                    d = d1
                else:
                    d = d2
                    
                if abs(d) <= abs(0.5 * olde):
                    u = x + d
                    if (u - a < tol2) or (b - u < tol2):
                        d = math.copysign(tol1,xm-x)
                else:
                    # choose segment by the sign of the derivative
                    e = a - x if dx >= 0.0 else b - x
                    d = 0.5*e
            else:
                e = a - x if dx >= 0.0 else b - x
                d = 0.5*e
                
        else:
            e = a - x if dx >= 0.0 else b - x
            d = 0.5*e
            
        if abs(d) >= tol1:
            u = x + d
            
            _u_ = ureal(u,1.0)
            fn_u = fn(_u_)
            
            fu = value(fn_u)
            du = deriv(fn_u,_u_)
            
        else:
            # Smallest step possible 
            u = x + math.copysign(tol1,d)
            
            _u_ = ureal(u,1.0)
            fn_u = fn(_u_)
            
            fu = value(fn_u)
            du = deriv(fn_u,_u_)
            
            # If the minimum sized step downhill 
            # goes up, then we are done!
            if fu > fx:
                return u, fu, du
                
        assert a <= u and u <= b, (a,u,b)   # invariant
    
        if fu <= fx:
            # Found a new best point 
            
            # Update the bracket on one side so that the previous 
            # `x` value is now the limit and the new `x` value is 
            # contained. 
            if u >= x:
                a = x
            else:
                b = x
                
            v, fv, dv = w, fw, dw
            w, fw, dw = x, fx, dx
            x, fx, dx = u, fu, du
            
            assert a <= x and x <= b, (a,x,b)   # invariant
            
        else:
            # The point `x` has not been bettered
            
            # `x` does not change, but `u` was inside the 
            # bracket so we can tighten the noose.
            if u < x:
                a = u
            else:
                b = u
             
            # `w` is the second best point and `v` is the 3rd best 
            if (fu <= fw) or (x == w):
                v, fv, dv = w, fw, dw
                w, fw, dw = u, fu, du
                
            elif (fu < fv) or (v == x) or (v == w):            
                v, fv, dv = u, fu, du
   
            assert a <= x and x <= b, (a,x,b)   # invariant
            
        assert fx <= fw and fx <= fv # invariant
        
    raise RuntimeError('Exceeded iteration limit in `_dbrent`')

#--------------------------------------------------------------------
def _arrays(sin_a,cos_a,sin_2a,cos_2a,x,y,u2_x,u2_y,cov):
    """Returns the set of arrays needed for the Chi-sq calculation

    Equations reference: M Krystek and M Anton,
    Meas. Sci. Technol. 22 (2011) 035101 (9pp)

    Note, this utility function can work equally with real numbers
    or uncertain real numbers. This allows us to us it in the
    type_a module and checking routines too.
    
    """
    sin_a_2 = sin_a**2
    cos_a_2 = cos_a**2
    two_sin_cos_a = 2.0*sin_a*cos_a

    # Note an alternative (perhaps more stable numerically is the following
    # it will need cos_2a and sin_2a to be passed in as arguments too.
    # However, it fails one of our test cases because an element goes to zero!
    
    # eqn(53)
    g_k = [
        (u2_x_i + u2_y_i)/2.0 - (u2_x_i - u2_y_i)*cos_2a/2.0 - 2.0*cov_i*sin_2a
            for u2_x_i, u2_y_i, cov_i in izip(u2_x,u2_y,cov)
    ]

##    # eqn(32)
##    g_k = [
##        u2_x_i*sin_a_2 + u2_y_i*cos_a_2 - two_sin_cos_a*cov_i
##            for u2_x_i, u2_y_i, cov_i in izip(u2_x,u2_y,cov)
##    ]

    N = len(g_k)

    fix_div_by_zero = lambda y,x: y/x if x != 0 else MAX

    # eqn(33), but without sqrt
    u2 = 1.0/(
        sum( fix_div_by_zero(1.0,g_k_i) for g_k_i in g_k ) / N
    )
    # eqn(34)
    w_k = [ fix_div_by_zero(u2,g_k_i) for g_k_i in g_k ]

    # Eqns (35,36,43)
    x_bar = sum(
        w_k_i*x_i for w_k_i,x_i in izip(w_k,x)
    ) / N
    
    y_bar = sum(
        w_k_i*y_i for w_k_i,y_i in izip(w_k,y)
    ) / N

    p_hat = y_bar*cos_a - x_bar*sin_a    
    
    # eqn(31)
    v_k = [ y_i*cos_a - x_i*sin_a - p_hat for x_i,y_i in izip(x,y) ]  

    return v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat

#--------------------------------------------------------------------
class ChiSq(object):

    """
    A callable object representing Chi-squared as a function of alpha.
    
    ``x`` and ``y`` are lists of uncertain real numbers used to 
    initialise the object. 
    
    Equations reference: M Krystek and M Anton,
    Meas. Sci. Technol. 22 (2011) 035101 (9pp)
    """
    
    def __init__(self,x,y,u_x,u_y,r_xy):
        self.x_u = x
        self.y_u = y
        
        self.x = [ x_i.x for x_i in x ]
        self.y = [ y_i.x for y_i in y ]

        if u_x is not None:
            self.u2_x = [ u_i**2 for u_i in u_x ]
            self.u2_y = [ u_i**2 for u_i in u_y ]
            self.cov = [ u_x_i*u_y_i*r_i
                for u_x_i,u_y_i,r_i in izip(u_x,u_y,r_xy) 
            ]  
        else:    
            self.u2_x = [ x_i.v for x_i in x ]
            self.u2_y = [ y_i.v for y_i in y ]
        
            self.cov = [ x_i.get_covariance(y_i)
                for x_i,y_i in izip(x,y)
            ]
        
    #--------------------------------------------------------------------
    def arrays(self,alpha):
        """Return v_k,g_k      
        """
        sin_a = alpha._sin()
        sin_2a = (2.0*alpha)._sin()
        cos_a = alpha._cos()
        cos_2a = (2.0*alpha)._cos()

        v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat = _arrays(
            sin_a,cos_a,sin_2a,cos_2a,self.x,self.y,self.u2_x,self.u2_y,self.cov
        )

        return v_k, g_k

    #--------------------------------------------------------------------
    def p_hat(self,alpha):
        """Eqn (43)"""
        sin_a = alpha._sin()
        sin_2a = (2.0*alpha)._sin()
        cos_a = alpha._cos()
        cos_2a = (2.0*alpha)._cos()

        v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat = _arrays(
            sin_a,cos_a,sin_2a,cos_2a,self.x_u,self.y_u,self.u2_x,self.u2_y,self.cov
        )

        return p_hat
        
    #--------------------------------------------------------------------
    def __call__(self,alpha):
        """Returns ChiSq(alpha)"""
        v_k,g_k = self.arrays(alpha)
            
        # Eqn (30)
        chi_2 = sum(
            v_k_i**2/g_k_i for v_k_i,g_k_i in izip(v_k,g_k)
        )
        
        return chi_2

#--------------------------------------------------------------------
class dChiSq_dalpha(object):

    """
    Callable object representing the derivative of Chi-squared wrt alpha. 

    Equations reference: M Krystek and M Anton,
    Meas. Sci. Technol. 22 (2011) 035101 (9pp)

    Some definitions (equation numbers):
        v_k :  (31)
        v_ka : (51)
        v_kaa : (52)
        g_k : (53)
        g_ka : 54        
    """
    
    def __init__(self,x,y,u_x,u_y,r_xy):
        self.x = x
        self.y = y

        if u_x is not None:
            self.u2_x = [ u_i**2 for u_i in u_x ]
            self.u2_y = [ u_i**2 for u_i in u_y ]
            self.cov = [ u_x_i*u_y_i*r_i
                for u_x_i,u_y_i,r_i in izip(u_x,u_y,r_xy) 
            ]  
        else:
            self.u2_x = [ x_i.v for x_i in x ]
            self.u2_y = [ y_i.v for y_i in y ]
            self.cov = [ x_i.get_covariance(y_i)
                for x_i,y_i in izip(x,y) 
            ]      
        
    #--------------------------------------------------------------------
    def arrays(self,alpha):
        """Return v_k, v_ka, g_k, g_ka (eqns: 31, 51, 53, 54)
        """
        sin_a = alpha._sin()
        sin_2a = (2.0*alpha)._sin()
        cos_a = alpha._cos()
        cos_2a = (2.0*alpha)._cos()

        v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat = _arrays(
            sin_a,cos_a,sin_2a,cos_2a,self.x,self.y,self.u2_x,self.u2_y,self.cov
        )

        # d(v_k)_d(alpha)
        v_ka = [ -y_i*sin_a - x_i*cos_a for x_i,y_i in izip(self.x,self.y) ]
        
        # d(g_k)_d(alpha)
        g_ka = [ sin_2a*(u2_x_i-u2_y_i) - 2.0*cov_i*cos_2a
            for u2_x_i,u2_y_i,cov_i in izip(u2_x,u2_y,self.cov) ]
        
        return v_k, v_ka, g_k, g_ka

        
    #--------------------------------------------------------------------
    def __call__(self,alpha):
        """Returns d(ChiSq)_d(alpha)( alpha ) 
        
        `v_k` (31); `v_ka` (51); `v_kaa` (52)
        `g_k` (53); `g_ka` (54); 
        """
        v_k, v_ka, g_k, g_ka = self.arrays(alpha)       
        
        # Eqn (56)
        dy_dx = sum(
           (2.0*v_k_i*g_k_i*v_ka_i - v_k_i**2*g_ka_i)/g_k_i**2
                for v_k_i,v_ka_i,g_k_i,g_ka_i
                    in izip(v_k, v_ka, g_k, g_ka)
        )
        
        return dy_dx
     
#--------------------------------------------------------------------
def line_fit_wtls(x,y,u_x=None,u_y=None,a_b=None,r_xy=None):
    """Perform straight-line regression with uncertainty in ``x`` and ``y``

    :arg x: list of uncertain real numbers for the independent variable
    :arg y: list of uncertain real numbers for the dependent variable
    :arg u_x: a sequence of uncertainties for the ``x`` data
    :arg u_y: a sequence of uncertainties for the ``y`` data
    :arg a_b: a pair of initial estimates for the intercept and slope
    :arg r_xy: correlation between x-y pairs [default: 0]

    Returns a :class:`~function.LineFitWTLS` object

    The elements of ``x`` and ``y`` must be uncertain numbers
    with non-zero uncertainties. The uncertainties of the ``x`` 
    and ``y`` sequences are used to calculate weights 
    for the regression unless the optional arguments
    ``u_x`` and ``u_y`` are specified.
    
    Implements a Weighted Total Least Squares algorithm
    that allows for correlation between x-y pairs. See reference: 
    
    M Krystek and M Anton, *Meas. Sci. Technol.* **22** (2011) 035101 (9pp)
        
    **Example**::

        # Pearson-York test data
        # see, e.g., Lybanon, M. in Am. J. Phys 52 (1), January 1984 
        >>> xin=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        >>> wx=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        >>> yin=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        >>> wy=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        # Convert weights to standard uncertainties 
        >>> uxin=[1./math.sqrt(wx_i) for wx_i in wx ]
        >>> uyin=[1./math.sqrt(wy_i) for wy_i in wy ]

        # Define uncertain numbers
        >>> x = [ ureal(xin_i,uxin_i) for xin_i,uxin_i in zip(xin,uxin) ]
        >>> y = [ ureal(yin_i,uyin_i) for yin_i,uyin_i in zip(yin,uyin) ]

        # TLS returns uncertain numbers
        >>> a,b = function.line_fit_wtls(x,y).a_b
        >>> a
        ureal(5.4799101832830...,0.2919334989452...,inf)
        >>> b
        ureal(-0.480533399108...,0.05761674075939...,inf)

    """
    ureal = lambda x,u: UncertainReal._elementary(
            x,u,inf,None,True
    )    

    if (u_x is not None or u_y is not None):
        if (u_x is None or u_y is None):
            raise RuntimeError(
                "You must supply ``u_x`` and ``u_y``"
            )
        elif (r_xy is None):
            # default value will be uncorrelated
            r_xy = [0] * len(u_x)
            
    for x_i,y_i in izip(x,y):
        assert isinstance(x_i,UncertainReal), 'uncertain real required'
        assert isinstance(y_i,UncertainReal), 'uncertain real required'

    if a_b is None:
        a_b = line_fit(x, y).a_b

    a0 = value(a_b[0])
    b0 = value(a_b[1])
    
    # initial value for `alpha`
    alpha0 = math.atan(b0)

    # chi_sq(alpha0) -> chisquared
    chi_sq = ChiSq(x,y,u_x,u_y,r_xy)   

    # Search for the minimum chi-squared wrt alpha
    x1 = alpha0 - HALF_PI
    x2 = alpha0 + HALF_PI

    # `brent` requires three points that bracket the minimum.
    # the `x1`, `alpha0`, `x2` parameters should be real, 
    # but `data` will return an uncertain number 
    # and expects an uncertain number argument.
    #
    # Returns x, fn(x) and df_dx(x), all floats
    alpha1,fn_alpha1,df_alpha1 = _dbrent(x1,alpha0,x2,chi_sq)

    # dChiSq_a( alpha ) will return dChiSq_dalpha(`alpha`)
    # dChiSq_a(alpha0) -> 1st partial derivative of chisquared at alpha0
    dChiSq_a = dChiSq_dalpha(x,y,u_x,u_y,r_xy)   
    
    # Need the partial derivative of dChiSq_a wrt alpha 
    alpha = ureal(alpha1,1) 
    F_alpha = dChiSq_a( alpha ) 
    dalpha_dF = -1.0/F_alpha.sensitivity(alpha)

    # Now we define `alpha` with sensitivity to the ``x`` and ``y`` data,
    # via the object ``F_alpha``, which represents the 1st partial derivative
    # of chi-squared at alpha1 (ideally zero, but really only close to the root).
    F_alpha = dChiSq_a( UncertainReal._constant(alpha1) )
    alpha = UncertainReal(
        alpha1
    ,   scale_vector(F_alpha._u_components,dalpha_dF)
    ,   scale_vector(F_alpha._d_components,dalpha_dF)
    ,   scale_vector(F_alpha._i_components,dalpha_dF)
    )

    # The sensitivity of p_hat to the x and y data is via 
    # `alpha`, `x_bar` and `y_bar` in eqn (43)
    p_hat = chi_sq.p_hat( alpha )

    # Note we have reversed the definitions of `a` and `b` here
    b = alpha._tan()
    a = p_hat/alpha._cos()

    N = len(x)

    ssr = chi_sq( UncertainReal._constant(alpha1) ).x

    return LineFitWTLS(a,b,ssr,N)

# ===========================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *
    doctest.testmod(  optionflags=doctest.NORMALIZE_WHITESPACE  )
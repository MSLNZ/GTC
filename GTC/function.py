"""
Utility functions
-----------------
Functions :func:`complex_to_seq` and :func:`seq_to_complex` 
are useful to convert between the matrix representation of 
complex numbers and Python :obj:`complex`.

The function :func:`mean` evaluates the mean of a sequence.

The function :func:`implicit` will evaluate the solution 
to :math:`fn(x) = 0`

Module contents
---------------

"""
from __future__ import division

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
    inf ,
    EPSILON
)

from GTC.lib import (
    UncertainReal, UncertainComplex, 
    mult_2nd_real_pair, mult_2nd_complex_pair,mult_2nd_real_complex
)

from GTC import vector
        
__all__ = (
    'complex_to_seq',
    'seq_to_complex',
    'mean',
    'implicit'
)
    
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
    
#---------------------------------------------------------------------------
def _simple_variance(v):
    v11, v12, v21, v22 = v
    if (
        abs(v11-v22) > 1E-15
    or  abs(v12) > 1E-15
    or  abs(v21) > 1E-15
    ):
        raise RuntimeError(
            "equal diagonal variance required, got: {}".format(v)
        )
#---------------------------------------------------------------------------
def mul2(arg1,arg2,estimated=False):
    """
    Return the product of ``arg1`` and ``arg2``

    Extends the usual calculation of a product, by
    using second-order contributions to uncertainty.
    
    :arg arg1: uncertain real or complex number
    :arg arg2: uncertain real or complex number
    :arg estimated: Boolean

    When both arguments are uncertain numbers 
    that have the same fixed values then 
    ``estimated`` should be set ``False``. 
    For instance, residual errors are often associated 
    with the value 0, or 1, which is not measured, in
    that case ``estimated=False`` is appropriate. 
    However, if either or both arguments are based on 
    measured values set ``estimated=True``.
    
    .. note::
    
        When ``estimated`` is ``True``, and the 
        product is close to zero, the result of a  
        second-order uncertainty calculation is 
        smaller than the uncertainty calculated by 
        the usual first-order method. In some cases, 
        an uncertainty of zero will be obtained.
    
    There are fairly strict limitations on the use of this
    function, especially for uncertain complex numbers:
    
    1) Arguments must be independent (have no common influence  
    quantities) and there can be no correlation between any 
    of the quantities that influence `arg1` or `arg2`. 

    2) If either argument is uncertain complex, the real and 
    imaginary components of that argument must have equal uncertainties  
    (i.e., the covariance matrix must be diagonal with equal elements 
    along the diagonal) and be independent (no common influences).

    A :class:`RuntimeError` exception is raised if  
    these conditions are not met.

    .. note::
    
        This function has been developed to improve the
        accuracy of uncertainty calculations where one or  
        both multiplicands are zero. In such cases, the 
        usual method of uncertainty propagation fails.

        For example ::
                
            >>> x1 = ureal(0,1,label='x1')
            >>> x2 = ureal(0,1,label='x2')
            >>> y = x1 * x2
            >>> y
            ureal(0.0,0.0,inf)
            >>> for cpt in rp.budget(y,trim=0):
            ... 	print("  {0.label}: {0.u}".format(cpt) )
            ... 	
              x1: 0.0
              x2: 0.0
              
        we see that none of the uncertainty in ``x1`` or ``x2`` 
        is propagated to ``y``. However, we may calculate 
        the second-order contribution ::
        
            >>> y = fn.mul2(x1,x2)
            >>> y
            ureal(0.0,1.0,inf)
            >>> for cpt in rp.budget(y,trim=0):
            ... 	print("  {0.label}: {0.u}".format(cpt) )
            ... 	
              x1: 0.70710678...
              x2: 0.70710678...
    
        The product now has a standard uncertainty of unity.
        
    .. warning::
    
        :func:`mul2` departs from the first-order linear  
        calculation of uncertainty described in the GUM.

        In particular, the strict proportionality between 
        components of uncertainty and first-order partial
        derivatives no longer holds.
        
        As a consequence, the calculation of first-order partial
        derivatives using the ``sensitivity`` method are also 
        incorrect.
        
    """
    reals = []
    comp = []
    for arg in (arg1,arg2):
        if not isinstance(arg,(UncertainReal,UncertainComplex)):
            raise RuntimeError(
                "uncertain number required, got: {!r}".format( arg )
            )

    if isinstance(arg1,UncertainReal):
        if isinstance(arg2,UncertainReal):
            return mult_2nd_real_pair(arg1,arg2,estimated)
        elif isinstance(arg2,UncertainComplex):
            _simple_variance(arg2.v)
            return mult_2nd_real_complex(arg1,arg2,estimated)
        else:
            raise RuntimeError(
                "uncertain number required, got: {!r}".format( arg2 )
            )
    elif isinstance(arg1,UncertainComplex):
        _simple_variance(arg1.v)
        if isinstance(arg2,UncertainReal):
            return mult_2nd_real_complex(arg2,arg1,estimated)
        elif isinstance(arg2,UncertainComplex):
            _simple_variance(arg2.v)
            return mult_2nd_complex_pair(arg1,arg2,estimated)
        else:
            raise RuntimeError(
                "uncertain number required, got: {!r}".format( arg2 )
            )
    else:
        raise RuntimeError(
            "uncertain number required, got: {!r}".format( arg1 )
        )
 
#---------------------------------------------------------------------------
def implicit(fn,x_min,x_max,epsilon=1E-13):
    """Return the solution to :math:`fn(x) = 0` 
        
    :arg fn: a user-defined function of one argument
    
    :arg x_min: lower limit of search range
    :type x_min: float

    :arg x_max: upper limit of search range     
    :type x_max: float

    :arg epsilon: tolerance for algorithm convergence   
    :type epsilon: float

    ``x_min`` and ``x_max`` delimit a range containing a single root 
    (ie, the function must cross the x-axis just once inside the range).

    .. note::
    
        *   A :class:`RuntimeError` is raised if the search algorithm fails to converge.
        
        *   An :class:`AssertionError` is raised if preconditions are not satisfied.

    **Example**::
    
        >>> near_unity = ureal(1,0.05)
        >>> fn = lambda x: x**2 - near_unity
        >>> function.implicit(fn,0,2)
        ureal(1.0,0.025...,inf)

    .. versionadded:: 1.3.4
    
    """
    return implicit_real(fn,x_min,x_max,epsilon)
    
#------------------------------------------------------------------------
def implicit_real(fn,x_min,x_max,epsilon):
    """Return the uncertain real number ``x``, that solves :math:`fn(x) = 0`

    The function fn() must take a single argument and x_min and
    x_max must define a range in which there is one (and only one)
    sign change in fn(x).

    The number 'epsilon' is a tolerance for accepting convergence.
    
    A RuntimeError will be raised if the root-search algorithm fails.

    An AssertionError will be raised if the preconditions for a 
    search to begin are not satisfied.

    Parameters
    ----------
    fn : a function of one argument
    x_min, x_max, epsilon : float

    Returns
    -------
    UncertainReal

    .. versionadded:: 1.3.4

    """
    xk,dy_dx = nr_get_root(fn,x_min,x_max,epsilon)

    # In an implicit function F(x,...) = 0, where
    # we solve F = 0 by finding a value for `x`,
    # `x` depends implicitly on the other arguments. 
    # The influence set of `F` and the influence set 
    # of `x` should are the same. 
    # `x` is not an elementary uncertain number;  
    # it is implicitly a function of the other 
    # arguments to F().

    # The components of uncertainty of `x` are related to 
    # the components of `F` as follows:
    #       u_i(x) = -( dF/dx_i / dF/dx ) * u_i(xi)

    y = fn( UncertainReal._constant(xk) )
    dx_dy = -1/dy_dx
    
    return UncertainReal(
        xk
    ,   vector.scale_vector(y._u_components,dx_dy)
    ,   vector.scale_vector(y._d_components,dx_dy)
    ,   vector.scale_vector(y._i_components,dx_dy)
    )     
    
#---------------------------------------------------------
# Newton-Raphson method with bisection.
# Based on Numerical Recipes in C, Ch 9, Section 4,
# but using uncertain numbers to evaluate the derivatives.
#
def nr_get_root(fn,x_min,x_max,epsilon):
    """Return the x-location of the root and the derivative at that point.
    
    This is a utility function used by implicit_real(). 
    It searches within the range for a real root.
    
    Parameters
    ----------
    fn : a function with a single argument
    x_min, x_max, epsilon : float

    Returns
    -------
    (float,float)

    .. versionadded:: 1.3.4

    """
    if x_max <= x_min:      
        raise RuntimeError(
            "Invalid search range: {!s}".format((x_min,x_max))
        )
           
    
    lower, upper = x_min, x_max

    ureal = lambda x,u: UncertainReal._elementary(x,u,inf,None,True)
    value = lambda x: x.x if isinstance(x,UncertainReal) else float(x)
    
    x = ureal(lower,1.0)
    f_x = fn(x) 
    fl = value(f_x)
    
    assert isinstance(f_x,UncertainReal),\
           "fn() must return an UncertainReal, got: %s" % type(f_x)
    
    if abs( fl ) < epsilon:
        return fl,f_x.sensitivity(x)
    
    x = ureal(upper,1.0)
    f_x = fn(x) 
    fu = value(f_x)

    if abs( fu ) < epsilon:
            return fu,f_x.sensitivity(x)

    if fl * fu >= 0.0:
        raise RuntimeError(
           "range does not appear to contain a root: {}".format((fl,fu))
        )

    if fl > 0.0:
        lower, upper = upper, lower

    # First place to look is the middle
    xk = (lower+upper) / 2.0  
    dx2 = abs(upper - lower)
    dx = dx2

    x = ureal(xk,1.0)
    f_x = fn(x) 
    f = value( f_x )
    df = f_x.sensitivity(x)

    for i in xrange(100): 
        if  (((xk-upper) * df-f) * ((xk-lower) * df - f) > 0.0
            or  ( abs(2.0 * f) > abs(dx2 * df) )):
            # Bisect if Newton out of range or
            # not decreasing fast enough.
            dx2 = dx
            dx = (upper - lower) / 2.0

            # If the bisection is too small then the root is found
            if(abs(dx) <= epsilon):
                return xk,df
            else:
                xk = lower + dx

        else:
            # Use Newton step
            dx2 = dx
            dx = f / df

            # If the change is ~ 0 then accept the root
            if(abs(dx) <= epsilon):
                return xk,df
            else:
                xk -= dx

        # Test convergence
        if(abs(dx) <= epsilon):
            return xk,df
  
        # Evaluate for next iteration;
        x = ureal(xk,1.0)
        f_x = fn(x) 
        f = value( f_x )
        df = f_x.sensitivity(x)

        if(f < 0.0):
            lower = xk
        else:
            upper = xk
            
    raise RuntimeError("Failed to converge") 
# ===========================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *
    doctest.testmod(  optionflags= doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS  )
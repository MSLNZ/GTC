try:
    xrange  # Python 2
except NameError:
    xrange = range

from GTC import inf, vector
from GTC.lib import UncertainReal 

__all__ = ('implicit',)

#---------------------------------------------------------------------------
def implicit(fn,x_min,x_max,epsilon=1E-13):
    """Return the solution to :math:`f(x) = 0` 
        
    :arg fn: a user-defined function
    
    :arg x_min: lower limit of search range
    :type x_min: float

    :arg x_max: upper limit of search range     
    :type x_max: float

    :arg epsilon: tolerance for algorithm convergence   
    :type epsilon: float

    The user-defined function ``fn`` takes a single uncertain real 
    number argument.

    ``x_min`` and ``x_max`` delimit a range containing a single root 
    (ie, the function must cross the x-axis just once inside the range).

    .. note::
    
        *   A :class:`RuntimeError` is raised if the search algorithm fails to converge.
        
        *   An :class:`AssertionError` is raised if preconditions are not satisfied.

    **Example**::
    
        >>> near_unity = ureal(1,0.05)
        >>> fn = lambda x: x**2 - near_unity
        >>> function.implicit(fn,0,2)
        ureal(1,0.025000000000000001,inf)

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
    fn : a function taking one argument
    x_min, x_max, epsilon : float

    Returns
    -------
    UncertainReal
    
    """
    xk,dy_dx = nr_get_root(fn,x_min,x_max,epsilon)

    # `x` depends implicitly on the other arguments, 
    # so the influence set of `y` and the influence set
    # of `x` are the same. But `x` is not an elementary 
    # uncertain number.

    # The components of uncertainty of `x` are related to 
    # the components of `y` as follows:
    #       u_i(x) = -( dy/dx_i / dy/dx ) * u_i(xi)

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
            dx2 = dx;
            dx = (upper - lower) / 2.0;

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
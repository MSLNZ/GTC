import sys 

from GTC import *
from GTC import context

__all__ = ('implicit',)

#---------------------------------------------------------------------------
def implicit(fn,x_min,x_max,epsilon=sys.float_info.epsilon):
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
    fn : a function with a single argument
    x_min, x_max, epsilon : float

    Returns
    -------
    UncertainReal
    
    """
    # TODO: make sure this still does what we want
    xk,dy_dx = nr_get_root(fn,x_min,x_max,epsilon,context)

    # `x` depends implicitly on the other arguments, 
    # so the influence set of `y` and the influence set
    # of `x` are the same (except for `x` itself).

    # The components of uncertainty of `x` are related to 
    # the components of `y` as follows:
    #       u_i(x) = -( dy/dx_i / dy/dx ) * u_i(xi)

    # A constant is: UncertainReal(x,Vector(),Vector(),Node(),context)
    # Now: UncertainReal._constant(float(x),label)
    y = fn( context.constant_real(xk,None) )
    dx_dy = -1/dy_dx
    
    return UncertainReal(
        xk
    ,   scale_vector(y._u_components,dx_dy)
    ,   scale_vector(y._d_components,dx_dy)
    ,   scale_vector(y._i_components,dx_dy)
    ,   Node( (y._node,dx_dy) )
    # ,   context
    )
    
    # In the Node constructor, the tuple is a node and the derivative of
    # the parent with respect to the child. This made the `x` returned 
    # a function of the nodes than influenced `y`.
    
    # In the new GTC Node() would be an intermediate result and 
    # the second argument its combined standard uncertainty. This 
    # probably won't do what we want.
    
#---------------------------------------------------------
# Newton-Raphson method with bisection.
# See NR Ch 9, Section 4.
#
def nr_get_root(fn,x_min,x_max,epsilon,context):
    """Return the x-location of the root and the derivative at that point.
    
    A utility function used by implicit_real(). It searches within the
    range for a real root.
    
    Parameters
    ----------
    fn : a function with a single argument
    x_min, x_max, epsilon : float
    context : Context

    Returns
    -------
    (float,float)

    """
    # TODO
    #    how to handle: elementary_real, ._node.partial_derivative, value 
    assert x_max > x_min,\
           "Invalid search range: %s" % str((x_min,x_max))
    
    lower, upper = x_min, x_max

    # UncertainReal._elementary(
                # float(x),
                # float(u),
                # float(df),
                # label,
                # independent
            # )
    # "elementary_real(self,x,u,df,label,dependent)"
    ureal = lambda x,u: context.elementary_real(
            x,u,inf,None,False # No label and independent
    )
    value = lambda x: float(x)
    
    x = ureal(lower,1.0)
    f_x = fn(x) 
    fl = value(f_x)
    assert isinstance(f_x,UncertainReal),\
           "fn() must return an UncertainReal, got: %s" % type(f_x)
    
    # Don't have ._node.partial_derivative() now
    # But there is y.sensitivity(x)
    # if we know that `x` is elementary and independent then 
    # sensitivity = y._u_components.get(n,0.0) / n.u, where n = x._node
    #
    if abs( fl ) < epsilon:
        return fl,f_x._node.partial_derivative(x._node)
    
    x = ureal(upper,1.0)
    f_x = fn(x) 
    fu = value(f_x)
    if abs( fu ) < epsilon:
        return fu,f_x._node.partial_derivative(x._node)

    assert fl * fu < 0.0,\
           "range does not appear to contain a root: %s" % str((fl,fu))

    if fl > 0.0:
        lower, upper = upper, lower

    # First place to look is the middle
    xk = (lower+upper) / 2.0  
    dx2 = abs(upper - lower)
    dx = dx2

    x = ureal(xk,1.0)
    f_x = fn(x) 
    f = value( f_x )
    df = f_x._node.partial_derivative(x._node)

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
        df = f_x._node.partial_derivative(x._node)

        if(f < 0.0):
            lower = xk
        else:
            upper = xk
            
    raise RuntimeError,"Failed to converge"
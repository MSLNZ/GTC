"""
Utility functions
-----------------
Functions :func:`complex_to_seq` and :func:`seq_to_complex` 
are useful to convert between the matrix representation of 
complex numbers and Python :obj:`complex`.

The function :func:`mean` evaluates the mean of a sequence.

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
    xrange = range

try:
    import builtins    # Python 3
except ImportError: 
    import __builtin__ as builtins
    
try:
    from collections.abc import Iterable  # Python 3
except ImportError:
    from collections import Iterable

from GTC import is_sequence
from GTC.named_tuples import InterceptSlope

def value(x):
    try:
        return x.x
    except AttributeError:
        return x
        
__all__ = (
    'complex_to_seq',
    'seq_to_complex',
    'mean',
    'line_fit',    
)

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
  Intercept: {}
  Slope: {}
  Correlation: {:.2G}
  Sum of the squared residuals: {}
  Number of points: {}
'''.format(
    a.s,
    b.s,
    get_correlation_real(a,b),
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

#--------------------------------------------------------------------
#
def line_fit(x,y):
    """Least-squares fit intercept and slope 
    
    :arg x:     sequence of independent variable data 
    :arg y:     sequence of dependent variable data

    Returns a :class:`~function.LineFitOLS` object
    
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
        analyses can be merged (see :func:`type_a.merge_components`).
        
    **Example**::

        >>> a0 =10
        >>> b0 = -3
        >>> u0 = .2

        >>> x = [ float(x_i) for x_i in xrange(10) ]
        >>> y = [ ureal(b0*x_i + a0,u0) for x_i in x ]

        >>> a,b = fn.line_fit(x,y).a_b
        >>> a
        ureal(10,0.1175507627290518,inf)
        >>> b
        ureal(-3,0.022019275302527213,inf)
        
    """  
    S = len(x) 
    S_x = sum( x ) 
    S_y = sum( y )

    k = S_x / S
    t = [ x_i - k for x_i in x ]

    S_tt = sum( t_i*t_i for t_i in t )
    
    b = sum( t_i*y_i/S_tt for t_i,y_i in izip(t,y) )
    a = (S_y - b*S_x)/S
    
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
    
    If ``z = x + yj``, then an array of the form ``[[x,-y],[y,x]]`` 
    can be used to represent ``z`` in matrix computations. 

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
    TOL = 1E-16
    if hasattr(seq,'shape'):
        if seq.shape != (2,2):
            raise RuntimeError("array shape illegal: {}".format(seq))
        elif (
            math.fabs( seq[0,0] - seq[1,1] ) > TOL
        or  math.fabs( seq[1,0] + seq[0,1] ) > TOL ):
            raise RuntimeError("ill-conditioned sequence: {}".format(seq))
        else:
            seq = list( seq.flat )
            
    elif is_sequence(seq):
        if len(seq) != 4:
            raise RuntimeError("sequence must have 4 elements: {}".format(seq))
        elif (
            math.fabs( seq[0] - seq[3] ) > TOL
        or  math.fabs( seq[1] + seq[2] ) > TOL ):
            raise RuntimeError("ill-conditioned sequence: {}".format(seq))
    
    else:
        raise RuntimeError("illegal argument: {}".format(seq))

    return complex(seq[0],seq[2])

# ===========================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *
    doctest.testmod(  optionflags=doctest.NORMALIZE_WHITESPACE  )
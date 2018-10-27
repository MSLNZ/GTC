"""
Utility functions
-----------------
The functions :func:`complex_to_seq` and :func:`seq_to_complex` 
support a matrix representation of complex numbers.

The function :func:`mean` evaluates the mean of a sequence.

Module contents
---------------

"""
from __future__ import division

import math
import collections
import itertools

from GTC.lib import (
    UncertainReal,
    UncertainComplex,
    z_to_seq
)

from GTC import is_sequence

__all__ = (
    'complex_to_seq',
    'seq_to_complex',
    'mean'
    )
        
#---------------------------------------------------------------------------
def mean(seq):
    """Return the arithmetic mean of elements in `seq`
    
    `seq` - an iterable
    
    If the elements of `seq` are uncertain numbers, 
    an uncertain number is returned.
    
    **Example** ::
    
        >>> seq = [ ureal(1,1), ureal(2,1), ureal(3,1) ]
        >>> function.mean(seq)
        ureal(2.0,0.5773502691896257,inf)
        
    """
    if is_sequence(seq):
        return sum(seq)/len(seq)
    elif isinstance(seq,collections.Iterable):
        seq = list(seq)
        return sum(seq)/len(seq)
    else:
        raise RuntimeError(
            "{!r} is not iterable".format(seq)
        )
#---------------------------------------------------------------------------
def complex_to_seq(z):
    """Transform a complex number into a 4-element sequence

    :arg z: a number

    If ``z = x + yj``, then ``matrix([[x,-y],[y,x]])`` 
    can be used to represent ``z`` in matrix computations. 

    **Examples**::
        >>> import numpy
        >>> z = 1 + 2j
        >>> function.complex_to_seq(z)
        (1.0, -2.0, 2.0, 1.0)
        
        >>> m = numpy.matrix( function.complex_to_seq(z) )
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

    A :obj:`RuntimeError` will be raised if ``seq`` is ill-conditioned.
    
    If ``z = x + yj``, then ``matrix([[x,-y],[y,x]])`` can
    be used to represent ``z`` in matrix computations. 

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
"""
Real-valued problems
--------------------
    
    Functions are provided that convert the half-width   
    of a one-dimensional distribution to a standard 
    uncertainty:
    
    *   :func:`uniform`
    *   :func:`triangular`
    *   :func:`u_shaped` 
    *   :func:`arcsine` 

Complex-valued problems
-----------------------
    
    The following functions convert information
    about two-dimensional distributions into
    standard uncertainties:
    
    *   :func:`uniform_ring`
    *   :func:`uniform_disk`
    *   :func:`unknown_phase_product`

A table of distributions
------------------------

    The mapping :obj:`distribution` is provided so that 
    the functions above can be selected by name. 
    For example, ::

        >>> a = 1.5
        >>> ureal( 1, type_b.distribution['gaussian'](a) )
        ureal(1.0,1.5,inf)
        >>> ureal( 1, type_b.distribution['uniform'](a) )
        ureal(1.0,0.8660254037844387,inf)
        >>> ureal( 1, type_b.distribution['arcsine'](a) )
        ureal(1.0,1.0606601717798212,inf)

    Keys to :obj:`distribution` are (case-sensitive):
    
    *   'gaussian'
    *   'uniform'
    *   'triangular'
    *   'arcsine' or 'u_shaped'
    *   'uniform_ring'
    *   'uniform_disk'
    
Module contents
---------------

"""
from __future__ import division

import math

__all__ = (
    'uniform',
    'triangular',
    'u_shaped',
    'arcsine',
    'uniform_ring',
    'uniform_disk',
    'unknown_phase_product',
    'distribution'
)

_root_2 = math.sqrt(2.0)
_root_3 = math.sqrt(3.0)
_root_6 = math.sqrt(6.0)

#---------------------------------------------------------------------------
def uniform(a):
    """Return the standard uncertainty for a uniform distribution. 

    :arg float a: the half-width

    **Example**::

        >>> x = ureal(1,type_b.uniform(1))
        >>> x
        ureal(1.0,0.5773502691896258,inf)
    
    """
    return a/_root_3

#---------------------------------------------------------------------------
def triangular(a):
    """Return the standard uncertainty for a triangular distribution. 
    
    :arg float a: the half-width 
    
    **Example**::

        >>> x = ureal(1,type_b.triangular(1))
        >>> x
        ureal(1.0,0.4082482904638631,inf)
        
    """
    return a/_root_6

#---------------------------------------------------------------------------
def arcsine(a):
    """Return the standard uncertainty for an arcsine distribution. 

    :arg float a: the half-width 
    
    **Example**::

        >>> x = ureal(1,type_b.arcsine(1))
        >>> x
        ureal(1.0,0.7071067811865475,inf)

    """
    return a/_root_2

# Aliases for the arcsine function
u_shaped = arcsine

#---------------------------------------------------------------------------
def uniform_ring(a):
    """Return the standard uncertainty for a uniform ring
    
    :arg float a: the radius
    
    Convert the radius of a uniform ring distribution ``a`` 
    to a standard uncertainty

    See reference: B D Hall, *Metrologia* **48** (2011) 324-332
    
    **Example**::

        >>> z = ucomplex( 0, type_b.uniform_ring(1) )
        >>> z
        ucomplex((0+0j), u=[0.7071067811865475,0.7071067811865475], r=0.0, df=inf)
        
    """
    return arcsine(a)

#---------------------------------------------------------------------------
def uniform_disk(a):
    """Return the standard uncertainty for a uniform disk 
    
    :arg float a: the radius
    
    Convert the radius of a uniform disk distribution ``a`` 
    to a standard uncertainty.
    
    See reference: B D Hall, *Metrologia* **48** (2011) 324-332

    **Example**::

        >>> z = ucomplex( 0, type_b.uniform_disk(1) )
        >>> z
        ucomplex((0+0j), u=[0.5,0.5], r=0.0, df=inf)
        
    """
    return a / 2.0

# #---------------------------------------------------------------------------
# # 
# def uncertain_ring(a_u_r):
    # """Return the standard uncertainty for an uncertain ring 
        
    # :arg a_u_r: the (estimated) radius with a standard uncertainty 
    # :type a_u_r: pair of float
    
    # Convert a radius estimate ``a``, with a standard
    # uncertainty ``u_r``, into a standard uncertainty.

    # See reference: B D Hall, *Metrologia* **48** (2011) 324-332

    # **Example**::

        # >>> estimate = (1,0.1)
        # >>> z = ucomplex( 0, type_b.uncertain_ring( estimate ) )
        # >>> z
        # ucomplex((0+0j), u=[0.714142842854285,0.714142842854285], r=0.0, df=inf)

    # .. note::
    
        # This function is deprecated. 
        
    # """
    # a, u_r = a_u_r
    # return math.sqrt( a**2/2.0 + u_r**2 )

#---------------------------------------------------------------------------
def unknown_phase_product(u1,u2):
    """Return the standard uncertainty for a product when phases are unknown

    :arg u1: the standard uncertainty of the first multiplicand
    :arg u2: the standard uncertainty of the second multiplicand
    
    Obtains the standard uncertainty associated
    with a complex product when estimates have unknown phase.

    The arguments ``u1`` and ``u2`` are the standard  
    uncertainties associated with each multiplicand.  
    
    See reference: B D Hall, *Metrologia* **48** (2011) 324-332

    **Example**::
    
        # X = Gamma1 * Gamma2
        >>> X = ucomplex( 0, type_b.unknown_phase_product(.1,.1) )
        >>> X
        ucomplex((0+0j), u=[0.014142135623730954,0.014142135623730954], r=0.0, df=inf)

    """
    return _root_2 * u1 * u2
   
#---------------------------------------------------------------------------
distribution = {
    'gaussian'  : lambda x: x,
    'uniform'   : uniform,
    'triangular': triangular,
    'arcsine'   : arcsine,
    'u_shaped'  : u_shaped,
    'uniform_ring'  : uniform_ring,
    'uniform_disk'  : uniform_disk,
    # 'uncertain_ring': uncertain_ring
}
   
#============================================================================
if __name__ == "__main__":
    import doctest
    from GTC import *
    doctest.testmod()

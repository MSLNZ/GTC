.. _core_functions:

==========================
Core Functions and Classes
==========================

.. contents::
   :local:

.. _core_module:

Core Functions
==============

A set of core mathematical functions, together with functions that create elementary uncertain numbers and functions that can be used to access uncertain number attributes, are defined in the :mod:`core` module. These functions are automatically imported into the ``GTC`` namespace, so they are available after performing ``from GTC import *``. 


.. automodule:: core
	:members: 
	:inherited-members:

Uncertain Number Types
======================

There are two types of uncertain number, one to represent real-valued quantities (:class:`~lib.UncertainReal`) and one to represent real-complex quantities (:class:`~lib.UncertainComplex`). 

.. _uncertain_real_number:

Uncertain Real Numbers
----------------------

    The :class:`~lib.UncertainReal` class defines an uncertain-number object with the attributes ``x``, ``u``, ``v`` and ``df``, 
    to obtain the value, uncertainty, variance and degrees-of-freedom for the uncertain number, respectively. 

    The function :func:`~core.ureal` creates elementary :class:`~lib.UncertainReal` objects. For example, ::
    
        >>> x = ureal(1.414141,0.01)
        >>> x
        ureal(1.414141,0.01,inf)

    All logical comparison operations (e.g., <, >, ==, etc) are applied to the *value* of an uncertain number. For example, ::
    
        >>> un = ureal(2.5,1)
        >>> un > 3
        False
        >>> un == 2.5
        True
    
    When the value of an :class:`~lib.UncertainReal` is converted to a string (e.g., by :class:`str`, or by :func:`print`), the precision depends on the uncertainty. The two least significant digits of the value correspond to the two most significant digits of the standard uncertainty. The value of standard uncertainty is appended to the string in parentheses.    
    
    For example, ::
	
        >>> x = ureal(1.414141,0.01)
        >>> str(x) 
        '1.414(10)'
        >>> print(x)
        1.414(10)
	
    When an :class:`~lib.UncertainReal` is converted to its Python *representation* (e.g., by :func:`repr`) a string is returned that shows  the representation of the elements that define the uncertain number.  
    
    For example, ::

        >>> x = ureal(1.4/3,0.01,5,label='x')
        >>> repr(x)
        "ureal(0.4666666666666666,0.01,5.0, label='x')"

.. autoclass:: lib.UncertainReal
   :members: conjugate, df, imag, real, label, u, v, x

Uncertain Complex Numbers
-------------------------
	
    The class :class:`~lib.UncertainComplex` defines an uncertain-number object with the attributes ``x``, ``u``, ``v`` and ``df``, 
    to obtain the value, uncertainty, variance-covariance matrix and degrees-of-freedom, respectively.

    The function :func:`~core.ucomplex` creates elementary :class:`~lib.UncertainComplex` objects, 
    for example ::  
    
        >>> z = ucomplex(1.333-0.121212j,(0.01,0.01))
        
    Equality comparison operations (``==`` and ``!=``) are applied to the *value* of uncertain complex numbers. 
    For example, ::
    
        >>> uc = ucomplex(3+3j,(1,1))
        >>> uc == 3+3j
        True
    
    The built-in function :func:`abs` returns the magnitude of the *value* of the uncertain number (use :func:`~core.magnitude` if uncertainty propagation is required). For example, ::

        >>> uc = ucomplex(1+1j,(1,1))
        >>> abs(uc)
        1.4142135623730951

        >>> magnitude(uc)
        ureal(1.4142135623730951,0.9999999999999999,inf)
                   
    When an :class:`~lib.UncertainComplex` is converted to a string (e.g., by the :class:`str` function or by :func:`print`), the precision depends on the uncertainty. 
    
    The lesser of the uncertainties in the real and imaginary components is used for formatting. The two least significant digits of the formated component values will correspond to the two most significant digits of this standard uncertainty. Values of standard uncertainty are appended to the component values in parentheses.
    
    For example, ::
	
        >>> z = ucomplex(1.333-0.121212j,(0.01,0.002))
        >>> print(z)
        (1.3330(100)-0.1212(20)j)
	
    
    When an :class:`~lib.UncertainComplex` is converted to its Python *representation* ( e.g., by :func:`repr` ), a string is returned that shows the representation of the elements that define the uncertain number. 
    
    For example, ::

        >>> z = ucomplex(1.333-0.121212j,(0.01,0.002))
        >>> repr(z)
        'ucomplex((1.333-0.121212j), u=[0.01,0.002], r=0.0, df=inf)'	

.. autoclass:: lib.UncertainComplex
    :members: conjugate, df, imag, real, label, u, v, x, r
   


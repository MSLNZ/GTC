.. _core_functions:

==========================
Core Functions and Classes
==========================

.. contents::
   :local:

.. _core_module:

Core Functions
==============

Functions that create elementary uncertain numbers and functions that access uncertain-number attributes, are defined in the :mod:`core` module. There is also a set of standard mathematical functions (e.g.: :func:`~core.sqrt`, :func:`~core.sin`, :func:`~core.log10`, etc) for uncertain numbers. These functions can be applied to the numeric Python types too.

All :mod:`core` functions are automatically imported into the ``GTC`` namespace (i.e., they are available after ``from GTC import *``).

.. automodule:: core
	:members: 
	:inherited-members:

.. _uncertain_number_types:

Uncertain Number Types
======================

There are two types of uncertain number, one to represent real-valued quantities (:class:`~lib.UncertainReal`) and one to represent real-complex quantities (:class:`~lib.UncertainComplex`). 

.. _uncertain_real_number:

Uncertain Real Numbers
----------------------

    :class:`~lib.UncertainReal` defines an uncertain-number object with attributes ``x``, ``u``, ``v`` and ``df``, 
    for the value, uncertainty, variance and degrees-of-freedom, respectively, of the uncertain number. 

    The function :func:`~core.ureal` creates elementary :class:`~lib.UncertainReal` objects. For example, ::
    
        >>> x = ureal(1.414141,0.01)
        >>> x
        ureal(1.414141,0.01,inf)

    All logical comparison operations (e.g., <, >, ==, etc) applied to uncertain-number objects use the *value* attribute. For example, ::
    
        >>> un = ureal(2.5,1)
        >>> un > 3
        False
        >>> un == 2.5
        True
    
    When the value of an :class:`~lib.UncertainReal` is converted to a string (e.g., by :class:`str`, or by :func:`print`), the precision displayed depends on the uncertainty. The two least significant digits of the value correspond to the two most significant digits of the standard uncertainty. The value of standard uncertainty is appended to the string between parentheses.    
    
    For example, ::
	
        >>> x = ureal(1.414141,0.01)
        >>> str(x) 
        ' 1.414(10)'
        >>> print(x)
        1.414(10)
	
    When an :class:`~lib.UncertainReal` is converted to its Python *representation* (e.g., by :func:`repr`) a string is returned that shows the representation of the elements that define the uncertain number.  
    
    For example, ::

        >>> x = ureal(1.4/3,0.01,5,label='x')
        >>> repr(x)
        "ureal(0.4666666666666666,0.01,5.0, label='x')"

.. autoclass:: lib.UncertainReal
   :members: conjugate, df, imag, real, label, u, v, x, uid

Uncertain Complex Numbers
-------------------------
	
    :class:`~lib.UncertainComplex` defines an uncertain-number object with attributes ``x``, ``u``, ``v`` and ``df``, 
    for the value, uncertainty, variance-covariance matrix and degrees-of-freedom, respectively.

    The function :func:`~core.ucomplex` creates elementary :class:`~lib.UncertainComplex` objects, 
    for example ::  
    
        >>> z = ucomplex(1.333-0.121212j,(0.01,0.01))
        
    Equality comparison operations (``==`` and ``!=``) applied to uncertain-complex-number objects use the *value* attribute. 
    For example, ::
    
        >>> uc = ucomplex(3+3j,(1,1))
        >>> uc == 3+3j
        True
    
    The built-in function :func:`abs` returns the magnitude of the *value* as a Python ``float`` (use :func:`~core.magnitude` if uncertainty propagation is required). For example, ::

        >>> uc = ucomplex(1+1j,(1,1))
        >>> abs(uc)
        1.4142135623730951

        >>> magnitude(uc)
        ureal(1.4142135623730951,0.9999999999999999,inf)
                   
    When an :class:`~lib.UncertainComplex` is converted to a string (e.g., by the :class:`str` function or by :func:`print`), the precision depends on the uncertainty. 
    
    The lesser of the uncertainties in the real and imaginary components will determine the precision displayed. The two least significant digits of the formated component values will correspond to the two most significant digits of this standard uncertainty. Values of standard uncertainty are appended to the component values between parentheses.
    
    For example, ::
	
        >>> z = ucomplex(1.333-0.121212j,(0.01,0.002))
        >>> print(z)
        (+1.3330(100)-0.1212(20)j)
	
    
    When an :class:`~lib.UncertainComplex` is converted to its Python *representation* ( e.g., by :func:`repr` ), a string is returned that shows the representation of the elements that define the uncertain number. 
    
    For example, ::

        >>> z = ucomplex(1.333-0.121212j,(0.01,0.002))
        >>> repr(z)
        'ucomplex((1.333-0.121212j), u=[0.01,0.002], r=0.0, df=inf)'	

.. autoclass:: lib.UncertainComplex
    :members: conjugate, df, imag, real, label, u, v, x, r, uid
   

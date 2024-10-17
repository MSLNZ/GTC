.. _linear_algebra:

==============
Linear Algebra
==============

This module provides support for calculations using arrays containing uncertain numbers.

The shorter name ``la`` has been defined as an alias for ``linear_algebra``,
to resolve the names of objects defined in this module.

Arrays of Uncertain Numbers
===========================

:class:`.UncertainArray` is a convenient container of uncertain numbers. The preferred way to create arrays is the function :func:`~.linear_algebra.uarray`.

An array can contain a mixture of :class:`~.lib.UncertainReal`, 
:class:`~.lib.UncertainComplex` and Python numbers (:class:`int`, :class:`float` and :class:`complex`).  

The usual mathematical operations can be applied to an array. For instance, if :code:`A` and :code:`B` have the same size, they can be added :code:`A + B`, subtracted :code:`A - B`, etc; or a function like :code:`sqrt(A)` can be applied. This vectorisation provides a succinct notation for repetitive operations 
but it does not offer a significant speed advantage over Python iteration. 

.. note::

    To evaluate the product of two-dimensional arrays representing matrices, the 
    function :func:`~linear_algebra.matmul` should be used (for Python 3.5 and above the 
    built-in binary operator ``@`` is an alternative). For example::
    
        >>> a = la.uarray([[1.1,.5],[ureal(3,1),.5]])
        >>> b = la.uarray([[5.2,ucomplex(4,1)],[.1,.1+3j]])
        >>> la.matmul(a,b)
        uarray([[5.7700000000000005,
                 ucomplex((4.45+1.5j), u=[1.1,1.1], r=0.0, df=inf)],
                [ureal(15.650000000000002,5.2,inf),
                 ucomplex((12.05+1.5j), u=[5.0,3.0], r=0.0, df=inf)]])

.. automodule:: linear_algebra
	:members: 
	:inherited-members:

.. autoclass:: uncertain_array.UncertainArray
   :members: value, uncertainty, variance, dof, label, real, imag, r, conjugate

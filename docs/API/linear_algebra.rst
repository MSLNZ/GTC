.. _linear_algebra:

==============
Linear Algebra
==============

This module provides support for calculations using arrays and matrices that contain uncertain numbers.

Arrays of Uncertain Numbers
===========================

:class:`.UncertainArray` is a convenient container of uncertain numbers. It provides succinct expressions for the application of standard mathematical operations to array elements. 
For instance, if :code:`A` and :code:`B` are arrays of the same size, they can be added :code:`A + B`, or subtracted :code:`A - B`, etc; or a function like :code:`sqrt(A)` can be applied to each element, and so on. An :class:`.UncertainArray` also provides excellent support for manipulating stored data.

The function :func:`~.linear_algebra.uarray`, which is available by default in the **GTC** namespace, is used to create a :class:`.UncertainArray`. 

:ref:`numpy-uarray` provides more information and some examples using :class:`.UncertainArray`.


.. autoclass:: uncertain_array.UncertainArray
   :members: value, uncertainty, variance, dof, label, real, imag, r, conjugate

The shorter name ``la`` has been defined as an alias for ``linear_algebra``,
to resolve the names of objects defined in this module.

.. automodule:: linear_algebra
	:members: 
	:inherited-members:

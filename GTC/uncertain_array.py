"""
The proper way to create an uncertain array is by calling :func:`.uarray`
"""
# Adding numpy arrays to GTC is not an easy exercise.
# Our need is to provide convenient containers for uncertain numbers.
# We do not try to integrate uncertain numbers in numpy's design.
from __future__ import division

import warnings

from numbers import Number, Real, Complex
from math import isnan, isinf
from cmath import isnan as cisnan
from cmath import isinf as cisinf
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

import numpy as np

from GTC import is_sequence
from GTC.linear_algebra import matmul

from GTC.core import (
    value,
    uncertainty,
    variance,
    dof,
    cos,
    sin,
    tan,
    acos,
    asin,
    atan,
    atan2,
    exp,
    log,
    log10,
    sqrt,
    sinh,
    cosh,
    tanh,
    acosh,
    asinh,
    atanh,
    mag_squared,
    magnitude,
    phase,
    result,
)

from GTC.lib import (
    UncertainReal,
    UncertainComplex
)


def _isnan(number):
    val = value(number)
    if isinstance(val, Real):
        return isnan(val)
    elif isinstance(val, Complex):
        return cisnan(val)
    else:
        raise TypeError('cannot calculate isnan of type {}'.format(type(number)))


def _isinf(number):
    val = value(number)
    if isinstance(val, Real):
        return isinf(val)
    elif isinstance(val, Complex):
        return cisinf(val)
    else:
        raise TypeError('cannot calculate isinf of type {}'.format(type(number)))

# Note numpy defines its own numeric types, instead of bool, int,
# float, complex, that have additional attributes. These types are needed by
# functions like `numpy.average`. (Uses `dtype` and `.size` attributes
# on the result returned by `mean`, as defined in a subclass if available.)

# One way to fix this is to add the required attributes
# to all the return values from `UncertainArray` methods.

# Another option is to ensure that array elements
# are always numpy-compatible and to ensure that all
# uncertain number objects are initialised with
#           a.dtype = np.dtype('O')
#           a.size = 1
#           a.shape = ()
#
# Our use of `dtype=object` for arrays means that numeric
# elements are not cast to numpy types when loaded into an array.
# To fix this would require iteration through all arrays as they
# are being created!

#--------------------------------------------------------------------
class UncertainArray(np.ndarray):
    """An :class:`UncertainArray` can contain elements of type
    :class:`int`, :class:`float`, :class:`complex`,
    :class:`.UncertainReal` or :class:`.UncertainComplex`.

    Do not instantiate this class directly. Use :func:`~.uarray` instead.

    Base: :class:`numpy.ndarray`

    .. versionadded:: 1.1

    """
    def __new__(cls, array, dtype=None, label=None):
        # The first case allows users to create uarray instances
        # with a definite numpy number type. This could be done
        # by wrapping a call to uarray() around an ndarray.
        # Without this, the type gets converted back to Python.
        if isinstance(array, np.ndarray):
            dtype = array.dtype
        elif dtype is None:
            dtype = np.dtype('O')

        obj = np.asarray(array, dtype=dtype).view(cls)
        obj._label = label
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._label = getattr(obj, 'label', None)

        # numpy looks at type().__name__ when preparing
        # a string representation of the object. This
        # change means we see `uarray` not `UncertainArray`.
        self.__class__.__name__ = 'uarray'

        self._broadcasted_shape = None

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        try:
            attr = getattr(self, '_' + ufunc.__name__)
        except AttributeError:
            # Want to raise a NotImplementedError without nested exceptions
            # In Python 3 this could be achieved by "raise Exception('...') from None"
            attr = None

        if attr is None:
            raise NotImplementedError(
                'The {} function has not been implemented'.format(ufunc)
            )

        if kwargs:
            warnings.warn('**kwargs, {}, are currently not supported'
                          .format(kwargs), stacklevel=2)

        case = len(inputs)
        if case == 1:
            pass  # Must be an UncertainArray
        elif case == 2:
            # At least 1 of the inputs must be an UncertainArray
            # If an input is not an ndarray then convert it to be an ndarray
            not0 = not isinstance(inputs[0], np.ndarray)
            if not0 or not isinstance(inputs[1], np.ndarray):
                # A tuple cannot be modified
                # This does not create a copy of the items
                inputs = list(inputs)
                # convert the input that is not an ndarray
                convert, keep = (0, 1) if not0 else (1, 0)
                if isinstance(inputs[convert], (Number, UncertainReal, UncertainComplex)):
                    inputs[convert] = np.full(inputs[keep].shape, inputs[convert], dtype=object)
                else:
                    inputs[convert] = np.asarray(inputs[convert], dtype=object)

            self._broadcasted_shape = None
            if inputs[0].shape != inputs[1].shape:
                broadcasted = np.broadcast(*inputs)
                inputs = broadcasted.iters
                self._broadcasted_shape = broadcasted.shape

        else:
            assert False, 'Should not occur: __array_ufunc__ received {} inputs'.format(case)

        return attr(*inputs)

    def __repr__(self):
        # Use the numpy formatting but hide the default dtype
        np_array_repr = np.array_repr(self)

        if self.dtype == object:
            # Truncate string from trailing ','
            i = np_array_repr.rfind(',')
            return np_array_repr[:i] + ')'
        else:
            return np_array_repr

    def __matmul__(self, other):
        # Implements the protocol used by the '@' operator defined in PEP 465.
        return matmul(self, other)

    def __rmatmul__(self, other):
        # Implements the protocol used by the '@' operator defined in PEP 465.
        return matmul(other, self)

    def _matmul(self, *inputs):
        # np.matmul became a ufunc in version 1.16.0
        return matmul(*inputs)

    def _create_empty(self, inputs=None, dtype=object):
        shape = self.shape if self._broadcasted_shape is None else self._broadcasted_shape
        a = np.empty(int(np.prod(shape)), dtype=dtype)
        if inputs is None:
            return a, self.flat, shape
        if len(inputs) == 1:
            return a, inputs[0].flat, shape
        if isinstance(inputs[0], np.ndarray):
            return a, izip(inputs[0].flat, inputs[1].flat), shape
        # then the inputs are already broadcasted iterators
        return a, izip(*inputs), shape

    @property
    def label(self):
        """The label that was assigned to the array when it was created.

        **Example**::

            >>> current = la.uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)], label='amps')
            >>> current.label
            'amps'

        :rtype: :class:`str`
        """
        return self._label

    @property
    def real(self):
        """The result of applying the attribute ``real`` to each
        element in the array.

        **Example**::

            >>> a = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
            >>> a.real
            uarray([ureal(1.2,0.6,inf), ureal(3.2,1.4,inf),
                    ureal(0.0,0.9,inf)])

        :rtype: :class:`UncertainArray`
        """
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = item.real
        return UncertainArray(out.reshape(shape))

    @property
    def imag(self):
        """The result of applying the attribute ``imag`` to each
        element in the array.

        **Example**::

            >>> a = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
            >>> a.imag
            uarray([ureal(-0.5,0.6,inf), ureal(1.2,0.2,inf),
                    ureal(1.5,0.9,inf)])

        :rtype: :class:`UncertainArray`
        """
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = item.imag
        return UncertainArray(out.reshape(shape))

    @property
    def r(self):
        """The result of applying the attribute ``r`` to  each element in the array.

        **Example**::

            >>> a = la.uarray([ucomplex(1.2-0.5j, (1.2, 0.7, 0.7, 2.2)),
            ...                ucomplex(-0.2+1.2j, (0.9, 0.4, 0.4, 1.5))])
            >>> a.r
            uarray([0.43082021842766455, 0.34426518632954817])

        :rtype: :class:`UncertainArray`
        """
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = item.r
        return UncertainArray(out.reshape(shape))

    @property
    def x(self):
        """The result of :func:`~.core.value` for each element in the array.

        **Example**::

            >>> a = la.uarray([0.57, ureal(0.45, 0.12), ucomplex(1.1+0.68j, 0.19)])
            >>> a.x
            uarray([0.57, 0.45, (1.1+0.68j)])

        :rtype: :class:`UncertainArray`
        """
        return self.value()

    def value(self):
        """The result of :func:`~.core.value` for each element in the array.

        **Example**::

            >>> a = la.uarray([0.57, ureal(0.45, 0.12), ucomplex(1.1+0.68j, 0.19)])
            >>> a.value()
            uarray([0.57, 0.45, (1.1+0.68j)])

        :rtype: :class:`UncertainArray`
        """
        # Note: in the future we might allow different `dtype` values.
        # However, this needs some thought. Should `dtype=float`
        # return complex numbers as a pair of reals, for example?
        # What are the most likely use-cases?
        # :param dtype: The data type of the returned array.
        # :type dtype: :class:`numpy.dtype`
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = value(item)
        return UncertainArray(out.reshape(shape))

    @property
    def u(self):
        """The result of :func:`~.core.uncertainty` for each element in the array.

        **Example**::

            >>> r = la.uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
            >>> r.u
            uarray([0.18, 0.12, 0.19])
            >>> c = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
            >>> c.u
            uarray([StandardUncertainty(real=0.6, imag=0.6),
                   StandardUncertainty(real=1.4, imag=0.2),
                   StandardUncertainty(real=0.9, imag=0.9)])

        :rtype: :class:`UncertainArray`
        """
        return self.uncertainty()

    def uncertainty(self):
        """The result of :func:`~.core.uncertainty` for each element in the array.

        **Example**::

            >>> r = la.uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
            >>> r.uncertainty()
            uarray([0.18, 0.12, 0.19])
            >>> c = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
            >>> c.uncertainty()
            uarray([StandardUncertainty(real=0.6, imag=0.6),
                   StandardUncertainty(real=1.4, imag=0.2),
                   StandardUncertainty(real=0.9, imag=0.9)])

        :rtype: :class:`UncertainArray`
        """
        # Note: in the future we might allow different `dtype` values.
        # However, we need to consider the use-cases carefully.
        # :param dtype: The data type of the returned array.
        # :type dtype: :class:`numpy.dtype`
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = uncertainty(item)
        return UncertainArray(out.reshape(shape))

    @property
    def v(self):
        """The result of :func:`~.core.variance` for each element in the array.

        **Example**::

            >>> r = la.uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
            >>> r.v
            uarray([0.0324, 0.0144, 0.0361])
            >>> c = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.5, 0.5)), ucomplex(1.5j, 0.9)])
            >>> c.v
            uarray([VarianceCovariance(rr=0.36, ri=0.0, ir=0.0, ii=0.36),
                    VarianceCovariance(rr=2.25, ri=0.0, ir=0.0, ii=0.25),
                    VarianceCovariance(rr=0.81, ri=0.0, ir=0.0, ii=0.81)])

        :rtype: :class:`UncertainArray`
        """
        return self.variance()

    def variance(self):
        """The result of :func:`~.core.variance` for each element in the array.

        **Example**::

            >>> r = la.uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
            >>> r.variance()
            uarray([0.0324, 0.0144, 0.0361])
            >>> c = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.5, 0.5)), ucomplex(1.5j, 0.9)])
            >>> c.variance()
            uarray([VarianceCovariance(rr=0.36, ri=0.0, ir=0.0, ii=0.36),
                    VarianceCovariance(rr=2.25, ri=0.0, ir=0.0, ii=0.25),
                    VarianceCovariance(rr=0.81, ri=0.0, ir=0.0, ii=0.81)])

        :rtype: :class:`UncertainArray`
        """
        # Note: in the future we might allow different `dtype` values.
        # However, we need to consider the use-cases carefully.
        # :param dtype: The data type of the returned array.
        # :type dtype: :class:`numpy.dtype`
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = variance(item)
        return UncertainArray(out.reshape(shape))

    @property
    def df(self):
        """The result of :func:`~.core.dof` for each element in the array.

        **Example**::

            >>> a = la.uarray([ureal(6, 2, df=3), ureal(4, 1, df=4), ureal(5, 3, df=7), ureal(1, 1)])
            >>> a.df
            uarray([3.0, 4.0, 7.0, inf])

        :rtype: :class:`UncertainArray`
        """
        return self.dof()

    def dof(self):
        """The result of :func:`~.core.dof` for each element in the array.

        **Example**::

            >>> a = la.uarray([ureal(6, 2, df=3), ureal(4, 1, df=4), ureal(5, 3, df=7), ureal(1, 1)])
            >>> a.dof()
            uarray([3.0, 4.0, 7.0, inf])

        :rtype: :class:`UncertainArray`
        """
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = dof(item)
        return UncertainArray(out.reshape(shape))

    def sensitivity(self, x):
        """The result of :func:`~.reporting.sensitivity` for each element in the array.

        :rtype: :class:`UncertainArray`
        """
        # Note, there is a case for introducing `dtype` or some other parameter.
        # The return types for complex cases may be multivariate.

        # `_create_empty()` handles only ndarray-like sequences
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        out, iterator, shape = self._create_empty((self, x))
        for i, (y, x) in enumerate(iterator):
            out[i] = y.sensitivity(x)
        return UncertainArray(out.reshape(shape))

    def u_component(self, x):
        """The result of :func:`~.reporting.u_component` for each element in the array.

        :rtype: :class:`UncertainArray`
        """
        # Note, there is a case for introducing `dtype` or some other parameter.
        # The return types for complex cases may be multivariate.

        # `_create_empty()` handles only ndarray-like sequences
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        out, iterator, shape = self._create_empty((self, x))
        for i, (y, x) in enumerate(iterator):
            out[i] = y.u_component(x)
        return UncertainArray(out.reshape(shape))

    def conjugate(self):
        """The result of applying the attribute ``conjugate`` to each element in the array.

        **Example**::

            >>> a = la.uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
            >>> a.conjugate()
            uarray([ucomplex((1.2+0.5j), u=[0.6,0.6], r=0.0, df=inf),
                    ucomplex((3.2-1.2j), u=[1.4,0.2], r=0.0, df=inf),
                    ucomplex((0-1.5j), u=[0.9,0.9], r=0.0, df=inf)])

        :rtype: :class:`UncertainArray`
        """
        # override this method because I wanted to create a custom __doc__
        return self._conjugate()

    def _conjugate(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = item.conjugate()
        return UncertainArray(out.reshape(shape))

    def _positive(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = +item
        return UncertainArray(out.reshape(shape))

    def _negative(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = -item
        return UncertainArray(out.reshape(shape))

    def _add(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a + b
        return UncertainArray(out.reshape(shape))

    def _subtract(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a - b
        return UncertainArray(out.reshape(shape))

    def _multiply(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a * b
        return UncertainArray(out.reshape(shape))

    def _divide(self, *inputs):
        return self._true_divide(*inputs)

    def _true_divide(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a / b
        return UncertainArray(out.reshape(shape))

    def _power(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a ** b
        return UncertainArray(out.reshape(shape))

    def _exp(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = exp(item)
        return UncertainArray(out.reshape(shape))

    def _log(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = log(item)
        return UncertainArray(out.reshape(shape))

    def _log10(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = log10(item)
        return UncertainArray(out.reshape(shape))

    def _sqrt(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = sqrt(item)
        return UncertainArray(out.reshape(shape))

    def _cos(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = cos(item)
        return UncertainArray(out.reshape(shape))

    def _sin(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = sin(item)
        return UncertainArray(out.reshape(shape))

    def _tan(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = tan(item)
        return UncertainArray(out.reshape(shape))

    def _arccos(self, *ignore):
        return self._acos()

    def _acos(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = acos(item)
        return UncertainArray(out.reshape(shape))

    def _arcsin(self, *ignore):
        return self._asin()

    def _asin(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = asin(item)
        return UncertainArray(out.reshape(shape))

    def _arctan(self, *ignore):
        return self._atan()

    def _atan(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = atan(item)
        return UncertainArray(out.reshape(shape))

    def _arctan2(self, *inputs):
        return self._atan2(inputs[1])

    def _atan2(self, *inputs):
        out, iterator, shape = self._create_empty((self, inputs[0]))
        for i, (a, b) in enumerate(iterator):
            out[i] = atan2(a, b)
        return UncertainArray(out.reshape(shape))

    def _sinh(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = sinh(item)
        return UncertainArray(out.reshape(shape))

    def _cosh(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = cosh(item)
        return UncertainArray(out.reshape(shape))

    def _tanh(self, *ignore):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = tanh(item)
        return UncertainArray(out.reshape(shape))

    def _arccosh(self, *ignore):
        return self._acosh()

    def _acosh(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = acosh(item)
        return UncertainArray(out.reshape(shape))

    def _arcsinh(self, *ignore):
        return self._asinh()

    def _asinh(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = asinh(item)
        return UncertainArray(out.reshape(shape))

    def _arctanh(self, *ignore):
        return self._atanh()

    def _atanh(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = atanh(item)
        return UncertainArray(out.reshape(shape))

    def _square(self, *ignore):
        return self._mag_squared()

    def _mag_squared(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = mag_squared(item)
        return UncertainArray(out.reshape(shape))

    def _magnitude(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = magnitude(item)
        return UncertainArray(out.reshape(shape))

    def _phase(self):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = phase(item)
        return UncertainArray(out.reshape(shape))

    def _intermediate(self, labels):
        # Default second argument of calling function is `None`
        if labels is None:
            out, iterator, shape = self._create_empty()
            for i, x in enumerate(iterator):
                out[i] = result(x)
        else:
            # `_create_empty()` handles only ndarray-like sequences
            if not is_sequence(labels):
                # Add index notation to the label base
                labels = [
                    "{}[{}]".format(labels, i)
                    for i in xrange(self.size)
                ]

            labels = np.asarray(labels)
            out, iterator, shape = self._create_empty((self, labels))
            for i, (x, lbl) in enumerate(iterator):
                out[i] = result(x, lbl)

        return UncertainArray(out.reshape(shape))

    def _equal(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, (a, b) in enumerate(iterator):
            out[i] = a == b
        return out.reshape(shape)

    def _not_equal(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, (a, b) in enumerate(iterator):
            out[i] = a != b
        return out.reshape(shape)

    def _less(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, (a, b) in enumerate(iterator):
            out[i] = a < b
        return out.reshape(shape)

    def _less_equal(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, (a, b) in enumerate(iterator):
            out[i] = a <= b
        return out.reshape(shape)

    def _greater(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, (a, b) in enumerate(iterator):
            out[i] = a > b
        return out.reshape(shape)

    def _greater_equal(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, (a, b) in enumerate(iterator):
            out[i] = a >= b
        return out.reshape(shape)

    def _maximum(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            if _isnan(a):
                out[i] = a
            elif _isnan(b):
                out[i] = b
            elif a > b:
                out[i] = a
            else:
                out[i] = b
        return UncertainArray(out.reshape(shape))

    def _minimum(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            if _isnan(a):
                out[i] = a
            elif _isnan(b):
                out[i] = b
            elif a < b:
                out[i] = a
            else:
                out[i] = b
        return UncertainArray(out.reshape(shape))

    def _logical_and(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a and b
        return UncertainArray(out.reshape(shape))

    def _logical_or(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, (a, b) in enumerate(iterator):
            out[i] = a or b
        return UncertainArray(out.reshape(shape))

    def _logical_xor(self, *inputs):
        raise TypeError(
            "Boolean bitwise operations are not defined for `UncertainArray`"
        )
        # out, iterator, shape = self._create_empty(inputs, dtype=bool)
        # for i, (a, b) in enumerate(iterator):
        #     out[i] = bool(a) ^ bool(b)
        # return out.reshape(shape)

    def _logical_not(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, item in enumerate(iterator):
            out[i] = not bool(item)
        return out.reshape(shape)

    def _isinf(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, item in enumerate(iterator):
            out[i] = _isinf(item)
        return out.reshape(shape)

    def _isnan(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, item in enumerate(iterator):
            out[i] = _isnan(item)
        return out.reshape(shape)

    def _isfinite(self, *inputs):
        out, iterator, shape = self._create_empty(inputs, dtype=bool)
        for i, item in enumerate(iterator):
            out[i] = not (_isnan(item) or _isinf(item))
        return out.reshape(shape)

    def _reciprocal(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, item in enumerate(iterator):
            out[i] = 1.0/item
        return UncertainArray(out.reshape(shape))

    def _absolute(self, *inputs):
        out, iterator, shape = self._create_empty(inputs)
        for i, item in enumerate(iterator):
            out[i] = abs(item)
        return UncertainArray(out.reshape(shape))

    def copy(self, order='C'):
        out, iterator, shape = self._create_empty()
        for i, item in enumerate(iterator):
            out[i] = +item
        return UncertainArray(out.reshape(shape, order=order), label=self.label)

    def round(self, *args, **kwargs):
        raise TypeError(
            "`round` is not defined for `UncertainArray`"
        )

    def sum(self, *args, **kwargs):
        raise TypeError(
            "`sum` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).sum(*args, **kwargs))

    def mean(self, *args, **kwargs):
        raise TypeError(
            "`mean` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).mean(*args, **kwargs))

    def std(self, *args, **kwargs):
        # If this is to be implemented we need to be clear about
        # what is calculated. This will not be an uncertain-number
        # calculation, it will take the values of a sample of uncertain
        # numbers and evaluate the SD. This will probably be clearer
        # if the function is in the `type_a` module.
        # Note we would also want a similar function to calculate
        # the standard error (ie the type-A uncertainty).
        raise TypeError(
            "`std` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).std(*args, **kwargs))

    def var(self, *args, **kwargs):
        # If this is to be implemented we need to be clear about
        # what is calculated. This will not be an uncertain-number
        # calculation, it will take the values of a sample of uncertain
        # numbers and evaluate the SD. This will probably be clearer
        # if the function is in the `type_a` module.
        # Note we would also want a similar function to calculate
        # the standard variance (ie the type-A uncertainty squared).
        raise TypeError(
            "`var` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).var(*args, **kwargs))

    def max(self, *args, **kwargs):
        raise TypeError(
            "`max` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).max(*args, **kwargs))

    def min(self, *args, **kwargs):
        raise TypeError(
            "`min` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).min(*args, **kwargs))

    def trace(self, *args, **kwargs):
        raise TypeError(
            "`trace` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).trace(*args, **kwargs))

    def cumprod(self, *args, **kwargs):
        # numpy catches ``TypeError`` and uses its
        # internal implementation of this method
        raise RuntimeError(
            "`cumprod` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).cumprod(*args, **kwargs))

    def cumsum(self, *args, **kwargs):
        # numpy catches ``TypeError`` and uses its
        # internal implementation of this method
        raise RuntimeError(
            "`cumsum` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).cumsum(*args, **kwargs))

    def prod(self, *args, **kwargs):
        raise TypeError(
            "`prod` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).prod(*args, **kwargs))

    def ptp(self, *args, **kwargs):
        raise TypeError(
            "`ptp` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self).ptp(*args, **kwargs))

    def any(self, *args, **kwargs):
        raise TypeError(
            "`any` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self, dtype=bool).any(*args, **kwargs))

    def all(self, *args, **kwargs):
        raise TypeError(
            "`all` is not defined for `UncertainArray`"
        )
        # return UncertainArray(np.asarray(self, dtype=bool).all(*args, **kwargs))


# Allows pickle to understand the class name 'uarray'
uarray = UncertainArray

"""
The proper way to create an uncertain array is by calling uarray(...)

This module was written in the following way so that numpy >= 1.13.0
does not have to be installed in order for someone to use GTC.
"""
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

from GTC.core import (
    value,
    uncertainty,
    variance,
    dof,
)

from GTC.lib import (
    UncertainReal,
    UncertainComplex
)

try:
    import numpy as np
except ImportError:
    UncertainArray = None
else:
    if np.__version__ < '1.13.0':
        # The __array_ufunc__ method was not introduced until v1.13.0
        UncertainArray = None
    else:

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

        class UncertainArray(np.ndarray):
            """Base: :class:`numpy.ndarray`

            An :class:`UncertainArray` contains elements that are of type
            :class:`.UncertainReal` or :class:`.UncertainComplex`.

            Do not instantiate this class directly. Use :func:`~.core.uarray` instead.

            """
            def __new__(cls, array, dtype=None, label=None):
                if dtype is None:
                    obj = np.asarray(array).view(cls)
                else:
                    obj = np.asarray(array, dtype=dtype).view(cls)
                obj._label = label
                return obj

            def __array_finalize__(self, obj):
                if obj is None: return
                self._label = getattr(obj, 'label', None)

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
                            inputs[convert] = np.full(inputs[keep].shape, inputs[convert])
                        else:
                            inputs[convert] = np.asarray(inputs[convert])
                    if inputs[0].shape != inputs[1].shape:
                        # this is the error type and message that numpy would display
                        raise ValueError(
                            'operands could not be broadcast together with shapes {} {}'
                            .format(inputs[0].shape, inputs[1].shape)
                        )
                else:
                    assert False, 'Should not occur: __array_ufunc__ received {} inputs'.format(case)

                return attr(*inputs)

            @property
            def label(self):
                """:class:`str` - The label that was assigned to the array when it was created.

                **Example**::

                    >>> current = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)], label='amps')
                    >>> current.label
                    'amps'

                """
                return self._label

            @property
            def x(self):
                """The result of :func:`~.core.value` for each element in the array.

                **Example**::

                    >>> a = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
                    >>> a.x
                    array([0.57, 0.45, 0.68])

                """
                if self.ndim == 0:
                    return value(self.item(0))
                item_get = self.item
                out = np.asarray([value(item_get(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            @property
            def u(self):
                """The result of :func:`~.core.uncertainty` for each element in the array.

                **Example**::

                    >>> r = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
                    >>> r.u
                    array([0.18, 0.12, 0.19])
                    >>> c = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> c.u
                    array([StandardUncertainty(real=0.6, imag=0.6),
                           StandardUncertainty(real=1.4, imag=0.2),
                           StandardUncertainty(real=0.9, imag=0.9)], dtype=object)

                """
                if self.ndim == 0:
                    return uncertainty(self.item(0))
                # for an UncertainArray with ucomplex elements the uncertainty
                # returns a StandardUncertainty namedtuple and therefore we
                # want the StandardUncertainty to be returned as a namedtuple
                # "object" and not as a tuple of floats
                out = np.empty(self.shape, dtype=object)
                item_set = out.itemset
                item_get = self.item
                for i in xrange(self.size):
                    item_set(i, uncertainty(item_get(i)))
                try:
                    # if this works then there are no StandardUncertainty
                    # elements in the array
                    return out.astype(np.float64)
                except ValueError:
                    return out

            @property
            def real(self):
                """The result of :attr:`UncertainReal.real <.lib.UncertainReal.real>` or
                :attr:`UncertainComplex.real <.lib.UncertainComplex.real>` for each
                element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> a.real
                    UncertainArray([ureal(1.2,0.6,inf), ureal(3.2,1.4,inf),
                                    ureal(0.0,0.9,inf)], dtype=object)

                """
                if self.ndim == 0:
                    return self.item(0).real
                return UncertainArray([item.real for item in self])

            @property
            def imag(self):
                """The result of :attr:`UncertainReal.imag <.lib.UncertainReal.imag>` or
                :attr:`UncertainComplex.imag <.lib.UncertainComplex.imag>` for each
                element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> a.imag
                    UncertainArray([ureal(-0.5,0.6,inf), ureal(1.2,0.2,inf),
                                    ureal(1.5,0.9,inf)], dtype=object)

                """
                if self.ndim == 0:
                    return self.item(0).imag
                return UncertainArray([item.imag for item in self])

            @property
            def v(self):
                """The result of :func:`~.core.variance` for each element in the array.

                **Example**::

                    >>> r = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
                    >>> r.v
                    array([0.0324, 0.0144, 0.0361])
                    >>> c = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.5, 0.5)), ucomplex(1.5j, 0.9)])
                    >>> c.v
                    array([VarianceCovariance(rr=0.36, ri=0.0, ir=0.0, ii=0.36),
                           VarianceCovariance(rr=2.25, ri=0.0, ir=0.0, ii=0.25),
                           VarianceCovariance(rr=0.81, ri=0.0, ir=0.0, ii=0.81)], dtype=object)

                """
                if self.ndim == 0:
                    return variance(self.item(0))
                # for an UncertainArray with ucomplex elements the variance()
                # returns a VarianceCovariance namedtuple and therefore we
                # want the VarianceCovariance to be returned as a namedtuple
                # "object" and not as a tuple of floats
                out = np.empty(self.shape, dtype=object)
                item_set = out.itemset
                item_get = self.item
                for i in xrange(self.size):
                    item_set(i, variance(item_get(i)))
                try:
                    # if this works then there are no VarianceCovariance
                    # elements in the array
                    return out.astype(np.float64)
                except ValueError:
                    return out

            @property
            def df(self):
                """The result of :func:`~.core.dof` for each element in the array.

                **Example**::

                    >>> a = uarray([ureal(0.6, 0.2, df=3), ureal(0.4, 0.1, df=4), ureal(0.5, 0.5, df=7)])
                    >>> a.df
                    array([3., 4., 7.])

                """
                if self.ndim == 0:
                    return dof(self.item(0))
                item_get = self.item
                out = np.asarray([dof(item_get(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            @property
            def r(self):
                """The result of :attr:`UncertainComplex.r <.lib.UncertainComplex.r>`
                for each element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, (1.2, 0.7, 0.7, 2.2)),
                    ...             ucomplex(-0.2+1.2j, (0.9, 0.4, 0.4, 1.5))])
                    >>> a.r
                    array([0.26515152, 0.2962963 ])

                """
                if self.ndim == 0:
                    return self.item(0).r
                return np.asarray([item.r for item in self])

            def _positive(self, *inputs):
                return UncertainArray([+val for val in inputs[0]])

            def _negative(self, *inputs):
                return UncertainArray([-val for val in inputs[0]])

            def _add(self, *inputs):
                return UncertainArray([lhs + rhs for lhs, rhs in izip(*inputs)])

            def _subtract(self, *inputs):
                return UncertainArray([lhs - rhs for lhs, rhs in izip(*inputs)])

            def _multiply(self, *inputs):
                return UncertainArray([lhs * rhs for lhs, rhs in izip(*inputs)])

            def _divide(self, *inputs):
                return UncertainArray([lhs / rhs for lhs, rhs in izip(*inputs)])

            def _true_divide(self, *inputs):
                return UncertainArray([lhs / rhs for lhs, rhs in izip(*inputs)])

            def _power(self, *inputs):
                return UncertainArray([lhs ** rhs for lhs, rhs in izip(*inputs)])

            def _equal(self, *inputs):
                return np.asarray([lhs == rhs for lhs, rhs in izip(*inputs)])

            def _not_equal(self, *inputs):
                return np.asarray([lhs != rhs for lhs, rhs in izip(*inputs)])

            def _less(self, *inputs):
                return np.asarray([lhs < rhs for lhs, rhs in izip(*inputs)])

            def _less_equal(self, *inputs):
                return np.asarray([lhs <= rhs for lhs, rhs in izip(*inputs)])

            def _greater(self, *inputs):
                return np.asarray([lhs > rhs for lhs, rhs in izip(*inputs)])

            def _greater_equal(self, *inputs):
                return np.asarray([lhs >= rhs for lhs, rhs in izip(*inputs)])

            def _maximum(self, *inputs):
                out = self.copy()
                # create references outside of the loop to avoid multiple attrib lookups
                out_set, i0, i1 = out.itemset, inputs[0].item, inputs[1].item
                for i in xrange(self.size):
                    a, b = i0(i), i1(i)
                    if _isnan(a):
                        out_set(i, a)
                    elif _isnan(b):
                        out_set(i, b)
                    elif a > b:
                        out_set(i, a)
                    else:
                        out_set(i, b)
                return out

            def _minimum(self, *inputs):
                out = self.copy()
                # create references outside of the loop to avoid multiple attrib lookups
                out_set, i0, i1 = out.itemset, inputs[0].item, inputs[1].item
                for i in xrange(self.size):
                    a, b = i0(i), i1(i)
                    if _isnan(a):
                        out_set(i, a)
                    elif _isnan(b):
                        out_set(i, b)
                    elif a < b:
                        out_set(i, a)
                    else:
                        out_set(i, b)
                return out

            def _logical_and(self, *inputs):
                a, b = inputs[0].item, inputs[1].item
                out = np.asarray([bool(a(i)) and bool(b(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _logical_or(self, *inputs):
                a, b = inputs[0].item, inputs[1].item
                out = np.asarray([bool(a(i)) or bool(b(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _logical_xor(self, *inputs):
                a, b = inputs[0].item, inputs[1].item
                out = np.asarray([bool(a(i)) ^ bool(b(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _logical_not(self, *inputs):
                a = inputs[0].item
                out = np.asarray([not bool(a(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _isinf(self, *inputs):
                item = inputs[0].item
                out = np.asarray([_isinf(item(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _isnan(self, *inputs):
                item = inputs[0].item
                out = np.asarray([_isnan(item(i)) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _isfinite(self, *inputs):
                item = inputs[0].item
                out = np.asarray([not (_isnan(item(i)) or _isinf(item(i))) for i in xrange(self.size)])
                return out.reshape(self.shape)

            def _reciprocal(self, *inputs):
                item = inputs[0].item
                out = np.asarray([1.0/item(i) for i in xrange(self.size)])
                return UncertainArray(out.reshape(self.shape))

            def _absolute(self, *inputs):
                return UncertainArray([abs(value) for value in inputs[0]])

            def _conjugate(self, *inputs):
                # use self instead of inputs[0]
                # I wanted to create a custom __doc__ for this method
                return UncertainArray([value.conjugate() for value in self])

            def conjugate(self):
                """The result of :meth:`UncertainReal.conjugate() <.lib.UncertainReal.conjugate>`
                or :meth:`UncertainComplex.conjugate() <.lib.UncertainComplex.conjugate>`
                for each element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> a.conjugate()
                    UncertainArray([ucomplex((1.2+0.5j), u=[0.6,0.6], r=0.0, df=inf),
                                    ucomplex((3.2-1.2j), u=[1.4,0.2], r=0.0, df=inf),
                                    ucomplex((0-1.5j), u=[0.9,0.9], r=0.0, df=inf)],
                                   dtype=object)

                """
                # override this method because I wanted to create a custom __doc__
                return self._conjugate()

            def _cos(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.cos(x)
                return UncertainArray([value._cos() for value in self])

            def _sin(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.sin(x)
                return UncertainArray([value._sin() for value in self])

            def _tan(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.tan(x)
                return UncertainArray([value._tan() for value in self])

            def _arccos(self, *inputs):
                return UncertainArray([value._acos() for value in inputs[0]])

            def _acos(self):
                return UncertainArray([value._acos() for value in self])

            def _arcsin(self, *inputs):
                return UncertainArray([value._asin() for value in inputs[0]])

            def _asin(self):
                return UncertainArray([value._asin() for value in self])

            def _arctan(self, *inputs):
                return UncertainArray([value._atan() for value in inputs[0]])

            def _atan(self):
                return UncertainArray([value._atan() for value in self])

            def _arctan2(self, *inputs):
                return UncertainArray([lhs._atan2(rhs) for lhs, rhs in izip(*inputs)])

            def _atan2(self, *inputs):
                return UncertainArray([lhs._atan2(rhs) for lhs, rhs in izip(self, inputs[0])])

            def _exp(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.exp(x)
                return UncertainArray([value._exp() for value in self])

            def _log(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.log(x)
                return UncertainArray([value._log() for value in self])

            def _log10(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.log10(x)
                return UncertainArray([value._log10() for value in self])

            def _sqrt(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.sqrt(x)
                return UncertainArray([value._sqrt() for value in self])

            def _sinh(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.sinh(x)
                return UncertainArray([value._sinh() for value in self])

            def _cosh(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.cosh(x)
                return UncertainArray([value._cosh() for value in self])

            def _tanh(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.tanh(x)
                return UncertainArray([value._tanh() for value in self])

            def _arccosh(self, *inputs):
                return UncertainArray([value._acosh() for value in inputs[0]])

            def _acosh(self):
                return UncertainArray([value._acosh() for value in self])

            def _arcsinh(self, *inputs):
                return UncertainArray([value._asinh() for value in inputs[0]])

            def _asinh(self):
                return UncertainArray([value._asinh() for value in self])

            def _arctanh(self, *inputs):
                return UncertainArray([value._atanh() for value in inputs[0]])

            def _atanh(self):
                return UncertainArray([value._atanh() for value in self])

            def _mag_squared(self):
                return UncertainArray([value._mag_squared() for value in self])

            def _square(self, *inputs):
                return UncertainArray([value._mag_squared() for value in inputs[0]])

            def _magnitude(self):
                return UncertainArray([value._magnitude() for value in self])

            def _phase(self):
                return UncertainArray([value._phase() for value in self])

            def copy(self, order='C'):
               out = np.empty(self.shape, dtype=self.dtype, order=order)
               item_set = out.itemset
               item_get = self.item
               for i in xrange(self.size):
                   item_set(i, +item_get(i))
               return UncertainArray(out, label=self.label)

            def sum(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).sum(*args, **kwargs))

            def mean(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).mean(*args, **kwargs))

            def std(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).std(*args, **kwargs))

            def var(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).var(*args, **kwargs))

            def max(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).max(*args, **kwargs))

            def min(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).min(*args, **kwargs))

            def trace(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).trace(*args, **kwargs))

            def cumprod(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).cumprod(*args, **kwargs))

            def cumsum(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).cumsum(*args, **kwargs))

            def prod(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).prod(*args, **kwargs))

            def ptp(self, *args, **kwargs):
                return UncertainArray(np.asarray(self).ptp(*args, **kwargs))

            def any(self, *args, **kwargs):
                return np.asarray(self, dtype=np.bool).any(*args, **kwargs)

            def all(self, *args, **kwargs):
                return np.asarray(self, dtype=np.bool).all(*args, **kwargs)

            def round(self, decimals=0, **kwargs):
                digits = kwargs.get('digits', decimals)
                df_decimals = kwargs.get('df_decimals', digits)
                out = np.empty(self.shape, dtype=self.dtype)
                item_set = out.itemset
                item_get = self.item
                # do not use list comprehension because the returned value from
                # _round is a tuple (it's actually a GroomedUncertain... namedtuple)
                for i in xrange(self.size):
                    item_set(i, item_get(i)._round(digits, df_decimals))
                return UncertainArray(out)

            def __matmul__(self, other):
                # Implements the protocol used by the '@' operator defined in PEP 465.
                if not isinstance(other, np.ndarray):
                    other = np.asarray(other)
                try:
                    # first, see if support for dtype=object was added
                    return UncertainArray(np.matmul(self, other))
                except TypeError:
                    return UncertainArray(self._matmul(self, other))

            def __rmatmul__(self, other):
                # Implements the protocol used by the '@' operator defined in PEP 465.
                if not isinstance(other, np.ndarray):
                    other = np.asarray(other)
                try:
                    # first, see if support for dtype=object was added
                    return UncertainArray(np.matmul(other, self))
                except TypeError:
                    return UncertainArray(self._matmul(other, self))

            def _matmul(self, lhs, rhs):
                # Must re-implement matrix multiplication because np.matmul
                # does not currently (as of v1.15.3) support dtype=object arrays.
                # A fix is planned for v1.16.0

                nd1, nd2 = lhs.ndim, rhs.ndim
                if nd1 == 0 or nd2 == 0:
                    raise ValueError("Scalar operands are not allowed, use '*' instead")

                if nd1 <= 2 and nd2 <= 2:
                    return lhs.dot(rhs)

                broadcast = np.broadcast(np.empty(lhs.shape[:-2]), np.empty(rhs.shape[:-2]))
                ranges = [np.arange(s) for s in broadcast.shape]
                grid = np.meshgrid(*ranges, sparse=False, indexing='ij')
                indices = np.array([item.ravel() for item in grid]).transpose()

                i1 = indices.copy()
                i2 = indices.copy()
                for i in range(len(indices[0])):
                    i1[:, i] = indices[:, i].clip(max=lhs.shape[i]-1)
                    i2[:, i] = indices[:, i].clip(max=rhs.shape[i]-1)

                slices = np.array([[slice(None), slice(None)]]).repeat(len(indices), axis=0)
                i1 = np.hstack((i1, slices))
                i2 = np.hstack((i2, slices))
                out = np.array([self._matmul(lhs[tuple(a)], rhs[tuple(b)]) for a, b in zip(i1, i2)])
                return out.reshape(list(broadcast.shape) + [lhs.shape[-2], rhs.shape[-1]])

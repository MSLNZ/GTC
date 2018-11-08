"""
The proper way to create an uncertain array is by calling uarray(...)

This module was written in the following way so that numpy >= 1.13.0
does not have to be installed in order for someone to use GTC.
"""
from __future__ import division
import warnings
from numbers import Number
from math import isnan, isinf
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

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
            try:
                return isnan(number.x)
            except AttributeError:
                return isnan(number)

        def _isinf(number):
            try:
                return isinf(number.x)
            except AttributeError:
                return isinf(number)

        class UncertainArray(np.ndarray):

            def __new__(cls, input_array, label=None):
                obj = np.asarray(input_array).view(cls)
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
                return self._label

            @property
            def x(self):
                if self.ndim == 0:
                    return self.item(0).x
                return np.asarray([item.x for item in self])

            @property
            def u(self):
                if self.ndim == 0:
                    return self.item(0).u
                if isinstance(self.item(0), UncertainComplex):
                    # for an UncertainArray with ucomplex elements the .u attribute
                    # returns a StandardUncertainty namedtuple and therefore we cannot
                    # use list comprehension to create the returned array because we
                    # want the StandardUncertainty to be returned as a namedtuple
                    # "object" and not as a tuple of floats
                    out = np.empty(self.shape, dtype=object)
                    item_set = out.itemset
                    item_get = self.item
                    for i in xrange(self.size):
                        item_set(i, item_get(i).u)
                    return out
                return np.asarray([item.u for item in self])

            @property
            def real(self):
                if self.ndim == 0:
                    return self.item(0).real
                return UncertainArray([item.real for item in self])

            @property
            def imag(self):
                if self.ndim == 0:
                    return self.item(0).imag
                return UncertainArray([item.imag for item in self])

            @property
            def v(self):
                if self.ndim == 0:
                    return self.item(0).v
                if isinstance(self.item(0), UncertainComplex):
                    # for an UncertainArray with ucomplex elements the .v attribute
                    # returns a VarianceCovariance namedtuple and therefore we cannot
                    # use list comprehension to create the returned array because we
                    # want the VarianceCovariance to be returned as a namedtuple
                    # "object" and not as a tuple of floats
                    out = np.empty(self.shape, dtype=object)
                    item_set = out.itemset
                    item_get = self.item
                    for i in xrange(self.size):
                        item_set(i, item_get(i).v)
                    return out
                return np.asarray([item.v for item in self])

            @property
            def df(self):
                if self.ndim == 0:
                    return self.item(0).df
                return np.asarray([item.df for item in self])

            @property
            def r(self):
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
                return UncertainArray([value.conjugate() for value in inputs[0]])

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

            def sum(self, **kwargs):
                return UncertainArray(np.asarray(self).sum(**kwargs))

            def mean(self, **kwargs):
                return UncertainArray(np.asarray(self).mean(**kwargs))

            def std(self, **kwargs):
                return UncertainArray(np.asarray(self).std(**kwargs))

            def var(self, **kwargs):
                return UncertainArray(np.asarray(self).var(**kwargs))

            def max(self, **kwargs):
                return UncertainArray(np.asarray(self).max(**kwargs))

            def min(self, **kwargs):
                return UncertainArray(np.asarray(self).min(**kwargs))

            def trace(self, **kwargs):
                return UncertainArray(np.asarray(self).trace(**kwargs))

            def cumprod(self, **kwargs):
                return UncertainArray(np.asarray(self).cumprod(**kwargs))

            def cumsum(self, **kwargs):
                return UncertainArray(np.asarray(self).cumsum(**kwargs))

            def prod(self, **kwargs):
                return UncertainArray(np.asarray(self).prod(**kwargs))

            def ptp(self, **kwargs):
                return UncertainArray(np.asarray(self).ptp(**kwargs))

            def any(self, **kwargs):
                return np.asarray(self, dtype=np.bool).any(**kwargs)

            def all(self, **kwargs):
                return np.asarray(self, dtype=np.bool).all(**kwargs)

            def round(self, decimals=0, **kwargs):
                digits = kwargs.get('digits', decimals)
                df_decimals = kwargs.get('df_decimals', digits)
                out = self.copy()
                item_set = out.itemset
                item_get = self.item
                # do not use list comprehension because the returned value from
                # _round is a tuple (it's actually a namedtuple)
                for i in xrange(self.size):
                    item_set(i, item_get(i)._round(digits, df_decimals))
                return out

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

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

        #--------------------------------------------------------------------
        class UncertainArray(np.ndarray):
            """Base: :class:`numpy.ndarray`

            An :class:`UncertainArray` can contain elements that are of type
            :class:`int`, :class:`float`, :class:`complex`,
            :class:`.UncertainReal` or :class:`.UncertainComplex`.

            Do not instantiate this class directly. Use :func:`~.uarray` instead.

            """
            def __new__(cls, array, dtype=None, label=None):
                if dtype is None: 
                    dtype = object
                obj = np.asarray(array, dtype=dtype).view(cls)
                obj._label = label
                return obj

            def __array_finalize__(self, obj):
                if obj is None: return
                self._label = getattr(obj, 'label', None)
                
                # numpy looks at type().__name__ when preparing
                # a string representation of the object. This 
                # change means we see `uarray` not `UncertainArray`.
                self.__class__.__name__ = 'uarray'

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

            def _create_empty(self, inputs=None, dtype=None, order='C'):
                dtype=object if dtype is None else dtype
                a = np.empty(self.shape, dtype=dtype, order=order)
                # create references to "itemset" and "item" to avoid multiple
                # attribute lookups in the "for" loop that the calling method uses
                if inputs is None:
                    return a, a.itemset, self.item
                if len(inputs) == 1:
                    return a, a.itemset, inputs[0].item
                return a, a.itemset, inputs[0].item, inputs[1].item

            def __repr__(self):
                # Use the numpy formating but hide the default dtype
                np_array_repr = np.array_repr(self)
                
                if np_array_repr.find('uarray(') == 0:
                    i = np_array_repr.rfind('dtype=object')
                    if i == -1:
                        # if dtype is not `object`, show it
                        output = prefix + np_array_repr[14:]
                    else:
                        i = np_array_repr[:i].rfind(',') # trailing ',' 
                        output = np_array_repr[:i] + ')'
                                    
                    return output 
                    
                # if np_array_repr.find('UncertainArray(') == 0:
                    # assert False
                    # prefix = 'uarray'

                    # i = np_array_repr.rfind('dtype=object')
                    # if i == -1:
                        # # if dtype is not `object`, show it
                        # output = prefix + np_array_repr[14:]
                    # else:
                        # i = np_array_repr[:i].rfind(',') # trailing ',' 
                        # output = prefix + np_array_repr[14:i] + ')'
                    
                    # # numpy used the width of the class name in formating new
                    # # line indents but we have replaced this with 'uarray'!
                    # extra_space = ' '*(len('UncertainArray') - len('uarray'))
                    # return ''.join([ 
                        # l_i.replace(extra_space,'',1) 
                            # for l_i in output.splitlines(True) # Retains \n's
                    # ])
                    
                else:
                    return np_array_repr
                
            @property
            def label(self):
                """The label that was assigned to the array when it was created.

                **Example**::

                    >>> current = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)], label='amps')
                    >>> current.label
                    'amps'

                :rtype: :class:`str`
                """
                return self._label

            @property
            def real(self):
                """The result of :attr:`UncertainReal.real <.lib.UncertainReal.real>` or
                :attr:`UncertainComplex.real <.lib.UncertainComplex.real>` for each
                element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> a.real
                    uarray([ureal(1.2,0.6,inf), ureal(3.2,1.4,inf),
                            ureal(0.0,0.9,inf)])

                :rtype: :class:`UncertainArray`
                """
                if self.ndim == 0:
                    return self.item(0).real
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, item(i).real)
                return UncertainArray(arr)

            @property
            def imag(self):
                """The result of :attr:`UncertainReal.imag <.lib.UncertainReal.imag>` or
                :attr:`UncertainComplex.imag <.lib.UncertainComplex.imag>` for each
                element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> a.imag
                    uarray([ureal(-0.5,0.6,inf), ureal(1.2,0.2,inf),
                            ureal(1.5,0.9,inf)])

                :rtype: :class:`UncertainArray`
                """
                if self.ndim == 0:
                    return self.item(0).imag
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, item(i).imag)
                return UncertainArray(arr)

            @property
            def r(self):
                """The result of :attr:`UncertainComplex.r <.lib.UncertainComplex.r>`
                for each element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, (1.2, 0.7, 0.7, 2.2)),
                    ...             ucomplex(-0.2+1.2j, (0.9, 0.4, 0.4, 1.5))])
                    >>> a.r
                    array([0.26515152, 0.2962963 ])

                :rtype: :class:`numpy.ndarray`
                """
                if self.ndim == 0:
                    return self.item(0).r
                arr, itemset, item = self._create_empty(dtype=float)
                for i in xrange(self.size):
                    itemset(i, item(i).r)
                return arr

            def value(self, dtype=None):
                """The result of :func:`~.core.value` for each element in the array.

                **Example**::

                    >>> a = uarray([0.57, ureal(0.45, 0.12), ucomplex(1.1+0.68j, 0.19)])
                    >>> a.value()
                    array([0.57, 0.45, (1.1+0.68j)], dtype=object)
                    >>> a.value(complex)
                    array([0.57+0.j  , 0.45+0.j  , 1.1 +0.68j])

                :param dtype: The data type of the returned array.
                :type dtype: :class:`numpy.dtype`
                :rtype: :class:`numpy.ndarray`
                """
                if self.ndim == 0:
                    return value(self.item(0))
                arr, itemset, item = self._create_empty(dtype=dtype)
                for i in xrange(self.size):
                    itemset(i, value(item(i)))
                return arr

            def uncertainty(self, dtype=None):
                """The result of :func:`~.core.uncertainty` for each element in the array.

                **Example**::

                    >>> r = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
                    >>> r.uncertainty(float)
                    array([0.18, 0.12, 0.19])
                    >>> c = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> c.uncertainty()
                    array([StandardUncertainty(real=0.6, imag=0.6),
                           StandardUncertainty(real=1.4, imag=0.2),
                           StandardUncertainty(real=0.9, imag=0.9)], dtype=object)

                :param dtype: The data type of the returned array.
                :type dtype: :class:`numpy.dtype`
                :rtype: :class:`numpy.ndarray`
                """
                if self.ndim == 0:
                    return uncertainty(self.item(0))
                arr, itemset, item = self._create_empty(dtype=dtype)
                for i in xrange(self.size):
                    itemset(i, uncertainty(item(i)))
                return arr

            def variance(self, dtype=None):
                """The result of :func:`~.core.variance` for each element in the array.

                **Example**::

                    >>> r = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
                    >>> r.variance(float)
                    array([0.0324, 0.0144, 0.0361])
                    >>> c = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.5, 0.5)), ucomplex(1.5j, 0.9)])
                    >>> c.variance()
                    array([VarianceCovariance(rr=0.36, ri=0.0, ir=0.0, ii=0.36),
                           VarianceCovariance(rr=2.25, ri=0.0, ir=0.0, ii=0.25),
                           VarianceCovariance(rr=0.81, ri=0.0, ir=0.0, ii=0.81)], dtype=object)

                :param dtype: The data type of the returned array.
                :type dtype: :class:`numpy.dtype`
                :rtype: :class:`numpy.ndarray`
                """
                if self.ndim == 0:
                    return variance(self.item(0))
                arr, itemset, item = self._create_empty(dtype=dtype)
                for i in xrange(self.size):
                    itemset(i, variance(item(i)))
                return arr

            def dof(self):
                """The result of :func:`~.core.dof` for each element in the array.

                **Example**::

                    >>> a = uarray([ureal(6, 2, df=3), ureal(4, 1, df=4), ureal(5, 3, df=7), ureal(1, 1)])
                    >>> a.dof()
                    array([ 3.,  4.,  7., inf])

                :rtype: :class:`numpy.ndarray`
                """
                if self.ndim == 0:
                    return dof(self.item(0))
                arr, itemset, item = self._create_empty(dtype=float)
                for i in xrange(self.size):
                    itemset(i, dof(item(i)))
                return arr

            def conjugate(self):
                """The result of :meth:`UncertainReal.conjugate() <.lib.UncertainReal.conjugate>`
                or :meth:`UncertainComplex.conjugate() <.lib.UncertainComplex.conjugate>`
                for each element in the array.

                **Example**::

                    >>> a = uarray([ucomplex(1.2-0.5j, 0.6), ucomplex(3.2+1.2j, (1.4, 0.2)), ucomplex(1.5j, 0.9)])
                    >>> a.conjugate()
                    uarray([ucomplex((1.2+0.5j), u=[0.6,0.6], r=0.0, df=inf),
                            ucomplex((3.2-1.2j), u=[1.4,0.2], r=0.0, df=inf),
                            ucomplex((0-1.5j), u=[0.9,0.9], r=0.0, df=inf)])

                :rtype: :class:`UncertainArray`
                """
                # override this method because I wanted to create a custom __doc__
                return self._conjugate()

            def _conjugate(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, item(i).conjugate())
                return UncertainArray(arr)

            def _positive(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, +item(i))
                return UncertainArray(arr)

            def _negative(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, -item(i))
                return UncertainArray(arr)

            def _add(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, lhs(i) + rhs(i))
                return UncertainArray(arr)

            def _subtract(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, lhs(i) - rhs(i))
                return UncertainArray(arr)

            def _multiply(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, lhs(i) * rhs(i))
                return UncertainArray(arr)

            def _divide(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, lhs(i) / rhs(i))
                return UncertainArray(arr)

            def _true_divide(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, lhs(i) / rhs(i))
                return UncertainArray(arr)

            def _power(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, lhs(i) ** rhs(i))
                return UncertainArray(arr)

            def _exp(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, exp(item(i)))
                return UncertainArray(arr)

            def _log(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, log(item(i)))
                return UncertainArray(arr)

            def _log10(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, log10(item(i)))
                return UncertainArray(arr)

            def _sqrt(self, *inputs):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, sqrt(item(i)))
                return UncertainArray(arr)

            def _cos(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, cos(item(i)))
                return UncertainArray(arr)

            def _sin(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, sin(item(i)))
                return UncertainArray(arr)

            def _tan(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, tan(item(i)))
                return UncertainArray(arr)

            def _arccos(self, *ignore):
                return self._acos()

            def _acos(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, acos(item(i)))
                return UncertainArray(arr)

            def _arcsin(self, *ignore):
                return self._asin()

            def _asin(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, asin(item(i)))
                return UncertainArray(arr)

            def _arctan(self, *ignore):
                return self._atan()

            def _atan(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, atan(item(i)))
                return UncertainArray(arr)

            def _arctan2(self, *inputs):
                return self._atan2(inputs[1])

            def _atan2(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty((self, inputs[0]))
                for i in xrange(self.size):
                    itemset(i, atan2(lhs(i), rhs(i)))
                return UncertainArray(arr)

            def _sinh(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, sinh(item(i)))
                return UncertainArray(arr)

            def _cosh(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, cosh(item(i)))
                return UncertainArray(arr)

            def _tanh(self, *ignore):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, tanh(item(i)))
                return UncertainArray(arr)

            def _arccosh(self, *ignore):
                return self._acosh()

            def _acosh(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, acosh(item(i)))
                return UncertainArray(arr)

            def _arcsinh(self, *ignore):
                return self._asinh()

            def _asinh(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, asinh(item(i)))
                return UncertainArray(arr)

            def _arctanh(self, *ignore):
                return self._atanh()

            def _atanh(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, atanh(item(i)))
                return UncertainArray(arr)

            def _square(self, *ignore):
                return self._mag_squared()

            def _mag_squared(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, mag_squared(item(i)))
                return UncertainArray(arr)

            def _magnitude(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, magnitude(item(i)))
                return UncertainArray(arr)

            def _phase(self):
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    itemset(i, phase(item(i)))
                return UncertainArray(arr)

            def copy(self, order='C'):
                arr, itemset, item = self._create_empty(order=order)
                for i in xrange(self.size):
                    itemset(i, +item(i))
                return UncertainArray(arr, label=self.label)

            def _equal(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, lhs(i) == rhs(i))
                return arr

            def _not_equal(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, lhs(i) != rhs(i))
                return arr

            def _less(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, lhs(i) < rhs(i))
                return arr

            def _less_equal(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, lhs(i) <= rhs(i))
                return arr

            def _greater(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, lhs(i) > rhs(i))
                return arr

            def _greater_equal(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, lhs(i) >= rhs(i))
                return arr

            def _maximum(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    a, b = lhs(i), rhs(i)
                    if _isnan(a):
                        itemset(i, a)
                    elif _isnan(b):
                        itemset(i, b)
                    elif a > b:
                        itemset(i, a)
                    else:
                        itemset(i, b)
                return UncertainArray(arr)

            def _minimum(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs)
                for i in xrange(self.size):
                    a, b = lhs(i), rhs(i)
                    if _isnan(a):
                        itemset(i, a)
                    elif _isnan(b):
                        itemset(i, b)
                    elif a < b:
                        itemset(i, a)
                    else:
                        itemset(i, b)
                return UncertainArray(arr)

            def _logical_and(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, bool(lhs(i)) and bool(rhs(i)))
                return arr

            def _logical_or(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, bool(lhs(i)) or bool(rhs(i)))
                return arr

            def _logical_xor(self, *inputs):
                arr, itemset, lhs, rhs = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, bool(lhs(i)) ^ bool(rhs(i)))
                return arr

            def _logical_not(self, *inputs):
                arr, itemset, item = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, not bool(item(i)))
                return arr

            def _isinf(self, *inputs):
                arr, itemset, item = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, _isinf(item(i)))
                return arr

            def _isnan(self, *inputs):
                arr, itemset, item = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, _isnan(item(i)))
                return arr

            def _isfinite(self, *inputs):
                arr, itemset, item = self._create_empty(inputs, dtype=bool)
                for i in xrange(self.size):
                    itemset(i, not (_isnan(item(i)) or _isinf(item(i))))
                return arr

            def _reciprocal(self, *inputs):
                arr, itemset, item = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, 1.0/item(i))
                return UncertainArray(arr)

            def _absolute(self, *inputs):
                arr, itemset, item = self._create_empty(inputs)
                for i in xrange(self.size):
                    itemset(i, abs(item(i)))
                return UncertainArray(arr)

            def __matmul__(self, other):
                # Implements the protocol used by the '@' operator defined in PEP 465.
                # import here to avoid circular imports
                from GTC.linear_algebra import matmul
                return matmul(self, other)

            def __rmatmul__(self, other):
                # Implements the protocol used by the '@' operator defined in PEP 465.
                # import here to avoid circular imports
                from GTC.linear_algebra import matmul
                return matmul(other, self)

            def round(self, decimals=0, **kwargs):
                digits = kwargs.get('digits', decimals)
                df_decimals = kwargs.get('df_decimals', digits)
                arr, itemset, item = self._create_empty()
                for i in xrange(self.size):
                    try:
                        itemset(i, item(i)._round(digits, df_decimals))
                    except AttributeError:
                        itemset(i, round(item(i), digits))
                return UncertainArray(arr)

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

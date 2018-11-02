"""
The are no public methods for an UncertainArray.

The proper way to create an uncertain array is by calling uarray(...)

This module was written in the following way so that numpy >= 1.13.0
does not have to be installed in order for someone to use GTC.
"""
from __future__ import division
from numbers import Number
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

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
        class UncertainArray(np.ndarray):

            def __new__(cls, input_array):
                return np.asarray(input_array).view(cls)

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

                return UncertainArray(attr(*inputs))

            def _negative(self, *inputs):
                return [-val for val in inputs[0]]

            def _add(self, *inputs):
                return [lhs + rhs for lhs, rhs in izip(*inputs)]

            def _subtract(self, *inputs):
                return [lhs - rhs for lhs, rhs in izip(*inputs)]

            def _multiply(self, *inputs):
                return [lhs * rhs for lhs, rhs in izip(*inputs)]

            def _divide(self, *inputs):
                return [lhs / rhs for lhs, rhs in izip(*inputs)]

            def _true_divide(self, *inputs):
                return [lhs / rhs for lhs, rhs in izip(*inputs)]

            def _power(self, *inputs):
                return [lhs ** rhs for lhs, rhs in izip(*inputs)]

            def _equal(self, *inputs):
                return [lhs == rhs for lhs, rhs in izip(*inputs)]

            def _not_equal(self, *inputs):
                return [lhs != rhs for lhs, rhs in izip(*inputs)]

            def _less(self, *inputs):
                return [lhs < rhs for lhs, rhs in izip(*inputs)]

            def _less_equal(self, *inputs):
                return [lhs <= rhs for lhs, rhs in izip(*inputs)]

            def _greater(self, *inputs):
                return [lhs > rhs for lhs, rhs in izip(*inputs)]

            def _greater_equal(self, *inputs):
                return [lhs >= rhs for lhs, rhs in izip(*inputs)]

            def _absolute(self, *inputs):
                return [abs(value) for value in inputs[0]]

            @property
            def real(self):
                return UncertainArray([item.real for item in self])

            @property
            def imag(self):
                return UncertainArray([item.imag for item in self])

            def _conjugate(self, *inputs):
                return [value.conjugate() for value in inputs[0]]

            def _cos(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.cos(x)
                return [value._cos() for value in self]

            def _sin(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.sin(x)
                return [value._sin() for value in self]

            def _tan(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.tan(x)
                return [value._tan() for value in self]

            def _arccos(self, *inputs):
                return [value._acos() for value in inputs[0]]

            def _acos(self):
                return UncertainArray([value._acos() for value in self])

            def _arcsin(self, *inputs):
                return [value._asin() for value in inputs[0]]

            def _asin(self):
                return UncertainArray([value._asin() for value in self])

            def _arctan(self, *inputs):
                return [value._atan() for value in inputs[0]]

            def _atan(self):
                return UncertainArray([value._atan() for value in self])

            def _arctan2(self, *inputs):
                return [lhs._atan2(rhs) for lhs, rhs in izip(*inputs)]

            def _atan2(self, *inputs):
                return UncertainArray([lhs._atan2(rhs) for lhs, rhs in izip(self, inputs[0])])

            def _exp(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.exp(x)
                return [value._exp() for value in self]

            def _log(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.log(x)
                return [value._log() for value in self]

            def _log10(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.log10(x)
                return [value._log10() for value in self]

            def _sqrt(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.sqrt(x)
                return [value._sqrt() for value in self]

            def _sinh(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.sinh(x)
                return [value._sinh() for value in self]

            def _cosh(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.cosh(x)
                return [value._cosh() for value in self]

            def _tanh(self, *inputs):
                # use self instead of inputs[0] for compatibility with GTC.core.tanh(x)
                return [value._tanh() for value in self]

            def _arccosh(self, *inputs):
                return [value._acosh() for value in inputs[0]]

            def _acosh(self):
                return UncertainArray([value._acosh() for value in self])

            def _arcsinh(self, *inputs):
                return [value._asinh() for value in inputs[0]]

            def _asinh(self):
                return UncertainArray([value._asinh() for value in self])

            def _arctanh(self, *inputs):
                return [value._atanh() for value in inputs[0]]

            def _atanh(self):
                return UncertainArray([value._atanh() for value in self])

            def _mag_squared(self):
                return UncertainArray([value._mag_squared() for value in self])

            def _magnitude(self):
                return UncertainArray([value._magnitude() for value in self])

            def _phase(self):
                return UncertainArray([value._phase() for value in self])

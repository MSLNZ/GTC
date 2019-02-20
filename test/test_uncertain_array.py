import unittest
import os
import math
import cmath
import tempfile
try:
    from itertools import izip  # Python 2
    import cPickle as pickle
    PY2 = True
except ImportError:
    izip = zip
    import pickle
    PY2 = False

import numpy as np

from GTC.core import (
    ureal,
    ucomplex,
    cos,
    sin,
    tan,
    acos,
    asin,
    atan,
    atan2,
    exp,
    pow,
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
    value,
    uncertainty,
    variance,
    dof
)

from GTC import inf, nan
from GTC.uncertain_array import UncertainArray
from GTC.named_tuples import GroomedUncertainReal, StandardUncertainty, VarianceCovariance
from GTC.lib import UncertainReal, UncertainComplex
from GTC import type_a, function
from GTC.linear_algebra import matmul, uarray

from testing_tools import *


class TestUncertainArray(unittest.TestCase):

    def setUp(self):
        self.x = [ureal(23, 2), ureal(26, 1), ureal(20, 3), ureal(25, 1), ureal(28, 2), ureal(24, 2)]
        self.y = [ureal(25, 2), ureal(24, 4), ureal(23, 1), ureal(25, 2), ureal(19, 6), ureal(27, 1)]
        self.xa = uarray(self.x)
        self.ya = uarray(self.y)

        self.xc = [ucomplex(23+2j, 2), ucomplex(26+1j, 1), ucomplex(20+0j, 3),
                   ucomplex(25+3j, 1), ucomplex(22+7j, 2), ucomplex(23+3j, 2)]
        self.yc = [ucomplex(25+1j, 2), ucomplex(24+4j, 4), ucomplex(23+2j, 1),
                   ucomplex(25+1j, 2), ucomplex(26+2j, 1), ucomplex(18+8j, 5)]
        self.xca = uarray(self.xc)
        self.yca = uarray(self.yc)

    def _ureal(self, x, u, df=inf, label=None, independent=True):
        # create an UncertainReal by bypassing exception handling
        return UncertainReal._elementary(x, u, df=df, label=label, independent=independent)

    def _ucomplex(self, z, u_r, u_i, r=None, df=inf, label=None, independent=True):
        # create an UncertainComplex by bypassing exception handling
        return UncertainComplex._elementary(z, u_r, u_i, r=r, df=df, label=label, independent=independent)

    def test_empty_array_like(self):
        for item in [list(), tuple()]:
            ua = uarray(item)
            self.assertTrue(isinstance(ua, UncertainArray))
            self.assertTrue(len(ua) == 0)
            self.assertTrue(ua.shape == (0,))
            self.assertTrue(ua.size == 0)

        ua = uarray([[[]]])  # an empty multi-dimensional array
        self.assertTrue(isinstance(ua, UncertainArray))
        self.assertTrue(len(ua) == 1)
        self.assertTrue(len(ua[0]) == 1)
        self.assertTrue(len(ua[0][0]) == 0)
        self.assertTrue(ua.shape == (1, 1, 0))
        self.assertTrue(ua[0].shape == (1, 0))
        self.assertTrue(ua[0, 0].shape == (0,))
        self.assertTrue(ua[0][0].shape == (0,))
        self.assertTrue(ua.size == 0)
        self.assertTrue(ua[0].size == 0)
        self.assertTrue(ua[0, 0].size == 0)
        self.assertTrue(ua[0][0].size == 0)

    def test_value_uncertainty_variance_r_dof_ureal(self):
        # make sure that a uarray of size==1 is okay
        a = uarray(ureal(1.2, 0.3, df=4.2))
        self.assertTrue(a.dtype == np.object)
        self.assertTrue(equivalent(value(a), 1.2))
        self.assertTrue(equivalent(uncertainty(a), 0.3))
        self.assertTrue(equivalent(variance(a), 0.09))
        self.assertTrue(equivalent(dof(a), 4.2))

        # a 1D array
        a = uarray([ureal(1.2, 0.3, df=3.3), ureal(2.5, 0.8, df=7)])
        self.assertTrue(a.dtype == np.object)
        self.assertTrue(equivalent(a[0].x, 1.2))
        self.assertTrue(equivalent(a[0].u, 0.3))
        self.assertTrue(equivalent(a[0].v, 0.09))
        self.assertTrue(equivalent(a[0].df, 3.3))
        self.assertTrue(equivalent(a[1].x, 2.5))
        self.assertTrue(equivalent(a[1].u, 0.8))
        self.assertTrue(equivalent(a[1].v, 0.64))
        self.assertTrue(equivalent(a[1].df, 7.0))

        # x = a.value(float)
        # self.assertTrue(x.dtype == np.float64)
        x = value(a)
        self.assertTrue(equivalent(x[0], 1.2))
        self.assertTrue(equivalent(x[1], 2.5))

        # u = a.uncertainty(float)
        # self.assertTrue(u.dtype == np.float64)
        u = uncertainty(a)
        self.assertTrue(equivalent(u[0], 0.3))
        self.assertTrue(equivalent(u[1], 0.8))

        # v = a.variance(float)
        # self.assertTrue(v.dtype == np.float64)
        v = variance(a)
        self.assertTrue(equivalent(v[0], 0.09))
        self.assertTrue(equivalent(v[1], 0.64))

        # df = a.dof()
        # self.assertTrue(df.dtype == np.float64)
        df = dof(a)
        self.assertTrue(equivalent(df[0], 3.3))
        self.assertTrue(equivalent(df[1], 7.0))

        # a 2D array
        a = uarray([[ureal(1.2, 0.3, df=1), ureal(2.5, 0.8, df=2)],
                    [ureal(-3.1, 1.1, df=3), ureal(0.3, 0.05, df=4)]])
        self.assertTrue(a.dtype == np.object)
        self.assertTrue(equivalent(a[0, 0].x, 1.2))
        self.assertTrue(equivalent(a[0, 0].u, 0.3))
        self.assertTrue(equivalent(a[0, 0].v, 0.09))
        self.assertTrue(equivalent(a[0, 0].df, 1.0))
        self.assertTrue(equivalent(a[0, 1].x, 2.5))
        self.assertTrue(equivalent(a[0, 1].u, 0.8))
        self.assertTrue(equivalent(a[0, 1].v, 0.64))
        self.assertTrue(equivalent(a[0, 1].df, 2.0))
        self.assertTrue(equivalent(a[1, 0].x, -3.1))
        self.assertTrue(equivalent(a[1, 0].u, 1.1))
        self.assertTrue(equivalent(a[1, 0].v, 1.21))
        self.assertTrue(equivalent(a[1, 0].df, 3.0))
        self.assertTrue(equivalent(a[1, 1].x, 0.3))
        self.assertTrue(equivalent(a[1, 1].u, 0.05))
        self.assertTrue(equivalent(a[1, 1].v, 0.0025))
        self.assertTrue(equivalent(a[1, 1].df, 4.0))

        # x = a.value(float)
        # self.assertTrue(x.dtype == np.float64)
        x = value(a)
        self.assertTrue(equivalent(x[0, 0], 1.2))
        self.assertTrue(equivalent(x[0, 1], 2.5))
        self.assertTrue(equivalent(x[1, 0], -3.1))
        self.assertTrue(equivalent(x[1, 1], 0.3))

        # u = a.uncertainty(float)
        # self.assertTrue(u.dtype == np.float64)
        u = uncertainty(a)
        self.assertTrue(equivalent(u[0, 0], 0.3))
        self.assertTrue(equivalent(u[0, 1], 0.8))
        self.assertTrue(equivalent(u[1, 0], 1.1))
        self.assertTrue(equivalent(u[1, 1], 0.05))

        # v = a.variance(float)
        # self.assertTrue(v.dtype == np.float64)
        v = variance(a)
        self.assertTrue(equivalent(v[0, 0], 0.09))
        self.assertTrue(equivalent(v[0, 1], 0.64))
        self.assertTrue(equivalent(v[1, 0], 1.21))
        self.assertTrue(equivalent(v[1, 1], 0.0025))

        # df = a.dof()
        # self.assertTrue(df.dtype == np.float64)
        df = dof(a)
        self.assertTrue(equivalent(df[0, 0], 1.0))
        self.assertTrue(equivalent(df[0, 1], 2.0))
        self.assertTrue(equivalent(df[1, 0], 3.0))
        self.assertTrue(equivalent(df[1, 1], 4.0))

        # # the ucomplex class has a `r` property, but ureal does not
        # with self.assertRaises(AttributeError) as err:
            # _ = a.r
        # self.assertTrue(str(err.exception) == "'UncertainReal' object has no attribute 'r'")

    def test_value_uncertainty_variance_r_dof_ucomplex(self):
        # # make sure that a uarray of size==1 is okay
        # a = uarray(ucomplex(1.2-0.5j, (1.2, 0.7, 0.7, 2.2), df=4.2))
        # self.assertTrue(a.dtype == np.object)
        # self.assertTrue(equivalent_complex(value(a), 1.2-0.5j))
        # self.assertTrue(isinstance(uncertainty(a), StandardUncertainty))
        # self.assertTrue(equivalent(uncertainty(a).real, 1.0954451150103321))
        # self.assertTrue(equivalent(uncertainty(a).imag, 1.4832396974191326))
        # self.assertTrue(isinstance(variance(a), VarianceCovariance))
        # self.assertTrue(equivalent(variance(a).rr, 1.2))
        # self.assertTrue(equivalent(variance(a).ri, 0.7))
        # self.assertTrue(equivalent(variance(a).ir, 0.7))
        # self.assertTrue(equivalent(variance(a).ii, 2.2))
        # self.assertTrue(equivalent(a.r, 0.2651515151515152))
        # self.assertTrue(equivalent(dof(a), 4.2))

        a = uarray([ucomplex(1.2-0.5j, (1.2, 0.7, 0.7, 2.2), df=4.2),
                    ucomplex(-0.2+1.2j, (0.9, 0.4, 0.4, 1.5), df=2.6)])
        self.assertTrue(a.dtype == np.object)
        self.assertTrue(equivalent_complex(a[0].x, 1.2-0.5j))
        self.assertTrue(isinstance(a[0].u, StandardUncertainty))
        self.assertTrue(equivalent(a[0].u.real, 1.0954451150103321))
        self.assertTrue(equivalent(a[0].u.imag, 1.4832396974191326))
        self.assertTrue(isinstance(a[0].v, VarianceCovariance))
        self.assertTrue(equivalent(a[0].v.rr, 1.2))
        self.assertTrue(equivalent(a[0].v.ri, 0.7))
        self.assertTrue(equivalent(a[0].v.ir, 0.7))
        self.assertTrue(equivalent(a[0].v.ii, 2.2))
        self.assertTrue(equivalent(a[0].r, 0.2651515151515152))
        self.assertTrue(equivalent(a[0].df, 4.2))
        self.assertTrue(equivalent_complex(a[1].x, -0.2+1.2j))
        self.assertTrue(isinstance(a[1].u, StandardUncertainty))
        self.assertTrue(equivalent(a[1].u.real, 0.9486832980505138))
        self.assertTrue(equivalent(a[1].u.imag, 1.224744871391589))
        self.assertTrue(isinstance(a[1].v, VarianceCovariance))
        self.assertTrue(equivalent(a[1].v.rr, 0.9))
        self.assertTrue(equivalent(a[1].v.ri, 0.4))
        self.assertTrue(equivalent(a[1].v.ir, 0.4))
        self.assertTrue(equivalent(a[1].v.ii, 1.5))
        self.assertTrue(equivalent(a[1].r, 0.29629629629629634))
        self.assertTrue(equivalent(a[1].df, 2.6))

        # x = a.value(complex)
        # self.assertTrue(x.dtype == np.complex128)
        x = value(a)
        self.assertTrue(equivalent_complex(x[0], 1.2-0.5j))
        self.assertTrue(equivalent_complex(x[1], -0.2+1.2j))

        # u = a.uncertainty()
        # self.assertTrue(u.dtype == np.object)
        u = uncertainty(a)
        self.assertTrue(isinstance(u[0], StandardUncertainty))
        self.assertTrue(equivalent(u[0].real, 1.0954451150103321))
        self.assertTrue(equivalent(u[0].imag, 1.4832396974191326))
        self.assertTrue(isinstance(u[1], StandardUncertainty))
        self.assertTrue(equivalent(u[1].real, 0.9486832980505138))
        self.assertTrue(equivalent(u[1].imag, 1.224744871391589))

        # v = a.variance()
        # self.assertTrue(v.dtype == np.object)
        v = variance(a)
        self.assertTrue(isinstance(v[0], VarianceCovariance))
        self.assertTrue(equivalent(v[0].rr, 1.2))
        self.assertTrue(equivalent(v[0].ri, 0.7))
        self.assertTrue(equivalent(v[0].ir, 0.7))
        self.assertTrue(equivalent(v[0].ii, 2.2))
        self.assertTrue(isinstance(v[1], VarianceCovariance))
        self.assertTrue(equivalent(v[1].rr, 0.9))
        self.assertTrue(equivalent(v[1].ri, 0.4))
        self.assertTrue(equivalent(v[1].ir, 0.4))
        self.assertTrue(equivalent(v[1].ii, 1.5))

        # df = a.dof()
        # self.assertTrue(df.dtype == np.float64)
        df = dof(a)
        self.assertTrue(equivalent(df[0], 4.2))
        self.assertTrue(equivalent(df[1], 2.6))

        r = a.r
        self.assertTrue(r.dtype == np.float_)
        self.assertTrue(equivalent(r[0], 0.2651515151515152))
        self.assertTrue(equivalent(r[1], 0.29629629629629634))

        # a 2D array
        a = uarray([[ucomplex(1.2-0.5j, (1.2, 0.7, 0.7, 2.2), df=4.2),
                     ucomplex(-0.2+1.2j, (0.9, 0.4, 0.4, 1.5), df=2.6)],
                    [ucomplex(6.3-1.5j, (3.4, 0.21, 0.21, 2.3), df=10.3),
                     ucomplex(8.7j, (1.4, 0.85, 0.85, 1.8), df=8.8)]])
        self.assertTrue(a.dtype == np.object)
        self.assertTrue(equivalent_complex(a[0, 0].x, 1.2-0.5j))
        self.assertTrue(isinstance(a[0, 0].u, StandardUncertainty))
        self.assertTrue(equivalent(a[0, 0].u.real, 1.0954451150103321))
        self.assertTrue(equivalent(a[0, 0].u.imag, 1.4832396974191326))
        self.assertTrue(isinstance(a[0, 0].v, VarianceCovariance))
        self.assertTrue(equivalent(a[0, 0].v.rr, 1.2))
        self.assertTrue(equivalent(a[0, 0].v.ri, 0.7))
        self.assertTrue(equivalent(a[0, 0].v.ir, 0.7))
        self.assertTrue(equivalent(a[0, 0].v.ii, 2.2))
        self.assertTrue(equivalent(a[0, 0].r, 0.2651515151515152))
        self.assertTrue(equivalent(a[0, 0].df, 4.2))
        self.assertTrue(equivalent_complex(a[0, 1].x, -0.2+1.2j))
        self.assertTrue(isinstance(a[0, 1].u, StandardUncertainty))
        self.assertTrue(equivalent(a[0, 1].u.real, 0.9486832980505138))
        self.assertTrue(equivalent(a[0, 1].u.imag, 1.224744871391589))
        self.assertTrue(isinstance(a[0, 1].v, VarianceCovariance))
        self.assertTrue(equivalent(a[0, 1].v.rr, 0.9))
        self.assertTrue(equivalent(a[0, 1].v.ri, 0.4))
        self.assertTrue(equivalent(a[0, 1].v.ir, 0.4))
        self.assertTrue(equivalent(a[0, 1].v.ii, 1.5))
        self.assertTrue(equivalent(a[0, 1].r, 0.29629629629629634))
        self.assertTrue(equivalent(a[0, 1].df, 2.6))
        self.assertTrue(equivalent_complex(a[1, 0].x, 6.3-1.5j))
        self.assertTrue(isinstance(a[1, 0].u, StandardUncertainty))
        self.assertTrue(equivalent(a[1, 0].u.real, 1.8439088914585775))
        self.assertTrue(equivalent(a[1, 0].u.imag, 1.51657508881031))
        self.assertTrue(isinstance(a[1, 0].v, VarianceCovariance))
        self.assertTrue(equivalent(a[1, 0].v.rr, 3.4))
        self.assertTrue(equivalent(a[1, 0].v.ri, 0.21))
        self.assertTrue(equivalent(a[1, 0].v.ir, 0.21))
        self.assertTrue(equivalent(a[1, 0].v.ii, 2.3))
        self.assertTrue(equivalent(a[1, 0].r, 0.026854219948849102))
        self.assertTrue(equivalent(a[1, 0].df, 10.3))
        self.assertTrue(equivalent_complex(a[1, 1].x, 8.7j))
        self.assertTrue(isinstance(a[1, 1].u, StandardUncertainty))
        self.assertTrue(equivalent(a[1, 1].u.real, 1.1832159566199232))
        self.assertTrue(equivalent(a[1, 1].u.imag, 1.3416407864998738))
        self.assertTrue(isinstance(a[1, 1].v, VarianceCovariance))
        self.assertTrue(equivalent(a[1, 1].v.rr, 1.4))
        self.assertTrue(equivalent(a[1, 1].v.ri, 0.85))
        self.assertTrue(equivalent(a[1, 1].v.ir, 0.85))
        self.assertTrue(equivalent(a[1, 1].v.ii, 1.8))
        self.assertTrue(equivalent(a[1, 1].r, 0.33730158730158727))
        self.assertTrue(equivalent(a[1, 1].df, 8.8))

        # x = a.value(complex)
        # self.assertTrue(x.dtype == np.complex128)
        x = value(a)
        self.assertTrue(equivalent_complex(x[0, 0], 1.2-0.5j))
        self.assertTrue(equivalent_complex(x[0, 1], -0.2+1.2j))
        self.assertTrue(equivalent_complex(x[1, 0], 6.3-1.5j))
        self.assertTrue(equivalent_complex(x[1, 1], 8.7j))

        # u = a.uncertainty()
        # self.assertTrue(u.dtype == np.object)
        u = uncertainty(a)
        self.assertTrue(isinstance(u[0, 0], StandardUncertainty))
        self.assertTrue(equivalent(u[0, 0].real, 1.0954451150103321))
        self.assertTrue(equivalent(u[0, 0].imag, 1.4832396974191326))
        self.assertTrue(isinstance(u[0, 1], StandardUncertainty))
        self.assertTrue(equivalent(u[0, 1].real, 0.9486832980505138))
        self.assertTrue(equivalent(u[0, 1].imag, 1.224744871391589))
        self.assertTrue(isinstance(u[1, 0], StandardUncertainty))
        self.assertTrue(equivalent(u[1, 0].real, 1.8439088914585775))
        self.assertTrue(equivalent(u[1, 0].imag, 1.51657508881031))
        self.assertTrue(isinstance(u[1, 1], StandardUncertainty))
        self.assertTrue(equivalent(u[1, 1].real, 1.1832159566199232))
        self.assertTrue(equivalent(u[1, 1].imag, 1.3416407864998738))

        # v = a.variance()
        # self.assertTrue(v.dtype == np.object)
        v = variance(a)
        self.assertTrue(isinstance(v[0, 0], VarianceCovariance))
        self.assertTrue(equivalent(v[0, 0].rr, 1.2))
        self.assertTrue(equivalent(v[0, 0].ri, 0.7))
        self.assertTrue(equivalent(v[0, 0].ir, 0.7))
        self.assertTrue(equivalent(v[0, 0].ii, 2.2))
        self.assertTrue(isinstance(v[0, 1], VarianceCovariance))
        self.assertTrue(equivalent(v[0, 1].rr, 0.9))
        self.assertTrue(equivalent(v[0, 1].ri, 0.4))
        self.assertTrue(equivalent(v[0, 1].ir, 0.4))
        self.assertTrue(equivalent(v[0, 1].ii, 1.5))
        self.assertTrue(isinstance(v[1, 0], VarianceCovariance))
        self.assertTrue(equivalent(v[1, 0].rr, 3.4))
        self.assertTrue(equivalent(v[1, 0].ri, 0.21))
        self.assertTrue(equivalent(v[1, 0].ir, 0.21))
        self.assertTrue(equivalent(v[1, 0].ii, 2.3))
        self.assertTrue(isinstance(v[1, 1], VarianceCovariance))
        self.assertTrue(equivalent(v[1, 1].rr, 1.4))
        self.assertTrue(equivalent(v[1, 1].ri, 0.85))
        self.assertTrue(equivalent(v[1, 1].ir, 0.85))
        self.assertTrue(equivalent(v[1, 1].ii, 1.8))

        # df = a.dof()
        # self.assertTrue(df.dtype == np.float64)
        df = dof(a)
        self.assertTrue(equivalent(df[0, 0], 4.2))
        self.assertTrue(equivalent(df[0, 1], 2.6))
        self.assertTrue(equivalent(df[1, 0], 10.3))
        self.assertTrue(equivalent(df[1, 1], 8.8))

        r = a.r
        self.assertTrue(r.dtype == np.float_)
        self.assertTrue(equivalent(r[0, 0], 0.2651515151515152))
        self.assertTrue(equivalent(r[0, 1], 0.29629629629629634))
        self.assertTrue(equivalent(r[1, 0], 0.026854219948849102))
        self.assertTrue(equivalent(r[1, 1], 0.33730158730158727))

    def test_value_uncertainty_variance_dof_mixed(self):
        a = uarray([ureal(1, 1, df=7), ucomplex(1+1j, (1, 0.5, 0.5, 1), df=4),
                    7, 3.2, 8-3j, self._ureal(inf, 9, df=nan), nan])

        # x = a.value()
        # self.assertTrue(x.dtype == np.object)
        x = value(a)
        self.assertTrue(equivalent(x[0], 1))
        self.assertTrue(equivalent_complex(x[1], 1+1j))
        self.assertTrue(equivalent(x[2], 7))
        self.assertTrue(equivalent(x[3], 3.2))
        self.assertTrue(equivalent_complex(x[4], 8-3j))
        self.assertTrue(cmath.isinf(x[5]))
        self.assertTrue(math.isnan(x[6]))

        # x = a.value(complex)
        # self.assertTrue(x.dtype == np.complex128)
        x = value(a)
        self.assertTrue(equivalent_complex(x[0], 1+0j))
        self.assertTrue(equivalent_complex(x[1], 1+1j))
        self.assertTrue(equivalent_complex(x[2], 7+0j))
        self.assertTrue(equivalent_complex(x[3], 3.2+0j))
        self.assertTrue(equivalent_complex(x[4], 8-3j))
        self.assertTrue(cmath.isinf(x[5]))
        self.assertTrue(cmath.isnan(x[6]))

        u = uncertainty(a)
        self.assertTrue(equivalent(u[0], 1))
        self.assertTrue(isinstance(u[1], StandardUncertainty))
        self.assertTrue(equivalent(u[1].real, 1))
        self.assertTrue(equivalent(u[1].imag, 1))
        self.assertTrue(equivalent(u[2], 0))
        self.assertTrue(equivalent(u[3], 0))
        self.assertTrue(equivalent(u[4], 0))
        self.assertTrue(equivalent(u[5], 9))
        self.assertTrue(equivalent(u[6], 0))

        v = variance(a)
        self.assertTrue(equivalent(v[0], 1.0))
        self.assertTrue(isinstance(v[1], VarianceCovariance))
        self.assertTrue(equivalent(v[1].rr, 1.0))
        self.assertTrue(equivalent(v[1].ri, 0.5))
        self.assertTrue(equivalent(v[1].ir, 0.5))
        self.assertTrue(equivalent(v[1].ii, 1.0))
        self.assertTrue(equivalent(v[2], 0.0))
        self.assertTrue(equivalent(v[3], 0.0))
        self.assertTrue(equivalent(v[4], 0.0))
        self.assertTrue(equivalent(v[5], 81.0))
        self.assertTrue(equivalent(v[6], 0))

        df = dof(a)
        self.assertTrue(equivalent(df[0], 7))
        self.assertTrue(equivalent(df[1], 4))
        self.assertTrue(math.isinf(df[2]))
        self.assertTrue(math.isinf(df[3]))
        self.assertTrue(math.isinf(df[4]))
        self.assertTrue(math.isnan(df[5]))
        self.assertTrue(math.isinf(df[6]))

    def test_label(self):
        # no label
        ur = uarray(np.arange(5) * ureal(1, 1))
        self.assertTrue(isinstance(ur, UncertainArray))
        self.assertTrue(ur.label is None)

        uc = uarray(np.arange(5) * ucomplex(1j, 1))
        self.assertTrue(isinstance(uc, UncertainArray))
        self.assertTrue(uc.label is None)

        # ureal elements -> the label gets passed along with copies, views, slices, ...
        a = uarray(np.arange(5) * ureal(1, 1), label='ureal')
        self.assertTrue(isinstance(a, UncertainArray))
        self.assertTrue(a.label == 'ureal')

        sub = a[1:]
        self.assertTrue(isinstance(sub, UncertainArray))
        self.assertTrue(sub.label == 'ureal')

        v = a.view()
        self.assertTrue(isinstance(v, UncertainArray))
        self.assertTrue(v.label == 'ureal')

        c = a.copy()
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(c.label == 'ureal')

        t = np.take(a, [1, 2])
        self.assertTrue(isinstance(t, UncertainArray))
        self.assertTrue(t.label == 'ureal')

        r = np.ravel(a)
        self.assertTrue(isinstance(r, UncertainArray))
        self.assertTrue(r.label == 'ureal')

        real = a.real
        self.assertTrue(isinstance(real, UncertainArray))
        self.assertTrue(real.label is None)

        imag = a.imag
        self.assertTrue(isinstance(imag, UncertainArray))
        self.assertTrue(imag.label is None)

        pos = +self.xa
        self.assertTrue(isinstance(pos, UncertainArray))
        self.assertTrue(pos.label is None)

        neg = +self.xa
        self.assertTrue(isinstance(pos, UncertainArray))
        self.assertTrue(neg.label is None)

        a2 = a * a
        self.assertTrue(isinstance(a2, UncertainArray))
        self.assertTrue(a2.label is None)

        with self.assertRaises(AttributeError):
            a2.label = 'a squared'  # AttributeError: can't set attribute

        # ucomplex elements -> the label gets passed along with copies, views, slices, ...
        a = uarray(np.arange(5) * ucomplex(1+1j, 1), label='ucomplex')
        self.assertTrue(isinstance(a, UncertainArray))
        self.assertTrue(a.label == 'ucomplex')

        sub = a[1:]
        self.assertTrue(isinstance(sub, UncertainArray))
        self.assertTrue(sub.label == 'ucomplex')

        v = a.view()
        self.assertTrue(isinstance(v, UncertainArray))
        self.assertTrue(v.label == 'ucomplex')

        c = a.copy()
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(c.label == 'ucomplex')

        t = np.take(a, [1, 2])
        self.assertTrue(isinstance(t, UncertainArray))
        self.assertTrue(t.label == 'ucomplex')

        r = np.ravel(a)
        self.assertTrue(isinstance(r, UncertainArray))
        self.assertTrue(r.label == 'ucomplex')

        real = a.real
        self.assertTrue(isinstance(real, UncertainArray))
        self.assertTrue(real.label is None)

        imag = a.imag
        self.assertTrue(isinstance(imag, UncertainArray))
        self.assertTrue(imag.label is None)

        pos = +self.xa
        self.assertTrue(isinstance(pos, UncertainArray))
        self.assertTrue(pos.label is None)

        neg = +self.xa
        self.assertTrue(isinstance(pos, UncertainArray))
        self.assertTrue(neg.label is None)

        a2 = a * a
        self.assertTrue(isinstance(a2, UncertainArray))
        self.assertTrue(a2.label is None)

        with self.assertRaises(AttributeError):
            a2.label = 'a squared'  # AttributeError: can't set attribute

    def test_real(self):
        # # make sure that a uarray of size==1 is okay
        # a = uarray(ureal(1.2, 0.3))
        # self.assertTrue(isinstance(a, UncertainArray))
        # self.assertTrue(equivalent(a.real.x, 1.2))
        # self.assertTrue(equivalent(a.real.u, 0.3))

        # # make sure that a uarray of size==1 is okay
        # a = uarray(ucomplex(1.2+3j, (0.3, 0.1)))
        # self.assertTrue(isinstance(a, UncertainArray))
        # self.assertTrue(equivalent(a.real.x, 1.2))
        # self.assertTrue(equivalent(a.real.u, 0.3))

        n = len(self.xc)
        z = [x.real for x in self.x]
        zc = [x.real for x in self.xc]

        # call np.real
        za = np.real(self.xa)
        self.assertTrue(isinstance(za, UncertainArray))
        zca = np.real(self.xca)
        self.assertTrue(isinstance(zca, UncertainArray))
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

        # call UncertainArray.real
        za = self.xa.real
        self.assertTrue(isinstance(za, UncertainArray))
        zca = self.xca.real
        self.assertTrue(isinstance(zca, UncertainArray))
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

    def test_imag(self):
        # # make sure that a uarray of size==1 is okay
        # a = uarray(ureal(1.2, 0.3))
        # self.assertTrue(isinstance(a, UncertainArray))
        # self.assertTrue(equivalent(a.imag.x, 0.0))
        # self.assertTrue(equivalent(a.imag.u, 0.0))

        # # make sure that a uarray of size==1 is okay
        # a = uarray(ucomplex(1.2+3j, (0.3, 0.1)))
        # self.assertTrue(isinstance(a, UncertainArray))
        # self.assertTrue(equivalent(a.imag.x, 3))
        # self.assertTrue(equivalent(a.imag.u, 0.1))

        n = len(self.xc)
        z = [x.imag for x in self.x]
        zc = [x.imag for x in self.xc]

        # call np.imag
        za = np.imag(self.xa)
        self.assertTrue(isinstance(za, UncertainArray))
        zca = np.imag(self.xca)
        self.assertTrue(isinstance(zca, UncertainArray))
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

        # call UncertainArray.imag
        za = self.xa.imag
        self.assertTrue(isinstance(za, UncertainArray))
        zca = self.xca.imag
        self.assertTrue(isinstance(zca, UncertainArray))
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

    def test_not_number_like(self):
        # don't initially care that these are not a number type
        # an error will eventually be raised

        bad1 = uarray(['hi'])
        bad2 = uarray([None])
        a = uarray([ureal(1, 1)])
        with self.assertRaises(TypeError):
            _ = a + bad1  # can only concatenate str (not "UncertainReal") to str

        with self.assertRaises(TypeError):
            _ = a + bad2  # unsupported operand type(s) for +: 'NoneType' and 'UncertainReal'

    def test_positive_unary(self):
        # # single element case 
        # a = uarray(ureal(1.2, 0.3, df=4.2))
        # pos = +a 
        # self.assertTrue( isinstance(pos,UncertainReal) )
        # self.assertTrue( equivalent(pos.x,1.2) ) 
        # self.assertTrue( equivalent(pos.u,0.3) )  
        
        pos = +self.xa
        self.assertTrue(pos is not self.xa)
        for i in range(len(self.xa)):
            self.assertTrue(equivalent(self.x[i].x, pos[i].x))
            self.assertTrue(equivalent(self.x[i].u, pos[i].u))
            self.assertTrue(equivalent(self.x[i].df, pos[i].df))

        pos = np.positive(self.xa)
        self.assertTrue(pos is not self.xa)
        for i in range(len(self.xa)):
            self.assertTrue(equivalent(self.x[i].x, pos[i].x))
            self.assertTrue(equivalent(self.x[i].u, pos[i].u))
            self.assertTrue(equivalent(self.x[i].df, pos[i].df))

    def test_negative_unary(self):
        # # single element case 
        # a = uarray(ureal(1.2, 0.3, df=4.2))
        # neg = -a 
        # self.assertTrue( isinstance(neg,UncertainReal) )
        # self.assertTrue( equivalent(neg.x,-1.2) ) 
        # self.assertTrue( equivalent(neg.u,0.3) )  
        
        neg = -self.xa
        self.assertTrue(neg is not self.xa)
        for i in range(len(self.xa)):
            self.assertTrue(equivalent(-self.x[i].x, neg[i].x))
            self.assertTrue(equivalent(self.x[i].u, neg[i].u))
            self.assertTrue(equivalent(self.x[i].df, neg[i].df))

        neg = np.negative(self.xa)
        self.assertTrue(neg is not self.xa)
        for i in range(len(self.xa)):
            self.assertTrue(equivalent(-self.x[i].x, neg[i].x))
            self.assertTrue(equivalent(self.x[i].u, neg[i].u))
            self.assertTrue(equivalent(self.x[i].df, neg[i].df))

    def test_add(self): 
        # # Single element case 
        # l = uarray( ureal(1.2,0.3,4) )
        # r = uarray( ureal(-0.2,0.4,7) )
        # y = l + r 
        # self.assertTrue( isinstance(y,UncertainReal) )
        # self.assertTrue( equivalent(y.x,1.0) )
        # self.assertTrue( equivalent(y.u,math.sqrt(0.25)) )
        
        n = len(self.x)

        # 1D array of uncertain numbers, no vectorization
        z = [x + y for x, y in izip(self.x, self.y)]
        zc = [x + y for x, y in izip(self.xc, self.yc)]
        za = self.xa + self.ya
        zca = self.xca + self.yca
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # 1D array of uncertain numbers, with vectorization
        # also switch the addition order to be y + x
        for m in range(n):
            z = [y + x for y, x in izip(self.y[m:], self.x[m:])]
            zc = [y + x for y, x in izip(self.yc[m:], self.xc[m:])]
            za = self.ya[m:] + self.xa[m:]
            zca = self.yca[m:] + self.xca[m:]
            for i in range(n-m):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # reshape the x and y arrays to be 2D
        z = [[self.x[0]+self.y[0], self.x[1]+self.y[1]],
             [self.x[2]+self.y[2], self.x[3]+self.y[3]],
             [self.x[4]+self.y[4], self.x[5]+self.y[5]]]
        zc = [[self.xc[0]+self.yc[0], self.xc[1]+self.yc[1]],
              [self.xc[2]+self.yc[2], self.xc[3]+self.yc[3]],
              [self.xc[4]+self.yc[4], self.xc[5]+self.yc[5]]]
        xa = self.xa.reshape(3, 2)
        ya = self.ya.reshape(3, 2)
        xca = self.xca.reshape(3, 2)
        yca = self.yca.reshape(3, 2)

        # 2D array of uncertain numbers, no vectorization
        za = xa + ya
        zca = xca + yca
        for i in range(3):
            for j in range(2):
                self.assertTrue(equivalent(z[i][j].x, za[i, j].x))
                self.assertTrue(equivalent(z[i][j].u, za[i, j].u))
                self.assertTrue(equivalent_complex(zc[i][j].x, zca[i, j].x))
                self.assertTrue(equivalent(zc[i][j].u.real, zca[i, j].u.real))
                self.assertTrue(equivalent(zc[i][j].u.imag, zca[i, j].u.imag))

        # 2D array of uncertain numbers, vectorization with the first column
        za = xa[:, 0] + ya[:, 0]
        zca = xca[:, 0] + yca[:, 0]
        for i in range(3):
            self.assertTrue(equivalent(z[i][0].x, za[i].x))
            self.assertTrue(equivalent(z[i][0].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i][0].x, zca[i].x))
            self.assertTrue(equivalent(zc[i][0].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i][0].u.imag, zca[i].u.imag))

        # 2D array of uncertain numbers, vectorization with the second row
        za = xa[1, :] + ya[1, :]
        zca = xca[1, :] + yca[1, :]
        for j in range(2):
            self.assertTrue(equivalent(z[1][j].x, za[j].x))
            self.assertTrue(equivalent(z[1][j].u, za[j].u))
            self.assertTrue(equivalent_complex(zc[1][j].x, zca[j].x))
            self.assertTrue(equivalent(zc[1][j].u.real, zca[j].u.real))
            self.assertTrue(equivalent(zc[1][j].u.imag, zca[j].u.imag))

        # check addition with a number, UncertainArray[ureal] + number
        for number in [88, 23.32, ureal(1.2, 0.4)]:
            z = [x + number for x in self.x]
            za = self.xa + number
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # check addition with a number, UncertainArray[ucomplex] + number
        for number in [88, 23.32, 16.3-5.2j, ucomplex(1.2j, 0.4)]:
            zc = [x + number for x in self.xc]
            zca = self.xca + number
            for i in range(n):
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check addition with a number, number + UncertainArray[ureal]
        for number in [-5, 1.8e4, ureal(-0.3, 0.015)]:
            z = [number + x for x in self.x]
            za = number + self.xa
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # check addition with a number, number + UncertainArray[ucomplex]
        for number in [88, 23.32, 16.3-5.2j, ucomplex(1.2j, 0.4)]:
            zc = [number + x for x in self.xc]
            zca = number + self.xca
            for i in range(n):
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check addition with a "regular" ndarray -> UncertainArray[ureal] + ndarray
        my_array = np.ones(len(z)) * 8.12
        z = [x + 8.12 for x in self.x]
        za = self.xa + my_array
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check addition with a "regular" ndarray -> UncertainArray[ucomplex] + ndarray
        my_array = np.ones(len(z)) * (8.12+4.3j)
        zc = [x + (8.12+4.3j) for x in self.xc]
        zca = self.xca + my_array
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check addition with a "regular" ndarray -> ndarray + UncertainArray[ureal]
        my_array = np.ones(len(z)) * -6.2
        z = [-6.2 + x for x in self.x]
        za = my_array + self.xa
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check addition with a "regular" ndarray -> ndarray + UncertainArray[ucomplex]
        my_array = np.ones(len(z)) * (-6.2-0.3j)
        zc = [(-6.2-0.3j) + x for x in self.xc]
        zca = my_array + self.xca
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check addition with a list -> UncertainArray[ureal] + list
        my_list = list(range(n))
        z = [x + val for x, val in izip(self.x, my_list)]
        za = self.xa + my_list
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check addition with a list -> UncertainArray[ucomplex] + list
        my_list = [1+3j, 5j, -3+2.2j, 0.1+0.4j, 8., 1.9+3.4j]
        zc = [x + val for x, val in izip(self.xc, my_list)]
        zca = self.xca + my_list
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check addition with a list -> list + UncertainArray[ureal]
        my_list = list(range(n))
        z = [val + x for val, x in izip(my_list, self.x)]
        za = my_list + self.xa
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check addition with a list -> list + UncertainArray[ucomplex]
        my_list = [3.1-2.3j, 4+5j, 3.2+7.3j, 5.1-0.4j, 0.1, 6.1+3.7j]
        zc = [val + x for val, x in izip(my_list, self.xc)]
        zca = my_list + self.xca
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        a = uarray([nan])
        b = uarray([ureal(1, 1)])
        c = a + b
        self.assertTrue(math.isnan(value(c)))
        self.assertTrue(equivalent(uncertainty(c), 1))
        self.assertTrue(math.isinf(dof(c)))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            _ = self.xa + self.ya[3:]

        with self.assertRaises(ValueError):
            _ = self.xa + [1, 2]

    def test_subtract(self):
        # # Single element case 
        # l = uarray( ureal(1.2,0.3,4) )
        # r = uarray( ureal(-0.2,0.4,7) )
        # y = l - r 
        # self.assertTrue( isinstance(y,UncertainReal) )
        # self.assertTrue( equivalent(y.x,1.4) )
        # self.assertTrue( equivalent(y.u,math.sqrt(0.25)) )

        n = len(self.x)

        # 1D array of uncertain numbers, no vectorization
        z = [x - y for x, y in izip(self.x, self.y)]
        zc = [x - y for x, y in izip(self.xc, self.yc)]
        za = self.xa - self.ya
        zca = self.xca - self.yca
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # 1D array of uncertain numbers, with vectorization
        # also switch the subtraction order to be y - x
        for m in range(n):
            z = [y - x for y, x in izip(self.y[m:], self.x[m:])]
            zc = [y - x for y, x in izip(self.yc[m:], self.xc[m:])]
            za = self.ya[m:] - self.xa[m:]
            zca = self.yca[m:] - self.xca[m:]
            for i in range(n-m):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # reshape the x and y arrays to be 2D
        z = [[self.x[0]-self.y[0], self.x[1]-self.y[1]],
             [self.x[2]-self.y[2], self.x[3]-self.y[3]],
             [self.x[4]-self.y[4], self.x[5]-self.y[5]]]
        zc = [[self.xc[0]-self.yc[0], self.xc[1]-self.yc[1]],
              [self.xc[2]-self.yc[2], self.xc[3]-self.yc[3]],
              [self.xc[4]-self.yc[4], self.xc[5]-self.yc[5]]]
        xa = self.xa.reshape(3, 2)
        ya = self.ya.reshape(3, 2)
        xca = self.xca.reshape(3, 2)
        yca = self.yca.reshape(3, 2)

        # 2D array of uncertain numbers, no vectorization
        za = xa - ya
        zca = xca - yca
        for i in range(3):
            for j in range(2):
                self.assertTrue(equivalent(z[i][j].x, za[i, j].x))
                self.assertTrue(equivalent(z[i][j].u, za[i, j].u))
                self.assertTrue(equivalent_complex(zc[i][j].x, zca[i, j].x))
                self.assertTrue(equivalent(zc[i][j].u.real, zca[i, j].u.real))
                self.assertTrue(equivalent(zc[i][j].u.imag, zca[i, j].u.imag))

        # 2D array of uncertain numbers, vectorization with the first column
        za = xa[:, 0] - ya[:, 0]
        zca = xca[:, 0] - yca[:, 0]
        for i in range(3):
            self.assertTrue(equivalent(z[i][0].x, za[i].x))
            self.assertTrue(equivalent(z[i][0].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i][0].x, zca[i].x))
            self.assertTrue(equivalent(zc[i][0].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i][0].u.imag, zca[i].u.imag))

        # 2D array of uncertain numbers, vectorization with the second row
        za = xa[1, :] - ya[1, :]
        zca = xca[1, :] - yca[1, :]
        for j in range(2):
            self.assertTrue(equivalent(z[1][j].x, za[j].x))
            self.assertTrue(equivalent(z[1][j].u, za[j].u))
            self.assertTrue(equivalent_complex(zc[1][j].x, zca[j].x))
            self.assertTrue(equivalent(zc[1][j].u.real, zca[j].u.real))
            self.assertTrue(equivalent(zc[1][j].u.imag, zca[j].u.imag))

        # check subtraction with a number, UncertainArray[ureal] - number
        for number in [88, 23.32, ureal(1.2, 0.4)]:
            z = [x - number for x in self.x]
            za = self.xa - number
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # check subtraction with a number, UncertainArray[ucomplex] - number
        for number in [88, 23.32, 16.3-5.2j, ucomplex(1.2j, 0.4)]:
            zc = [x - number for x in self.xc]
            zca = self.xca - number
            for i in range(n):
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check subtraction with a number, number - UncertainArray[ureal]
        for number in [-5, 1.8e4, ureal(-0.3, 0.015)]:
            z = [number - x for x in self.x]
            za = number - self.xa
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # check subtraction with a number, number - UncertainArray[ucomplex]
        for number in [88, 23.32, 16.3-5.2j, ucomplex(1.2j, 0.4)]:
            zc = [number - x for x in self.xc]
            zca = number - self.xca
            for i in range(n):
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check subtraction with a "regular" ndarray -> UncertainArray[ureal] - ndarray
        my_array = np.ones(len(z)) * 8.12
        z = [x - 8.12 for x in self.x]
        za = self.xa - my_array
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check subtraction with a "regular" ndarray -> UncertainArray[ucomplex] - ndarray
        my_array = np.ones(len(z)) * (8.12+4.3j)
        zc = [x - (8.12+4.3j) for x in self.xc]
        zca = self.xca - my_array
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check subtraction with a "regular" ndarray -> ndarray - UncertainArray[ureal]
        my_array = np.ones(len(z)) * -6.2
        z = [-6.2 - x for x in self.x]
        za = my_array - self.xa
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check subtraction with a "regular" ndarray -> ndarray - UncertainArray[ucomplex]
        my_array = np.ones(len(z)) * (-6.2-0.3j)
        zc = [(-6.2-0.3j) - x for x in self.xc]
        zca = my_array - self.xca
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check subtraction with a list -> UncertainArray[ureal] - list
        my_list = list(range(n))
        z = [x - val for x, val in izip(self.x, my_list)]
        za = self.xa - my_list
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check subtraction with a list -> UncertainArray[ucomplex] - list
        my_list = [1+3j, 5j, -3+2.2j, 0.1+0.4j, 8., 1.9+3.4j]
        zc = [x - val for x, val in izip(self.xc, my_list)]
        zca = self.xca - my_list
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # check subtraction with a list -> list - UncertainArray[ureal]
        my_list = list(range(n))
        z = [val - x for val, x in izip(my_list, self.x)]
        za = my_list - self.xa
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # check subtraction with a list -> list - UncertainArray[ucomplex]
        my_list = [3.1-2.3j, 4+5j, 3.2+7.3j, 5.1-0.4j, 0.1, 6.1+3.7j]
        zc = [val - x for val, x in izip(my_list, self.xc)]
        zca = my_list - self.xca
        for i in range(n):
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        a = uarray([nan])
        b = uarray([ureal(1, 1)])
        c = a - b
        self.assertTrue(math.isnan(value(c)))
        self.assertTrue(equivalent(uncertainty(c), 1))
        self.assertTrue(math.isinf(dof(c)))

    def test_multiply(self):
        # # Single element case 
        # l = uarray( ureal(1.2,0.3,4) )
        # r = uarray( ureal(-0.2,0.4,7) )
        # y = l * r 
        # self.assertTrue( isinstance(y,UncertainReal) )
        # self.assertTrue( equivalent(y.x,-0.24) )
        # self.assertTrue( equivalent(y.u,math.sqrt( (1.2*.4)**2 + (0.2*.3)**2)) )

        n = len(self.x)

        # x * y
        z = [x * y for x, y in izip(self.x, self.y)]
        za = self.xa * self.ya
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

    def test_divide(self):
        # # Single element case 
        # l = uarray( ureal(1.2,0.3,4) )
        # r = uarray( ureal(-0.2,0.4,7) )
        # y = l / r 
        # self.assertTrue( isinstance(y,UncertainReal) )
        # q = 1.2/-0.2
        # self.assertTrue( equivalent(y.x,q) )
        # self.assertTrue( equivalent(y.u,math.sqrt( (q/1.2*.3)**2 + (q/-0.2*.4)**2)) )
        
        n = len(self.x)

        # x / y
        z = [x / y for x, y in izip(self.x, self.y)]
        za = self.xa / self.ya
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # true_divide
        np.true_divide(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

    def test_comparisons(self):
        # # Single element case 
        # l = uarray( ureal(1.2,0.3,4) )
        # r = uarray( ureal(-0.2,0.4,7) )
        # rr = uarray( ureal(1.2,0.4,7) )
        # # ==
        # y = l == r 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( not y )
        # y = l == rr 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( y )
        # # !=
        # y = l != r 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( y )
        # y = l != rr 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( not y )
        # # <
        # y = l < r 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( not y )
        # y = l < rr 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( not y )

        # # >
        # y = l > r 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( y )
        # y = l > rr 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( not y )

        # # <=
        # y = l <= r 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( not y )
        # y = l <= rr 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( y )

        # # >=
        # y = l >= r 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( y )
        # y = l >= rr 
        # self.assertTrue( isinstance(y,np.bool_) )
        # self.assertTrue( y )
        
        n = len(self.x)

        # ==
        a = self.xa + self.ya
        b = self.xa + self.ya
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(not np.array_equal(self.xa, self.ya))

        # == element-wise
        z = [x == y for x, y in izip(self.x, self.y)]
        za = np.equal(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(z[i] == za[i])

        # != element-wise
        z = [x != y for x, y in izip(self.x, self.y)]
        za = np.not_equal(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(z[i] == za[i])

        # < element-wise
        z = [x < y for x, y in izip(self.x, self.y)]
        za = np.less(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(z[i] == za[i])

        # <= element-wise
        z = [x <= y for x, y in izip(self.x, self.y)]
        za = np.less_equal(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(z[i] == za[i])

        # > element-wise
        z = [x > y for x, y in izip(self.x, self.y)]
        za = np.greater(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(z[i] == za[i])

        # >= element-wise
        z = [x >= y for x, y in izip(self.x, self.y)]
        za = np.greater_equal(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(z[i] == za[i])

    def test_abs(self):
        # # Single element case 
        # a = uarray( ureal(-0.2,0.4,7) )
        # x = abs(a)
        # self.assertTrue( isinstance(x,float) )
        # self.assertTrue( equivalent(x,0.2) )
        
        n = len(self.x)
        z = [abs(x) for x in self.x]
        zc = [abs(x) for x in self.xc]

        # uses np.abs
        za = np.abs(self.xa)
        zca = np.abs(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i], za[i]))
            self.assertTrue(equivalent(zc[i], zca[i]))

        # uses np.absolute
        za = np.absolute(self.xa)
        zca = np.absolute(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i], za[i]))
            self.assertTrue(equivalent(zc[i], zca[i]))

    def test_conjugate(self):
        # # Single element case 
        # a = uarray( ureal(-0.2,0.4,7) )
        # x = a.conjugate()
        # self.assertTrue( isinstance(x,UncertainReal) )
        # self.assertTrue( equivalent(x.x,-0.2) )
 
        # a = uarray( ucomplex(-0.2+0.1j,0.4,7) )
        # x = a.conjugate()
        # self.assertTrue( isinstance(x,UncertainComplex) )
        # self.assertTrue( equivalent(x.x.real,-0.2) )
        # self.assertTrue( equivalent(x.x.imag,-0.1) )
 
        n = len(self.xc)
        z = [x.conjugate() for x in self.x]
        zc = [x.conjugate() for x in self.xc]

        # call np.conjugate
        za = np.conjugate(self.xa)
        zca = np.conjugate(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call UncertainArray.conjugate()
        za = self.xa.conjugate()
        zca = self.xca.conjugate()
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_cos(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = cos(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.cos(x)) )
    
        n = len(self.x)
        z = [cos(x) for x in self.x]
        zc = [cos(x) for x in self.xc]

        # call np.cos
        za = np.cos(self.xa)
        zca = np.cos(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.cos
        za = cos(self.xa)
        zca = cos(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_sin(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = sin(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.sin(x)) )
    
        n = len(self.x)
        z = [sin(x) for x in self.x]
        zc = [sin(x) for x in self.xc]

        # call np.sin
        za = np.sin(self.xa)
        zca = np.sin(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.sin
        za = sin(self.xa)
        zca = sin(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_tan(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = tan(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.tan(x)) )
    
        n = len(self.x)
        z = [tan(x) for x in self.x]
        zc = [tan(x) for x in self.xc]

        # call np.tan
        za = np.tan(self.xa)
        zca = np.tan(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.tan
        za = tan(self.xa)
        zca = tan(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_acos(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = acos(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.acos(x)) )
    
        x = [ureal(0.4, 0.02), ureal(-0.3, 0.01), ureal(-0.2, 0.03), ureal(0.8, 0.03)]
        xa = uarray(x)
        n = len(x)
        z = [acos(val) for val in x]

        # call np.arccos
        za = np.arccos(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.acos
        za = acos(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the array
        xa = uarray(x).reshape(2, 2)

        # call np.arccos
        m = np.arccos(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.acos
        m = acos(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_asin(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = asin(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.asin(x)) )
    
        x = [ureal(0.4, 0.02), ureal(-0.3, 0.01), ureal(-0.2, 0.03), ureal(0.8, 0.03)]
        xa = uarray(x)
        n = len(x)
        z = [asin(val) for val in x]

        # call np.arcsin
        za = np.arcsin(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.asin
        za = asin(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the array
        xa = uarray(x).reshape(2, 2)

        # call np.arcsin
        m = np.arcsin(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.asin
        m = asin(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_atan(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = atan(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.atan(x)) )
    
        x = [ureal(0.4, 0.02), ureal(-0.3, 0.01), ureal(-0.2, 0.03), ureal(0.8, 0.03)]
        xa = uarray(x)
        n = len(x)
        z = [atan(val) for val in x]

        # call np.arctan
        za = np.arctan(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.atan
        za = atan(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the array
        xa = uarray(x).reshape(2, 2)

        # call np.arctan
        m = np.arctan(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.atan
        m = atan(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_atan2(self):
        # # Single element case
        # x = uarray( ureal(0.4, 0.02) )
        # y = uarray( ureal(0.3, 0.01) ) 
        # z = atan2(x,y) 
        # self.assertTrue( isinstance(z,UncertainReal) )
        # self.assertTrue( equivalent( z.x, math.atan2(0.4,0.3)) )
        
        # z = np.arctan2(x,y)
        # self.assertTrue( isinstance(z,UncertainReal) )
        # self.assertTrue( equivalent( z.x, math.atan2(0.4,0.3)) )
        
        x = [ureal(0.4, 0.02), ureal(-0.3, 0.01), ureal(-0.2, 0.03), ureal(0.8, 0.03)]
        y = [ureal(0.3, 0.01), ureal(0.1, 0.06), ureal(0.16, 0.02), ureal(0.21, 0.07)]
        xa = uarray(x)
        ya = uarray(y)
        n = len(x)
        z = [atan2(v1, v2) for v1, v2 in izip(x, y)]

        # call np.arctan2 
        za = np.arctan2(xa, ya)
       
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.atan2
        za = atan2(xa, ya)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the arrays
        xa = uarray(x).reshape(2, 2)
        ya = uarray(y).reshape(2, 2)

        # call np.arctan2
        m = np.arctan2(xa, ya)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.atan2
        m = atan2(xa, ya)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_exp(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = exp(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.exp(x)) )
    
        n = len(self.x)
        z = [exp(x) for x in self.x]
        zc = [exp(x) for x in self.xc]

        # call np.exp
        za = np.exp(self.xa)
        zca = np.exp(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.exp
        za = exp(self.xa)
        zca = exp(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_pow(self):
        # # Single element case
        # x = uarray( ureal(0.4, 0.02) )
        # y = uarray( ureal(0.3, 0.01) ) 
        # z = x ** y 
        # self.assertTrue( isinstance(z,UncertainReal) )
        # self.assertTrue( equivalent( z.x, 0.4 ** 0.3) )

        n = len(self.x)

        # use the "**" syntax -> UncertainArray ** UncertainArray
        z = [x ** y for x, y in izip(self.x, self.y)]
        zc = [x ** y for x, y in izip(self.xc, self.yc)]
        za = self.xa ** self.ya
        zca = self.xca ** self.yca
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # use the np.power function -> UncertainArray ** UncertainArray
        za = np.power(self.xa, self.ya)
        zca = np.power(self.xca, self.yca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # UncertainArray[ureal] ** number
        for item in [3, -2.4, 5.3, ureal(2.3, 0.9)]:
            z = [x ** item for x in self.x]
            za = self.xa ** item
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # UncertainArray[ucomplex] ** number
        for item in [3, -2.4, 5.3j, ucomplex(2.3+0.5j, (0.9, 0.4))]:
            zc = [x ** item for x in self.xc]
            zca = self.xca ** item
            for i in range(n):
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real, tol=1e-9))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag, tol=1e-9))

        # number ** UncertainArray[ureal]
        for item in [-2, 0.38, 2.1, ureal(6.2, 3.2)]:
            z = [item ** x for x in self.x]
            za = item ** self.xa
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # number ** UncertainArray[ucomplex]
        for item in [-2, 0.38, 2.1j, ucomplex(6.2+2.1j, (3.2, 1.1))]:
            zc = [item ** x for x in self.xc]
            zca = item ** self.xca
            for i in range(n):
                self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
                self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
                self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # use GTC.core.pow -> UncertainArray ** UncertainArray
        z = [pow(x, y) for x, y in izip(self.x, self.y)]
        za = pow(self.xa, self.ya)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # use GTC.core.pow -> UncertainArray ** number
        for item in [23.1, 7.21, 4, ureal(4.2, 0.9)]:
            z = [pow(x, item) for x in self.x]
            za = pow(self.xa, item)
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

        # use GTC.core.pow -> number ** UncertainArray
        for item in [-11.1, 0.002, 1.01, ureal(8.2, 4.4)]:
            z = [pow(item, x) for x in self.x]
            za = pow(item, self.xa)
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

    def test_log(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = log(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.log(x)) )
    
        n = len(self.x)
        z = [log(x) for x in self.x]
        zc = [log(x) for x in self.xc]

        # call np.log
        za = np.log(self.xa)
        zca = np.log(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.log
        za = log(self.xa)
        zca = log(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_log10(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = log10(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.log10(x)) )
    
        n = len(self.x)
        z = [log(x) for x in self.x]
        zc = [log(x) for x in self.xc]

        n = len(self.x)
        z = [log10(x) for x in self.x]
        zc = [log10(x) for x in self.xc]

        # call np.log10
        za = np.log10(self.xa)
        zca = np.log10(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.log10
        za = log10(self.xa)
        zca = log10(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_sqrt(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = sqrt(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.sqrt(x)) )
    
        n = len(self.x)
        z = [sqrt(x) for x in self.x]
        zc = [sqrt(x) for x in self.xc]

        # call np.sqrt
        za = np.sqrt(self.xa)
        zca = np.sqrt(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.sqrt
        za = sqrt(self.xa)
        zca = sqrt(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_sinh(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = sinh(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.sinh(x)) )
    
        n = len(self.x)
        z = [sinh(x) for x in self.x]
        zc = [sinh(x) for x in self.xc]

        # call np.sinh
        za = np.sinh(self.xa)
        zca = np.sinh(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.sinh
        za = sinh(self.xa)
        zca = sinh(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_cosh(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = cosh(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.cosh(x)) )
    
        n = len(self.x)
        z = [sinh(x) for x in self.x]
        zc = [sinh(x) for x in self.xc]

        n = len(self.x)
        z = [cosh(x) for x in self.x]
        zc = [cosh(x) for x in self.xc]

        # call np.cosh
        za = np.cosh(self.xa)
        zca = np.cosh(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.cosh
        za = cosh(self.xa)
        zca = cosh(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_tanh(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = tanh(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.tanh(x)) )
    
        n = len(self.x)
        z = [tanh(x) for x in self.x]
        zc = [tanh(x) for x in self.xc]

        # call np.tanh
        za = np.tanh(self.xa)
        zca = np.tanh(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

        # call GTC.core.tanh
        za = tanh(self.xa)
        zca = tanh(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent_complex(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u.real, zca[i].u.real))
            self.assertTrue(equivalent(zc[i].u.imag, zca[i].u.imag))

    def test_acosh(self):
        # # Single element case 
        # x = 1.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = acosh(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.acosh(x)) )
    
        x = [ureal(1.2, 0.02), ureal(3.1, 0.01), ureal(4.1, 0.03), ureal(2.2, 0.03)]
        xa = uarray(x)
        n = len(x)
        z = [acosh(val) for val in x]

        # call np.arccosh
        za = np.arccosh(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.acosh
        za = acosh(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the array
        xa = uarray(x).reshape(2, 2)

        # call np.arccosh
        m = np.arccosh(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.acosh
        m = acosh(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_asinh(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = asinh(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.asinh(x)) )
    
        x = [ureal(1.2, 0.02), ureal(3.1, 0.01), ureal(4.1, 0.03), ureal(2.2, 0.03)]
        xa = uarray(x)
        n = len(x)
        z = [asinh(val) for val in x]

        # call np.arcsinh
        za = np.arcsinh(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.asinh
        za = asinh(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the array
        xa = uarray(x).reshape(2, 2)

        # call np.arcsinh
        m = np.arcsinh(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.asinh
        m = asinh(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_atanh(self):
        # # Single element case 
        # x = 0.2
        # a = uarray( ureal(x,0.4,7) )
        # ux = atanh(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,math.atanh(x)) )
    
        x = [ureal(0.4, 0.02), ureal(-0.3, 0.01), ureal(-0.2, 0.03), ureal(0.8, 0.03)]
        xa = uarray(x)
        n = len(x)
        z = [atanh(val) for val in x]

        # call np.arctanh
        za = np.arctanh(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # call GTC.core.atanh
        za = atanh(xa)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # reshape the array
        xa = uarray(x).reshape(2, 2)

        # call np.arctanh
        m = np.arctanh(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

        # call GTC.core.atanh
        m = atanh(xa)
        self.assertTrue(m.shape == (2, 2))
        self.assertTrue(equivalent(z[0].x, m[0, 0].x))
        self.assertTrue(equivalent(z[0].u, m[0, 0].u))
        self.assertTrue(equivalent(z[1].x, m[0, 1].x))
        self.assertTrue(equivalent(z[1].u, m[0, 1].u))
        self.assertTrue(equivalent(z[2].x, m[1, 0].x))
        self.assertTrue(equivalent(z[2].u, m[1, 0].u))
        self.assertTrue(equivalent(z[3].x, m[1, 1].x))
        self.assertTrue(equivalent(z[3].u, m[1, 1].u))

    def test_mag_squared(self):
        # # Single element case 
        # x = 0.4-1.1j
        # a = uarray( ucomplex(x,1,7) )
        # ux = mag_squared(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,abs(x*x)) )
    
        for item1, item2 in [(self.x, self.xa), (self.xc, self.xca)]:
            for func in (mag_squared, np.square):
                n = len(item1)
                z = [mag_squared(x) for x in item1]
                za = func(item2)
                for i in range(n):
                    self.assertTrue(equivalent(z[i].x, za[i].x))
                    self.assertTrue(equivalent(z[i].u, za[i].u))

                # reshape the array and make sure we get the same shape back
                m = func(item2.reshape(2, 3))
                self.assertTrue(m.shape == (2, 3))
                self.assertTrue(equivalent(z[0].x, m[0, 0].x))
                self.assertTrue(equivalent(z[1].x, m[0, 1].x))
                self.assertTrue(equivalent(z[2].x, m[0, 2].x))
                self.assertTrue(equivalent(z[3].x, m[1, 0].x))
                self.assertTrue(equivalent(z[4].x, m[1, 1].x))
                self.assertTrue(equivalent(z[5].x, m[1, 2].x))
                self.assertTrue(equivalent(z[0].u, m[0, 0].u))
                self.assertTrue(equivalent(z[1].u, m[0, 1].u))
                self.assertTrue(equivalent(z[2].u, m[0, 2].u))
                self.assertTrue(equivalent(z[3].u, m[1, 0].u))
                self.assertTrue(equivalent(z[4].u, m[1, 1].u))
                self.assertTrue(equivalent(z[5].u, m[1, 2].u))

        z = [mag_squared(ucomplex(-1j, 0.21)), mag_squared(ucomplex(1, 0.5))]
        za = uarray([ucomplex(-1j, 0.21), ucomplex(1, 0.5)])
        out = np.square(za)
        for i in range(2):
            self.assertTrue(equivalent(z[i].x, out[i].x))
            self.assertTrue(equivalent(z[i].u, out[i].u))

    def test_magnitude(self):
        # # Single element case 
        # x = 0.4-1.1j
        # a = uarray( ucomplex(x,1,7) )
        # ux = magnitude(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,abs(x)) )
    
        for item1, item2 in [(self.x, self.xa), (self.xc, self.xca)]:
            n = len(item1)
            z = [magnitude(x) for x in item1]
            za = magnitude(item2)
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

            # reshape the array and make sure we get the same shape back
            m = magnitude(item2.reshape(2, 3))
            self.assertTrue(m.shape == (2, 3))
            self.assertTrue(equivalent(z[0].x, m[0, 0].x))
            self.assertTrue(equivalent(z[1].x, m[0, 1].x))
            self.assertTrue(equivalent(z[2].x, m[0, 2].x))
            self.assertTrue(equivalent(z[3].x, m[1, 0].x))
            self.assertTrue(equivalent(z[4].x, m[1, 1].x))
            self.assertTrue(equivalent(z[5].x, m[1, 2].x))
            self.assertTrue(equivalent(z[0].u, m[0, 0].u))
            self.assertTrue(equivalent(z[1].u, m[0, 1].u))
            self.assertTrue(equivalent(z[2].u, m[0, 2].u))
            self.assertTrue(equivalent(z[3].u, m[1, 0].u))
            self.assertTrue(equivalent(z[4].u, m[1, 1].u))
            self.assertTrue(equivalent(z[5].u, m[1, 2].u))

    def test_phase(self):
        # # Single element case 
        # x = 0.4-1.1j
        # a = uarray( ucomplex(x,1,7) )
        # ux = phase(a)
        # self.assertTrue( isinstance(ux,UncertainReal) )
        # self.assertTrue( equivalent(ux.x,cmath.phase(x)) )
    
        for item1, item2 in [(self.x, self.xa), (self.xc, self.xca)]:
            n = len(item1)
            z = [phase(x) for x in item1]
            za = phase(item2)
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

            # reshape the array and make sure we get the same shape back
            m = phase(item2.reshape(2, 3))
            self.assertTrue(m.shape == (2, 3))
            self.assertTrue(equivalent(z[0].x, m[0, 0].x))
            self.assertTrue(equivalent(z[1].x, m[0, 1].x))
            self.assertTrue(equivalent(z[2].x, m[0, 2].x))
            self.assertTrue(equivalent(z[3].x, m[1, 0].x))
            self.assertTrue(equivalent(z[4].x, m[1, 1].x))
            self.assertTrue(equivalent(z[5].x, m[1, 2].x))
            self.assertTrue(equivalent(z[0].u, m[0, 0].u))
            self.assertTrue(equivalent(z[1].u, m[0, 1].u))
            self.assertTrue(equivalent(z[2].u, m[0, 2].u))
            self.assertTrue(equivalent(z[3].u, m[1, 0].u))
            self.assertTrue(equivalent(z[4].u, m[1, 1].u))
            self.assertTrue(equivalent(z[5].u, m[1, 2].u))

    #
    # The following tests are for specific numpy ufuncs that
    # have no direct GTC equivalent function
    #

    def test_sum(self):
        self.assertRaises(TypeError,self.xa.sum)
        self.assertRaises(TypeError,np.sum,self.xa)
        # # 1D array
        # xlist = [ureal(i, i*0.1) for i in range(100)]
        # xarray = uarray(xlist)
        # self.assertTrue(xarray.shape == (100,))

        # b = 0
        # for x in xlist:
            # b += x
        # for a in [xarray.sum(), function.sum(xarray)]:
            # self.assertTrue(equivalent(value(a), b.x))
            # self.assertTrue(equivalent(uncertainty(a), b.u))

        # # 3D array
        # xlist = [[[ureal(i*j*k, i*j*k*0.1) for k in range(1, 5)] for j in range(7, 10)] for i in range(3, 9)]
        # xarray = uarray(xlist)
        # self.assertTrue(xarray.shape == (6, 3, 4))

        # axis_none = 0.0
        # axis_0 = [[0.0 for i in range(4)] for j in range(3)]
        # axis_1 = [[0.0 for i in range(4)] for j in range(6)]
        # axis_2 = [[0.0 for i in range(3)] for j in range(6)]
        # for i in range(6):
            # for j in range(3):
                # for k in range(4):
                    # _value = xlist[i][j][k]
                    # axis_none += _value
                    # axis_0[j][k] += _value
                    # axis_1[i][k] += _value
                    # axis_2[i][j] += _value

        # # axis=None
        # for a in [xarray.sum(), function.sum(xarray)]:
            # self.assertTrue(equivalent(value(a), axis_none.x))
            # self.assertTrue(equivalent(uncertainty(a), axis_none.u))

        # # axis=0
        # m, n = len(axis_0), len(axis_0[0])
        # for a in [xarray.sum(axis=0), function.sum(xarray, axis=0)]:
            # self.assertTrue(a.shape == (m, n))
            # for j in range(m):
                # for k in range(n):
                    # self.assertTrue(equivalent(a[j, k].x, axis_0[j][k].x))
                    # self.assertTrue(equivalent(a[j, k].u, axis_0[j][k].u))

        # # axis=1
        # m, n = len(axis_1), len(axis_1[0])
        # for a in [xarray.sum(axis=1), function.sum(xarray, axis=1)]:
            # self.assertTrue(a.shape == (m, n))
            # for i in range(m):
                # for k in range(n):
                    # self.assertTrue(equivalent(a[i, k].x, axis_1[i][k].x))
                    # self.assertTrue(equivalent(a[i, k].u, axis_1[i][k].u))

        # # axis=2
        # m, n = len(axis_2), len(axis_2[0])
        # for a in [xarray.sum(axis=2), function.sum(xarray, axis=2)]:
            # self.assertTrue(a.shape == (m, n))
            # for i in range(m):
                # for j in range(n):
                    # self.assertTrue(equivalent(a[i, j].x, axis_2[i][j].x))
                    # self.assertTrue(equivalent(a[i, j].u, axis_2[i][j].u))

    def test_mean(self):
        self.assertRaises(TypeError,self.xa.mean)
        self.assertRaises(TypeError,np.mean,self.xa.mean)

        # for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            # ave = 0.0
            # for val in x:
                # ave += val
            # ave = ave/float(len(x))

            # for m in [xa.mean(), np.mean(xa), np.average(xa)]:
                # if j == 0:
                    # self.assertTrue(equivalent(value(m), ave.x))
                    # self.assertTrue(equivalent(uncertainty(m), ave.u))
                # else:
                    # self.assertTrue(equivalent_complex(value(m), ave.x))
                    # self.assertTrue(equivalent(uncertainty(m).real, ave.u.real))
                    # self.assertTrue(equivalent(uncertainty(m).imag, ave.u.imag))

            # xa = xa.reshape(2, 3)

            # for m in [xa.mean(), np.mean(xa), np.average(xa)]:
                # if j == 0:
                    # self.assertTrue(equivalent(value(m), ave.x))
                    # self.assertTrue(equivalent(uncertainty(m), ave.u))
                # else:
                    # self.assertTrue(equivalent_complex(value(m), ave.x))
                    # self.assertTrue(equivalent(uncertainty(m).real, ave.u.real))
                    # self.assertTrue(equivalent(uncertainty(m).imag, ave.u.imag))

            # for m in [xa.mean(axis=0), np.mean(xa, axis=0), np.average(xa, axis=0)]:
                # aves = [(x[0] + x[3])/2.0, (x[1] + x[4])/2.0, (x[2] + x[5])/2.0]
                # for idx in range(3):
                    # if j == 0:
                        # self.assertTrue(equivalent(m[idx].x, aves[idx].x))
                        # self.assertTrue(equivalent(m[idx].u, aves[idx].u))
                    # else:
                        # self.assertTrue(equivalent_complex(m[idx].x, aves[idx].x))
                        # self.assertTrue(equivalent(m[idx].u.real, aves[idx].u.real))
                        # self.assertTrue(equivalent(m[idx].u.imag, aves[idx].u.imag))

            # for m in [xa.mean(axis=1), np.mean(xa, axis=1), np.average(xa, axis=1)]:
                # aves = [(x[0] + x[1] + x[2])/3.0, (x[3] + x[4] + x[5])/3.0]
                # for idx in range(2):
                    # if j == 0:
                        # self.assertTrue(equivalent(m[idx].x, aves[idx].x))
                        # self.assertTrue(equivalent(m[idx].u, aves[idx].u))
                    # else:
                        # self.assertTrue(equivalent_complex(m[idx].x, aves[idx].x))
                        # self.assertTrue(equivalent(m[idx].u.real, aves[idx].u.real))
                        # self.assertTrue(equivalent(m[idx].u.imag, aves[idx].u.imag))

    def test_std(self):
        self.assertRaises(TypeError,self.xa.std)
        self.assertRaises(TypeError,np.std,self.xa)
        # for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            # ave = 0.0
            # for item in x:
                # ave += item
            # ave = ave/float(len(x))

            # stdev = 0.0
            # for item in [(val - ave)**2 for val in x]:
                # stdev += item

            # for ddof in range(5):
                # a = sqrt(stdev/(float(len(x)-ddof)))
                # b = xa.std(ddof=ddof)
                # if j == 0:
                    # self.assertTrue(equivalent(a.x, b.value()))
                    # self.assertTrue(equivalent(a.u, b.uncertainty()))
                # else:
                    # self.assertTrue(equivalent_complex(a.x, b.value()))
                    # self.assertTrue(equivalent(a.u.real, b.uncertainty().real))
                    # self.assertTrue(equivalent(a.u.imag, b.uncertainty().imag))

        # x = [[ureal(i*j, i*j*0.05) for j in range(1, 11)] for i in range(5, 11)]
        # xa = uarray(x)

        # # axis-0 -> columns
        # aves = [0.0 for _ in range(len(x[0]))]
        # for i in range(len(x)):
            # for j in range(len(x[0])):
                # aves[j] += x[i][j]
        # n = float(len(x))
        # aves = [a/n for a in aves]

        # for ddof in range(5):
            # stdevs = [0.0 for _ in range(len(aves))]
            # for idx, ave in enumerate(aves):
                # for i in range(len(x)):
                    # stdevs[idx] += (x[i][idx] - ave) ** 2
                # stdevs[idx] = sqrt(stdevs[idx] / (n - ddof))
            # for a in [xa.std(axis=0, ddof=ddof), np.std(xa, axis=0, ddof=ddof)]:
                # for i in range(len(stdevs)):
                    # self.assertTrue(equivalent(a[i].x, stdevs[i].x))
                    # self.assertTrue(equivalent(a[i].u, stdevs[i].u))

        # # axis-1 -> rows
        # aves = [0.0 for _ in range(len(x))]
        # for i in range(len(x)):
            # for j in range(len(x[0])):
                # aves[i] += x[i][j]
        # n = float(len(x[0]))
        # aves = [a/n for a in aves]

        # for ddof in range(5):
            # stdevs = [0.0 for _ in range(len(aves))]
            # for idx, ave in enumerate(aves):
                # for item in [(val - ave) ** 2 for val in x[idx]]:
                    # stdevs[idx] += item
                # stdevs[idx] = sqrt(stdevs[idx] / (n - ddof))
            # for a in [xa.std(axis=1, ddof=ddof), np.std(xa, axis=1, ddof=ddof)]:
                # for i in range(len(stdevs)):
                    # self.assertTrue(equivalent(a[i].x, stdevs[i].x))
                    # self.assertTrue(equivalent(a[i].u, stdevs[i].u))

    def test_var(self):
        self.assertRaises(TypeError,self.xa.var)
        self.assertRaises(TypeError,np.var,self.xa)
        # for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            # ave = 0.0
            # for item in x:
                # ave += item
            # ave = ave/float(len(x))

            # var = 0.0
            # for item in [(val - ave)**2 for val in x]:
                # var += item

            # for ddof in range(5):
                # a = var/(float(len(x)-ddof))
                # b = xa.var(ddof=ddof)
                # if j == 0:
                    # self.assertTrue(equivalent(a.x, b.value()))
                    # self.assertTrue(equivalent(a.u, b.uncertainty()))
                # else:
                    # self.assertTrue(equivalent_complex(a.x, b.value()))
                    # self.assertTrue(equivalent(a.u.real, b.uncertainty().real))
                    # self.assertTrue(equivalent(a.u.imag, b.uncertainty().imag))

        # x = [[ureal(i*j, i*j*0.05) for j in range(1, 11)] for i in range(5, 11)]
        # xa = uarray(x)

        # # axis-0 -> columns
        # aves = [0.0 for _ in range(len(x[0]))]
        # for i in range(len(x)):
            # for j in range(len(x[0])):
                # aves[j] += x[i][j]
        # n = float(len(x))
        # aves = [a/n for a in aves]

        # for ddof in range(5):
            # vars = [0.0 for _ in range(len(aves))]
            # for idx, ave in enumerate(aves):
                # for i in range(len(x)):
                    # vars[idx] += (x[i][idx] - ave) ** 2
                # vars[idx] = vars[idx] / (n - ddof)
            # for a in [xa.var(axis=0, ddof=ddof), np.var(xa, axis=0, ddof=ddof)]:
                # for i in range(len(vars)):
                    # self.assertTrue(equivalent(a[i].x, vars[i].x))
                    # self.assertTrue(equivalent(a[i].u, vars[i].u))

        # # axis-1 -> rows
        # aves = [0.0 for _ in range(len(x))]
        # for i in range(len(x)):
            # for j in range(len(x[0])):
                # aves[i] += x[i][j]
        # n = float(len(x[0]))
        # aves = [a/n for a in aves]

        # for ddof in range(5):
            # vars = [0.0 for _ in range(len(aves))]
            # for idx, ave in enumerate(aves):
                # for item in [(val - ave) ** 2 for val in x[idx]]:
                    # vars[idx] += item
                # vars[idx] = vars[idx] / (n - ddof)
            # for a in [xa.var(axis=1, ddof=ddof), np.var(xa, axis=1, ddof=ddof)]:
                # for i in range(len(vars)):
                    # self.assertTrue(equivalent(a[i].x, vars[i].x))
                    # self.assertTrue(equivalent(a[i].u, vars[i].u))

    def test_max(self):
        self.assertRaises(TypeError,self.xa.max)
        self.assertRaises(TypeError,np.max,self.xa)
        # b = max(self.x)
        # for a in [self.xa.max(), np.max(self.xa)]:
            # self.assertTrue(equivalent(value(a), b.x))
            # self.assertTrue(equivalent(uncertainty(a), b.u))

        # xa = self.xa.reshape(3, 2)

        # for a in [xa.max(), np.max(xa)]:
            # self.assertTrue(equivalent(value(a), b.x))
            # self.assertTrue(equivalent(uncertainty(a), b.u))

        # for a in [xa.max(axis=0), np.max(xa, axis=0)]:
            # for i, b in enumerate([self.x[::2], self.x[1::2]]):
                # b = max(b)
                # self.assertTrue(equivalent(a[i].x, b.x))
                # self.assertTrue(equivalent(a[i].u, b.u))

        # for a in [xa.max(axis=1), np.max(xa, axis=1)]:
            # for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
                # b = max(b)
                # self.assertTrue(equivalent(a[i].x, b.x))
                # self.assertTrue(equivalent(a[i].u, b.u))

    def test_min(self):
        self.assertRaises(TypeError,self.xa.min)
        self.assertRaises(TypeError,np.min,self.xa)
        # b = min(self.x)
        # for a in [self.xa.min(), np.min(self.xa)]:
            # self.assertTrue(equivalent(value(a), b.x))
            # self.assertTrue(equivalent(uncertainty(a), b.u))

        # xa = self.xa.reshape(3, 2)

        # for a in [xa.min(), np.min(xa)]:
            # self.assertTrue(equivalent(value(a), b.x))
            # self.assertTrue(equivalent(uncertainty(a), b.u))

        # for a in [xa.min(axis=0), np.min(xa, axis=0)]:
            # for i, b in enumerate([self.x[::2], self.x[1::2]]):
                # b = min(b)
                # self.assertTrue(equivalent(a[i].x, b.x))
                # self.assertTrue(equivalent(a[i].u, b.u))

        # for a in [xa.min(axis=1), np.min(xa, axis=1)]:
            # for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
                # b = min(b)
                # self.assertTrue(equivalent(a[i].x, b.x))
                # self.assertTrue(equivalent(a[i].u, b.u))

    def test_argmax(self):
        b = self.x.index(max(self.x))
        for a in [self.xa.argmax(), np.argmax(self.xa)]:
            self.assertTrue(a == b)

        xa = self.xa.reshape(3, 2)

        for a in [xa.argmax(), np.argmax(xa)]:
            self.assertTrue(equivalent(a, b))

        for a in [xa.argmax(axis=0), np.argmax(xa, axis=0)]:
            for i, b in enumerate([self.x[::2], self.x[1::2]]):
                self.assertTrue(a[i] == b.index(max(b)))

        for a in [xa.argmax(axis=1), np.argmax(xa, axis=1)]:
            for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
                self.assertTrue(a[i] == b.index(max(b)))

    def test_argmin(self):
        b = self.x.index(min(self.x))
        for a in [self.xa.argmin(), np.argmin(self.xa)]:
            self.assertTrue(a == b)

        xa = self.xa.reshape(3, 2)

        for a in [xa.argmin(), np.argmin(xa)]:
            self.assertTrue(equivalent(a, b))

        for a in [xa.argmin(axis=0), np.argmin(xa, axis=0)]:
            for i, b in enumerate([self.x[::2], self.x[1::2]]):
                self.assertTrue(a[i] == b.index(min(b)))

        for a in [xa.argmin(axis=1), np.argmin(xa, axis=1)]:
            for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
                self.assertTrue(a[i] == b.index(min(b)))

    def test_transpose(self):
        a = uarray([ureal(i, i*0.1) for i in range(5*7)]).reshape(5, 7)
        for item in [a.T, a.transpose(), np.transpose(a)]:
            for i in range(5):
                for j in range(7):
                    self.assertTrue(equivalent(a[i, j].x, item[j, i].x))
                    self.assertTrue(equivalent(a[i, j].u, item[j, i].u))

    def test_argsort(self):
        b = [self.x.index(a) for a in sorted(self.x)]
        for a in [self.xa.argsort(), np.argsort(self.xa)]:
            for i in range(len(a)):
                self.assertTrue(a[i] == b[i])

        xa = self.xa.reshape(2, 3)
        b = [[item.index(x) for x in sorted(item)] for item in [self.x[:3], self.x[3:]]]
        for a in [xa.argsort(), np.argsort(xa)]:
            for i in range(2):
                for j in range(3):
                    self.assertTrue(a[i, j] == b[i][j])

        b = [[item.index(x) for x in sorted(item)] for item in [self.x[::3], self.x[1::3], self.x[2::3]]]
        for a in [xa.argsort(axis=0), np.argsort(xa, axis=0)]:
            for i in range(2):
                for j in range(3):
                    self.assertTrue(a[i, j] == b[j][i])

        b = [[item.index(x) for x in sorted(item)] for item in [self.x[:3], self.x[3:]]]
        for a in [xa.argsort(axis=1), np.argsort(xa, axis=1)]:
            for i in range(2):
                for j in range(3):
                    self.assertTrue(a[i, j] == b[i][j])

    def test_where(self):
        a = self.xa[np.where(self.xa.x > 25)]
        b = [x for x in self.x if x.x > 25]
        self.assertTrue(len(a) == len(b))
        for i in range(len(a)):
            self.assertTrue(equivalent(a[i].x, b[i].x))
            self.assertTrue(equivalent(a[i].u, b[i].u))

        a = self.xca[np.where(self.xca.real > 24)]
        b = [x for x in self.xc if x.real > 24]
        self.assertTrue(len(a) == len(b))
        for i in range(len(a)):
            self.assertTrue(equivalent_complex(a[i].x, b[i].x))
            self.assertTrue(equivalent(a[i].u.real, b[i].u.real))
            self.assertTrue(equivalent(a[i].u.imag, b[i].u.imag))

        a = self.xca[np.where(self.xca.imag > 3)]
        b = [x for x in self.xc if x.imag > 3]
        self.assertTrue(len(a) == len(b))
        for i in range(len(a)):
            self.assertTrue(equivalent_complex(a[i].x, b[i].x))
            self.assertTrue(equivalent(a[i].u.real, b[i].u.real))
            self.assertTrue(equivalent(a[i].u.imag, b[i].u.imag))

    def test_shape(self):
        x = [ureal(i, i*0.1) for i in range(5*9*3)]

        xa = uarray(x)
        self.assertTrue(xa.shape == (5*9*3,))

        xa = xa.reshape(5, 9, 3)
        self.assertTrue(xa.shape == (5, 9, 3))

    def test_reshape(self):
        a = uarray([ureal(i, i*0.1) for i in range(100)])

        with self.assertRaises(ValueError):
            a.reshape(50, 20)

        a.reshape(2, 5, 10)  # should not raise ValueError

    def test_size(self):

        xa = uarray([])
        self.assertTrue(xa.size == 0)

        x = [ureal(i, i*0.1) for i in range(5*9*3)]

        xa = uarray(x)
        self.assertTrue(xa.size == 5*9*3)

        xa = xa.reshape(5, 9, 3)
        self.assertTrue(xa.size == 5*9*3)

    def test_view(self):
        v = self.xa.view()
        for i in range(len(self.x)):
            self.assertTrue(v[i] is self.xa[i])

    def test_diagonal(self):
        x = [ucomplex(complex(0, i), i) for i in range(25)]
        xa = uarray(x)

        with self.assertRaises(ValueError):
            xa.diagonal()  # diag requires an array of at least two dimensions

        a = xa.reshape(5, 5).diagonal()
        for i in range(5):
            idx = i*5+i
            self.assertTrue(equivalent_complex(a[i].x, x[idx].x))
            self.assertTrue(equivalent(a[i].u.real, x[idx].u.real))
            self.assertTrue(equivalent(a[i].u.imag, x[idx].u.imag))

    def test_trace(self):
        self.assertRaises(TypeError,self.xa.trace)
        self.assertRaises(TypeError,np.trace,self.xa)
        # # UncertainArray[ureal]
        # xlist = [ureal(i, i * 0.1) for i in range(9)]
        # xarray = uarray(xlist)

        # with self.assertRaises(ValueError):
            # xarray.trace()  # diag requires an array of at least two dimensions

        # t = xarray.reshape(3, 3).trace()
        # xt = xlist[0] + xlist[4] + xlist[8]
        # self.assertTrue(equivalent(value(t), xt.x))
        # self.assertTrue(equivalent(uncertainty(t), xt.u))

        # t = np.trace(xarray.reshape(3, 3))
        # self.assertTrue(equivalent(value(t), xt.x))
        # self.assertTrue(equivalent(uncertainty(t), xt.u))

        # # UncertainArray[ucomplex]
        # xlist = [ucomplex(complex(i, i*2), (i*0.2, i * 0.1)) for i in range(9)]
        # xarray = uarray(xlist)

        # with self.assertRaises(ValueError):
            # xarray.trace()  # diag requires an array of at least two dimensions

        # t = xarray.reshape(3, 3).trace()
        # xt = xlist[0] + xlist[4] + xlist[8]
        # self.assertTrue(equivalent_complex(value(t), xt.x))
        # self.assertTrue(equivalent(uncertainty(t.real), xt.u.real))
        # self.assertTrue(equivalent(uncertainty(t.imag), xt.u.imag))

        # t = np.trace(xarray.reshape(3, 3))
        # self.assertTrue(equivalent_complex(value(t), xt.x))
        # self.assertTrue(equivalent(uncertainty(t.real), xt.u.real))
        # self.assertTrue(equivalent(uncertainty(t.imag), xt.u.imag))

    def test_ndim(self):
        a = uarray([ureal(1, 1) for _ in range(1000)])
        self.assertTrue(a.ndim == 1)
        self.assertTrue(a.reshape(10, 100).ndim == 2)
        self.assertTrue(a.reshape(10, 10, 10).ndim == 3)
        self.assertTrue(a.reshape(10, 10, 2, 5).ndim == 4)
        self.assertTrue(a.reshape(2, 5, 10, 2, 5).ndim == 5)
        self.assertTrue(a.reshape(2, 5, 2, 5, 2, 5).ndim == 6)

    def test_base(self):
        for i in range(len(self.x)):
            self.assertTrue(self.xa.base[i] is self.x[i])

    def test_data(self):
        if PY2:
            self.assertTrue(isinstance(self.xa.data, buffer))
        else:
            self.assertTrue(isinstance(self.xa.data, memoryview))

    def test_copy(self):
        c = self.xa.copy()
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(c.shape == self.xa.shape)
        for i in range(len(self.xa)):
            self.assertTrue(c[i] is not self.xa[i])
            # self.assertTrue(c[i].x is not self.xa[i].x)
            self.assertTrue(c[i].x == self.xa[i].x)
            self.assertTrue(c[i].u is not self.xa[i].u)
            self.assertTrue(c[i].u == self.xa[i].u)
            self.assertTrue(self.xa[i] is self.x[i])  # original is unchanged

        x = uarray([
            [ureal(1, 0.1, df=10, label='1', independent=True),
             ureal(2, 0.2, df=20, label='2', independent=False)],
            [ureal(3, 0.3, df=30, label='3', independent=False),
             ureal(4, 0.4, df=40, label='4', independent=True)]])
        c = x.copy()
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(c.shape == x.shape)
        for i in range(2):
            for j in range(2):
                self.assertTrue(c[i, j] is not x[i, j])
                # self.assertTrue(c[i, j].x is not x[i, j].x)
                self.assertTrue(c[i, j].x == x[i, j].x)
                self.assertTrue(c[i, j].u is not x[i, j].u)
                self.assertTrue(c[i, j].u == x[i, j].u)
                self.assertTrue(c[i, j].df is not x[i, j].df)
                self.assertTrue(c[i, j].df == x[i, j].df)
                self.assertTrue(c[i, j].label is None)  # labels don't get copied, see UncertainReal.__pos__
                self.assertTrue(x[i, j].label is not None)
                self.assertTrue(c[i, j]._u_components._index is not x[i, j]._u_components._index)
                self.assertTrue(c[i, j]._u_components._index == x[i, j]._u_components._index)
                self.assertTrue(c[i, j]._u_components._value is not x[i, j]._u_components._value)
                self.assertTrue(c[i, j]._u_components._value == x[i, j]._u_components._value)
                self.assertTrue(c[i, j]._d_components._index is not x[i, j]._d_components._index)
                self.assertTrue(c[i, j]._d_components._index == x[i, j]._d_components._index)
                self.assertTrue(c[i, j]._d_components._value is not x[i, j]._d_components._value)
                self.assertTrue(c[i, j]._d_components._value == x[i, j]._d_components._value)
                self.assertTrue(c[i, j]._i_components._index is not x[i, j]._i_components._index)
                self.assertTrue(c[i, j]._i_components._index == x[i, j]._i_components._index)
                self.assertTrue(c[i, j]._i_components._value is not x[i, j]._i_components._value)
                self.assertTrue(c[i, j]._i_components._value == x[i, j]._i_components._value)

    def test_nbytes(self):
        self.assertTrue(self.xa.nbytes == self.ya.nbytes)
        self.assertTrue(self.xca.nbytes == self.yca.nbytes)

    def test_flatten(self):
        xa = uarray([[ureal(i, j) for i in range(5)] for j in range(5)])
        self.assertTrue(xa.shape == (5, 5))
        self.assertTrue(xa.flatten().shape == (25,))

    def test_flat(self):
        xa = uarray([[ureal(1, 1) for _ in range(5)] for _ in range(5)])
        self.assertTrue(isinstance(xa.flat, np.flatiter))
        xa.flat[[1, 4]] = ureal(9, 9)
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 1) or (i == 0 and j == 4):
                    self.assertTrue(xa[i, j].x == 9)
                else:
                    self.assertTrue(xa[i, j].x == 1)

    def test_fill(self):
        xa = uarray([ureal(1, 1) for _ in range(10)])
        self.assertTrue(sum(xa) == 10)
        xa.fill(ureal(3, 1))
        self.assertTrue(sum(xa) == 30)

    def test_compress(self):
        a = uarray([[ureal(1, 1), ureal(2, 2)], [ureal(3, 3), ureal(4, 4)], [ureal(5, 5), ureal(6, 6)]])
        c = np.compress([0, 1], a, axis=0)
        self.assertTrue(c[0, 0].x == 3)
        self.assertTrue(c[0, 0].u == 3)
        self.assertTrue(c[0, 1].x == 4)
        self.assertTrue(c[0, 1].u == 4)

    def test_cumprod(self):
        # Note numpy will catch and handle TypeError in this case
        self.assertRaises(RuntimeError,self.xa.cumprod)
        self.assertRaises(RuntimeError,np.cumprod,self.xa)
        # x = [[ureal(1, 0.1), ureal(2, 0.2), ureal(3, 0.3)], [ureal(4, 0.4), ureal(5, 0.5), ureal(6, 0.6)]]
        # xa = uarray(x)

        # # flattened
        # a = np.cumprod(xa)
        # b = [x[0][0], x[0][0]*x[0][1], x[0][0]*x[0][1]*x[0][2], x[0][0]*x[0][1]*x[0][2]*x[1][0],
             # x[0][0]*x[0][1]*x[0][2]*x[1][0]*x[1][1], x[0][0]*x[0][1]*x[0][2]*x[1][0]*x[1][1]*x[1][2]]
        # for i in range(6):
            # self.assertTrue(equivalent(a[i].x, b[i].x))
            # self.assertTrue(equivalent(a[i].u, b[i].u))

        # a = np.cumprod(xa, axis=0)
        # b = [[x[0][0], x[0][1], x[0][2]], [x[0][0]*x[1][0], x[0][1]*x[1][1], x[0][2]*x[1][2]]]
        # for i in range(2):
            # for j in range(3):
                # self.assertTrue(equivalent(a[i, j].x, b[i][j].x))
                # self.assertTrue(equivalent(a[i, j].u, b[i][j].u))

        # a = xa.cumprod(axis=1)
        # b = [[x[0][0], x[0][0]*x[0][1], x[0][0]*x[0][1]*x[0][2]],
             # [x[1][0], x[1][0]*x[1][1], x[1][0]*x[1][1]*x[1][2]]]
        # for i in range(2):
            # for j in range(3):
                # self.assertTrue(equivalent(a[i, j].x, b[i][j].x))
                # self.assertTrue(equivalent(a[i, j].u, b[i][j].u))

    def test_cumsum(self):
        # Note numpy will catch and handle TypeError in this case
        self.assertRaises(RuntimeError,self.xa.cumsum)
        self.assertRaises(RuntimeError,np.cumsum,self.xa)
        # x = [[ureal(1, 0.1), ureal(2, 0.2), ureal(3, 0.3)], [ureal(4, 0.4), ureal(5, 0.5), ureal(6, 0.6)]]
        # xa = uarray(x)

        # # flattened
        # a = np.cumsum(xa)
        # b = [x[0][0], x[0][0]+x[0][1], x[0][0]+x[0][1]+x[0][2], x[0][0]+x[0][1]+x[0][2]+x[1][0],
             # x[0][0]+x[0][1]+x[0][2]+x[1][0]+x[1][1], x[0][0]+x[0][1]+x[0][2]+x[1][0]+x[1][1]+x[1][2]]
        # for i in range(6):
            # self.assertTrue(equivalent(a[i].x, b[i].x))
            # self.assertTrue(equivalent(a[i].u, b[i].u))

        # a = xa.cumsum(axis=0)
        # b = [[x[0][0], x[0][1], x[0][2]], [x[0][0]+x[1][0], x[0][1]+x[1][1], x[0][2]+x[1][2]]]
        # for i in range(2):
            # for j in range(3):
                # self.assertTrue(equivalent(a[i, j].x, b[i][j].x))
                # self.assertTrue(equivalent(a[i, j].u, b[i][j].u))

        # a = np.cumsum(xa, axis=1)
        # b = [[x[0][0], x[0][0]+x[0][1], x[0][0]+x[0][1]+x[0][2]],
             # [x[1][0], x[1][0]+x[1][1], x[1][0]+x[1][1]+x[1][2]]]
        # for i in range(2):
            # for j in range(3):
                # self.assertTrue(equivalent(a[i, j].x, b[i][j].x))
                # self.assertTrue(equivalent(a[i, j].u, b[i][j].u))

    def test_dot(self):
        x = [ucomplex(2j, 0.2), ucomplex(3j, 0.3)]
        y = [ucomplex(2j, 1.2), ucomplex(3j, 0.9)]
        xa = uarray(x)
        ya = uarray(y)
        z = x[0]*y[0] + x[1]*y[1]
        za = xa.dot(ya)
        self.assertTrue(equivalent_complex(z.x, za.x))
        self.assertTrue(equivalent(z.u.real, za.u.real))
        self.assertTrue(equivalent(z.u.imag, za.u.imag))

        x = [[ureal(1, 0.1), ureal(2, 0.2)], [ureal(3, 0.3), ureal(4, 0.4)]]
        y = [[ureal(5, 0.5), ureal(6, 0.6)], [ureal(7, 0.7), ureal(8, 0.8)]]
        xa = uarray(x)
        ya = uarray(y)
        z = [[x[0][0]*y[0][0] + x[0][1]*y[1][0], x[0][0]*y[0][1] + x[0][1]*y[1][1]],
             [x[1][0]*y[0][0] + x[1][1]*y[1][0], x[1][0]*y[0][1] + x[1][1]*y[1][1]]]
        za = xa.dot(ya)
        for i in range(2):
            for j in range(2):
                self.assertTrue(equivalent(z[i][j].x, za[i,j].x))
                self.assertTrue(equivalent(z[i][j].u, za[i,j].u))

    def test_matmul(self):
        # From Python 3.5+ the @ symbol can be used for an ndarray with dtype=object
        # NOTE: We can only run this test if Python >= 3.5
        import sys
        if sys.version_info >= (3, 5):
            from uarray_matmul import run
            run()
        else:
            m = [[ureal(5, 1), ureal(-1, 0.3), ureal(3, 1.3)],
                 [ureal(1, 0.1), ureal(2, 0.8), ureal(-3, 1)],
                 [ureal(-1, 0.5), ureal(2, 1.1), ureal(4, 0.3)]]
            b = [ureal(1, 0.2), ureal(2, 1.1), ureal(3, 0.4)]

            ma = uarray(m)
            ba = uarray(b)

            # vector * vector

            z = b[0] * 1 + b[1] * 2 + b[2] * 3
            za = matmul(ba, [1, 2, 3])
            assert equivalent(z.x, value(za))
            assert equivalent(z.u, uncertainty(za))

            try:
                matmul(ba, [1, 2])
            except ValueError:  # Expect this error -> shapes (3,) and (2,) not aligned: 3 (dim 0) != 2 (dim 0)
                pass
            else:
                raise ValueError('this should not work -> matmul(ba, [1, 2])')

            # vector * matrix

            z = [1 * m[0][0] + 2 * m[1][0] + 3 * m[2][0],
                 1 * m[0][1] + 2 * m[1][1] + 3 * m[2][1],
                 1 * m[0][2] + 2 * m[1][2] + 3 * m[2][2]]
            za = matmul([1, 2, 3], ma)
            for i in range(3):
                assert equivalent(z[i].x, za[i].x)
                assert equivalent(z[i].u, za[i].u)

            try:
                matmul([1, 2], ma)
            except ValueError:  # Expect this error -> shapes (2,) and (3,3) not aligned: 2 (dim 0) != 3 (dim 0)
                pass
            else:
                raise ValueError('this should not work -> matmul([1, 2], ma)')

            # matrix * vector

            z = [m[0][0] * b[0] + m[0][1] * b[1] + m[0][2] * b[2],
                 m[1][0] * b[0] + m[1][1] * b[1] + m[1][2] * b[2],
                 m[2][0] * b[0] + m[2][1] * b[1] + m[2][2] * b[2]]

            za = matmul(ma, ba)
            for i in range(3):
                assert equivalent(z[i].x, za[i].x)
                assert equivalent(z[i].u, za[i].u)

            try:
                matmul(ma, np.arange(4))
            except ValueError:  # Expect this error -> shapes (3,3) and (4,) not aligned: 3 (dim 1) != 4 (dim 0)
                pass
            else:
                raise ValueError('this should not work -> matmul(ma, np.arange(4))')

            # matrix * matrix

            na = np.arange(10 * 10).reshape(10, 10) * -3.1
            nb = np.arange(10 * 10).reshape(10, 10) * 2.3
            nc = matmul(na, nb)

            ua = uarray(na.copy() * ureal(1, 0))
            ub = uarray(nb.copy() * ureal(1, 0))
            uc = matmul(ua, ub)
            assert nc.shape == uc.shape

            i, j = nc.shape
            for ii in range(i):
                for jj in range(j):
                    assert equivalent(na[ii, jj], ua[ii, jj].x)
                    assert equivalent(nb[ii, jj], ub[ii, jj].x)
                    assert equivalent(nc[ii, jj], uc[ii, jj].x, tol=1e-10)

            try:
                matmul(ma, np.arange(4 * 4).reshape(4, 4))
            except ValueError:  # Expect this error -> shapes (3,3) and (4,4) not aligned: 3 (dim 1) != 4 (dim 0)
                pass
            else:
                raise ValueError('this should not work -> matmul(ma, np.arange(4*4).reshape(4,4))')

            # test a bunch of different dimensions
            test_dims = [
                [(), ()],
                [(0,), (1, 3)],
                [(1,), (1, 3)],
                [(4,), (4, 3)],
                [(2, 4), (4,)],
                [(2, 4), (3,)],
                [(2, 4), (3, 2)],
                [(2, 4), (4, 2)],
                [(1, 2, 4), (1, 4, 2)],
                [(2, 2, 4), (1, 4, 2)],
                [(1, 2, 4), (2, 4, 2)],
                [(2, 2, 4), (2, 4, 2)],
                [(3, 2, 4), (3, 4, 2)],
                [(6, 2, 4), (3, 2, 2)],
                [(6, 2, 4), (3, 4, 8)],
                [(6, 2, 4), (6, 4, 8)],
                [(5, 3, 2, 4), (5, 3, 4, 2)],
                [(3, 2, 2, 4), (3, 9, 4, 2)],
                [(8, 3, 1, 2, 4), (8, 3, 9, 4, 2)],
            ]

            for s1, s2 in test_dims:
                na = np.arange(int(np.prod(np.array(s1)))).reshape(s1)
                nb = np.arange(int(np.prod(np.array(s2)))).reshape(s2)
                try:
                    nc = matmul(na, nb)
                except:
                    nc = None

                ua = uarray(na.copy() * ureal(1, 0))
                ub = uarray(nb.copy() * ureal(1, 0))
                try:
                    uc = matmul(ua, ub)
                except:
                    if nc is not None:
                        raise AssertionError('The regular matmul PASSED, the custom-written matmul FAILED')
                else:
                    if nc is None:
                        raise AssertionError('The regular matmul FAILED, the custom-written matmul PASSED')
                    assert np.array_equal(nc, uc), 'The arrays are not equal\n{}\n{}'.format(nc, uc)

    def test_astype(self):
        # make sure that the following is not allowed

        # TypeError: float() argument must be a string or a number, not 'UncertainReal'
        with self.assertRaises(TypeError):
            _ = self.xa.astype(np.float32)

        # TypeError: float() argument must be a string or a number, not 'UncertainComplex'
        with self.assertRaises(TypeError):
            _ = self.xca.astype(np.float32)

    def test_prod(self):
        self.assertRaises(TypeError,self.xa.prod)
        self.assertRaises(TypeError,np.prod,self.xa)
        # # 1D array
        # xlist = [ureal(i, i * 0.1) for i in range(100)]
        # xarray = uarray(xlist)
        # self.assertTrue(xarray.shape == (100,))

        # a = xarray.prod()
        # b = 1.0
        # for x in xlist:
            # b *= x
        # self.assertTrue(equivalent(value(a), b.x))
        # self.assertTrue(equivalent(uncertainty(a), b.u))

        # # 3D array
        # xlist = [[[ureal(i * j * k, i * j * k * 0.1) for k in range(1, 5)] for j in range(7, 10)] for i in range(3, 9)]
        # xarray = uarray(xlist)
        # self.assertTrue(xarray.shape == (6, 3, 4))

        # axis_none = 1.0
        # axis_0 = [[1.0 for i in range(4)] for j in range(3)]
        # axis_1 = [[1.0 for i in range(4)] for j in range(6)]
        # axis_2 = [[1.0 for i in range(3)] for j in range(6)]
        # for i in range(6):
            # for j in range(3):
                # for k in range(4):
                    # _value = xlist[i][j][k]
                    # axis_none *= _value
                    # axis_0[j][k] *= _value
                    # axis_1[i][k] *= _value
                    # axis_2[i][j] *= _value

        # # axis=None
        # a = xarray.prod()
        # self.assertTrue(equivalent(value(a), axis_none.x))
        # self.assertTrue(equivalent(uncertainty(a), axis_none.u))

        # # axis=0
        # a = xarray.prod(axis=0)
        # m, n = len(axis_0), len(axis_0[0])
        # self.assertTrue(a.shape == (m, n))
        # for j in range(m):
            # for k in range(n):
                # self.assertTrue(equivalent(a[j, k].x, axis_0[j][k].x))
                # self.assertTrue(equivalent(a[j, k].u, axis_0[j][k].u))

        # # axis=1
        # a = xarray.prod(axis=1)
        # m, n = len(axis_1), len(axis_1[0])
        # self.assertTrue(a.shape == (m, n))
        # for i in range(m):
            # for k in range(n):
                # self.assertTrue(equivalent(a[i, k].x, axis_1[i][k].x))
                # self.assertTrue(equivalent(a[i, k].u, axis_1[i][k].u))

        # # axis=2
        # a = xarray.prod(axis=2)
        # m, n = len(axis_2), len(axis_2[0])
        # self.assertTrue(a.shape == (m, n))
        # for i in range(m):
            # for j in range(n):
                # self.assertTrue(equivalent(a[i, j].x, axis_2[i][j].x))
                # self.assertTrue(equivalent(a[i, j].u, axis_2[i][j].u))

    def test_dtype(self):
        self.assertTrue(isinstance(self.xa.dtype, object))
        self.assertTrue(isinstance(self.xca.dtype, object))

    def test_item(self):
        for i in range(len(self.x)):
            self.assertTrue(equivalent(self.xa.item(i).x, self.x[i]))
            self.assertTrue(equivalent(self.xa.item(i).x, self.xa[i].x))
            self.assertTrue(equivalent(self.xa.item(i).u, self.xa[i].u))

        x = np.arange(2*6*4).reshape(2, 6, 4) * ureal(1, 1)
        xa = uarray(x)
        idx = 0
        for i in range(2):
            for j in range(6):
                for k in range(4):
                    self.assertTrue(equivalent(xa.item(idx).x, x[i, j, k]))
                    self.assertTrue(xa.item(idx).x is xa[i, j, k].x)
                    self.assertTrue(xa.item(idx).u is xa[i, j, k].u)
                    idx += 1

    def test_itemset(self):
        xa = uarray(np.ones(9).reshape(3, 3) * ureal(1, 0.1))
        self.assertTrue(xa.item(4).x == 1)
        self.assertTrue(xa.item(4).u == 0.1)
        self.assertTrue(xa[1, 1].x == 1)
        self.assertTrue(xa[1, 1].u == 0.1)

        xa.itemset(4, ureal(99, .9))
        self.assertTrue(xa.item(4).x == 99.)
        self.assertTrue(xa.item(4).u == 0.9)
        self.assertTrue(xa[1, 1].x == 99.)
        self.assertTrue(xa[1, 1].u == 0.9)

        xa.itemset((2, 2), ureal(-99, 9.9))
        self.assertTrue(xa.item(xa.size-1).x == -99.)
        self.assertTrue(xa.item(xa.size-1).u == 9.9)
        self.assertTrue(xa[2, 2].x == -99.)
        self.assertTrue(xa[2, 2].u == 9.9)

    def test_itemsize(self):
        # just check that the itemsize attribute doesn't raise an exception
        self.assertTrue(isinstance(self.xa.itemsize, int))

    def test_get_set_field(self):
        # TypeError: Cannot get/set field of an object array
        with self.assertRaises(TypeError):
            _ = self.xa.getfield(np.float64)

        with self.assertRaises(TypeError):
            _ = self.xa.setfield(3, np.int32)

    def test_put(self):
        xa = uarray(np.arange(5) * ureal(1, 1))
        xa.put([0, 2], [ureal(-44, 4), ureal(-55, 5)])
        self.assertTrue(xa[0].x == -44)
        self.assertTrue(xa[0].u == 4)
        self.assertTrue(xa[1].x == 1)
        self.assertTrue(xa[1].u == 1)
        self.assertTrue(xa[2].x == -55)
        self.assertTrue(xa[2].u == 5)
        self.assertTrue(xa[3].x == 3)
        self.assertTrue(xa[3].u == 3)
        self.assertTrue(xa[4].x == 4)
        self.assertTrue(xa[4].u == 4)

    def test_take(self):
        xa = uarray([ureal(4, 1), ureal(3, 1), ureal(5, 1), ureal(7, 1), ureal(6, 1), ureal(8, 1)])
        indices = [0, 1, 4]
        out = np.take(xa, indices)
        self.assertTrue(out[0].x == 4)
        self.assertTrue(out[1].x == 3)
        self.assertTrue(out[2].x == 6)

    def test_repeat(self):
        xa = uarray([[ureal(1, .1), ureal(2, .2)], [ureal(3, .3), ureal(4, .4)]])
        xa = np.repeat(xa, 2)
        self.assertTrue(xa[0].x == 1)
        self.assertTrue(xa[1].x == 1)
        self.assertTrue(xa[2].x == 2)
        self.assertTrue(xa[3].x == 2)
        self.assertTrue(xa[4].x == 3)
        self.assertTrue(xa[5].x == 3)
        self.assertTrue(xa[6].x == 4)
        self.assertTrue(xa[7].x == 4)

    def test_resize(self):
        xa = uarray([ureal(i, 0.1*i) for i in range(1000)])
        self.assertTrue(isinstance(xa, UncertainArray))
        self.assertTrue(xa.shape == (1000,))
        xa.resize(100, 2, 5)
        self.assertTrue(xa.shape == (100, 2, 5))
        self.assertTrue(isinstance(xa, UncertainArray))

    def test_ravel(self):
        xa = uarray([[[ureal(1, 1) for _ in range(10)] for _ in range(5)] for _ in range(8)])
        self.assertTrue(isinstance(xa, UncertainArray))
        self.assertTrue(xa.shape == (8, 5, 10))
        xa = xa.ravel()
        self.assertTrue(xa.shape == (8*5*10,))
        self.assertTrue(isinstance(xa, UncertainArray))

    def test_tostring_tobytes(self):
        # this just tests that one could call these attributes
        self.assertTrue(self.xca.tostring() == self.xca.tobytes())

    def test_tolist(self):
        xlist = self.xa.tolist()
        self.assertTrue(isinstance(xlist, list))
        self.assertTrue(isinstance(self.xa, UncertainArray))
        for i in range(len(self.x)):
            self.assertTrue(self.x[i].x == xlist[i])

    def test_swapaxes(self):
        xa = uarray([[[ureal(1, 1) for _ in range(10)] for _ in range(5)] for _ in range(8)])
        self.assertTrue(xa.shape == (8, 5, 10))
        xa = np.swapaxes(xa, 0, 1)
        self.assertTrue(isinstance(xa, UncertainArray))
        self.assertTrue(xa.shape == (5, 8, 10))

    def test_strides(self):
        i, j, k = 7, 3, 5
        xa = uarray([[[ureal(1, 1) for _ in range(k)] for _ in range(j)] for _ in range(i)])
        self.assertTrue(xa.strides == (xa.itemsize*j*k, xa.itemsize*k, xa.itemsize))

    def test_squeeze(self):
        xa = uarray([[[ureal(0, 0)], [ureal(1, 1)], [ureal(2, 2)]]])
        self.assertTrue(xa.shape == (1, 3, 1))
        a = xa.squeeze()
        self.assertTrue(isinstance(a, UncertainArray))
        self.assertTrue(a.shape == (3,))
        a = xa.squeeze(axis=0)
        self.assertTrue(isinstance(a, UncertainArray))
        self.assertTrue(a.shape == (3, 1))
        with self.assertRaises(ValueError):
            _ = xa.squeeze(axis=1)
        a = xa.squeeze(axis=2)
        self.assertTrue(isinstance(a, UncertainArray))
        self.assertTrue(a.shape == (1, 3))

    def test_sort(self):
        a = uarray([[ureal(1, 0), ureal(4, 0)], [ureal(3, 0), ureal(1, 0)]])

        a.sort(axis=1)
        self.assertTrue(a[0, 0].x == 1)
        self.assertTrue(a[0, 1].x == 4)
        self.assertTrue(a[1, 0].x == 1)
        self.assertTrue(a[1, 1].x == 3)

        a.sort(axis=0)
        self.assertTrue(a[0, 0].x == 1)
        self.assertTrue(a[0, 1].x == 3)
        self.assertTrue(a[1, 0].x == 1)
        self.assertTrue(a[1, 1].x == 4)

    def test_flags_setflags(self):
        xa = uarray([ureal(i, 0.1 * i) for i in range(10)])
        self.assertTrue(xa.flags.aligned)
        self.assertTrue(xa.flags.writeable)
        self.assertTrue(isinstance(xa, UncertainArray))

        xa.setflags(write=False, align=False)
        self.assertTrue(not xa.flags.aligned)
        self.assertTrue(not xa.flags.writeable)
        self.assertTrue(isinstance(xa, UncertainArray))

    def test_searchsorted(self):
        # sorting should be independent of the value of UN.u
        a = uarray([ureal(1,5), ureal(2,4), ureal(3,3), ureal(4,2), ureal(5,1)])

        self.assertTrue(a.searchsorted(3) == 2)
        self.assertTrue(a.searchsorted(3, side='right') == 3)

        out = a.searchsorted([ureal(-10, 88), ureal(10, 53), ureal(2, 61), ureal(3, 0.5)])
        self.assertTrue(out[0] == 0)
        self.assertTrue(out[1] == 5)
        self.assertTrue(out[2] == 1)
        self.assertTrue(out[3] == 2)

    def test_ptp(self):
        self.assertRaises(TypeError,self.xa.ptp)
        self.assertRaises(TypeError,np.ptp,self.xa)
        # xa = uarray([[[ureal(i*j*k, 0.8) for k in range(2)] for j in range(3)] for i in range(4)])

        # ptp = xa.ptp()
        # self.assertTrue(value(ptp) == 6)

        # ptp = xa.ptp(axis=0)
        # self.assertTrue(value( function.sum( ptp[:, 0] ) ) == 0)
        # self.assertTrue(value( function.sum( ptp[0, :] ) ) == 0)
        # self.assertTrue(value( function.sum( ptp[:, 1] ) ) == 9)
        # self.assertTrue(value( function.sum( ptp[1, :] ) ) == 3)
        # self.assertTrue(value( function.sum( ptp[2, :] ) ) == 6)

        # ptp = xa.ptp(axis=1)
        # self.assertTrue(value( function.sum( ptp[:, 0] ) ) == 0)
        # self.assertTrue(value( function.sum( ptp[0, :] ) ) == 0)
        # self.assertTrue(value( function.sum( ptp[:, 1] ) ) == 12)
        # self.assertTrue(value( function.sum( ptp[1, :] ) ) == 2)
        # self.assertTrue(value( function.sum( ptp[2, :] ) ) == 4)
        # self.assertTrue(value( function.sum( ptp[3, :] ) ) == 6)

        # ptp = xa.ptp(axis=2)
        # self.assertTrue(value( function.sum( ptp[:, 0] ) ) == 0)
        # self.assertTrue(value( function.sum( ptp[0, :] ) ) == 0)
        # self.assertTrue(value( function.sum( ptp[:, 1] ) ) == 6)
        # self.assertTrue(value( function.sum( ptp[:, 2] ) ) == 12)
        # self.assertTrue(value( function.sum( ptp[1, :] ) ) == 3)
        # self.assertTrue(value( function.sum( ptp[2, :] ) ) == 6)
        # self.assertTrue(value( function.sum( ptp[3, :] ) ) == 9)

    def test_round(self):
        a = uarray([[ureal(0.378384871, 0.1831984, df=12.44649822), ureal(1.649863876, 1.28794362876, df=9.2184761424)],
                    [ureal(64.17441638, 2.4987163, df=inf), ureal(-472.974793166, 7.812474106, df=87.1683249)]])

        self.assertTrue(isinstance(a, UncertainArray))
        self.assertTrue(isinstance(a[0], UncertainArray))
        self.assertTrue(isinstance(a[0,0], UncertainReal))

        # `decimals` parameter for np.round is really the `digits` parameter of UncertainReal._round
        for i in range(3):
            if i == 0:
                out = np.round(a, decimals=2)
            elif i == 1:
                out = a.round(decimals=2)
            else:
                out = a.round(digits=2)
            self.assertTrue(out is not a)
            self.assertTrue(out[0] is not a[0])
            self.assertTrue(out[0][0] is not a[0][0])
            self.assertTrue(isinstance(out[0], UncertainArray))
            self.assertTrue(isinstance(out[0, 0], GroomedUncertainReal))
            self.assertTrue(equivalent(out[0, 0].x, 0.38))
            self.assertTrue(equivalent(out[0, 0].u, 0.18))
            self.assertTrue(equivalent(out[0, 0].df, 12.44))
            self.assertTrue(equivalent(out[0, 1].x, 1.6))
            self.assertTrue(equivalent(out[0, 1].u, 1.3))
            self.assertTrue(equivalent(out[0, 1].df, 9.21))
            self.assertTrue(equivalent(out[1, 0].x, 64.2))
            self.assertTrue(equivalent(out[1, 0].u, 2.5))
            self.assertTrue(math.isinf(out[1, 0].df))
            self.assertTrue(equivalent(out[1, 1].x, -473.0))
            self.assertTrue(equivalent(out[1, 1].u, 7.8))
            self.assertTrue(equivalent(out[1, 1].df, 87.16))

        out = a.round(digits=4, df_decimals=0)
        self.assertTrue(equivalent(out[0, 0].x, 0.3784))
        self.assertTrue(equivalent(out[0, 0].u, 0.1832))
        self.assertTrue(equivalent(out[0, 0].df, 12))
        self.assertTrue(equivalent(out[0, 1].x, 1.650))
        self.assertTrue(equivalent(out[0, 1].u, 1.2880))
        self.assertTrue(equivalent(out[0, 1].df, 9))
        self.assertTrue(equivalent(out[1, 0].x, 64.174))
        self.assertTrue(equivalent(out[1, 0].u, 2.499))
        self.assertTrue(math.isinf(out[1, 0].df))
        self.assertTrue(equivalent(out[1, 1].x, -472.975))
        self.assertTrue(equivalent(out[1, 1].u, 7.812))
        self.assertTrue(equivalent(out[1, 1].df, 87))

    # # TODO the partition and argpartition tests do not produce the expected results
    # # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.partition.html?highlight=partition#numpy.ndarray.partition
    # def test_partition(self):
    #     a = uarray([ureal(3, 33), ureal(4, 44), ureal(2, 22), ureal(1, 11)])
    #     a.partition(3)
    #     self.assertTrue(equivalent(a[0].x, 2.0))
    #     self.assertTrue(equivalent(a[0].u, 22.0))
    #     self.assertTrue(equivalent(a[1].x, 1.0))
    #     self.assertTrue(equivalent(a[1].u, 11.0))
    #     self.assertTrue(equivalent(a[2].x, 3.0))
    #     self.assertTrue(equivalent(a[2].u, 33.0))
    #     self.assertTrue(equivalent(a[3].x, 4.0))
    #     self.assertTrue(equivalent(a[3].u, 44.0))
    #
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.argpartition.html#numpy.argpartition
    # def test_argpartition(self):
    #     a = uarray([ureal(3, 33), ureal(4, 44), ureal(2, 22), ureal(1, 11)])
    #     print(a[np.argpartition(a, 3)])  -> expect array([2, 1, 3, 4])

    def test_nonzero(self):
        x = uarray([[ureal(1, 1), ureal(0, 1), ureal(0, 1)],
                    [ureal(0, 1), ureal(2, 1), ureal(0, 1)],
                    [ureal(1, 1), ureal(1, 1), ureal(0, 1)]])

        indices = x.nonzero()
        self.assertTrue(np.array_equal(indices[0], [0, 1, 2, 2]))
        self.assertTrue(np.array_equal(indices[1], [0, 1, 0, 1]))

        indices = np.nonzero(x)
        self.assertTrue(np.array_equal(indices[0], [0, 1, 2, 2]))
        self.assertTrue(np.array_equal(indices[1], [0, 1, 0, 1]))

    def test_newbyteorder(self):
        # just testing that calling newbyteorder does not raise an exception
        # calling this method doesn't do anything to the uarray
        self.assertTrue(isinstance(self.xa.newbyteorder(), UncertainArray))
        self.assertTrue(np.array_equal(self.xa.newbyteorder('S'), self.xa))
        self.assertTrue(np.array_equal(self.xa.newbyteorder('L'), self.xa))
        self.assertTrue(np.array_equal(self.xa.newbyteorder('N'), self.xa))

    def test_ctypes(self):
        # just testing that calling ctypes does not raise an exception
        self.assertTrue(isinstance(self.xa.ctypes, object))

    def test_dump_dumps(self):
        d = self.xa.dumps()
        xa = pickle.loads(d)
        self.assertTrue(np.array_equal(self.xa, xa))

        path = os.path.join(tempfile.gettempdir(), 'uarray-dump.dat')
        with open(path, 'wb') as fp:
            self.xa.dump(fp)
        with open(path, 'rb') as fp:
            xa = pickle.load(fp)
        self.assertTrue(np.array_equal(self.xa, xa))
        os.remove(path)

    def test_tofile(self):
        path = os.path.join(tempfile.gettempdir(), 'uarray-tofile.txt')
        self.xa.tofile(path, sep=' ')
        with open(path, 'rt') as fp:
            text = fp.read()
        self.assertTrue(text.startswith('ureal('))
        os.remove(path)

    def test_byteswap(self):
        # just testing that calling byteswap does not raise an exception
        # calling this method doesn't do anything to the uarray
        a = uarray([ureal(1, 1), ureal(2, 2)])
        bs = a.byteswap()
        self.assertTrue(isinstance(bs, UncertainArray))
        self.assertTrue(np.array_equal(bs, a))

    def test_choose(self):
        choices = uarray([[ureal(0, 0), ureal(1, 1), ureal(2, 2), ureal(3, 3)],
                          [ureal(10, 10), ureal(11, 11), ureal(12, 12), ureal(13, 13)],
                          [ureal(20, 20), ureal(21, 21), ureal(22, 22), ureal(23, 23)],
                          [ureal(30, 30), ureal(31, 31), ureal(32, 32), ureal(33, 33)]])

        chosen = np.choose([2, 3, 1, 0], choices)
        self.assertTrue(equivalent(chosen[0].x, 20))
        self.assertTrue(equivalent(chosen[1].x, 31))
        self.assertTrue(equivalent(chosen[2].x, 12))
        self.assertTrue(equivalent(chosen[3].x, 3))

        chosen = np.choose([2, 4, 1, 0], choices, mode='clip')
        self.assertTrue(equivalent(chosen[0].x, 20))
        self.assertTrue(equivalent(chosen[1].x, 31))
        self.assertTrue(equivalent(chosen[2].x, 12))
        self.assertTrue(equivalent(chosen[3].x, 3))

        chosen = np.choose([2, 4, 1, 0], choices, mode='wrap')
        self.assertTrue(equivalent(chosen[0].x, 20))
        self.assertTrue(equivalent(chosen[1].x, 1))
        self.assertTrue(equivalent(chosen[2].x, 12))
        self.assertTrue(equivalent(chosen[3].x, 3))

        choices = uarray([ureal(-10, 1), ureal(10, 2)])
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        chosen = a.choose(choices)  # access the ndarray choose() method, just to try something different
        self.assertTrue(equivalent(chosen[0, 0].x, 10))
        self.assertTrue(equivalent(chosen[0, 1].x, -10))
        self.assertTrue(equivalent(chosen[0, 2].x, 10))
        self.assertTrue(equivalent(chosen[1, 0].x, -10))
        self.assertTrue(equivalent(chosen[1, 1].x, 10))
        self.assertTrue(equivalent(chosen[1, 2].x, -10))
        self.assertTrue(equivalent(chosen[2, 0].x, 10))
        self.assertTrue(equivalent(chosen[2, 1].x, -10))
        self.assertTrue(equivalent(chosen[2, 2].x, 10))

    def test_clip(self):
        a = uarray([ureal(i, 0.1*i) for i in range(10)])
        for i in range(10):
            self.assertTrue(equivalent(a[i].x, i))
            self.assertTrue(equivalent(a[i].u, i*0.1))

        out = np.clip(a, a_min=ureal(1, 1), a_max=ureal(8, 8))
        self.assertTrue(equivalent(out[0].x, 1))
        self.assertTrue(equivalent(out[0].u, 1))
        for i in range(1, 9):
            self.assertTrue(equivalent(a[i].x, i))
            self.assertTrue(equivalent(a[i].u, i*0.1))
        self.assertTrue(equivalent(out[9].x, 8))
        self.assertTrue(equivalent(out[9].u, 8))

        a = uarray([[ureal(i*j, 0.1*i*j) for j in range(1, 5)] for i in range(2,4)])
        self.assertTrue(equivalent(a[0, 0].x, 2))
        self.assertTrue(equivalent(a[0, 1].x, 4))
        self.assertTrue(equivalent(a[0, 2].x, 6))
        self.assertTrue(equivalent(a[0, 3].x, 8))
        self.assertTrue(equivalent(a[1, 0].x, 3))
        self.assertTrue(equivalent(a[1, 1].x, 6))
        self.assertTrue(equivalent(a[1, 2].x, 9))
        self.assertTrue(equivalent(a[1, 3].x, 12))
        out = np.clip(a, a_min=ureal(5, 5), a_max=ureal(7, 7))
        self.assertTrue(equivalent(out[0, 0].x, 5))
        self.assertTrue(equivalent(out[0, 1].x, 5))
        self.assertTrue(equivalent(out[0, 2].x, 6))
        self.assertTrue(equivalent(out[0, 3].x, 7))
        self.assertTrue(equivalent(out[1, 0].x, 5))
        self.assertTrue(equivalent(out[1, 1].x, 6))
        self.assertTrue(equivalent(out[1, 2].x, 7))
        self.assertTrue(equivalent(out[1, 3].x, 7))

    def test_minimum(self):
        a = uarray([ureal(2, 6), ureal(3, 1), ureal(4, 5)])
        b = uarray([ureal(1, 0.05), ureal(5, 2), ureal(2, 9)])
        out = np.minimum(a, b)
        self.assertTrue(isinstance(out, UncertainArray))
        self.assertTrue(equivalent(out[0].x, 1))
        self.assertTrue(equivalent(out[1].x, 3))
        self.assertTrue(equivalent(out[2].x, 2))

        # comparisons with NaN and INF
        a = uarray([self._ureal(nan, nan), self._ureal(-inf, 0), ureal(7, 0.7), self._ureal(-inf, inf)])
        b = uarray([ureal(7, 2), self._ureal(nan, nan), 10, self._ureal(inf, inf)])
        out = np.minimum(a, b)
        self.assertTrue(math.isnan(out[0].x))
        self.assertTrue(math.isnan(out[1].x))
        self.assertTrue(equivalent(out[2].x, 7))
        self.assertTrue(out[3].x == -inf)

        a = a.reshape(2, 2)
        b = b.reshape(2, 2)
        out = np.minimum(a, b)
        self.assertTrue(math.isnan(out[0, 0].x))
        self.assertTrue(math.isnan(out[0, 1].x))
        self.assertTrue(equivalent(out[1, 0].x, 7))
        self.assertTrue(out[1, 1].x == -inf)

    def test_maximum(self):
        a = uarray([ureal(2, 6), ureal(3, 1), ureal(4, 5)])
        b = uarray([ureal(1, 0.05), ureal(5, 2), ureal(2, 9)])
        out = np.maximum(a, b)
        self.assertTrue(isinstance(out, UncertainArray))
        self.assertTrue(equivalent(out[0].x, 2))
        self.assertTrue(equivalent(out[1].x, 5))
        self.assertTrue(equivalent(out[2].x, 4))

        # comparisons with NaN and INF
        a = uarray([self._ureal(nan, nan), self._ureal(-inf, 0), ureal(7, 0.7), self._ureal(-inf, inf)])
        b = uarray([ureal(7, 2), self._ureal(nan, nan), 10, self._ureal(inf, inf)])
        out = np.maximum(a, b)
        self.assertTrue(math.isnan(out[0].x))
        self.assertTrue(math.isnan(out[1].x))
        self.assertTrue(equivalent(out[2], 10))
        self.assertTrue(out[3].x == inf)

        a = a.reshape(2, 2)
        b = b.reshape(2, 2)
        out = np.maximum(a, b)
        self.assertTrue(math.isnan(out[0, 0].x))
        self.assertTrue(math.isnan(out[0, 1].x))
        self.assertTrue(equivalent(out[1, 0], 10))
        self.assertTrue(out[1, 1].x == inf)

    def test_logical_or(self):
        # Python logical 'and' applied to numbers returns 
        # a number. Eg, a = 0, b = 1: `a or b` evaluates to 1.
        # So with uncertain numbers we do the same
        self.assertTrue(all(np.logical_or(self.xa, self.ya)))

        a = uarray([ureal(0, 1), ureal(0, 1), ureal(1, 1), ureal(1, 1)])
        b = uarray([ureal(0, 1), ureal(1, 1), ureal(0, 1), ureal(1, 1)])
        c = np.logical_or(a, b)
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(not c[0])
        self.assertTrue(c[0] is b[0])
        self.assertTrue(c[1])
        self.assertTrue(c[1] is b[1])
        self.assertTrue(c[2])
        self.assertTrue(c[2] is a[2])
        self.assertTrue(c[3])
        self.assertTrue(c[3] is a[3])

        c = np.logical_or(a.reshape(2, 2), b.reshape(2, 2))
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(not c[0, 0])
        self.assertTrue(c[0, 1])
        self.assertTrue(c[1, 0])
        self.assertTrue(c[1, 1])

        x = uarray(np.arange(5) * ureal(1, 1))
        out = np.logical_or(x < 1, x > 3)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(out[0])
        self.assertTrue(not out[1])
        self.assertTrue(not out[2])
        self.assertTrue(not out[3])
        self.assertTrue(out[4])

    def test_logical_and(self):
        # Python logical 'and' applied to numbers returns 
        # a number. Eg, a = 3, b = 1: `a and b` evaluates to 3.
        # So with uncertain numbers we do the same
        self.assertTrue(all(np.logical_and(self.xa, self.ya)))

        a = uarray([ureal(0, 1), ureal(0, 1), ureal(1, 1), ureal(1, 1)])
        b = uarray([ureal(0, 1), ureal(1, 1), ureal(0, 1), ureal(1, 1)])
        c = np.logical_and(a, b)
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(not c[0])
        self.assertTrue(c[0] is a[0])
        self.assertTrue(not c[1])
        self.assertTrue(c[1] is a[1])
        self.assertTrue(not c[2])
        self.assertTrue(c[2] is b[2])
        self.assertTrue(c[3])
        self.assertTrue(c[3] is b[3])

        c = np.logical_and(a.reshape(2, 2), b.reshape(2, 2))
        self.assertTrue(isinstance(c, UncertainArray))
        self.assertTrue(not c[0, 0])
        self.assertTrue(not c[0, 1])
        self.assertTrue(not c[1, 0])
        self.assertTrue(c[1, 1])

        x = uarray(np.arange(5) * ureal(1, 1))
        out = np.logical_and(x > 1, x < 4)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0])
        self.assertTrue(not out[1])
        self.assertTrue(out[2])
        self.assertTrue(out[3])
        self.assertTrue(not out[4])

    def test_logical_not(self):
        self.assertTrue(not all(np.logical_not(self.xa)))

        a = uarray([ureal(0, 1), ureal(0, 1), ureal(1, 1), ureal(1, 1)])
        c = np.logical_not(a)
        self.assertTrue(not isinstance(c, UncertainArray))
        self.assertTrue(c[0])
        self.assertTrue(c[1])
        self.assertTrue(not c[2])
        self.assertTrue(not c[3])

        c = np.logical_not(a.reshape(2, 2))
        self.assertTrue(not isinstance(c, UncertainArray))
        self.assertTrue(c[0, 0])
        self.assertTrue(c[0, 1])
        self.assertTrue(not c[1, 0])
        self.assertTrue(not c[1, 1])

        x = uarray(np.arange(5) * ureal(1, 1))
        out = np.logical_not(x < 3)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0])
        self.assertTrue(not out[1])
        self.assertTrue(not out[2])
        self.assertTrue(out[3])
        self.assertTrue(out[4])

    # def test_logical_xor(self):
        # c = np.logical_xor(uarray(ureal(1, 0)), uarray(ureal(0, 1)))
        # self.assertTrue(not isinstance(c, UncertainArray))
        # self.assertTrue(c)

        # a = uarray([ucomplex(1, 0), ucomplex(2, 0), ucomplex(0, 1), ucomplex(0, 2)])
        # b = uarray([ucomplex(1, 0), ucomplex(0, 0), ucomplex(1, 1), ucomplex(0, 2)])
        # c = np.logical_xor(a, b)
        # self.assertTrue(not isinstance(c, UncertainArray))
        # self.assertTrue(not c[0])
        # self.assertTrue(c[1])
        # self.assertTrue(c[2])
        # self.assertTrue(not c[3])

        # a = uarray(np.eye(5) * ureal(1, 1))
        # b = uarray(np.zeros((5, 5)) * ureal(1, 1))
        # c = np.logical_xor(a, b)
        # self.assertTrue(not isinstance(c, UncertainArray))
        # for i in range(5):
            # for j in range(5):
                # if i == j:
                    # self.assertTrue(c[i, j])
                # else:
                    # self.assertTrue(not c[i, j])

    def test_any(self):
        self.assertRaises(TypeError,self.xa.any)
        self.assertRaises(TypeError,np.any,self.xa)
        # self.assertTrue(self.xa.any())

        # a = uarray([[ureal(1, 0), ureal(0, 0)], [ureal(-121, 7), ureal(3, 1)]])
        # self.assertTrue(np.any(a))

        # a = uarray([[ureal(1, 0), ureal(0, 0)], [ureal(0, 7), ureal(0, 1)]])
        # c = np.any(a, axis=0)
        # self.assertTrue(isinstance(c, UncertainArray))
        # self.assertTrue(c[0])
        # self.assertTrue(not c[1])

        # c = np.any(uarray([ucomplex(-1j, 1), ucomplex(0, 0), ucomplex(5 + 2j, 1)]))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)

        # # NaN, +INF, -INF evaluate to True because these are not equal to zero.
        # c = np.any(uarray(self._ureal(np.nan, 0)))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)
        # c = np.any(uarray(self._ureal(np.inf, 0)))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)
        # c = np.any(uarray(self._ureal(-np.inf, 0)))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)

    def test_all(self):
        self.assertRaises(TypeError,self.xa.all)
        self.assertRaises(TypeError,np.all,self.xa)
        # self.assertTrue(self.xa.all())

        # a = uarray([[ureal(1, 0), ureal(0, 0)], [ureal(-121, 7), ureal(3, 1)]])
        # self.assertTrue(not a.all())

        # a = uarray([[ureal(1, 0), ureal(0, 0)], [ureal(-10, 7), ureal(9, 1)]])
        # c = np.all(a, axis=0)
        # self.assertTrue(isinstance(c, UncertainArray))
        # self.assertTrue(c[0])
        # self.assertTrue(not c[1])

        # c = np.all(uarray([ucomplex(-1j, 1), ucomplex(2.1-3.2j, 0.004), ucomplex(5 + 2j, 2.3)]))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)

        # # NaN, +INF, -INF evaluate to True because these are not equal to zero.
        # c = np.all(uarray([self._ureal(np.nan, 0), ureal(2, 1)]))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)
        # c = np.all(uarray(self._ureal(np.inf, 0)))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)
        # c = np.all(uarray(self._ureal(-np.inf, 0)))
        # self.assertTrue(isinstance(c, np.bool_))
        # self.assertTrue(c)

    def test_isnan(self):
        self.assertTrue(not all(np.isnan(self.xa)))
        self.assertTrue(not all(np.isnan(self.xca)))

        a = uarray([self._ureal(nan, nan), self._ureal(nan, 0), ureal(7, 7), self._ureal(inf, 0)])
        out = np.isnan(a)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(out[0])
        self.assertTrue(out[1])
        self.assertTrue(not out[2])
        self.assertTrue(not out[3])

        out = np.isnan(a.reshape(2, 2))
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(out[0, 0])
        self.assertTrue(out[0, 1])
        self.assertTrue(not out[1, 0])
        self.assertTrue(not out[1, 1])

        a = uarray([self._ucomplex(nan, nan, nan), self._ucomplex(nan, 0, 0),
                    ucomplex(7, 7, 7), self._ucomplex(inf, 0, 0)])
        out = np.isnan(a)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(out[0])
        self.assertTrue(out[1])
        self.assertTrue(not out[2])
        self.assertTrue(not out[3])

        out = np.isnan(a.reshape(2, 2))
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(out[0, 0])
        self.assertTrue(out[0, 1])
        self.assertTrue(not out[1, 0])
        self.assertTrue(not out[1, 1])

    def test_isinf(self):
        self.assertTrue(not all(np.isinf(self.xa)))
        self.assertTrue(not all(np.isinf(self.xca)))

        a = uarray([self._ureal(nan, nan), self._ureal(-inf, 0), ureal(7, 7), self._ureal(inf, 0)])
        out = np.isinf(a)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0])
        self.assertTrue(out[1])
        self.assertTrue(not out[2])
        self.assertTrue(out[3])

        out = np.isinf(a.reshape(2, 2))
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0, 0])
        self.assertTrue(out[0, 1])
        self.assertTrue(not out[1, 0])
        self.assertTrue(out[1, 1])

        a = uarray([self._ucomplex(nan+1j, nan, nan), self._ucomplex(-inf, 0, 0),
                    ucomplex(7, 7, 7), self._ucomplex(inf, 0, 0)])
        out = np.isinf(a)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0])
        self.assertTrue(out[1])
        self.assertTrue(not out[2])
        self.assertTrue(out[3])

        out = np.isinf(a.reshape(2, 2))
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0, 0])
        self.assertTrue(out[0, 1])
        self.assertTrue(not out[1, 0])
        self.assertTrue(out[1, 1])

    def test_isfinite(self):
        self.assertTrue(all(np.isfinite(self.xa)))

        a = uarray([self._ureal(nan, nan), self._ureal(-inf, 0), ureal(7, 7), self._ureal(inf, 0)])
        out = np.isfinite(a)
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0])
        self.assertTrue(not out[1])
        self.assertTrue(out[2])
        self.assertTrue(not out[3])

        out = np.isfinite(a.reshape(2, 2))
        self.assertTrue(not isinstance(out, UncertainArray))
        self.assertTrue(not out[0, 0])
        self.assertTrue(not out[0, 1])
        self.assertTrue(out[1, 0])
        self.assertTrue(not out[1, 1])

    def test_reciprocal(self):
        z = [1./val for val in self.x]
        za = np.reciprocal(self.xa)
        for i in range(len(z)):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        za = np.reciprocal(self.xa.reshape(2, 3))
        self.assertTrue(equivalent(z[0].x, za[0, 0].x))
        self.assertTrue(equivalent(z[0].u, za[0, 0].u))
        self.assertTrue(equivalent(z[1].x, za[0, 1].x))
        self.assertTrue(equivalent(z[1].u, za[0, 1].u))
        self.assertTrue(equivalent(z[2].x, za[0, 2].x))
        self.assertTrue(equivalent(z[2].u, za[0, 2].u))
        self.assertTrue(equivalent(z[3].x, za[1, 0].x))
        self.assertTrue(equivalent(z[3].u, za[1, 0].u))
        self.assertTrue(equivalent(z[4].x, za[1, 1].x))
        self.assertTrue(equivalent(z[4].u, za[1, 1].u))
        self.assertTrue(equivalent(z[5].x, za[1, 2].x))
        self.assertTrue(equivalent(z[5].u, za[1, 2].u))

    def test_names(self):
        # the elements in array are not tuples
        with self.assertRaises(TypeError):
            _ = uarray([], names=['a', 'b', 'c'])
        with self.assertRaises(TypeError):
            _ = uarray(ureal(1, 1), names=['a', 'b', 'c'])
        with self.assertRaises(TypeError):
            _ = uarray([ureal(1, 1), ureal(2, 2), ureal(3, 3)], names=['a', 'b', 'c'])

        # len(array[0]) != len(names)
        with self.assertRaises(ValueError):
            _ = uarray((), names=['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            _ = uarray((ureal(1, 1),), names=['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            _ = uarray((ureal(1, 1), 2, 3, 4), names=['a', 'b', 'c'])

        # all items are of type UncertainReal
        a = [(ureal(1, 1), ureal(2, 2), ureal(3, 3)),
             (ureal(4, 4), ureal(5, 5), ureal(6, 6))]
        ua = uarray(a, names=['x', 'y', 'z'])

        self.assertTrue(ua.dtype.names == ('x', 'y', 'z'))
        for field in sorted(ua.dtype.fields):
            self.assertTrue(ua.dtype.fields[field][0].str == '|O')

        x = ua['x']
        self.assertTrue(x[0] is a[0][0])
        self.assertTrue(x[1] is a[1][0])

        y = ua['y']
        self.assertTrue(y[0] is a[0][1])
        self.assertTrue(y[1] is a[1][1])

        z = ua['z']
        self.assertTrue(z[0] is a[0][2])
        self.assertTrue(z[1] is a[1][2])

        # mix of UncertainReal, float, int and complex
        a = [(4j, ureal(1, 1), 7.0, ureal(2, 2), 8, ureal(3, 3)),
             (1-0.2j, ureal(4, 4), 9.0, ureal(5, 5), 10, ureal(6, 6))]
        ua = uarray(a, names=['u', 'v', 'w', 'x', 'y', 'z'], label='apple')

        self.assertTrue(ua.dtype.names == ('u', 'v', 'w', 'x', 'y', 'z'))
        for i, field in enumerate(sorted(ua.dtype.fields)):
            kind = ua.dtype.fields[field][0].kind
            if i == 0:
                self.assertTrue(kind == 'c')
            elif i == 2:
                self.assertTrue(kind == 'f')
            elif i == 4:
                self.assertTrue(kind == 'i')
            else:
                self.assertTrue(kind == 'O')

        u = ua['u']
        self.assertTrue(u[0] == a[0][0])
        self.assertTrue(u[1] == a[1][0])
        self.assertTrue(u.label == 'apple')

        v = ua['v']
        self.assertTrue(v[0] is a[0][1])
        self.assertTrue(v[1] is a[1][1])
        self.assertTrue(v.label == 'apple')

        w = ua['w']
        self.assertTrue(w[0] == a[0][2])
        self.assertTrue(w[1] == a[1][2])
        self.assertTrue(w.label == 'apple')

        x = ua['x']
        self.assertTrue(x[0] is a[0][3])
        self.assertTrue(x[1] is a[1][3])
        self.assertTrue(x.label == 'apple')

        y = ua['y']
        self.assertTrue(y[0] == a[0][4])
        self.assertTrue(y[1] == a[1][4])
        self.assertTrue(y.label == 'apple')

        z = ua['z']
        self.assertTrue(z[0] is a[0][5])
        self.assertTrue(z[1] is a[1][5])
        self.assertTrue(z.label == 'apple')

        # # dtype gets precedence over names
        # ua = uarray([(ureal(1, 1), ureal(2, 2))],
                    # names=['a', 'b'],
                    # dtype=([('x', np.object), ('y', np.object)]))
        # self.assertTrue(ua.dtype.names == ('x', 'y'))

    def test_example1(self):
        v = [ureal(4.937, 0.012), ureal(5.013, 0.008), ureal(4.986, 0.014)]
        i = [ureal(0.023, 0.003), ureal(0.019, 0.006), ureal(0.020, 0.004)]
        p = [ureal(1.0442, 2e-4), ureal(1.0438, 5e-4), ureal(1.0441, 3e-4)]
        va = uarray(v)
        ia = uarray(i)
        pa = uarray(p)

        r = [(v[idx] / i[idx]) * cos(p[idx]) for idx in range(len(v))]
        ra = (va / ia) * np.cos(pa)
        for idx in range(len(v)):
            self.assertTrue(equivalent(r[idx].x, ra[idx].x))
            self.assertTrue(equivalent(r[idx].u, ra[idx].u))

        self.assertTrue(equivalent(type_a.mean(r), function.mean( value( ra ) )))
        # self.assertTrue(equivalent(type_a.mean(r), value(ra).mean()))
        # self.assertTrue(equivalent(type_a.standard_deviation(r), np.std( value(ra), ddof=1)))

        weights = [(1.0/r[idx].u)**2 for idx in range(len(r))]
        wt_ave = sum(weights[idx] * r[idx].x for idx in range(len(r))) / sum(weights)
        
        # `np.average` does not support keywords when the array dtype is object.
        # This is awkward, because we want value() and variance() to return an 
        # array of objects.
        np_ave = np.average(  
            np.array(value(ra),dtype=float), 
            weights=1.0/ np.array(variance(ra),dtype=float) 
        )
        self.assertTrue( equivalent(wt_ave, np_ave) )

    def test_broadcasting(self):
        x = uarray([[ureal(1, 1)], [ureal(2, 2)], [ureal(3, 3)]])
        y = uarray([ureal(4, 4), ureal(5, 5), ureal(6, 6)])
        got = x + y

        expect = uarray(
            [[ureal(1, 1) + ureal(4, 4), ureal(1, 1) + ureal(5, 5), ureal(1, 1) + ureal(6, 6)],
             [ureal(2, 2) + ureal(4, 4), ureal(2, 2) + ureal(5, 5), ureal(2, 2) + ureal(6, 6)],
             [ureal(3, 3) + ureal(4, 4), ureal(3, 3) + ureal(5, 5), ureal(3, 3) + ureal(6, 6)]]
        )

        self.assertTrue(x.shape != y.shape)
        self.assertTrue(got.shape == expect.shape)
        for i in range(3):
            for j in range(3):
                self.assertTrue(equivalent(got[i, j].x, expect[i, j].x))
                self.assertTrue(equivalent(got[i, j].u, expect[i, j].u))

    #
    # The following is a list of all ufuncs
    #
    # <Math operations>
    # add - tested
    # subtract - tested
    # multiply - tested
    # divide - tested
    # logaddexp
    # logaddexp2
    # true_divide - tested (with divide)
    # floor_divide
    # negative - tested
    # positive - tested
    # power - tested
    # remainder
    # mod
    # fmod
    # divmod
    # absolute - tested
    # fabs
    # rint
    # sign
    # heaviside
    # conj - tested (as conjugate)
    # exp - tested
    # exp2
    # log - tested
    # log2
    # log10 - tested
    # expm1
    # log1p
    # sqrt - tested
    # square - tested (with mag_squared)
    # cbrt
    # reciprocal - tested
    #
    # <Trigonometric functions>
    # sin - tested
    # cos -tested
    # tan - tested
    # arcsin - tested
    # arccos - tested
    # arctan - tested
    # arctan2 - tested
    # hypot
    # sinh - tested
    # cosh - tested
    # tanh - tested
    # arcsinh - tested
    # arccosh - tested
    # arctanh - tested
    # deg2rad
    # rad2deg
    #
    # <Bit-twiddling functions>
    # bitwise_and
    # bitwise_or
    # bitwise_xor
    # invert
    # left_shift
    # right_shift
    #
    # <Comparison functions>
    # greater - tested
    # greater_equal - tested
    # less - tested
    # less_equal - tested
    # not_equal - tested
    # equal - tested
    # logical_and - tested
    # logical_or - tested
    # logical_xor - tested
    # logical_not - tested
    # maximum - tested
    # minimum - tested
    # fmax
    # fmin
    #
    # <Floating functions>
    # isfinite - tested
    # isinf - tested
    # isnan - tested
    # isnat
    # fabs
    # signbit
    # copysign
    # nextafter
    # spacing
    # modf
    # ldexp
    # frexp
    # fmod
    # floor
    # ceil
    # trunc


if __name__ == '__main__':
    unittest.main()  # Runs all test methods starting with 'test'

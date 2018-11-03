import unittest
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

import numpy as np

from GTC.core import (
    ureal,
    ucomplex,
    uarray,
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
    phase
)

from GTC.uncertain_array import UncertainArray

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

    def test_empty_array_like(self):
        for item in [list(), tuple()]:
            ua = uarray(item)
            self.assertTrue(isinstance(ua, UncertainArray))
            self.assertTrue(len(ua) == 0)
            # the following also checks that the shape and size attributes are available for an UncertainArray
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

    # # TODO: ignore these test for now. Must decide if they should pass of fail
    # def test_not_array_like(self):
    #     self.assertRaises(TypeError, uarray, None)
    #     self.assertRaises(TypeError, uarray, True)
    #     self.assertRaises(TypeError, uarray, -7)
    #     self.assertRaises(TypeError, uarray, 1.)
    #     self.assertRaises(TypeError, uarray, 6j)
    #     self.assertRaises(TypeError, uarray, 1-3j)
    #     self.assertRaises(TypeError, uarray, 'Hi!')
    #     self.assertRaises(TypeError, uarray, dict())
    #     self.assertRaises(TypeError, uarray, set())
    #     self.assertRaises(TypeError, uarray, ureal(23, 2))
    #     self.assertRaises(TypeError, uarray, ucomplex(1+2j, 0.1))

    # def test_not_filled_with_uncertain_numbers(self):
    #     self.assertRaises(TypeError, uarray, [None])
    #     self.assertRaises(TypeError, uarray, [True])
    #     self.assertRaises(TypeError, uarray, [-7])
    #     self.assertRaises(TypeError, uarray, [1.])
    #     self.assertRaises(TypeError, uarray, [6j])
    #     self.assertRaises(TypeError, uarray, [1-3j])
    #     self.assertRaises(TypeError, uarray, ['Hi!'])
    #     self.assertRaises(TypeError, uarray, [dict()])
    #     self.assertRaises(TypeError, uarray, [set()])
    #     self.assertRaises(TypeError, uarray, [[None]])
    #     self.assertRaises(TypeError, uarray, [[[1,2,3]]])

    def test_positive_unary(self):
        pos = +self.xa
        self.assertTrue(pos is not self.xa)
        for i in range(len(self.xa)):
            self.assertTrue(equivalent(self.x[i].x, pos[i].x))
            self.assertTrue(equivalent(self.x[i].u, pos[i].u))
            self.assertTrue(equivalent(self.x[i].df, pos[i].df))

    def test_negative_unary(self):
        neg = -self.xa
        self.assertTrue(neg is not self.xa)
        for i in range(len(self.xa)):
            self.assertTrue(equivalent(-self.x[i].x, neg[i].x))
            self.assertTrue(equivalent(self.x[i].u, neg[i].u))
            self.assertTrue(equivalent(self.x[i].df, neg[i].df))

    def test_add(self):
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
        xa = self.xa.reshape(3, 2)  # also checks that the reshape() method is available for an UncertainArray
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

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            _ = self.xa + self.ya[3:]

        with self.assertRaises(ValueError):
            _ = self.xa + [1, 2]

    def test_subtract(self):
        n = len(self.x)

        # x - y
        z = [x - y for x, y in izip(self.x, self.y)]
        za = self.xa - self.ya
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

        # y - x
        z = [y - x for y, x in izip(self.y, self.x)]
        za = self.ya - self.xa
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

    def test_multiply(self):
        n = len(self.x)

        # x * y
        z = [x * y for x, y in izip(self.x, self.y)]
        za = self.xa * self.ya
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

    def test_divide(self):
        n = len(self.x)

        # x / y
        z = [x / y for x, y in izip(self.x, self.y)]
        za = self.xa / self.ya
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))

    def test_comparisons(self):
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

    def test_x_u(self):
        # make sure that a uarray of size==1 is okay
        a = uarray(ureal(1.2, 0.3))
        self.assertTrue(equivalent(a.x, 1.2))
        self.assertTrue(equivalent(a.u, 0.3))

        # make sure that a uarray of size==1 is okay
        a = uarray(ucomplex(1.2+3j, (0.3, 0.1)))
        self.assertTrue(equivalent_complex(a.x, 1.2+3j))
        self.assertTrue(equivalent(a.u.real, 0.3))
        self.assertTrue(equivalent(a.u.imag, 0.1))

        a = uarray([ureal(1.2, 0.3), ureal(2.5, 0.8)])
        self.assertTrue(equivalent(a[0].x, 1.2))
        self.assertTrue(equivalent(a[0].u, 0.3))
        self.assertTrue(equivalent(a[1].x, 2.5))
        self.assertTrue(equivalent(a[1].u, 0.8))

        a = uarray([[ureal(1.2, 0.3), ureal(2.5, 0.8)], [ureal(-3.1, 1.1), ureal(0.3, 0.05)]])
        self.assertTrue(equivalent(a[0, 0].x, 1.2))
        self.assertTrue(equivalent(a[0, 0].u, 0.3))
        self.assertTrue(equivalent(a[0, 1].x, 2.5))
        self.assertTrue(equivalent(a[0, 1].u, 0.8))
        self.assertTrue(equivalent(a[1, 0].x, -3.1))
        self.assertTrue(equivalent(a[1, 0].u, 1.1))
        self.assertTrue(equivalent(a[1, 1].x, 0.3))
        self.assertTrue(equivalent(a[1, 1].u, 0.05))

    def test_real(self):
        # make sure that a uarray of size==1 is okay
        a = uarray(ureal(1.2, 0.3))
        self.assertTrue(equivalent(a.real.x, 1.2))
        self.assertTrue(equivalent(a.real.u, 0.3))

        # make sure that a uarray of size==1 is okay
        a = uarray(ucomplex(1.2+3j, (0.3, 0.1)))
        self.assertTrue(equivalent(a.real.x, 1.2))
        self.assertTrue(equivalent(a.real.u, 0.3))

        n = len(self.xc)
        z = [x.real for x in self.x]
        zc = [x.real for x in self.xc]

        # call np.real
        za = np.real(self.xa)
        zca = np.real(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

        # call UncertainArray.real
        za = self.xa.real
        zca = self.xca.real
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

    def test_imag(self):
        # make sure that a uarray of size==1 is okay
        a = uarray(ureal(1.2, 0.3))
        self.assertTrue(equivalent(a.imag.x, 0.0))
        self.assertTrue(equivalent(a.imag.u, 0.0))

        # make sure that a uarray of size==1 is okay
        a = uarray(ucomplex(1.2+3j, (0.3, 0.1)))
        self.assertTrue(equivalent(a.imag.x, 3))
        self.assertTrue(equivalent(a.imag.u, 0.1))

        n = len(self.xc)
        z = [x.imag for x in self.x]
        zc = [x.imag for x in self.xc]

        # call np.imag
        za = np.imag(self.xa)
        zca = np.imag(self.xca)
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

        # call UncertainArray.imag
        za = self.xa.imag
        zca = self.xca.imag
        for i in range(n):
            self.assertTrue(equivalent(z[i].x, za[i].x))
            self.assertTrue(equivalent(z[i].u, za[i].u))
            self.assertTrue(equivalent(zc[i].x, zca[i].x))
            self.assertTrue(equivalent(zc[i].u, zca[i].u))

    def test_conjugate(self):
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
        for item1, item2 in [(self.x, self.xa), (self.xc, self.xca)]:
            n = len(item1)
            z = [mag_squared(x) for x in item1]
            za = mag_squared(item2)
            for i in range(n):
                self.assertTrue(equivalent(z[i].x, za[i].x))
                self.assertTrue(equivalent(z[i].u, za[i].u))

            # reshape the array and make sure we get the same shape back
            m = mag_squared(item2.reshape(2, 3))
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

    def test_magnitude(self):
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
        # 1D array
        xlist = [ureal(i, i*0.1) for i in range(100)]
        xarray = uarray(xlist)
        self.assertTrue(xarray.shape == (100,))

        a = xarray.sum()
        b = 0
        for x in xlist:
            b += x
        self.assertTrue(equivalent(a.x, b.x))
        self.assertTrue(equivalent(a.u, b.u))

        # 3D array
        xlist = [[[ureal(i*j*k, i*j*k*0.1) for k in range(1, 5)] for j in range(7, 10)] for i in range(3, 9)]
        xarray = uarray(xlist)
        self.assertTrue(xarray.shape == (6, 3, 4))

        axis_none = 0.0
        axis_0 = [[0.0 for i in range(4)] for j in range(3)]
        axis_1 = [[0.0 for i in range(4)] for j in range(6)]
        axis_2 = [[0.0 for i in range(3)] for j in range(6)]
        for i in range(6):
            for j in range(3):
                for k in range(4):
                    value = xlist[i][j][k]
                    axis_none += value
                    axis_0[j][k] += value
                    axis_1[i][k] += value
                    axis_2[i][j] += value

        # axis=None
        a = xarray.sum()
        self.assertTrue(equivalent(a.x, axis_none.x))
        self.assertTrue(equivalent(a.u, axis_none.u))

        # axis=0
        a = xarray.sum(axis=0)
        m, n = len(axis_0), len(axis_0[0])
        self.assertTrue(a.shape == (m, n))
        for j in range(m):
            for k in range(n):
                self.assertTrue(equivalent(a[j, k].x, axis_0[j][k].x))
                self.assertTrue(equivalent(a[j, k].u, axis_0[j][k].u))

        # axis=1
        a = xarray.sum(axis=1)
        m, n = len(axis_1), len(axis_1[0])
        self.assertTrue(a.shape == (m, n))
        for i in range(m):
            for k in range(n):
                self.assertTrue(equivalent(a[i, k].x, axis_1[i][k].x))
                self.assertTrue(equivalent(a[i, k].u, axis_1[i][k].u))

        # axis=2
        a = xarray.sum(axis=2)
        m, n = len(axis_2), len(axis_2[0])
        self.assertTrue(a.shape == (m, n))
        for i in range(m):
            for j in range(n):
                self.assertTrue(equivalent(a[i, j].x, axis_2[i][j].x))
                self.assertTrue(equivalent(a[i, j].u, axis_2[i][j].u))

    def test_mean(self):
        for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            ave = 0.0
            for val in x:
                ave += val
            ave = ave/float(len(x))

            m = xa.mean()
            if j == 0:
                self.assertTrue(equivalent(m.x, ave.x))
                self.assertTrue(equivalent(m.u, ave.u))
            else:
                self.assertTrue(equivalent_complex(m.x, ave.x))
                self.assertTrue(equivalent(m.u.real, ave.u.real))
                self.assertTrue(equivalent(m.u.imag, ave.u.imag))

            xa = xa.reshape(2, 3)

            m = xa.mean()
            if j == 0:
                self.assertTrue(equivalent(m.x, ave.x))
                self.assertTrue(equivalent(m.u, ave.u))
            else:
                self.assertTrue(equivalent_complex(m.x, ave.x))
                self.assertTrue(equivalent(m.u.real, ave.u.real))
                self.assertTrue(equivalent(m.u.imag, ave.u.imag))

            m = xa.mean(axis=0)
            aves = [(x[0] + x[3])/2.0, (x[1] + x[4])/2.0, (x[2] + x[5])/2.0]
            for idx in range(3):
                if j == 0:
                    self.assertTrue(equivalent(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u, aves[idx].u))
                else:
                    self.assertTrue(equivalent_complex(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u.real, aves[idx].u.real))
                    self.assertTrue(equivalent(m[idx].u.imag, aves[idx].u.imag))

            m = xa.mean(axis=1)
            aves = [(x[0] + x[1] + x[2])/3.0, (x[3] + x[4] + x[5])/3.0]
            for idx in range(2):
                if j == 0:
                    self.assertTrue(equivalent(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u, aves[idx].u))
                else:
                    self.assertTrue(equivalent_complex(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u.real, aves[idx].u.real))
                    self.assertTrue(equivalent(m[idx].u.imag, aves[idx].u.imag))

    def test_std(self):
        for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            ave = 0.0
            for item in x:
                ave += item
            ave = ave/float(len(x))

            stdev = 0.0
            for item in [(val - ave)**2 for val in x]:
                stdev += item

            for ddof in range(5):
                a = sqrt(stdev/(float(len(x)-ddof)))
                b = xa.std(ddof=ddof)
                if j == 0:
                    self.assertTrue(equivalent(a.x, b.x))
                    self.assertTrue(equivalent(a.u, b.u))
                else:
                    self.assertTrue(equivalent_complex(a.x, b.x))
                    self.assertTrue(equivalent(a.u.real, b.u.real))
                    self.assertTrue(equivalent(a.u.imag, b.u.imag))

        x = [[ureal(i*j, i*j*0.05) for j in range(1, 11)] for i in range(5, 11)]
        xa = uarray(x)

        # axis-0 -> columns
        aves = [0.0 for _ in range(len(x[0]))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                aves[j] += x[i][j]
        n = float(len(x))
        aves = [a/n for a in aves]

        for ddof in range(5):
            stdevs = [0.0 for _ in range(len(aves))]
            for idx, ave in enumerate(aves):
                for i in range(len(x)):
                    stdevs[idx] += (x[i][idx] - ave) ** 2
                stdevs[idx] = sqrt(stdevs[idx] / (n - ddof))
            a = xa.std(axis=0, ddof=ddof)
            for i in range(len(stdevs)):
                self.assertTrue(equivalent(a[i].x, stdevs[i].x))
                self.assertTrue(equivalent(a[i].u, stdevs[i].u))

        # axis-1 -> rows
        aves = [0.0 for _ in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                aves[i] += x[i][j]
        n = float(len(x[0]))
        aves = [a/n for a in aves]

        for ddof in range(5):
            stdevs = [0.0 for _ in range(len(aves))]
            for idx, ave in enumerate(aves):
                for item in [(val - ave) ** 2 for val in x[idx]]:
                    stdevs[idx] += item
                stdevs[idx] = sqrt(stdevs[idx] / (n - ddof))
            a = xa.std(axis=1, ddof=ddof)
            for i in range(len(stdevs)):
                self.assertTrue(equivalent(a[i].x, stdevs[i].x))
                self.assertTrue(equivalent(a[i].u, stdevs[i].u))

    def test_var(self):
        for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            ave = 0.0
            for item in x:
                ave += item
            ave = ave/float(len(x))

            var = 0.0
            for item in [(val - ave)**2 for val in x]:
                var += item

            for ddof in range(5):
                a = var/(float(len(x)-ddof))
                b = xa.var(ddof=ddof)
                if j == 0:
                    self.assertTrue(equivalent(a.x, b.x))
                    self.assertTrue(equivalent(a.u, b.u))
                else:
                    self.assertTrue(equivalent_complex(a.x, b.x))
                    self.assertTrue(equivalent(a.u.real, b.u.real))
                    self.assertTrue(equivalent(a.u.imag, b.u.imag))

        x = [[ureal(i*j, i*j*0.05) for j in range(1, 11)] for i in range(5, 11)]
        xa = uarray(x)

        # axis-0 -> columns
        aves = [0.0 for _ in range(len(x[0]))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                aves[j] += x[i][j]
        n = float(len(x))
        aves = [a/n for a in aves]

        for ddof in range(5):
            vars = [0.0 for _ in range(len(aves))]
            for idx, ave in enumerate(aves):
                for i in range(len(x)):
                    vars[idx] += (x[i][idx] - ave) ** 2
                vars[idx] = vars[idx] / (n - ddof)
            a = xa.var(axis=0, ddof=ddof)
            for i in range(len(vars)):
                self.assertTrue(equivalent(a[i].x, vars[i].x))
                self.assertTrue(equivalent(a[i].u, vars[i].u))

        # axis-1 -> rows
        aves = [0.0 for _ in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                aves[i] += x[i][j]
        n = float(len(x[0]))
        aves = [a/n for a in aves]

        for ddof in range(5):
            vars = [0.0 for _ in range(len(aves))]
            for idx, ave in enumerate(aves):
                for item in [(val - ave) ** 2 for val in x[idx]]:
                    vars[idx] += item
                vars[idx] = vars[idx] / (n - ddof)
            a = xa.var(axis=1, ddof=ddof)
            for i in range(len(vars)):
                self.assertTrue(equivalent(a[i].x, vars[i].x))
                self.assertTrue(equivalent(a[i].u, vars[i].u))

    def test_max(self):
        a = self.xa.max()
        b = max(self.x)
        self.assertTrue(equivalent(a.x, b.x))
        self.assertTrue(equivalent(a.u, b.u))

        xa = self.xa.reshape(3, 2)

        a = xa.max()
        self.assertTrue(equivalent(a.x, b.x))
        self.assertTrue(equivalent(a.u, b.u))

        a = xa.max(axis=0)
        for i, b in enumerate([self.x[::2], self.x[1::2]]):
            b = max(b)
            self.assertTrue(equivalent(a[i].x, b.x))
            self.assertTrue(equivalent(a[i].u, b.u))

        a = xa.max(axis=1)
        for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
            b = max(b)
            self.assertTrue(equivalent(a[i].x, b.x))
            self.assertTrue(equivalent(a[i].u, b.u))

    def test_min(self):
        a = self.xa.min()
        b = min(self.x)
        self.assertTrue(equivalent(a.x, b.x))
        self.assertTrue(equivalent(a.u, b.u))

        xa = self.xa.reshape(3, 2)

        a = xa.min()
        self.assertTrue(equivalent(a.x, b.x))
        self.assertTrue(equivalent(a.u, b.u))

        a = xa.min(axis=0)
        for i, b in enumerate([self.x[::2], self.x[1::2]]):
            b = min(b)
            self.assertTrue(equivalent(a[i].x, b.x))
            self.assertTrue(equivalent(a[i].u, b.u))

        a = xa.min(axis=1)
        for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
            b = min(b)
            self.assertTrue(equivalent(a[i].x, b.x))
            self.assertTrue(equivalent(a[i].u, b.u))

    def test_argmax(self):
        a = self.xa.argmax()
        b = self.x.index(max(self.x))
        self.assertTrue(a == b)

        xa = self.xa.reshape(3, 2)

        a = xa.argmax()
        self.assertTrue(equivalent(a, b))

        a = xa.argmax(axis=0)
        for i, b in enumerate([self.x[::2], self.x[1::2]]):
            self.assertTrue(a[i] == b.index(max(b)))

        a = xa.argmax(axis=1)
        for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
            self.assertTrue(a[i] == b.index(max(b)))

    def test_argmin(self):
        a = self.xa.argmin()
        b = self.x.index(min(self.x))
        self.assertTrue(a == b)

        xa = self.xa.reshape(3, 2)

        a = xa.argmin()
        self.assertTrue(equivalent(a, b))

        a = xa.argmin(axis=0)
        for i, b in enumerate([self.x[::2], self.x[1::2]]):
            self.assertTrue(a[i] == b.index(min(b)))

        a = xa.argmin(axis=1)
        for i, b in enumerate([self.x[:2], self.x[2:4], self.x[4:]]):
            self.assertTrue(a[i] == b.index(min(b)))

    def test_transpose(self):
        a = uarray([ureal(i, i*0.1) for i in range(5*7)]).reshape(5, 7)
        for item in [a.T, a.transpose()]:
            for i in range(5):
                for j in range(7):
                    self.assertTrue(equivalent(a[i, j].x, item[j, i].x))
                    self.assertTrue(equivalent(a[i, j].u, item[j, i].u))

    def test_argsort(self):
        b = [self.x.index(a) for a in sorted(self.x)]
        a = self.xa.argsort()
        for i in range(len(a)):
            self.assertTrue(a[i] == b[i])

        xa = self.xa.reshape(2, 3)
        a = xa.argsort()
        b = [[item.index(x) for x in sorted(item)] for item in [self.x[:3], self.x[3:]]]
        for i in range(2):
            for j in range(3):
                self.assertTrue(a[i, j] == b[i][j])

        a = xa.argsort(axis=0)
        b = [[item.index(x) for x in sorted(item)] for item in [self.x[::3], self.x[1::3], self.x[2::3]]]
        for i in range(2):
            for j in range(3):
                self.assertTrue(a[i, j] == b[j][i])

        a = xa.argsort(axis=1)
        b = [[item.index(x) for x in sorted(item)] for item in [self.x[:3], self.x[3:]]]
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

    def test_size(self):

        xa = uarray([])
        self.assertTrue(xa.size == 0)

        x = [ureal(i, i*0.1) for i in range(5*9*3)]

        xa = uarray(x)
        self.assertTrue(xa.size == 5*9*3)

        xa = xa.reshape(5, 9, 3)
        self.assertTrue(xa.size == 5*9*3)


    #
    # The following are attributes/methods of an ndarray
    # that do not have a test method written yet
    #

    # def test_all(self):
    # def test_any(self):
    # def test_argpartition(self):
    # def test_astype(self):
    # def test_base(self):
    # def test_byteswap(self):
    # def test_choose(self):
    # def test_clip(self):
    # def test_compress(self):
    # def test_copy(self):
    # def test_ctypes(self):
    # def test_cumprod(self):
    # def test_cumsum(self):
    # def test_data(self):
    # def test_diagonal(self):
    # def test_dot(self):
    # def test_dtype(self):
    # def test_dump(self):
    # def test_dumps(self):
    # def test_fill(self):
    # def test_flags(self):
    # def test_flat(self):
    # def test_flatten(self):
    # def test_getfield(self):
    # def test_item(self):
    # def test_itemset(self):
    # def test_itemsize(self):
    # def test_nbytes(self):
    # def test_ndim(self):
    # def test_newbyteorder(self):
    # def test_nonzero(self):
    # def test_partition(self):
    # def test_prod(self):
    # def test_ptp(self):
    # def test_put(self):
    # def test_ravel(self):
    # def test_repeat(self):
    # def test_resize(self):
    # def test_round(self):
    # def test_searchsorted(self):
    # def test_setfield(self):
    # def test_setflags(self):
    # def test_sort(self):
    # def test_squeeze(self):
    # def test_strides(self):
    # def test_swapaxes(self):
    # def test_take(self):
    # def test_tobytes(self):
    # def test_tofile(self):
    # def test_tolist(self):
    # def test_tostring(self):
    # def test_trace(self):
    # def test_view(self):


if __name__ == '__main__':
    unittest.main()  # Runs all test methods starting with 'test'

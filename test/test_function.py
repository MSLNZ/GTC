import unittest
import math
try:
    xrange  # Python 2
except NameError:
    xrange = range

try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

import numpy as np

from GTC import function

from GTC.core import (
    ureal,
    ucomplex,
    value,
    uncertainty,
    variance,
    dof
)
from GTC.linear_algebra import matmul, uarray

from testing_tools import *

TOL = 1E-13 
  
#-----------------------------------------------------
def simple_sigma_abr(x,y,u_y=None):
    """
    The uncertainty in `a` and `b` as well as `r`
    
    eqn 15.2.9 from NR, p 663
    
    """
    if u_y is None:
        weights = False
        u_y = [1.0] * len(x)
    else:
        weights = True
        
    N = len(x)
    v = [ u_i*u_i for u_i in u_y ]
        
    S = sum( 1.0/v_i for v_i in v)
    S_x = sum( x_i/v_i for x_i,v_i in izip(x,v) )
    S_xx = sum( x_i**2/v_i for x_i,v_i in izip(x,v) )

    S_y = sum( y_i/v_i for y_i,v_i in izip(y,v) )

    k = S_x / S
    t = [ (x_i - k)/u_y_i for x_i,u_y_i in izip(x,u_y) ]

    S_tt = sum( t_i*t_i for t_i in t )

    b = sum( t_i*y_i/u_y_i/S_tt for t_i,y_i,u_y_i in izip(t,y,u_y) )
    a = (S_y - b*S_x)/S

    delta = S*S_xx - S_x**2
    sigma_a = math.sqrt( S_xx/delta ) 
    sigma_b = math.sqrt( S/delta )
    r = -S_x/(delta*sigma_a*sigma_b)

    if not weights:
        # Need chi-square to adjust the values
        f = lambda x_i,y_i,u_y_i: ((y_i - a - b*x_i))**2 
        chisq = sum( f(x_i,y_i,u_y_i) for x_i,y_i,u_y_i in izip(x,y,u_y) )
        data_u = math.sqrt( chisq/(N-2) )
        sigma_a *= data_u
        sigma_b *= data_u

    return sigma_a, sigma_b, r
#---------------------------------------------------------------
class UncertainLineTests(unittest.TestCase):

    def test_simple(self):
        """Trivial straight line
        
        uncertain number in y only with unity weight

        """
        x = [ float(x_i) for x_i in xrange(10) ]
        y = [ ureal(y_i,1) for y_i in x ]

        a,b = function.line_fit(x,y).a_b # default weight is unity

        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)

        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            [1.0 for x_i in x ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)

        equivalent(r,a.get_correlation(b),TOL)        

    def test_simple_errors_in_x_and_y(self):
        """Trivial straight line
        
        uncertain number in x and y with unity weights.
        We expect the uncertainty to increase by sqrt(2).

        """
        N = 10
        y = [ ureal(y_i,1) for y_i in xrange(N) ]
        x = [ ureal(x_i,1) for x_i in xrange(N) ]

        a,b = function.line_fit( x,y ).a_b
        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)

        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            [ math.sqrt(2) for x_i,y_i in zip(x,y) ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)

        equivalent(r,a.get_correlation(b),TOL)        

    # uncomment when line_fit_wls is available
        # a,b = function.line_fit_wls( x,y ).a_b
        # equivalent( value(a) ,0.0,TOL)
        # equivalent( value(b) ,1.0,TOL)

        # sig_a, sig_b, r = simple_sigma_abr(
            # [value(x_i) for x_i in x],
            # [value(y_i) for y_i in y],
            # [ math.sqrt(2) for x_i,y_i in zip(x,y) ]
        # )
        # equivalent(uncertainty(a),sig_a,TOL)
        # equivalent(uncertainty(b),sig_b,TOL)

        # equivalent(r,a.get_correlation(b),TOL)        

    # def test_simple_scaled(self):
        # """Straight line with non-trivial `a` and `b`

        # uncertain number in y only, but non-unity weight

        # """
        # N = 10
        # a0 =10
        # b0 = -3
        # u0 = .2
        # u = [u0] * N
        
        # x = [ float(x_i) for x_i in xrange(10) ]
        # y = [ ureal(b0*x_i + a0,u_i) for x_i,u_i in itertools.izip(x,u) ]

        # a,b = function.line_fit_wls(x,y).a_b

        # equivalent(a,a0,TOL)
        # equivalent(b,b0,TOL)

        # # The uncertainties in `a` and `b` should match
        # sig_a, sig_b, r = simple_sigma_abr(
            # x,
            # [float(y_i) for y_i in y],
            # [ y_i.u for y_i in y ]
        # )
        # equivalent(uncertainty(a),sig_a,TOL)
        # equivalent(uncertainty(b),sig_b,TOL)
        
        # # So should the correlation between `a` and `b`
        # equivalent(r,get_correlation(a,b),TOL)             

    # def test_simple_scaled_errors_in_x_and_y(self):
        # """Straight line with non-trivial `a` and `b`

        # uncertain number in x and y, with non-unity weight.

        # """
        # a0 =10
        # b0 = -3
        # u0 = .2

        # N = 10        
        # x = [ ureal(x_i,u0) for x_i in xrange(N) ]
        # y = [ ureal( b0 * value(x_i) + a0, 1) for x_i in x ]    # unity uncertainty

        # u = [uncertainty(y_i - b0*x_i) for x_i,y_i in itertools.izip(x,y)]

        # a,b = function.line_fit_wls(x,y).a_b

        # equivalent(a,a0,TOL)
        # equivalent(b,b0,TOL)

        # # The uncertainties in `a` and `b` should match
        # # when we feed in the `d` array
        # sig_a, sig_b, r = simple_sigma_abr(
            # [float(x_i) for x_i in x],
            # [float(y_i) for y_i in y],
            # u
        # )
        # equivalent(uncertainty(b),sig_b,TOL)
        # equivalent(uncertainty(a),sig_a,TOL)

        # # So should the correlation between `a` and `b`
        # equivalent(r,get_correlation(a,b),TOL)        

        # # We can calculate the uncertainty sequence directly
        # d = [ math.sqrt(1 + (b0 * u0)**2) for i in xrange(N) ]
        # sig_a, sig_b, r = simple_sigma_abr(
            # [float(x_i) for x_i in x],
            # [float(y_i) for y_i in y],
            # d
        # )
        # equivalent(uncertainty(b),sig_b,TOL)
        # equivalent(uncertainty(a),sig_a,TOL)

class DoFLineTest(unittest.TestCase):
    """
    Test that the dof calculation is working correctly.
    """
    def test_willink(self):
        """
        This case is taken from
        R Willink, Metrologia 45 (2008) 63-67

        The test ensures that the DoF calculation, when using
        the LS to interpolate to another data point, is correct.
        
        """
        nu = 2
        u = 0.1
        x = [ value(x_i) for x_i in [-1,0,1] ]
        y = [ ureal(y_i,u,df=nu) for y_i in x ]

        a,b = function.line_fit(x,y).a_b # default weight is unity

        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)
        
        equivalent(0.0,a.get_correlation(b),TOL)        

        # See below eqns (12) and (13), in section 2.1
        x_13 = 2.0/3.0
        y_13 = a + b*x_13
        equivalent(y_13.v,5.0*u**2/9.0,TOL)        
        equivalent(y_13.df,25.0*nu/17.0,TOL)        
          
#---------------------------------------------------------
class StdDataSets(object):
    """
    See section 5 in:
    'Design and us of reference data sets for testing scientific software'
    M. G. Cox and P. M. Harris, Analytica Chemica Acta 380 (1999) 339-351.
    
    """
    
    def __init__(self,mu,h,q,n):
        self._mu = mu
        self._h = h
        self._q = q
        self._n = n

    def seq(self,k=1):
        self._k = k
        
        N = self._n
        a = np.array( xrange(-N,N+1) ) * self._h
        q = self._q ** self._k
        
        return (self._mu + q) - a

    def mean(self):
        return self._mu + (self._q ** self._k)

    def std(self):
        N = self._n
        return self._h * math.sqrt((N + 0.5)*(N+1)/3.0)
        
#-----------------------------------------------------
class TestComplexToSeq(unittest.TestCase):

    """
    tests the conversion between complex and matrix representations
    """

    def test(self):
        z = 1 + 2j
        zm = function.complex_to_seq(z)
        self.assertTrue( equivalent_sequence((z.real,-z.imag,z.imag,z.real),zm ) )

        z = -1.1 - 4.5j
        zm = (z.real,-z.imag,z.imag,z.real)
        self.assertTrue( equivalent_complex(z,function.seq_to_complex(zm) ) )

        z = 1 
        zm = function.complex_to_seq(z)
        self.assertTrue( equivalent_sequence((z.real,-z.imag,z.imag,z.real),zm ) )

        z = 0 - 4.5j
        zm = (z.real,-z.imag,z.imag,z.real)
        self.assertTrue( equivalent_complex(z,function.seq_to_complex(zm) ) )

        self.assertRaises(RuntimeError,function.seq_to_complex,"hell")
        self.assertRaises(RuntimeError,function.seq_to_complex,(1,2,3,1))   # ill-conditioned
        self.assertRaises(RuntimeError,function.seq_to_complex,(1,2,2,1))   # ill-conditioned
        self.assertRaises(RuntimeError,function.seq_to_complex,(2,-2,2,1))  # ill-conditioned
        self.assertRaises(RuntimeError,function.seq_to_complex,(1,2,3,4,5))
        self.assertRaises(RuntimeError,function.seq_to_complex,(1,2,5))
        self.assertRaises(RuntimeError,function.seq_to_complex,z)

        zm = np.array(zm)   
        self.assertTrue( zm.shape == (4,) )
        self.assertRaises(RuntimeError,function.seq_to_complex,zm)

#-----------------------------------------------------
class TestFunctionFunctions(unittest.TestCase):
    """
    Test functions in `function` module:
        - mean

    """
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
        
    def test_mean(self):
        # generator argument (iterable)
        m = function.mean( i for i in range(10) )
        self.assertTrue( equivalent(m,4.5) )
        
        # sequence argument 
        m = function.mean( range(10) )
        self.assertTrue( equivalent(m,4.5) )
        
        # anything else makes no sense
        self.assertRaises(RuntimeError,function.mean,3)
 
    def test_sum_ndarray(self):
        # 1D array
        xlist = [ureal(i, i*0.1) for i in range(100)]
        xarray = uarray(xlist)
        self.assertTrue(xarray.shape == (100,))

        b = 0
        for x in xlist:
            b += x
        for a in [function.sum(xarray)]:
            self.assertTrue(equivalent(value(a), b.x))
            self.assertTrue(equivalent(uncertainty(a), b.u))

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
                    _value = xlist[i][j][k]
                    axis_none += _value
                    axis_0[j][k] += _value
                    axis_1[i][k] += _value
                    axis_2[i][j] += _value

        # axis=None
        for a in [function.sum(xarray)]:
            self.assertTrue(equivalent(value(a), axis_none.x))
            self.assertTrue(equivalent(uncertainty(a), axis_none.u))

        # axis=0
        m, n = len(axis_0), len(axis_0[0])
        for a in [function.sum(xarray, axis=0)]:
            self.assertTrue(a.shape == (m, n))
            for j in range(m):
                for k in range(n):
                    self.assertTrue(equivalent(a[j, k].x, axis_0[j][k].x))
                    self.assertTrue(equivalent(a[j, k].u, axis_0[j][k].u))

        # axis=1
        m, n = len(axis_1), len(axis_1[0])
        for a in [function.sum(xarray, axis=1)]:
            self.assertTrue(a.shape == (m, n))
            for i in range(m):
                for k in range(n):
                    self.assertTrue(equivalent(a[i, k].x, axis_1[i][k].x))
                    self.assertTrue(equivalent(a[i, k].u, axis_1[i][k].u))

        # axis=2
        m, n = len(axis_2), len(axis_2[0])
        for a in [function.sum(xarray, axis=2)]:
            self.assertTrue(a.shape == (m, n))
            for i in range(m):
                for j in range(n):
                    self.assertTrue(equivalent(a[i, j].x, axis_2[i][j].x))
                    self.assertTrue(equivalent(a[i, j].u, axis_2[i][j].u)) 
                    
    def test_mean_ndarray(self):
        for j, (x, xa) in enumerate([(self.x, self.xa), (self.xc, self.xca)]):
            ave = 0.0
            for val in x:
                ave += val
            ave = ave/float(len(x))

        for m in [function.mean(xa)]:
            if j == 0:
                self.assertTrue(equivalent(value(m), ave.x))
                self.assertTrue(equivalent(uncertainty(m), ave.u))
            else:
                self.assertTrue(equivalent_complex(value(m), ave.x))
                self.assertTrue(equivalent(uncertainty(m).real, ave.u.real))
                self.assertTrue(equivalent(uncertainty(m).imag, ave.u.imag))

        xa = xa.reshape(2, 3)

        for m in [function.mean(xa)]:
            if j == 0:
                self.assertTrue(equivalent(value(m), ave.x))
                self.assertTrue(equivalent(uncertainty(m), ave.u))
            else:
                self.assertTrue(equivalent_complex(value(m), ave.x))
                self.assertTrue(equivalent(uncertainty(m).real, ave.u.real))
                self.assertTrue(equivalent(uncertainty(m).imag, ave.u.imag))

        for m in [function.mean(xa, axis=0)]:
            aves = [(x[0] + x[3])/2.0, (x[1] + x[4])/2.0, (x[2] + x[5])/2.0]
            for idx in range(3):
                if j == 0:
                    self.assertTrue(equivalent(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u, aves[idx].u))
                else:
                    self.assertTrue(equivalent_complex(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u.real, aves[idx].u.real))
                    self.assertTrue(equivalent(m[idx].u.imag, aves[idx].u.imag))

        for m in [function.mean(xa, axis=1)]:
            aves = [(x[0] + x[1] + x[2])/3.0, (x[3] + x[4] + x[5])/3.0]
            for idx in range(2):
                if j == 0:
                    self.assertTrue(equivalent(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u, aves[idx].u))
                else:
                    self.assertTrue(equivalent_complex(m[idx].x, aves[idx].x))
                    self.assertTrue(equivalent(m[idx].u.real, aves[idx].u.real))
                    self.assertTrue(equivalent(m[idx].u.imag, aves[idx].u.imag))
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
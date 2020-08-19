import unittest
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

import numpy

from GTC import *

from GTC.lib import (
    UncertainReal, 
    UncertainComplex,
    set_correlation_real,
    real_ensemble,
    complex_ensemble,
    append_real_ensemble
)

from testing_tools import *

TOL = 1E-13 
            
#-----------------------------------------------------
class TestZeroTimesZero(unittest.TestCase):

    def test_errors(self):
        """
        Illegal conditions:
            - correlation between arguments
            - shared influences
            - unequal real-imaginary component uncertainties
        """
        x1 = ureal(0,1,independent=False)
        x2 = ureal(0,1,independent=False)
        x3 = ureal(0,1)
        
        set_correlation(.5,x1,x2)
        
        x4 = x1 * x2
        x5 = x2 * x3

        self.assertRaises(RuntimeError,fn.mul2,x1,x2)          # correlated
        self.assertRaises(RuntimeError,fn.mul2,x4,x5)          # shared influence

        # complex case
        u1 = 2
        u2 = 4
        u3 = 6
        z1 = ucomplex(0,u1,independent=False)
        z2 = ucomplex(0,u2,independent=False)
        z3 = ucomplex(0,u3)

        set_correlation(.5,z1.real,z2.imag)

        z4 = z1 * z2
        z5 = z2 * z3

        self.assertRaises(RuntimeError,fn.mul2,z1,z2)               # correlated
        self.assertRaises(RuntimeError,fn.mul2,z4,z5)               # shared influence
        self.assertRaises(RuntimeError,fn.mul2,ucomplex(0,[1,2]),z5)# different component uncertainties
        self.assertRaises(RuntimeError,fn.mul2,z5,ucomplex(0,[1,2]))# different component uncertainties
        self.assertRaises(RuntimeError,fn.mul2,ureal(0,1),ucomplex(0,[1,2]))# different component uncertainties
        self.assertRaises(RuntimeError,fn.mul2,ucomplex(0,[1,2]),ureal(0,1))# different component uncertainties

    def test1(self):
        """
        Simple binary and tertiary real
        
        """
        root_2 = math.sqrt(2)
        x1 = ureal(0,2)
        x2 = ureal(0,6)

        y = fn.mul2(x1,x2)

        self.assertTrue( equivalent(y.x,0) )

        uc = x1.u * x2.u
        self.assertTrue( equivalent(y.u,uc) )

        self.assertTrue(
            equivalent( component(y,x1), uc/root_2 )
        )

        x3 = ureal(0,4)
        uc *= x3.u
        y = fn.mul2(y,x3)

        self.assertTrue( equivalent(y.u,uc) )

        self.assertTrue(
            equivalent( component(y,x1), uc/root_2**2 )
        )
        self.assertTrue(
            equivalent( component(y,x3), uc/root_2 )
        )

        # a different order should not affect uc
        y = fn.mul2(x1,fn.mul2(x2,x3))
        self.assertTrue( equivalent(y.u,uc) )

    def test2(self):
        """
        Simple binary complex
        
        """
        u1 = 2
        z1 = ucomplex(0,u1)
        u2 = 3
        z2 = ucomplex(0,u2)
        
        z = fn.mul2(z1,z2)

        self.assertTrue( equivalent_complex( value(z),0) )

        # z.real = z1_r * z2_r - z1_i * z2_i
        self.assertTrue(
            equivalent( rp.u_component(z.real,z1.real), u1*u2/math.sqrt(2))
        )
        self.assertTrue(
            equivalent( rp.u_component(z.real,z1.imag), -u1*u2/math.sqrt(2))
        )
        self.assertTrue(
            equivalent( rp.u_component(z.real,z2.real), u1*u2/math.sqrt(2))
        )
        self.assertTrue(
            equivalent( rp.u_component(z.real,z2.imag), -u1*u2/math.sqrt(2))
        )

        # combined standard uncertainty is RSS of components
        uc = math.sqrt( (u1*u2)**2 + (u1*u2)**2)
        self.assertTrue(
            equivalent( uncertainty(z.real), uc)
        )

        # z_im = z1_i * z2_r + z1_r * z2_i
        self.assertTrue(
            equivalent( rp.u_component(z.imag,z1.real), u1*u2/math.sqrt(2))
        )
        self.assertTrue(
            equivalent( rp.u_component(z.imag,z1.imag), u1*u2/math.sqrt(2))
        )
        self.assertTrue(
            equivalent( rp.u_component(z.imag,z2.real), u1*u2/math.sqrt(2))
        )
        self.assertTrue(
            equivalent( rp.u_component(z.imag,z2.imag), u1*u2/math.sqrt(2))
        )

        # combined standard uncertainty is RSS of components
        uc = math.sqrt(
            (u1*u2)**2
        +   (u1*u2)**2
        )
        self.assertTrue(
            equivalent( uncertainty(z.imag), uc)
        )

    def test3(self):
        """
        Simple binary real and complex
        
        """
        root2 = math.sqrt(2)
        
        u1 = 3
        z1 = ucomplex(0,u1)
        u2 = 4
        x2 = ureal(0,u2)
        
        z = fn.mul2(z1,x2)

        self.assertTrue( equivalent_complex( value(z),0) )

        # z.real = z1_r * x 
        self.assertTrue(
            equivalent( rp.u_component(z.real,x2), u1*u2/root2)
        )
        self.assertTrue(
            equivalent( rp.u_component(z.real,z1.real), u1*u2/root2)
        )
        # z.imag = z1_i * x 
        self.assertTrue(
            equivalent( rp.u_component(z.imag,x2), u1*u2/root2)
        )
        self.assertTrue(
            equivalent( rp.u_component(z.imag,z1.imag), u1*u2/root2)
        )

        # combined standard uncertainty is RSS of components
        uc = u1*u2
        self.assertTrue(
            equivalent( uncertainty(z.real), uc)
        )
        uc = u1*u2
        self.assertTrue(
            equivalent( uncertainty(z.imag), uc)
        )

        # Changing the order will change nothing
        z = fn.mul2(x2,z1)

        self.assertTrue( equivalent_complex( value(z),0) )

        # z.real = z1_r * x 
        self.assertTrue(
            equivalent( rp.u_component(z.real,x2), u1*u2/root2)
        )
        # z.imag = z1_i * x 
        self.assertTrue(
            equivalent( rp.u_component(z.imag,x2), u1*u2/root2)
        )

        # combined standard uncertainty is RSS of components
        uc = u1*u2
        self.assertTrue(
            equivalent( uncertainty(z.real), uc)
        )
        uc = u1*u2
        self.assertTrue(
            equivalent( uncertainty(z.imag), uc)
        )

    def test4(self):
        """
        Triple real
        
        """
        root2 = math.sqrt(2)
        
        u = [2,3,4]
        x1 = ureal(0,u[0])
        x2 = ureal(0,u[1])
        x3 = ureal(0,u[2])

        y1 = fn.mul2( fn.mul2(x1,x2),x3)

        self.assertTrue( equivalent( value(y1),0. ) )

        uc = 1
        for u_i in u: uc *= u_i

        self.assertTrue( equivalent(y1.u,uc) )

        self.assertTrue(
            equivalent( rp.u_component(y1,x1), uc/root2**2)
        )
        self.assertTrue(
            equivalent( rp.u_component(y1,x2), uc/root2**2)
        )
        self.assertTrue(
            equivalent( rp.u_component(y1,x3), uc/root2)
        )

        # Do in a different order
        y2 = fn.mul2( x1,fn.mul2(x2,x3))

        self.assertTrue( equivalent( value(y2),0.) )

        self.assertTrue( equivalent(y2.u,uc) )

        self.assertTrue(
            equivalent( rp.u_component(y2,x1), uc/root2)
        )
        self.assertTrue(
            equivalent( rp.u_component(y2,x2), uc/root2**2)
        )
        self.assertTrue(
            equivalent( rp.u_component(y2,x3), uc/root2**2)
        )
       
    def test5(self):
        """
        Triple complex

        I have not completely checked all the components
        because the algorithm doesn't require it.
        
        """
        root2 = math.sqrt(2)
        
        u1 = 2 
        u2 = 3 
        u3 = 4 
        
        z1 = ucomplex(0,u1)
        z2 = ucomplex(0,u2)
        z3 = ucomplex(0,u3)

        zz = fn.mul2(z1,fn.mul2(z2,z3))
        zzz = fn.mul2(fn.mul2(z1,z2),z3)
        zzzz = fn.mul2(fn.mul2(z1,z3),z2)

        self.assertTrue( equivalent_sequence(zz.u,zzz.u) )
        self.assertTrue( equivalent_sequence(zzz.u,zzzz.u) )

        uc = u1 * u2 * u3
        # u = sqrt(2)*(sqrt(2)*u1*u2)*u3
        self.assertTrue( equivalent(zz.u[0], 2*uc) )
        self.assertTrue( equivalent(zz.u[1], 2*uc) )

        # The working for these results is in the GTC folder
        self.assertTrue(
            equivalent( rp.u_component(zzz.real,z1.real), 0)
        )
        self.assertTrue(
            equivalent( rp.u_component(zzz.real,z1.imag), -uc)
        )
        self.assertTrue(
            equivalent( rp.u_component(zzz.real,z3.real), uc)
        )
        self.assertTrue(
            equivalent( rp.u_component(zzz.real,z3.imag), -uc)
        )

    def test_not_zero(self):
        """
        mul2 can evaluate products that are not zero too
        
        """
        x = ureal(1.0, 3.0)
        y = ureal(1.5, 2.0)
        z = fn.mul2(x,y)

        a = x.x
        b = y.x
        ua = x.u
        ub = y.u

        # The second order term is the last one
        variance = (a*ub)**2 + (b*ua)**2 + (ua*ub)**2
        uab = math.sqrt(variance)

        self.assertTrue( equivalent( value( z ),a*b ) )
        self.assertTrue( equivalent( uncertainty( z ),uab ) )
        # self.assertTrue( equivalent( rp.sensitivity(z,x),b) )
        # self.assertTrue( equivalent( rp.sensitivity(z,y),a) )
    
  
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
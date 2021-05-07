import unittest
import math

TOL = 1E-13
DIGITS = 13

from GTC import *
from testing_tools import *

implicit = fn.implicit

#-----------------------------------------------------
class TestImplicit(unittest.TestCase):
    """
    The implicit(fn,x_min,x_max,epsilon=1E-10) function returns an UncertainReal
    number 'x' that satisfies abs( fn(x) ) < epsilon.

    A RuntimeError will be raised if the root search does not converge.
    
    An exception will be raised is:
        * fn(x) does not return an UncertainReal when `x` is UncertainReal
        * [x_min,x_max] does not appear to include a zero (this is done by
            checking that fn(x_max) * fn(x_min) < 0)
        * x_min < x_max
    
    """
    def test_preconditions(self):
        fn = lambda x : x + ureal(0,1)
        self.assertRaises(RuntimeError,implicit,fn,10,1)
        self.assertRaises(RuntimeError,implicit,fn,10,20)
        
        # This will convert an uncertain number `x` into a float.
        fn = lambda x : value(x)
        self.assertRaises(AssertionError,implicit,fn,-1,1)

    def test_root_find(self):
        # solve 0 = cos(x), this has no uncertainty!
        fn = lambda x: cos(x)   
        x = implicit(fn,0,math.pi)
        self.assertTrue( equivalent(value(x), math.pi/2.0 ) )
        self.assertTrue( equivalent( value(fn(x)),0 ) )
        self.assertTrue( equivalent( rp.u_component(x,x),0.0 ) )

        # solve 0 = (x-x1) - y
        y = ureal(0.5,.1)
        x1 = ureal(1,1)
        fn = lambda x: (x-x1) - y
        x = implicit(fn,0.5,2.0)
        self.assertTrue( equivalent( value(x), 1.5 ) )
        self.assertTrue( equivalent( value(fn(x)),0 ) )
        self.assertTrue( equivalent( rp.u_component(x,x1),1.0 ) )
        self.assertTrue( equivalent( rp.u_component(x,y),.1 ) )
        self.assertTrue( equivalent( uncertainty(x), math.sqrt(variance(y)+variance(x1))) )
  
#===============================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
    
    
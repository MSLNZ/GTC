import unittest
import math
try:
    xrange  # Python 2
except NameError:
    xrange = range

import numpy

from GTC import function

from testing_tools import *

TOL = 1E-13 
                
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
        a = numpy.array( xrange(-N,N+1) ) * self._h
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

        zm = numpy.array(zm)   
        self.assertTrue( zm.shape == (4,) )
        self.assertRaises(RuntimeError,function.seq_to_complex,zm)

#-----------------------------------------------------
class TestFunctionFunctions(unittest.TestCase):
    """
    Test functions in `function` module:
        - mean

    """
    def test_mean(self):
        # generator argument (iterable)
        m = function.mean( i for i in range(10) )
        self.assertTrue( equivalent(m,4.5) )
        
        # sequence argument 
        m = function.mean( range(10) )
        self.assertTrue( equivalent(m,4.5) )
        
        # anything else makes no sense
        self.assertRaises(RuntimeError,function.mean,3)
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
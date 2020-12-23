import unittest
import sys
import math
import cmath
import numpy
import operator
import collections

try:
    xrange  # Python 2
except NameError:
    xrange = range

TOL = 1E-13
DIGITS = 13

from GTC import *

from testing_tools import *

#-----------------------------------------------------
class TestFFT(unittest.TestCase):
    
    """
    These tests use numpy as validation but note there 
    is a different sign convention in the TF definition.
    """
    
    def test_dc(self):
  
        N = 16
        
        d = numpy.ones(N,dtype='cfloat')        
        ft.fft(d)
        self.assertTrue( equivalent_complex(d[0],N) )

        d = numpy.ones(N,dtype='cfloat')        
        d = numpy.fft.fft(d)
        self.assertTrue( equivalent_complex(d[0],N) )

        d = numpy.ones(N,dtype='cfloat')
        ft.ifft(d)
        self.assertTrue( equivalent_complex(d[0],1) )

        d = numpy.ones(N,dtype='cfloat')
        d = numpy.fft.ifft(d)
        self.assertTrue( equivalent_complex(d[0],1) )
        
    def test_basis(self):
 
        N = 16
        d = numpy.zeros(N,dtype='cfloat')

        for k in xrange(N):
            
            for i in range(N):
                d[i] = complex( math.cos(2*math.pi*i*k/N),-math.sin(2*math.pi*i*k/N) )
            ft.fft(d)
            self.assertTrue( equivalent_complex(d[k],N) )

            # Reverse transform undoes FT 
            ft.ifft(d)
            for i in range(N):
                d_i = complex( math.cos(2*math.pi*i*k/N), -math.sin(2*math.pi*i*k/N) )
                self.assertTrue( equivalent_complex(d[i],d_i) )

            # Check with numpy implementation. Note that numpy adopts the 
            # engineering form, so the sign of the exponent is opposite
            for i in range(N):
                d[i] = complex( math.cos(2*math.pi*i*k/N),math.sin(2*math.pi*i*k/N) )
            d = numpy.fft.fft(d)
            self.assertTrue( equivalent_complex(d[k],N) )
            
            # Reverse transform undoes FT 
            d = numpy.fft.ifft(d)
            for i in range(N):
                d_i = complex( math.cos(2*math.pi*i*k/N), math.sin(2*math.pi*i*k/N) )
                self.assertTrue( equivalent_complex(d[i],d_i) )
            
#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
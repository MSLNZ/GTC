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
        d = ft.fft(d)
        self.assertTrue( equivalent_complex(d[0],N) )

        d = numpy.ones(N,dtype='cfloat')        
        d = numpy.fft.fft(d)
        self.assertTrue( equivalent_complex(d[0],N) )

        d = numpy.ones(N,dtype='cfloat')
        d = ft.ifft(d)
        self.assertTrue( equivalent_complex(d[0],1) )

        d = numpy.ones(N,dtype='cfloat')
        d = numpy.fft.ifft(d)
        self.assertTrue( equivalent_complex(d[0],1) )
        
    def test_basis(self):
 
        N = 16
        d = numpy.empty(N,dtype='cfloat')

        for k in xrange(N):
            
            for i in xrange(N):
                d[i] = complex( math.cos(2*math.pi*i*k/N),-math.sin(2*math.pi*i*k/N) )
            d = ft.fft(d)
            self.assertTrue( equivalent_complex(d[k],N) )

            # Reverse transform undoes FT 
            d = ft.ifft(d)
            for i in xrange(N):
                d_i = complex( math.cos(2*math.pi*i*k/N), -math.sin(2*math.pi*i*k/N) )
                self.assertTrue( equivalent_complex(d[i],d_i) )

            # Check with numpy implementation. Note that numpy adopts the 
            # engineering form, so the sign of the exponent is opposite
            for i in xrange(N):
                d[i] = complex( math.cos(2*math.pi*i*k/N),math.sin(2*math.pi*i*k/N) )
            d = numpy.fft.fft(d)
            self.assertTrue( equivalent_complex(d[k],N) )
            
            # Reverse transform undoes FT 
            d = numpy.fft.ifft(d)
            for i in xrange(N):
                d_i = complex( math.cos(2*math.pi*i*k/N), math.sin(2*math.pi*i*k/N) )
                self.assertTrue( equivalent_complex(d[i],d_i) )
  
#-----------------------------------------------------
class Test_FFT(unittest.TestCase):
    
    """
    Test the _fft routine. This is intended only 
    for internal use. It uses arrays of float rather 
    than complex. It remains to be seen whether this 
    has any numerical advantage in some applications. 
    """
    
    def test_dc(self):
  
        N = 32
        
        d = numpy.ones(N,dtype=float)        
        d = ft._fft(d)
        self.assertTrue( equivalent(d[0],N/2 ) )

        d = numpy.ones(N>>1,dtype=float)        
        d = numpy.fft.fft(d)
        self.assertTrue( equivalent_complex(d[0],N/2) )
        
    def test_basis(self):
 
        n = 32
        N = n >> 1
        d = numpy.empty(n,dtype='float')

        for k in xrange(N):
            
            for i in xrange(N):
                d[2*i] = math.cos(2*math.pi*i*k/N)
                d[2*i+1] = -math.sin(2*math.pi*i*k/N)    
            d = ft._fft(d)
            
            self.assertTrue( equivalent(d[2*k],N) )
            self.assertTrue( equivalent(d[2*k+1],0.0) )

            # Reverse transform undoes FT 
            d = ft._fft(d,True)
            for i in xrange(N):
                dr = math.cos(2*math.pi*i*k/N) * N
                di = -math.sin(2*math.pi*i*k/N) * N   
                self.assertTrue( equivalent(d[2*i],dr) )
                self.assertTrue( equivalent(d[2*i+1],di) )

#-----------------------------------------------------
class Test_REALFT(unittest.TestCase):
    
    """
    Test the _realft routine. This is intended only 
    for internal use. It uses arrays of float rather 
    than complex.  
    """
    
    def test_dc(self):
  
        N = 16
        
        d = numpy.ones(N,dtype=float)        
        d = ft._realft(d)
        self.assertTrue( equivalent(d[0],N ) )

        d = numpy.ones(N,dtype=float)        
        d = numpy.fft.rfft(d)
        self.assertTrue( equivalent_complex(d[0],N) )
        
    def test_basis(self):
 
        N = 16
        d = numpy.empty(N,dtype='float')

        for k in xrange(1,N<<1):
            
            for i in xrange(N):
                d[i] = math.cos(2*math.pi*i*k/N)
                
            d = ft._realft(d)
                        
            if 2*k < N:
                self.assertTrue( equivalent(d[2*k],N/2) )
                self.assertTrue( equivalent(d[2*k+1],0.0) )
            elif 2*k == N:
                self.assertTrue( equivalent(d[1],N) )
                
            # Reverse transform undoes the forward transform 
            d = ft._realft(d,True)
            for i in range(N):
                dr = math.cos(2*math.pi*i*k/N) 
                self.assertTrue( equivalent(d[i]*2/N,dr) )
  
#-----------------------------------------------------
class Test_TWOFT(unittest.TestCase):
    
    """
    Test the _twoft routine. This is intended only 
    for internal use. It uses arrays of float.  
    """  

    def test_dc(self):
  
        N = 16
        
        d1 = [1]*N
        d2 = [1]*N
        
        f1, f2 = ft._twoft(d1,d2)
        
        self.assertTrue( equivalent(f1[0],N ) )
        self.assertTrue( equivalent(f2[0],N ) )

    def test(self):
        N = 16
        k1 = 3
        k2 = 1

        d1 = [0]*N
        d2 = [0]*N
        
        for k1 in xrange(1,N//2):
            for k2 in xrange(1,N//2):
                for i in xrange(N):
                    d1[i] = math.cos(2*math.pi*i*k1/N)
                    d2[i] = -math.sin(2*math.pi*i*k2/N)
 
                f1, f2 = ft._twoft(d1,d2)  

                self.assertTrue( equivalent(f1[2*k1]*2/N,1) )                
                self.assertTrue( equivalent(f1[2*k1+1]*2/N,0) )                
                self.assertTrue( equivalent(f2[2*k2]*2/N,0) )                
                self.assertTrue( equivalent(f2[2*k2+1]*2/N,-1) )                
 
#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
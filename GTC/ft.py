from __future__ import division

import sys
import math

if (sys.version_info > (3, 0)):
    xrange = range

__all__ = (
    'fft', 
    'ifft',
)

# """
# This is based on the NR algorithm, but alternatives are out there.
# Faster FFT libraries may include: Intel's MKL library (C,C++,Fortran) 
# or MIT licensed FFTW (C, C++), or CenterSpaces' NMath library ...
# """

#----------------------------------------------------------------------------
def ifft(data):
    """
    Evaluate the inverse fast Fourier transform 
        
    ``data`` is an array of ``N`` complex values,
    where ``N`` must be a power of 2. 
    
    Note..

        ``data`` is modified in-place and also returned 


    """
    fft(data,True)
    
    N = len(data)
    for i,d_i in enumerate(data): 
        data[i] = d_i/N
        
    return data 
        
#----------------------------------------------------------------------------
def fft(data,inverse=False):
    """
    Evaluate in-place and return the fast Fourier transform of ``data`` 

    ``data`` is an array of ``N`` complex values,
    where ``N`` must be a power of 2. 
    
    Set ``inverse`` True to evaluate the inverse transform,
    but values must be re-scaled by ``N``.
    
    Note..

        ``data`` is modified by this function 
        
    Note..
    
        The 'forward' transform is defined here with a positive exponent, 
        whereas implementations used by engineers adopt 
        a negative exponent for the forward transform. 
    
    """
    isign = -1 if inverse else 1 
    N = len(data)   # Must be a power of 2

    n = N << 1      # times 2
      
    j = 0
    for i in xrange(n):
        if j > i: 
            data[i], data[j] = data[j], data[i]
        
        m = N >> 1  # divided by 2
        
        while m >= 2 and j+1>m: 
            j = j - m
            m = m >> 1 # divided by 2
          
        j = j + m 
               
    mmax = 2
    while n > mmax:
        # See: https://stackoverflow.com/questions/2220879/c-numerical-recipies-fft 
        theta_2 = isign*(math.pi/mmax) 
        wtemp = math.sin(theta_2)
        wpr = -2.0*wtemp*wtemp 
        wpi = math.sin(2.0*theta_2)
        wp = complex(wpr,wpi)     

        w = complex(1,0) 
        istep = mmax << 1    
        for m in xrange(1,mmax,2):
            for i in xrange(m,n+1,istep):
                j = i + mmax 
                
                # perhaps this can be re-factored?
                ii = i-1 >> 1 
                jj = j-1 >> 1 
                
                temp = w * data[jj]             
                data[jj] = data[ii] - temp 
                data[ii] = data[ii] + temp
                
            w += w*wp          
            
        mmax = istep
        
    return data 
 
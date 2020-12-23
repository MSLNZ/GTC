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
# This is the NR algorithm, but perhaps it is not so fast. Alternatives are out there:
# Fast FFT libraries: Intel's MKL library (C,C++,Fortran) or MIT licensed FFTW (C, C++), 
# or CenterSpaces' NMath library
# """
#----------------------------------------------------------------------------
def ifft(data):
    """
    Evaluate the inverse fast Fourier transform 
        
    """
    fft(data,True)
    
    N = len(data)
    for i,d_i in enumerate(data): 
        data[i] = d_i/N
        
#----------------------------------------------------------------------------
def fft(data,inverse=False):
    """
    Evaluate the fast Fourier transform of ``data`` in-place 

    ``data`` is treated as an array of ``N`` complex values,
    where ``N`` must be a power of 2. 
    
    ``inverse`` may be set True to evaluate the inverse transform,
    but the values must be re-scaled by ``N``.
    
    Note..
    
        The 'forward' transform here uses a positive exponent 
        whereas other implementations used by engineers often adopt 
        a negative exponent for the forward transform. 
    
    """
    N = len(data)   # Must be a power of 2
    n = N << 1      # times 2
    
    isign = -1 if inverse else 1 
    
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
                
                ii = i-1 >> 1 
                jj = j-1 >> 1 
                # print("i,j",ii,jj)
                
                # tempr = wr*data[j].real - wi*data[j].imag 
                # tempi = I*(wr*data[j].imag + wi*data[j].real) 
                temp = w * data[jj]
                
                # data[j] = data[i] - (tempr + tempi)
                # data[i] = data[i] + (tempr + tempi)
                data[jj] = data[ii] - temp 
                data[ii] = data[ii] + temp
                
            # wtemp = wr
            # wr = wr*wpr - wi*wpi + wr 
            # wi = wi*wpr + wtemp*wpi + wi
            w += w*wp          
            
        mmax = istep
        # print("mmax=",mmax)
 
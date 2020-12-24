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
# See: https://stackoverflow.com/questions/2220879/c-numerical-recipies-fft 
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
    
        This transform is defined with a positive exponent, 
        whereas engineering implementations use a negative exponent
        (i.e., the inverse transform).
        
    **Example**::
    
        >>> N = 16
        >>> k = 1 
        >>> d = [0] * N

        >>> for i in range(N):
        ...     d[i] = cmath.exp( -2j*math.pi*i*k/N )  
        ...
        
        >>> d = ft.fft(d)
        >>> d[1]
        (16+4.36419692839477e-15j)
        
    """
    isign = -1 if inverse else 1
    
    N = len(data)   # Must be a power of 2 -- not checked

    j = 0
    for i in xrange(2*N):
        if j > i: 
            data[i], data[j] = data[j], data[i]
        
        m = N >> 1  # divide by 2   
        while m >= 2 and j+1>m: 
            j = j - m
            m = m >> 1 
          
        j = j + m 
   
    mmax = 1
    while N > mmax:
        istep = mmax << 1 # times 2

        theta_2 = isign*(math.pi/(2*mmax)) 
        wtemp = math.sin(theta_2)        
        wp = complex(
            -2.0*wtemp*wtemp,
            math.sin(2.0*theta_2)
        )     

        w = complex(1,0)        
        for m in xrange(mmax):
            for i in xrange(m,N,istep):
                j = i + mmax 
                
                temp = w * data[j]             
                data[j] = data[i] - temp 
                data[i] = data[i] + temp
                                
            w += w*wp          
            
        mmax = istep
        
    return data 
 
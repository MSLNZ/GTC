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
# This is based on the NR FFT algorithm, but alternatives are out there.
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

    ``data`` is an array of ``N`` complex values, where ``N`` must be 
    a power of 2. 
    
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
        >>> "{0.real:.3f}+j{0.imag:.3f}".format( d[1] )
        '16.000+j0.000'
        
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
 
#----------------------------------------------------------------------------
def _fft(data,inverse=False):
    """
    ``data`` is an array of ``n`` real values,
    where ``n`` must be a power of 2. 
    
    Set ``inverse`` True to evaluate the inverse transform,
    but values must be re-scaled by ``N = n/2``.
    
    Note..

        ``data`` is modified by this function 

    """
    isign = -1 if inverse else 1
    
    n = len(data)
    N = n / 2

    j = 1
    for i in xrange(1,n,2):
        if j>i: 
            data[j-1], data[i-1] = data[i-1], data[j-1]
            data[j], data[i] = data[i], data[j]
            
        m = n >> 1 
        while m >= 2 and j > m:
            j = j - m
            m = m >> 1 
        j = j + m 

    mmax = 2
    while n > mmax:
        istep = mmax << 1
        
        theta_2 = isign*math.pi/mmax
        wtemp = math.sin(theta_2)
        wpr = -2.0*wtemp*wtemp 
        wpi = math.sin(2*theta_2) 
        
        wr = 1.0
        wi = 0.0 
        for m in xrange(1,mmax,2):
            for i in xrange(m,n+1,istep):
                j = i + mmax
                
                tempr = wr*data[j-1] - wi*data[j]
                tempi = wr*data[j] + wi*data[j-1] 
                
                data[j-1] = data[i-1] - tempr 
                data[j] = data[i] - tempi 
                
                data[i-1] = data[i-1] + tempr 
                data[i] = data[i] + tempi 
                
            wtemp = wr 
            wr = wr + wr*wpr - wi*wpi
            wi = wi + wi*wpr + wtemp*wpi 
            
        mmax = istep 
            
    return data      

#----------------------------------------------------------------------------
def _realft(data,inverse=False):
    """
    Evaluate the positive frequency half of the spectrum of ``data`` 
    
    ``data`` is an array of ``N`` real values,
    where ``N`` must be a power of 2. 
    
    ``data[0]`` is returned with the zero frequency term
    ``data[1]`` is returned with the Nyquist frequency term (pure real)
    
    Set ``inverse`` True to evaluate the inverse transform,
    but values must be re-scaled by ``N/2``.
    
    Note..

        ``data`` is modified by this function 

    """
    N = len(data)
    
    theta_2 = math.pi/N  
    c1 = 0.5 
    
    if inverse:
        c2 = 0.5 
        theta_2 = -theta_2 
    else:
        c2 = -0.5
        data = _fft(data,inverse)

    wtemp = math.sin(theta_2)
    wpr = -2.0*wtemp*wtemp 
    wpi = math.sin(2*theta_2) 
    
    wr = 1.0 + wpr 
    wi = wpi  
    np3 = N + 2
    
    for i in xrange(2,(N>>2) + 1):
        
        i1 = (i - 1) + (i - 1)
        i2 = i1 + 1
        i3 = np3 - i2 - 1
        i4 = i3 + 1
        
        h1r = c1*( data[i1] + data[i3] )
        h1i = c1*( data[i2] - data[i4] )
        
        h2r = -c2*( data[i2] + data[i4] )
        h2i = c2*( data[i1] - data [i3] )
        
        data[i1] = h1r + wr*h2r - wi*h2i 
        data[i2] = h1i + wr*h2i + wi*h2r
        data[i3] = h1r - wr*h2r + wi*h2i  
        data[i4] = -h1i + wr*h2i + wi*h2r
        
        wtemp = wr 
        wr = wr + wr*wpr - wi*wpi 
        wi = wi + wi*wpr + wtemp*wpi 
        
    if inverse:
        h1r = data[0]
        data[0] = c1*(h1r + data[1])
        data[1] = c1*(h1r - data[1])
        data = _fft(data,inverse)
        
    else:
        h1r = data[0]
        data[0] = h1r + data[1] 
        data[1] = h1r - data[1]
        
    return data 
    
#----------------------------------------------------------------------------
def _twoft(data1,data2):
    """
    Given real arrays ``data1`` and ``data2`` evaluate the corresponding 
    Fourier transforms and return these in a pair of complex arrays.
    
    Note..

        ``data1`` and ``data2`` are not modified by calling this function 
    
    """
    N = len(data1)
    assert N == len(data2)
    
    fft1 = [0]*2*N
    fft2 = [0]*2*N
    
    nn2 = N + N + 2
    nn3 = 1 + nn2 
    for j in xrange(1,N+1):
        jj = 2*j 
        fft1[jj-1-1] = data1[j-1]
        fft1[jj-1] = data2[j-1]
        
    fft1 = _fft(fft1)  
    
    fft2[0] = fft1[1]
    fft1[1] = fft2[1] = 0.0 
     
    for j in xrange(3,N+2,2):
        
        rep = 0.5*( fft1[j-1] + fft1[nn2-j-1] )
        rem = 0.5*( fft1[j-1] - fft1[nn2-j-1] )
        aip = 0.5*( fft1[j] + fft1[nn3-j-1] )
        aim = 0.5*( fft1[j] - fft1[nn3-j-1] )
        
        fft1[j-1] = rep 
        fft1[j] = aim 
        fft1[nn2-j-1] = rep 
        fft1[nn3-j-1] = -aim 
        
        fft2[j-1] = aip
        fft2[j] = -rem 
        fft2[nn2-j-1] = aip 
        fft2[nn3-j-1] = rem 
        
    return fft1, fft2
        
    
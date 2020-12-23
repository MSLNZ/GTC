from __future__ import division

import sys
import math

if (sys.version_info > (3, 0)):
    xrange = range

#----------------------------------------------------------------------------
def fft(data,isign=-1):
    """
    """
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
"""
A benchmark used in the past for uncertain complex number calculations

"""
from GTC import *
from GTC.vector import is_ordered

import math
import numpy
import time

import sys

def fn():
    
    if __debug__: print('debug on')

    lossless = False

    _t0 = time.time()

    _J_ = 1j

    # Physical Constants
    c0 = 299792458.     #(m/s)
    u0 = 4e-7*numpy.pi  #(Vs/Am)
    ur = 1.
    u  = u0*ur
    E0 = 1./(c0**2*u0)  #(As/Vm)
    Er = 1.000649
    E  = E0*Er
    sigma = 1.3e7       #(A/Vm)
    Zr = 50.0           #(Ohm)

    # Frequency List
    flist = 1e9 # numpy.arange(1e9 ,19e9) #(Hz)

    # Ideal 2-Port
    S11 = 0. #numpy.array([0.]*numpy.size(flist))
    S21 = 1. #numpy.array([1.]*numpy.size(flist))
    S12 = 1. #numpy.array([1.]*numpy.size(flist))
    S22 = 0. #numpy.array([0.]*numpy.size(flist))

    _t1 = time.time()

    # Airline Mechanical Data
    N  = 101 
    dl = 1e-3
    a  = numpy.array([3.040e-3]*N)
    b  = numpy.array([7.000e-3]*N)

    unc_i = numpy.zeros(N,object)   # N 0's with memory layout for an object
    unc_o = numpy.zeros(N,object)   # The initial values are over-written 
                                    # immediately below
    for i in range(N):
        name_i = 'unc_i[%s]' %i
        name_o = 'unc_o[%s]' %i
        unc_i[i] = ucomplex(0., [10e-6,0.0])
        unc_o[i] = ucomplex(0., [20e-6,0.0])
        
    a = a + unc_i   # A numpy array addition with N elements
    b = b + unc_o   

    _t2 = time.time()

    _show_info = True

    for i in range(N-1):
    
        # if _show_info:
            # _t2a = time.time()
            
        # Characterization of a single air line section
        _a = (a[i] + a[i+1])/2
        _b = (b[i] + b[i+1])/2        
        ba = _b/_a  
        ln_ba = log(ba)
        w  = 2*numpy.pi*flist
        if lossless:
            # Lossless
            R  = 0
            L  = u0/(2*numpy.pi)*ln_ba
            G  = 0
            C  = 2*numpy.pi*E/ln_ba
        else:
            # Lossy Case
            p  = 1./sigma
            ds = numpy.sqrt(2*p/(w*u))
            k  = w*numpy.sqrt(u*E)
            d0 = ds*(1 + ba)/(4*_b*ln_ba)
            F0 = (ba**2 - 1)/(2*ln_ba) - (ba*ln_ba)/(ba + 1) - 1./2*(ba + 1)
            C0 = 2*numpy.pi*E/ln_ba
            L0 = u*ln_ba/(2*numpy.pi)
            k2a2F0 = k**2*_a**2*F0
            R  = 2*w*L0*d0*(1 - k2a2F0/2)
            L  = L0*(1 + 2*d0*(1 - k2a2F0/2))
            G  = w*C0*d0*k2a2F0
            C  = C0*(1 + d0*k2a2F0)
            
        Z  = R + 1j*w*L
        Y  = G + 1j*w*C
        g  = sqrt(Z*Y)
        Z0 = sqrt(Z/Y)

        # S-Parameters of air line section
        _Ds  = 2*Z0*Zr*cosh(g*dl) + (Z0**2 + Zr**2)*sinh(g*dl)
        _S11 = (Z0**2 - Zr**2)*sinh(g*dl)/_Ds
        _S21 = 2*Z0*Zr/_Ds
        _S12 = _S21
        _S22 = _S11
        
        # Cascading
        # This was a bottleneck in earlier implementations
        _temp = (1 - S22*_S11)  
        S11 = S11 + S12*S21*_S11/_temp
        S21 = S21*_S21/_temp
        S12 = S12*_S12/_temp
        S22 = _S22 + S22*_S12*_S21/_temp

        # if _show_info:
            # _t2b = time.time()
            
    _t3 = time.time()

    vS11, cvS11 = value(S11), variance(S11)
    vS21, cvS21 = value(S21), variance(S21)
    vS12, cvS12 = value(S12), variance(S12)
    vS22, cvS22 = value(S22), variance(S22)

    _t4 = time.time()

    print('set up time: {}'.format(_t1 - _t0))
    print('initialisation time: {}'.format(_t2 - _t1))
    print('build-up time: {}'.format(_t3 - _t2))
    print('results time: {}'.format(_t4 - _t3))
    print('total time: {}'.format(_t4 - _t0))

#========================================================
if __name__ == '__main__':

    fn()    
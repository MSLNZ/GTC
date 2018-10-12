"""
Sample estimates
----------------
    *   :func:`estimate` returns an uncertain number defined from
        the statistics of a sample of data.
    *   :func:`multi_estimate_real` returns a sequence of related 
        uncertain real numbers defined from the multivariate statistics 
        calculated from a sample of data. 
    *   :func:`multi_estimate_complex` returns a sequence of related uncertain
        complex numbers defined from the multivariate statistics of a sample of data. 
    *   :func:`estimate_digitized` returns an uncertain number for 
        the mean of a sample of digitized data.
    *   :func:`mean` returns the mean of a sample of data.
    *   :func:`standard_uncertainty` evaluates the standard 
        uncertainty associated with the sample mean.
    *   :func:`standard_deviation` evaluates the standard 
        deviation of a sample of data.
    *   :func:`variance_covariance_complex` evaluates the variance
        and covariance associated with the mean real component 
        and mean imaginary component of the data.
      
.. note::

    Many functions in :mod:`type_a` treat the data as pure numbers. 
    Sequences of uncertain numbers can be passed to these 
    functions, but only the values of the uncertain numbers will be used.
    This allows type-B uncertainty components to be associated with
    observational data (e.g., the type-B uncertainty due to a systematic
    error) before a type-A analysis is performed, which is often
    convenient. 
    
Module contents
---------------

"""
from __future__ import division

import sys
import math
import numbers
import itertools

from GTC.lib_complex import UncertainComplex
from GTC.lib_real import UncertainReal
from GTC.lib_real import get_correlation_real
from GTC.context import _context 

from GTC.vector import merge_vectors
from GTC.nodes import Node

from GTC import inf

from GTC.named_tuples import (
    VarianceCovariance,
    StandardUncertainty,
    StandardDeviation,
    InterceptSlope
)

__all__ = (
    'estimate','multi_estimate_real','multi_estimate_complex',
    'estimate_digitized',
    'mean',
    'standard_deviation',
    'standard_uncertainty',
    'variance_covariance_complex',
    )

#-----------------------------------------------------------------------------------------
def estimate_digitized(seq,delta,label=None,truncate=False,context=_context):
    """
    Return an uncertain number for the mean of a sample of digitized data

    :arg seq: a sequence of real numbers or uncertain real numbers
    :arg delta: a real number for the digitization step size
    :arg label: a label for the returned uncertain number
    :arg truncate: if True, truncation is assumed

    When an instrument rounds, or truncates, readings to a 
    finite resolution ``delta``, the uncertainty in an estimate  
    of the mean of a sequence of readings depends on the amount  
    of scatter in the data and on the number of points in the sample.
    
    The argument ``truncate`` should be set ``True`` 
    if an instrument truncates readings instead of rounding them.
    
    See reference: R Willink, *Metrologia*, **44** (2007) 73-81

    **Examples**::
    
        # LSD = 0.0001, data varies between -0.0055 and -0.0057
        >>> seq = (-0.0056,-0.0055,-0.0056,-0.0056,-0.0056, 
        ...      -0.0057,-0.0057,-0.0056,-0.0056,-0.0057,-0.0057)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0056272727272727272874,1.9497827808661157478e-05,10)

        # LSD = 0.0001, data varies between -0.0056 and -0.0057
        >>> seq = (-0.0056,-0.0056,-0.0056,-0.0056,-0.0056,
        ... -0.0057,-0.0057,-0.0056,-0.0056,-0.0057,-0.0057)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0056363636363636364021,1.5212000482437778871e-05,10)

        # LSD = 0.0001, no spread in data values
        >>> seq = (-0.0056,-0.0056,-0.0056,-0.0056,-0.0056,
        ... -0.0056,-0.0056,-0.0056,-0.0056,-0.0056,-0.0056)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0055999999999999999431,2.8867513459481289171e-05,10)
        
        # LSD = 0.0001, no spread in data values, fewer points
        >>> seq = (-0.0056,-0.0056,-0.0056)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0055999999999999999431,3.2914029430219170322e-05,2)
        
    """
    N = len(seq)
    if N < 2:
        raise RuntimeError(
            "digitized data sequence must have more than one element"
        )

    try:
        seq = [ float(x_i) for x_i in seq ]
    except TypeError:
        seq = [ x_i.x for x_i in seq ]
    
    x_max = max(seq)
    x_min = min(seq)
    
    mean = math.fsum(seq)/N
        
    if x_max == x_min:
        # No scatter in the data
        if N == 2:
            root_c_12 = math.sqrt(6.4/12.0)
        elif N == 3:
            root_c_12 = math.sqrt(1.3/12.0)
        elif N >= 4:
            root_c_12 = math.sqrt(1.0/12.0)
        else:
            assert False,"should not occur"
            
        u = root_c_12*delta
    else:
        accum = lambda psum,x: psum + (x-mean)**2
        var = reduce(accum, seq, 0.0) / (N - 1)

        if abs(x_max - x_min - delta) < 10*sys.float_info.epsilon:
            # Scatter is LSD only
            x_mid = (x_max + x_min)/2.0
            u = math.sqrt(
                max(var/N,(x_mid - mean)**2/3.0)
            )
        else:
            u = math.sqrt(var/N)

    if truncate:
        mean += delta/2.0
        
    return context.elementary_real(mean,u,N-1,label,independent=True)
    
#-----------------------------------------------------------------------------------------
def estimate(seq,label=None,context=_context):
    """Obtain an uncertain number by type-A evaluation 

    :arg seq:   a sequence representing a sample of data
    :arg label: a label for the returned uncertain number
    
    :returns:   an uncertain real number, or an uncertain complex number
                
    The elements of ``seq`` may be real numbers, complex numbers, or
    uncertain real or complex numbers. Note that if uncertain numbers
    are used, only the value attribute is used.

    The sample mean is an estimate of the quantity of interest. 
    The uncertainty in this estimate is the standard deviation of
    the sample mean (or the sample covariance of the mean, 
    for the complex case).    
    
    Returns an uncertain real number when the mean of ``seq`` is real, 
    or an uncertain complex number when the mean is complex.

    **Examples**::

        >>> data = range(15)
        >>> type_a.estimate(data)
        ureal(7,1.15470053837925,14)
        
        >>> data = [(0.91518731126816899+1.5213442955575518j),
        ... (0.96572684493613492-0.18547192979059401j),
        ... (0.23216598132006649+1.6951311687588568j),
        ... (2.1642786101267397+2.2024333895672563j),
        ... (1.1812532664590505+0.59062101107787357j),
        ... (1.2259264339405165+1.1499373179910186j),
        ... (-0.99422341300318684+1.7359338393131392j),
        ... (1.2122867690240853+0.32535154897909946j),
        ... (2.0122536479379196-0.23283009302603963j),
        ... (1.6770229536619197+0.77195994890476838j)]

        >>> type_a.estimate(data)
        ucomplex(
            (1.059187840567141+0.9574410497332931j), 
            u=[0.2888166531024181,0.2655555630050262], 
            r=-0.314, 
            df=9
        )

    """
    df = len(seq)-1
    mu = mean(seq)
    
    if isinstance(mu,complex):
        u,r = standard_uncertainty(seq,mu)
        return context.elementary_complex(
            mu,u[0],u[1],r,df,
            label,
            independent = (r == 0.0)
        )
        
    else:
        u = standard_uncertainty(seq,mu)
        return context.elementary_real(
            mu,u,df,label,independent=True
        )

#-----------------------------------------------------------------------------------------
def mean(seq):
    """Return the arithmetic mean of data in ``seq``

    If ``seq`` contains real or uncertain real numbers,
    a real number is returned.

    If ``seq`` contains complex or uncertain complex
    numbers, a complex number is returned.
    
    **Example**::

        >>> data = range(15)
        >>> type_a.mean(data)
        7.0
            
    """
    mu = sum(seq) / len(seq)
    if isinstance(mu,(numbers.Real,UncertainReal) ):
        return float(mu)
    elif isinstance(mu,(numbers.Complex,UncertainComplex)):
        return complex(mu)
    else:
        raise RuntimeError(
            "Unexpected type: '%s'" % repr(mu)
        )

#-----------------------------------------------------------------------------------------
def standard_deviation(seq,mu=None):
    """Return the sample standard deviation
    
    :arg seq: sequence of numbers
    :arg mu: the arithmetic mean of ``seq``
        
    If ``seq`` contains complex or uncertain complex
    numbers, the standard deviation in the real and
    imaginary components is evaluated, as well as
    the sample correlation coefficient.

    Otherwise the sample standard deviation is returned.
    
    The calculation only uses the `value` attribute 
    of uncertain numbers.
    
    **Examples**::

        >>> data = range(15)
        >>> type_a.standard_deviation(data)
        4.47213595499958

        >>> data = [(0.91518731126816899+1.5213442955575518j),
        ... (0.96572684493613492-0.18547192979059401j),
        ... (0.23216598132006649+1.6951311687588568j),
        ... (2.1642786101267397+2.2024333895672563j),
        ... (1.1812532664590505+0.59062101107787357j),
        ... (1.2259264339405165+1.1499373179910186j),
        ... (-0.99422341300318684+1.7359338393131392j),
        ... (1.2122867690240853+0.32535154897909946j),
        ... (2.0122536479379196-0.23283009302603963j),
        ... (1.6770229536619197+0.77195994890476838j)]
        >>> sd,r = type_a.standard_deviation(data)
        >>> sd
        standard_deviation(real=0.913318449990377, imag=0.8397604244242309)
        >>> r
        -0.31374045124595246
        
    """
    N = len(seq)
    
    if mu is None:
        mu = mean(seq)

    # `mean` will only return a pure number type
    if isinstance(mu,numbers.Real):
        accum = lambda psum,x: psum + (float(x)-mu)**2
        variance = reduce(accum, seq, 0.0) / (N - 1)
        
        return math.sqrt( variance )
        
    elif isinstance(mu,numbers.Complex):
        cv_11,cv_12,cv_12,cv_22 = variance_covariance_complex(seq,mu)

        sd_re = math.sqrt(cv_11)
        sd_im = math.sqrt(cv_22)

        den = sd_re * sd_im
        
        if den == 0.0: 
            if cv_12 != 0.0 :
                raise RuntimeError,\
                    "numerical instability in covariance calculation"
            else:
                # no correlation
                r = 0.0
        else:
            r = cv_12 / den
            
        return StandardDeviation(sd_re,sd_im), r
        
    else:
        assert False,"should never occur"

#-----------------------------------------------------------------------------------------
def standard_uncertainty(seq,mu=None):
    """Return the standard uncertainty of the sample mean

    :arg seq: sequence of numbers
    :arg mu: the arithmetic mean of ``seq``
    
    :rtype: float
    
    If ``seq`` contains complex or uncertain complex
    numbers, the standard uncertainties of the real and
    imaginary components are evaluated, as well as the
    sample correlation coefficient.

    Otherwise the standard uncertainty of the sample mean 
    is returned.

    The calculation only uses the `value` attribute 
    of uncertain numbers. 

    **Example**::

        >>> data = range(15)
        >>> type_a.standard_uncertainty(data)
        1.1547005383792515

        >>> data = [(0.91518731126816899+1.5213442955575518j),
        ... (0.96572684493613492-0.18547192979059401j),
        ... (0.23216598132006649+1.6951311687588568j),
        ... (2.1642786101267397+2.2024333895672563j),
        ... (1.1812532664590505+0.59062101107787357j),
        ... (1.2259264339405165+1.1499373179910186j),
        ... (-0.99422341300318684+1.7359338393131392j),
        ... (1.2122867690240853+0.32535154897909946j),
        ... (2.0122536479379196-0.23283009302603963j),
        ... (1.6770229536619197+0.77195994890476838j)]
        >>> u,r = type_a.standard_uncertainty(data)
        >>> u
        standard_uncertainty(real=0.28881665310241805, imag=0.2655555630050262)
        >>> u.real
        0.28881665310241805
        >>> r
        -0.31374045124595246

    """
    ROOT_N = math.sqrt(len(seq))
    
    if mu is None:
        mu = mean(seq)

    if isinstance(mu,numbers.Real):
        sd = standard_deviation(seq,mu)
        return sd / ROOT_N  
        
    elif isinstance(mu,numbers.Complex):
        sd,r = standard_deviation(seq,mu)
        return StandardUncertainty(sd.real/ROOT_N,sd.imag/ROOT_N),r
        
    else:
        assert False,"should not occur : '%s'" % repr(mu)

#-----------------------------------------------------------------------------------------
def variance_covariance_complex(seq,mu=None):
    """Return the sample variance-covariance matrix

    :arg seq: sequence of numbers   
    :arg mu: the arithmetic mean of ``seq``

    :returns: a 4-element sequence

    If ``mu`` is not provided it will be evaluated
    (see :func:`~type_a.mean`).

    ``seq`` may contain numbers or uncertain numbers.
    However, the calculation only uses the `value` 
    of uncertain numbers.
    
    .. note:

        Variance-covariance matrix elements are returned  
        in a namedtuple; they can be accessed using the 
        attributes ``.rr``, ``.ri``, ``,ir`` and ``.ii``.
        
    **Example**::
    
        >>> data = [(0.91518731126816899+1.5213442955575518j),
        ... (0.96572684493613492-0.18547192979059401j),
        ... (0.23216598132006649+1.6951311687588568j),
        ... (2.1642786101267397+2.2024333895672563j),
        ... (1.1812532664590505+0.59062101107787357j),
        ... (1.2259264339405165+1.1499373179910186j),
        ... (-0.99422341300318684+1.7359338393131392j),
        ... (1.2122867690240853+0.32535154897909946j),
        ... (2.0122536479379196-0.23283009302603963j),
        ... (1.6770229536619197+0.77195994890476838j)]
        >>> type_a.variance_covariance_complex(data)
        variance_covariance(rr=0.8341505910928249, ri=-0.24062910264062262, 
            ir=-0.24062910264062262, ii=0.7051975704291644)

        >>> v = type_a.variance_covariance_complex(data)
        >>> v[0]
        0.8341505910928249
        >>> v.rr
        0.8341505910928249
        >>> v.ii
        0.7051975704291644

    """
    N = len(seq)
    
    zseq = [ complex(x) for x in seq ]
    
    if mu is None:
        mu = mean(zseq)        
    else:
        mu = complex( mu )

            
    accum_vr = lambda psum,z: psum + (z.real - mu.real)**2
    accum_vi = lambda psum,z: psum + (z.imag - mu.imag)**2
    accum_cv = lambda psum,z: psum + (z.imag - mu.imag)*(z.real - mu.real)
    
    cv_11 = reduce(accum_vr,zseq,0.0) / (N-1) 
    cv_22 = reduce(accum_vi,zseq,0.0) / (N-1)
    cv_12 = reduce(accum_cv,zseq,0.0) / (N-1)

    return VarianceCovariance(cv_11,cv_12,cv_12,cv_22)

#-----------------------------------------------------------------------------------------
def multi_estimate_real(seq_of_seq,labels=None,context=_context):
    """Return a sequence of related uncertain real numbers 

    :arg seq_of_seq: a sequence of real-valued sequences
    :arg labels: a sequence of labels 
    
    :returns: a sequence of uncertain real numbers

    The sequences in ``seq_of_seq`` must all be the same length.
    
    Defines uncertain numbers using the sample statistics from 
    the data sequences, including the sample covariance. 

    The uncertain numbers returned are considered related, so that a 
    degrees-of-freedom calculation can be performed even if there is 
    correlation between them. 

    **Example**::
    
        # From Appendix H2 in the GUM
        
        >>> V = [5.007,4.994,5.005,4.990,4.999]
        >>> I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        >>> phi = [1.0456,1.0438,1.0468,1.0428,1.0433]
        >>> v,i,p = type_a.multi_estimate_real((V,I,phi),labels=('V','I','phi'))
        >>> v
        ureal(4.99899999999999967,0.00320936130717617944,4,label='V')
        >>> i
        ureal(0.019661000000000001392,9.47100839404133456689e-06,4,label='I')
        >>> p
        ureal(1.044459999999999944,0.0007520638270785368149,4,label='phi')
        
        >>> r = v/i*cos(p)
        >>> r
        ureal(127.73216992810208,0.071071407396995398,4)
        
    """
    M = len(seq_of_seq)
    N = len(seq_of_seq[0])
    
    if labels is not None and len(labels) != M:
        raise RuntimeError("Incorrect number of labels: '{!r}'".format(labels) ) 
        
    # Calculate the deviations from the mean for each sequence
    means = [ ]
    dev = []
    for i,seq_i in enumerate(seq_of_seq):
        if len(seq_i) != N:
            raise RuntimeError( "{:d}th sequence length inconsistent".format(i) )

        mu_i =  math.fsum(seq_i) / N
        means.append( mu_i )
        dev.append([ float(x_j)-mu_i for x_j in seq_i])

    # calculate the covariance matrix
    N_N_1 = N*(N-1)
    u = []
    cv = [] # M elements of len M-1, M-2, ...
    for i,seq_i in enumerate(dev):
        u_i = math.sqrt(
            math.fsum(d_i**2 for d_i in seq_i)/N_N_1
        )
        u.append(u_i)
        cv.append([])
        for seq_j in dev[i+1:]:
            cv[i].append(
                math.fsum(
                    d_i*d_j
                        for d_i,d_j in itertools.izip(seq_i,seq_j)
                )/N_N_1
            )

    ureal = context.elementary_real
    set_correlation = context.set_correlation

    # Create a list of elementary uncertain numbers
    # to return a list of standard uncertainties
    # to normalise the CV matrix.
    df = N-1
    rtn = []
    for i in xrange(M):
        mu_i = means[i]
        u_i = u[i]
        l_i = labels[i] if labels is not None else ""
        rtn.append( ureal(mu_i,u_i,df,l_i,independent=False) )

    # Create the list of ensemble id's,
    # assign it to the register in the context,
    # set the correlation between nodes
    context.real_ensemble( rtn, df )
    
    for i in xrange(M):
        u_i = u[i]
        un_i = rtn[i]
        
        for j in xrange(M-1-i):
            cv_ij = cv[i][j]
            if cv_ij != 0.0:
                r =  cv_ij / (u_i*u[i+j+1])
                un_j = rtn[i+j+1]
                set_correlation(un_i,un_j,r)

    return rtn

#-----------------------------------------------------------------------------------------
def multi_estimate_complex(seq_of_seq,labels=None,context=_context):
    """
    Return a sequence of related uncertain complex numbers

    :arg seq_of_seq: a sequence of complex number sequences
    :arg labels: a sequence of labels for the uncertain numbers
    
    :returns: a sequence of uncertain complex numbers
        
    The sequences in ``seq_of_seq`` must all be the same length.
    
    Defines uncertain numbers using the sample statistics, including
    the sample covariance.  

    The uncertain complex numbers returned are considered related,
    so they may be used in a degrees-of-freedom calculation even 
    if there is correlation between them. 

    **Example**::
    
        # From Appendix H2 in the GUM
        
        >>> I = [ complex(x) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3) ]
        >>> V = [ complex(x) for x in (5.007,4.994,5.005,4.990,4.999)]
        >>> P = [ complex(0,p) for p in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        >>> v,i,p = type_a.multi_estimate_complex( (V,I,P) )

        >>> get_correlation(v.real,i.real)
        -0.355311219817512

        >>> z = v/i*exp(p)
        >>> z.real
        ureal(127.73216992810208,0.071071407396995398,4)
        >>> get_correlation(z.real,z.imag)
        -0.5884297844235157
        
    """
    M = len(seq_of_seq)
    N = len(seq_of_seq[0])
    
    if labels is not None and len(labels) != M:
        raise RuntimeError( "Incorrect number of labels: '{!r}'".format(labels) ) 

    # 1. Create a 2M sequence of sequences of real values
    x = []
    for i in xrange(M):
        x.append( [ float(z_i.real) for z_i in seq_of_seq[i] ] )
        x.append( [ float(z_i.imag) for z_i in seq_of_seq[i] ] )
        if len(x[-1]) != N:
            raise RuntimeError(
                "{:d}th sequence length is incorrect".format(i)
            )

    TWOM = 2*M
    N_1 = N-1
    N_N_1 = N*N_1

    # 2. Evaluate the means and uncertainties (keep the deviation sequences)
    x_mean = [ math.fsum(seq_i) / N for seq_i in x ]
    x_u = []
    for i in xrange(TWOM):
        mu_i = x_mean[i]
        x[i] = [ mu_i - x_ij for x_ij in x[i] ]
        x_u.append(
            math.sqrt(
                math.fsum( x_ij**2 for x_ij in x[i] )/N_N_1
            )
        )
    # 3. Define uncertain M complex numbers
    ucomplex = context.elementary_complex
    set_correlation = context.set_correlation

    x_influences = []
    rtn = []
    for i in xrange(M):
        j = 2*i
        uc = ucomplex(
            complex(x_mean[j],x_mean[j+1]),
            x_u[j],x_u[j+1],0.0,
            N_1,
            labels[i] if labels is not None else None,
            independent=False
        )
        rtn.append( uc )
        x_influences.extend( (uc.real,uc.imag) )
        

    # 4. Calculate covariances and set correlation coefficients
    for i in xrange(TWOM-1):
        x_i = x[i]
        un_i = x_influences[i]
        for j in xrange(i+1,TWOM):
            x_j = x[j]
            cv = math.fsum( 
                d_i*d_j for d_i,d_j in itertools.izip(x_i,x_j) 
            )/N_N_1
            if cv != 0.0:
                r = cv/(x_u[i]*x_u[j]) 
                # raises exception if |r| > 1
                # makes a symmetric entry
                set_correlation(un_i,x_influences[j],r)

    context.complex_ensemble( rtn, N_1 )
    
    return rtn

#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *
    
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

    

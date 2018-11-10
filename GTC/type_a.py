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

    Many functions in :mod:`type_a` treat  data as pure numbers. 
    Sequences of uncertain numbers can be passed to these 
    functions, but only the uncertain-number values will be used.
    
Module contents
---------------

"""
from __future__ import division

import sys
import math
import numbers
from functools import reduce

try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

from GTC.context import _context 

from GTC.lib import (
    UncertainReal, 
    UncertainComplex,
    set_correlation_real,
    real_ensemble,
    complex_ensemble
)

from GTC.named_tuples import (
    VarianceCovariance,
    StandardUncertainty,
    StandardDeviation
)

__all__ = (
    'estimate',
    'estimate_digitized',
    'multi_estimate_real',
    'multi_estimate_complex',
    'mean',
    'standard_deviation',
    'standard_uncertainty',
    'variance_covariance_complex',
)

#-----------------------------------------------------------------------------------------
def _as_value(x):
    try:
        return x.x 
    except AttributeError:
        return x 
        
#-----------------------------------------------------------------------------------------
def estimate_digitized(seq,delta,label=None,truncate=False,context=_context):
    """
    Return an uncertain number for the mean of digitized data

    :arg seq: data
    :type seq: float, :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`
    :arg float delta: digitization step size 
    :arg str label: label for uncertain number returned
    :arg bool truncate: if ``True``, truncation, rather than rounding, is assumed
    :rtype: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`

    A sequence of data that has been formatted with fixed precision  
    can completely conceal a small amount of variability in the original
    values, or merely obscure that variability.  
    
    This function recognises the possible interaction between truncation, or rounding,
    errors and random errors in the underlying data. The function 
    obtains the mean of the data sequence and evaluates the uncertainty 
    in this mean as an estimate of the mean of the process generating 
    the data.   
        
    Set the argument ``truncate`` to ``True`` 
    if data have been truncated, instead of rounded.
    
    See reference: R Willink, *Metrologia*, **44** (2007) 73-81

    **Examples**::
    
        # LSD = 0.0001, data varies between -0.0055 and -0.0057
        >>> seq = (-0.0056,-0.0055,-0.0056,-0.0056,-0.0056, 
        ...      -0.0057,-0.0057,-0.0056,-0.0056,-0.0057,-0.0057)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.005627272727272727,1.9497827808661157e-05,10)

        # LSD = 0.0001, data varies between -0.0056 and -0.0057
        >>> seq = (-0.0056,-0.0056,-0.0056,-0.0056,-0.0056,
        ... -0.0057,-0.0057,-0.0056,-0.0056,-0.0057,-0.0057)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.005636363636363636,1.5212000482437775e-05,10)

        # LSD = 0.0001, no spread in data values
        >>> seq = (-0.0056,-0.0056,-0.0056,-0.0056,-0.0056,
        ... -0.0056,-0.0056,-0.0056,-0.0056,-0.0056,-0.0056)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0056,2.886751345948129e-05,10)
        
        # LSD = 0.0001, no spread in data values, fewer points
        >>> seq = (-0.0056,-0.0056,-0.0056)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0056,3.291402943021917e-05,2)
        
    """
    N = len(seq)
    if N < 2:
        raise RuntimeError(
            "digitized data sequence must have more than one element"
        )

    try:
        seq = [ float(x_i) for x_i in seq ]
    except TypeError:
        # If not a float then an uncertain number?
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
        
    return UncertainReal._elementary(mean,u,N-1,label,independent=True)
    
#-----------------------------------------------------------------------------------------
def estimate(seq,label=None,context=_context):
    """Return an uncertain number for the mean of the data 

    :arg seq:   a sequence of data
    :arg str label: a label for the returned uncertain number
    
    :rtype:   :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`
                
    The elements of ``seq`` may be real numbers, complex numbers, or
    uncertain real or complex numbers. Note that only the value of uncertain 
    numbers will be used.

    In a type-A evaluation, the sample mean provides an estimate of the  
    quantity of interest. The uncertainty in this estimate 
    is the standard deviation of the sample mean (or the  
    sample covariance of the mean, in the complex case).    
    
    The function returns an :class:`~lib.UncertainReal` when 
    the mean of the data is real, and an :class:`~lib.UncertainComplex` 
    when the mean of the data is complex.

    **Examples**::

        >>> data = range(15)
        >>> type_a.estimate(data)
        ureal(7.0,1.1547005383792515,14)
        
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
        ucomplex((1.059187840567141+0.9574410497332932j), u=[0.28881665310241805,0.2655555630050262], r=-4.090655272692547, df=9)

    """
    df = len(seq)-1
    mu = mean(seq)
    
    if isinstance(mu,complex):
        u,r = standard_uncertainty(seq,mu)
        return UncertainComplex._elementary(
            mu,u[0],u[1],r,df,
            label,
            independent = (r == 0.0)
        )
        
    else:
        u = standard_uncertainty(seq,mu)
        return UncertainReal._elementary(
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
        return _as_value(mu)
    elif isinstance(mu,(numbers.Complex,UncertainComplex)):
        return _as_value(mu)
    else:
        raise TypeError(
            "Unexpected type: '%s'" % repr(mu)
        )

#-----------------------------------------------------------------------------------------
def standard_deviation(seq,mu=None):
    """Return the sample standard deviation
    
    :arg seq: sequence of data
    :arg mu: the arithmetic mean of ``seq``
        
    If ``seq`` contains real or uncertain real numbers, 
    the sample standard deviation is returned.
    
    If ``seq`` contains complex or uncertain complex
    numbers, the standard deviation in the real and
    imaginary components is evaluated, as well as
    the correlation coefficient between the components.
    The results are returned in a pair of objects: a
    :obj:`~named_tuples.StandardDeviation` namedtuple 
    and a correlation coefficient. 

    Only the values of uncertain numbers are used in calculations. 
    
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
        StandardDeviation(real=0.913318449990377, imag=0.8397604244242309)
        >>> r
        -0.31374045124595246
        
    """
    N = len(seq)
    
    if mu is None:
        mu = mean(seq)

    # `mean` returns either a real or complex
    if isinstance(mu,numbers.Real):
        accum = lambda psum,x: psum + (_as_value(x)-mu)**2
        variance = reduce(accum, seq, 0.0) / (N - 1)
        
        return math.sqrt( variance )
        
    elif isinstance(mu,numbers.Complex):
        cv_11,cv_12,cv_12,cv_22 = variance_covariance_complex(seq,mu)

        sd_re = math.sqrt(cv_11)
        sd_im = math.sqrt(cv_22)

        den = sd_re * sd_im
        
        if den == 0.0: 
            if cv_12 != 0.0 :
                raise RuntimeError(
                    "numerical instability in covariance calculation"
                )
            else:
                r = 0.0
        else:
            r = cv_12 / den
            
        return StandardDeviation(sd_re,sd_im), r
        
    else:
        assert False,"unexpected"

#-----------------------------------------------------------------------------------------
def standard_uncertainty(seq,mu=None):
    """Return the standard uncertainty of the sample mean

    :arg seq: sequence of data
    :arg mu: the arithmetic mean of ``seq``
    
    :rtype: float or :obj:`~named_tuples.StandardUncertainty`
    
    If ``seq`` contains real or uncertain real numbers,
    the standard uncertainty of the sample mean 
    is returned.

    If ``seq`` contains complex or uncertain complex
    numbers, the standard uncertainties of the real and
    imaginary components are evaluated, as well as the
    sample correlation coefficient are returned in a
    :obj:`~named_tuples.StandardUncertainty` namedtuple

    Only the values of uncertain numbers are used in calculations. 

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
        StandardUncertainty(real=0.28881665310241805, imag=0.2655555630050262)
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
        assert False,"unexpected, mu={!r}".format(mu)

#-----------------------------------------------------------------------------------------
def variance_covariance_complex(seq,mu=None):
    """Return the sample variance-covariance matrix

    :arg seq: sequence of data   
    :arg mu: the arithmetic mean of ``seq``

    :returns: a 4-element sequence

    If ``mu`` is ``None`` the mean will be evaluated 
    by :func:`~type_a.mean`.

    ``seq`` may contain numbers or uncertain numbers.
    Only the values of uncertain numbers are used in calculations. 
    
    Variance-covariance matrix elements are returned  
    in a :obj:`~named_tuples.VarianceCovariance` namedtuple; 
    they can be accessed using the 
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
        VarianceCovariance(rr=0.8341505910928249, ri=-0.24062910264062262, ir=-0.24062910264062262, ii=0.7051975704291644)

        >>> v = type_a.variance_covariance_complex(data)
        >>> v[0]
        0.8341505910928249
        >>> v.rr
        0.8341505910928249
        >>> v.ii
        0.7051975704291644

    """
    N = len(seq)
    
    zseq = [ _as_value(x) for x in seq ]
    
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
def multi_estimate_real(seq_of_seq,labels=None):
    """Return a sequence of uncertain real numbers 

    :arg seq_of_seq: a sequence of sequences of data
    :arg labels: a sequence of `str` labels 
    
    :rtype: seq of :class:`~lib.UncertainReal`

    The sequences in ``seq_of_seq`` must all be the same length.
    Each sequence is associated with a particular quantity and contains 
    a sample of data. An uncertain number for the quantity will be created  
    using the sample of data, using sample statistics. The covariance 
    between different quantities will also be evaluated from the data.
    
    A sequence of elementary uncertain numbers are returned. The uncertain numbers 
    are considered related, allowing a degrees-of-freedom calculations 
    to be performed on derived quantities. 

    **Example**::
    
        # From Appendix H2 in the GUM
        
        >>> V = [5.007,4.994,5.005,4.990,4.999]
        >>> I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        >>> phi = [1.0456,1.0438,1.0468,1.0428,1.0433]
        >>> v,i,p = type_a.multi_estimate_real((V,I,phi),labels=('V','I','phi'))
        >>> v
        ureal(4.999,0.0032093613071761794,4, label='V')
        >>> i
        ureal(0.019661,9.471008394041335e-06,4, label='I')
        >>> p
        ureal(1.04446,0.0007520638270785368,4, label='phi')
        
        >>> r = v/i*cos(p)
        >>> r
        ureal(127.73216992810208,0.0710714073969954,4.0)
        
    """
    M = len(seq_of_seq)
    N = len(seq_of_seq[0])
    
    if labels is not None and len(labels) != M:
        raise RuntimeError(
            "Incorrect number of labels: '{!r}'".format(labels) 
        ) 
        
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
                        for d_i,d_j in izip(seq_i,seq_j)
                )/N_N_1
            )

    ureal = UncertainReal._elementary

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
    real_ensemble( rtn, df )
    
    for i in xrange(M):
        u_i = u[i]
        un_i = rtn[i]
        
        for j in xrange(M-1-i):
            cv_ij = cv[i][j]
            if cv_ij != 0.0:
                r =  cv_ij / (u_i*u[i+j+1])
                un_j = rtn[i+j+1]
                set_correlation_real(un_i,un_j,r)

    return rtn

#-----------------------------------------------------------------------------------------
def multi_estimate_complex(seq_of_seq,labels=None,context=_context):
    """
    Return a sequence of uncertain complex numbers

    :arg seq_of_seq: a sequence of sequences of data
    :arg labels: a sequence of `str` labels
    
    :rtype: a sequence of :class:`~lib.UncertainComplex`
        
    The sequences in ``seq_of_seq`` must all be the same length.
    Each sequence contains a sample of data that is associated with 
    a particular quantity. An uncertain number for the quantity will  
    be created using this data from sample statistics. The covariance 
    between different quantities will also be evaluated from the data.
    
    A sequence of elementary uncertain complex numbers are returned. These   
    uncertain numbers are considered related, allowing a degrees-of-freedom  
    calculations to be performed on derived quantities. 
    
    Defines uncertain numbers using the sample statistics, including
    the sample covariance.  

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
        ureal(127.73216992810208,0.0710714073969954,4.0)
        >>> get_correlation(z.real,z.imag)
        -0.5884297844235157
        
    """
    M = len(seq_of_seq)
    N = len(seq_of_seq[0])
    
    if labels is not None and len(labels) != M:
        raise RuntimeError( 
            "Incorrect number of labels: '{!r}'".format(labels) 
        ) 

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
    ucomplex = UncertainComplex._elementary

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
                d_i*d_j for d_i,d_j in izip(x_i,x_j)
            )/N_N_1
            if cv != 0.0:
                r = cv/(x_u[i]*x_u[j]) 
                set_correlation_real(un_i,x_influences[j],r)

    complex_ensemble( rtn, N_1 )
    
    return rtn

#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *    
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

    

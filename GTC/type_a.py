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
    
Least squares regression
------------------------
    *   :func:`line_fit` performs an ordinary least-squares straight 
        line fit to a sample of data.  
    *   :func:`line_fit_wls` performs a weighted least-squares straight 
        line fit to a sample of data. The weights are assumed to be exact.
    *   :func:`line_fit_rwls` performs a weighted least-squares  
        straight line fit to a sample of data. The weights
        are only assumed normalise the variability of observations.
    *   :func:`line_fit_wtls` performs a weighted total least-squares straight 
        line fit to a sample of data.   
 
Merging uncertain components
----------------------------
    *   :func:`merge` combines results from a type-A and type-B analysis 
        of the same data. 
 
.. note::

    Many functions in :mod:`type_a` treat  data as pure numbers. 
    Sequences of uncertain numbers can be passed to these 
    functions, but only the uncertain-number values will be used.
    
    :func:`merge` is provided so that the results of 
    type-A and type-B analyses on the same data can be 
    combined. 

Module contents
---------------

"""
from __future__ import division

import sys
import math
import numbers
import numpy as np
from functools import reduce

try:
    from itertools import izip  # Python 2
    from collections import Iterable
except ImportError:
    izip = zip
    xrange = range
    from collections.abc import Iterable

from GTC import (
    inf, 
    function,
    is_sequence
)
from GTC.lib import (
    UncertainReal, 
    UncertainComplex,
    set_correlation_real,
    real_ensemble,
    complex_ensemble,
    append_real_ensemble,
    value,
    value_seq
)
ureal = UncertainReal._elementary
ucomplex = UncertainComplex._elementary
result = lambda un,label: un._intermediate(label)

from GTC import type_b
from GTC.type_b import (
    LineFit, LineFitWTLS,
)

from GTC.named_tuples import (
    VarianceCovariance,
    StandardUncertainty,
    StandardDeviation,
    InterceptSlope
)

EPSILON = sys.float_info.epsilon 
HALF_PI = math.pi/2.0

__all__ = (
    'estimate',
    'estimate_digitized',
    'multi_estimate_real',
    'multi_estimate_complex',
    'mean',
    'standard_deviation',
    'standard_uncertainty',
    'variance_covariance_complex',
    'line_fit', 'line_fit_wls', 'line_fit_rwls', 'line_fit_wtls',
    'LineFitOLS','LineFitRWLS','LineFitWLS','LineFitWTLS',
    'merge',
)

#-----------------------------------------------------------------------------------------
#
class LineFitOLS(LineFit):
    
    """
    Class to hold the results of an ordinary least-squares regression to data.

    It can also be used to apply the results of a regression analysis. 
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)
            
    def __str__(self):
        header = '''
Ordinary Least-Squares Results:
'''
        return header + LineFit.__str__(self)

    def x_from_y(self,yseq,x_label=None,y_label=None):
        """Estimate the stimulus ``x`` corresponding to the responses in ``yseq``

        :arg yseq: a sequence of ``y`` observations 
        :arg x_label: a label for the return uncertain number `x` 
        :arg y_label: a label for the estimate of `y` based on ``yseq`` 

        ..note::
            When ``x_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        **Example** ::
        
            >>> x_data = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5,
            ...                 0.7, 0.7, 0.7, 0.9, 0.9, 0.9]
            >>> y_data = [0.028, 0.029, 0.029, 0.084, 0.083, 0.081, 0.135, 0.131,
            ...                 0.133, 0.180, 0.181, 0.183, 0.215, 0.230, 0.216]

            >>> fit = type_a.line_fit(x_data,y_data)
            
            >>> x0 = fit.x_from_y( [0.0712, 0.0716] )            
            >>> x0
            ureal(0.2601659751037...,0.01784461112558...,13.0)

        """
        df = self._N - 2       
        a, b = self._a_b

        p = len(yseq)
        y = math.fsum( yseq ) / p
        
        y = ureal(
            y,
            math.sqrt( self._ssr/df/p ),
            df,
            label=y_label,
            independent=False
        )  

        append_real_ensemble(a,y)
        
        if x_label is None:
            x = (y - a)/b
        else:
            x = result( (y - a)/b, label=x_label )

        return x
        
    def y_from_x(self,x,s_label=None,y_label=None):
        """Return an uncertain number ``y`` that predicts the response to ``x``

        :arg x: a real number, or an uncertain real number
        :arg s_label: a label for an elementary uncertain number associated with observation variability  
        :arg y_label: a label for the return uncertain number `y` 

        This is a prediction of a single future response ``y`` to a stimulus ``x``
        
        The variability in observations is based on residuals obtained during regression.
        
        An uncertain real number can be used for ``x``, in which
        case the associated uncertainty will also be propagated into ``y``.

        ..note::
            When ``y_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)
        
        """
        a, b = self._a_b   
        df = self._N - 2
        
        u = math.sqrt( self._ssr/df )
        
        noise = ureal(
            0,u,df,label=s_label,independent=False
        )
        
        append_real_ensemble(a,noise)
                  
        if y_label is None:
            y = a + b*x + noise
        else:
            y = result( a + b*x + noise, label=y_label )
        
        return y

#-----------------------------------------------------------------------------------------
#
class LineFitRWLS(LineFit):
    
    """
    Class to hold the results of a relative weighted least-squares regression.
    The weights provided normalise the variability of observations.
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Relative Weighted Least-Squares Results:
'''
        return header + LineFit.__str__(self)

    def x_from_y(self,yseq,s_y,x_label=None,y_label=None):
        """Estimate the stimulus ``x`` corresponding to the responses in ``yseq``

        :arg yseq: a sequence of further observations of ``y``
        :arg s_y: a scale factor for the uncertainty of the ``yseq``
        :arg x_label: a label for the return uncertain number `x` 
        :arg y_label: a label for the estimate of `y` based on ``yseq``

        ..note::
            When ``x_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        """
        df = self._N - 2       
        a, b = self._a_b
        
        p = len(yseq)
        y = math.fsum( yseq ) / p
        
        y = ureal(
            y,
            s_y * math.sqrt( self._ssr/df/p ),
            df,
            label=y_label,
            independent=False
        )            

        append_real_ensemble(a,y)
        
        if x_label is None:
            x = (y - a)/b
        else:
            x = result( (y - a)/b, label=x_label )

        return x

    def y_from_x(self,x,s_y,s_label=None,y_label=None):
        """Return an uncertain number ``y`` that predicts the response to ``x``

        :arg x: a real number, or an uncertain real number
        :arg s_y: a scale factor for the response uncertainty
        :arg s_label: a label for an elementary uncertain number associated with observation variability  
        :arg y_label: a label for the return uncertain number `y` 

        Returns a single future response ``y`` predicted for a stimulus ``x``.

        Because there is different variability in 
        the response to different stimuli, the
        scale factor ``s_y`` is required. It is assumed 
        that the standard deviation in the ``y`` value is 
        proportional to ``s_y``.
        
        An uncertain real number can be used for ``x``, in which
        case the associated uncertainty is also propagated into ``y``.
        
        ..note::
            When ``y_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        """
        a, b = self._a_b   
        df = self._N - 2
        
        u = math.sqrt( s_y*self._ssr/df )
        
        noise = ureal(0,u,df,label=s_label)

        append_real_ensemble(a,noise)
                  
        if y_label is None:
            y = a + b*x + noise
        else:
            y = result( a + b*x + noise, label=y_label )
        
        return y
        
#-----------------------------------------------------------------------------------------
#
class LineFitWLS(LineFit):
    
    """
    Class to hold the results of a weighted least-squares regression.
    The weight factors provided are assumed to correspond exactly to the
    variability of observations.
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Weighted Least-Squares Results:
'''
        return header + LineFit.__str__(self)

    def x_from_y(self,yseq,u_yseq,x_label=None,y_label=None):
        """Estimate the stimulus ``x`` corresponding to the responses in ``yseq``

        :arg yseq: a sequence of further observations of ``y``
        :arg u_yseq: the standard uncertainty of the ``yseq`` data
        :arg x_label: a label for the return uncertain number `x` 
        :arg y_label: a label for the estimate of `y` based on ``yseq``

        The variations in ``yseq`` values are assumed to result from 
        independent random effects.
        
        ..note::
            When ``x_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        """
        a, b = self._a_b
        
        p = len(yseq)
        y = math.fsum( yseq ) / p
        
        y = ureal(
            y,
            u_yseq / math.sqrt( p ),
            inf,
            label=y_label,
            independent=False
        )            

        append_real_ensemble(a,y)
        
        if x_label is None:
            x = (y - a)/b
        else:
            x = result( (y - a)/b, label=x_label )

        return x

    def y_from_x(self,x,s_y,s_label=None,y_label=None):
        """Return an uncertain number ``y`` that predicts the response to ``x``

        :arg x: a real number, or an uncertain real number
        :arg s_y: response variability uncertainty
        :arg s_label: a label for an elementary uncertain number associated with response variability  
        :arg y_label: a label for the return uncertain number `y` 

        Returns a single future response ``y`` predicted for a stimulus ``x``.

        The standard uncertainty ``s_y`` is used to create an additive
        component of uncertainty associated with variability in the ``y`` value.
        
        An uncertain real number can be used for ``x``, in which
        case the associated uncertainty is also propagated into ``y``.
        
        ..note::
            When ``y_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        """
        a, b = self._a_b   
        df = self.N - 2
        
        noise = ureal(0,s_y,df,label=s_label)

        append_real_ensemble(a,noise)
                  
        if y_label is None:
            y = a + b*x + noise
        else:
            y = result( a + b*x + noise, label=y_label )
        
        return y
        
#-----------------------------------------------------------------------------------------
def line_fit(x,y,label=None):
    """Return a least-squares straight-line fit to the data
     
    .. versionadded:: 1.2
    
    :arg x:     sequence of stimulus data (independent-variable)  
    :arg y:     sequence of response data (dependent-variable)  
    :arg label: suffix to label the uncertain numbers `a` and `b`

    :returns:   an object containing regression results
    :rtype:     :class:`.LineFitOLS`

    Performs an ordinary least-squares regression of ``y`` to ``x``.
        
    **Example**::
    
        >>> x = [1,2,3,4,5,6,7,8,9]
        >>> y = [15.6,17.5,36.6,43.8,58.2,61.6,64.2,70.4,98.8]
        >>> result = type_a.line_fit(x,y)
        >>> a,b = result.a_b
        >>> a
        ureal(4.8138888888888...,4.8862063121833...,7)
        >>> b
        ureal(9.4083333333333...,0.8683016476563...,7)

        >>> y_p = a + b*5.5
        >>> dof(y_p)
        7.0
            
    """
    N = len(x)
    
    x = value_seq(x)
    y = value_seq(y)

    df = N-2

    S = N
    S_x = math.fsum( x )
    S_y = math.fsum( y )
        
    k = S_x / S
    t = [ (x_i - k) for x_i in x ]

    S_tt =  math.fsum( t_i*t_i for t_i in t )

    b_ =  math.fsum( t_i*y_i/S_tt for t_i,y_i in izip(t,y) )
    a_ = (S_y - b_*S_x)/S

    siga = math.sqrt( (1.0 + S_x*S_x/(S*S_tt))/S )
    sigb = math.sqrt( 1.0/S_tt )
    r_ab = -S_x/(S*S_tt*siga*sigb)
    
    # Sum of squared residuals needed to correctly calculate parameter uncertainties
    f = lambda x_i,y_i: (y_i - a_ - b_*x_i)**2 
    ssr =  math.fsum( f(x_i,y_i) for x_i,y_i in izip(x,y) )

    data_u = math.sqrt( ssr/df )
    siga *= data_u
    sigb *= data_u
            
    a = ureal(
        a_,
        siga,
        df=df,
        label='a_{}'.format(label) if label is not None else None,
        independent=False
    )
    b = ureal(
        b_,
        sigb,
        df=df,
        label='b_{}'.format(label) if label is not None else None,
        independent=False
    )
    
    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitOLS(a,b,ssr,N)

#-----------------------------------------------------------------------------------------
def _line_fit_wls(x,y,u_y):
    """Utility function
    
    NB client must supply sequences `x` and `y` of pure numbers
    """
    N = len(x)

    v = [ u_y_i*u_y_i for u_y_i in u_y ]
    S =  math.fsum( 1.0/v_i for v_i in v)

    S_x =  math.fsum( x_i/v_i for x_i,v_i in izip(x,v) )
    S_y =  math.fsum( y_i/v_i for y_i,v_i in izip(y,v) )

    k = S_x / S
    t = [ (x_i - k)/u_y_i for x_i,u_y_i in izip(x,u_y) ]

    S_tt =  math.fsum( t_i*t_i for t_i in t )

    b_ =  math.fsum( t_i*y_i/u_y_i/S_tt for t_i,y_i,u_y_i in izip(t,y,u_y) )
    a_ = (S_y - b_*S_x)/S

    siga = math.sqrt( (1.0 + S_x*S_x/(S*S_tt))/S )
    sigb = math.sqrt( 1.0/S_tt )
    r_ab = -S_x/(S*S_tt*siga*sigb)
    
    f = lambda x_i,y_i,u_y_i: ((y_i - a_ - b_*x_i)/u_y_i)**2 
    ssr =  math.fsum( f(x_i,y_i,u_y_i) for x_i,y_i,u_y_i in izip(x,y,u_y) )

    return a_,b_,siga,sigb,r_ab,ssr,N

#-----------------------------------------------------------------------------------------
def line_fit_wls(x,y,u_y,label=None):
    """Return a weighted least-squares straight-line fit
    
    .. versionadded:: 1.2
     
    :arg x:     sequence of stimulus data (independent-variable)  
    :arg y:     sequence of response data (dependent-variable)  
    :arg u_y:   sequence of uncertainties in the response data 
    :arg label: suffix to label the uncertain numbers `a` and `b`

    :returns:   an object containing regression results
    :rtype:     :class:`.LineFitWLS`

    **Example**::
    
        >>> x = [1,2,3,4,5,6]
        >>> y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
        >>> u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        
        >>> fit = type_a.line_fit_wls(x,y,u_y)
        >>> fit.a_b     
         InterceptSlope(a=ureal(0.8852320675105...,0.5297081435088...,inf),
         b=ureal(2.056962025316...,0.177892016741...,inf))
        
    """
    x = value_seq(x)
    y = value_seq(y)

    a_,b_,siga,sigb,r_ab,ssr,N = _line_fit_wls(x,y,u_y)
    
    a = ureal(
        a_,
        siga,
        inf,
        label='a_{}'.format(label) if label is not None else None,
        independent=False
    )
    b = ureal(
        b_,
        sigb,
        inf,
        label='b_{}'.format(label) if label is not None else None,
        independent=False
    )
    
    a.set_correlation(r_ab,b)

    return LineFitWLS(a,b,ssr,N)

#-----------------------------------------------------------------------------------------
def line_fit_rwls(x,y,s_y,label=None):
    """Return a relative weighted least-squares straight-line fit
    
    .. versionadded:: 1.2
    
    The ``s_y`` values are used to scale variability in the ``y`` data.
    It is assumed that the standard deviation of each ``y`` value is 
    proportional to the corresponding ``s_y`` scale factor.
    The unknown common factor in the uncertainties is estimated from the residuals.
    
    :arg x:     sequence of stimulus data (independent-variable)  
    :arg y:     sequence of response data (dependent-variable)  
    :arg s_y:   sequence of scale factors
    :arg label: suffix to label the uncertain numbers `a` and `b`

    :returns:   an object containing regression results
    :rtype:     :class:`.LineFitRWLS`

    **Example**::

        >>> x = [1,2,3,4,5,6]
        >>> y = [3.014,5.225,7.004,9.061,11.201,12.762]
        >>> s_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        >>> fit = type_a.line_fit_rwls(x,y,s_y)
        >>> a, b = fit.a_b
        >>>
        >>> print(fit)
        <BLANKLINE>
        Relative Weighted Least-Squares Results:
        <BLANKLINE>
          Intercept: 1.14(12)
          Slope: 1.973(41)
          Correlation: -0.87
          Sum of the squared residuals: 1.3395217958...
          Number of points: 6
        <BLANKLINE>
          
    """
    N = len(x)
    if N < 3:
        raise RuntimeError(
            "At least three data points are required, got {}".format(N)
        )
        
    x = value_seq(x)
    y = value_seq(y)
    
    a_,b_,siga,sigb,r_ab,ssr,N = _line_fit_wls(x,y,s_y)

    df = N-2
    sigma_hat = math.sqrt(ssr/df)
    siga *= sigma_hat
    sigb *= sigma_hat
    
    a = ureal(
        a_,
        siga,
        df,
        label='a_{}'.format(label) if label is not None else None,
        independent=False
    )
    b = ureal(
        b_,
        sigb,
        df,
        label='b_{}'.format(label) if label is not None else None,
        independent=False
    )
    
    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitRWLS(a,b,ssr,N)
    
#--------------------------------------------------------------------
#
def line_fit_wtls(x,y,u_x,u_y,a0_b0=None,r_xy=None,label=None):
    """Return a total least-squares straight-line fit 
    
    .. versionadded:: 1.2

    :arg x:     sequence of independent-variable data
    :arg y:     sequence of dependent-variable data 
    :arg u_x:   sequence of uncertainties in ``x``
    :arg u_y:   sequence of uncertainties in ``y``
    :arg a0_b0: a pair of initial estimates for the intercept and slope
    :arg r_xy:  correlation between x-y pairs
    :arg label: suffix labeling the uncertain numbers `a` and `b`

    :returns:   an object containing the fitting results
    :rtype:     :class:`.LineFitWTLS`

    The optional argument ``a_b`` can be used to provide a pair 
    of initial estimates for the intercept and slope. 

    Based on paper by M Krystek and M Anton,
    *Meas. Sci. Technol.* **22** (2011) 035101 (9pp)
    
    **Example**::

        # Pearson-York test data see, e.g., 
        # Lybanon, M. in Am. J. Phys 52 (1) 1984 
        >>> x=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        >>> wx=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]

        >>> y=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        >>> wy=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        # standard uncertainties required for weighting
        >>> ux=[1./math.sqrt(wx_i) for wx_i in wx ]
        >>> uy=[1./math.sqrt(wy_i) for wy_i in wy ]

        >>> result = ta.line_fit_wtls(x,y,ux,uy)
        >>> intercept, slope = result.a_b
        >>> intercept
        ureal(5.47991018...,0.29193349...,8)
        >>> slope
        ureal(-0.48053339...,0.057616740...,8)
    
    """
    independent = r_xy is None

    x_u = [ ureal( value(x_i),u_i,inf,None,independent=independent)
        for x_i, u_i in izip(x,u_x)
    ]
    y_u = [ ureal( value(y_i),u_i,inf,None,independent=independent)
        for y_i, u_i in izip(y,u_y)
    ]
    if not independent:
        for x_i,y_i,r_i in izip(x_u,y_u,r_xy):
            x_i.set_correlation(r_i,y_i)

    result = type_b.line_fit_wtls(x_u,y_u,a_b=a0_b0)

    a, b = result.a_b
    N = result.N
    ssr = result.ssr
    r_ab = a.get_correlation(b)
    
    df = N-2
    
    a = ureal(
        a.x,
        a.u,
        df,
        label='a_{}'.format(label) if label is not None else None,
        independent=False
    )
    b = ureal(
        b.x,
        b.u,
        df,
        label='b_{}'.format(label) if label is not None else None,
        independent=False
    )

    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitWTLS(a,b,ssr,N)
    
#-----------------------------------------------------------------------------------------
def estimate_digitized(seq,delta,label=None,truncate=False):
    """
    Return an uncertain number for the mean of digitized data in ``seq``

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
    evaluates the mean of the data and evaluates the uncertainty 
    in this mean.   
        
    Set the argument ``truncate`` to ``True`` 
    if data have been truncated, instead of rounded.
    
    See reference: R Willink, *Metrologia*, **44** (2007) 73-81

    **Examples**::
    
        # LSD = 0.0001, data varies between -0.0055 and -0.0057
        >>> seq = (-0.0056,-0.0055,-0.0056,-0.0056,-0.0056, 
        ...      -0.0057,-0.0057,-0.0056,-0.0056,-0.0057,-0.0057)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.005627272727272...,1.9497827808661...e-05,10)

        # LSD = 0.0001, data varies between -0.0056 and -0.0057
        >>> seq = (-0.0056,-0.0056,-0.0056,-0.0056,-0.0056,
        ... -0.0057,-0.0057,-0.0056,-0.0056,-0.0057,-0.0057)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.005636363636363...,1.52120004824377...e-05,10)

        # LSD = 0.0001, no spread in data values
        >>> seq = (-0.0056,-0.0056,-0.0056,-0.0056,-0.0056,
        ... -0.0056,-0.0056,-0.0056,-0.0056,-0.0056,-0.0056)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0056,2.8867513459481...e-05,10)
        
        # LSD = 0.0001, no spread in data values, fewer points
        >>> seq = (-0.0056,-0.0056,-0.0056)
        >>> type_a.estimate_digitized(seq,0.0001)
        ureal(-0.0056,3.2914029430219...e-05,2)
        
    """
    N = len(seq)
    if N < 2:
        raise RuntimeError(
            "digitized data sequence must have more than one element"
        )

    seq = value_seq(seq)
    
    x_max = max(seq)
    x_min = min(seq)
    
    mu = mean(seq)
        
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
        accum = lambda psum,x: psum + (x-mu)**2
        var = reduce(accum, seq, 0.0) / (N - 1)

        if abs(x_max - x_min - delta) < 10*sys.float_info.epsilon:
            # Scatter is LSD only
            x_mid = (x_max + x_min)/2.0
            u = math.sqrt(
                max(var/N,(x_mid - mu)**2/3.0)
            )
        else:
            u = math.sqrt(var/N)

    if truncate:
        mu += delta/2.0
        
    return ureal(mu,u,N-1,label,independent=True)
    
#-----------------------------------------------------------------------------------------
def estimate(seq,label=None):
    """Return an uncertain number for the mean of the data in ``seq``

    :arg seq:   a sequence of data
    :arg str label: a label for the returned uncertain number
    
    :rtype:   :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`
                
    The elements of ``seq`` may be real numbers, complex numbers, or
    uncertain real or complex numbers. Note that only the value of uncertain 
    numbers will be used.
    
    The function returns an :class:`~lib.UncertainReal` when 
    the mean of the data is real, and an :class:`~lib.UncertainComplex` 
    when the mean of the data is complex.

    In a type-A evaluation, the sample mean provides an estimate of the  
    quantity of interest. The uncertainty in this estimate 
    is the standard deviation of the sample mean (or the  
    sample covariance of the mean, in the complex case).    

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
        ucomplex((1.059187840567141+0.9574410497332932j), u=[0.28881665310241805,0.2655555630050262], r=-0.3137404512459525, df=9)

    """
    df = len(seq)-1
    
    seq = value_seq(seq)
    
    mu = mean(seq)
    
    if isinstance(mu,complex):
        u,r = standard_uncertainty(seq,mu)
        return ucomplex(
            mu,u[0],u[1],r,df,
            label,
            independent = (r == 0.0)
        )
       
    else:
        u = standard_uncertainty(seq,mu)
        return ureal(
            mu,u,df,label,independent=True
        )

#-----------------------------------------------------------------------------------------
def mean(seq,*args,**kwargs):
    """Return the arithmetic mean of data in ``seq``

    :arg seq: a sequence, :class:`~numpy.ndarray`, or iterable, of numbers or uncertain numbers
    :arg args: optional arguments when ``seq`` is an :class:`~numpy.ndarray`
    :arg kwargs: optional keyword arguments when ``seq`` is an :class:`~numpy.ndarray`
    
    If ``seq`` contains real or uncertain real numbers,
    a real number is returned.

    If ``seq`` contains complex or uncertain complex
    numbers, a complex number is returned.
    
    **Example**::

        >>> data = range(15)
        >>> type_a.mean(data)
        7.0
            
    """
    return value( type_b.mean(seq,*args,**kwargs) )
    
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

    # `type_a.mean` returns either a real or complex
    if isinstance(mu,numbers.Real):
        accum = lambda psum,x: psum + (value(x)-mu)**2
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
        raise RuntimeError(
            "unexpected type for mean value: {!r}".format(mu)
        )

#-----------------------------------------------------------------------------------------
def standard_uncertainty(seq,mu=None):
    """Return the standard uncertainty associated with the sample mean

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
    df = N-1
    
    zseq = value_seq(seq)
    
    if mu is None:
        mu = mean(zseq)         
        
    mu = complex( mu )
           
    accum_vr = lambda psum,z: psum + (z.real - mu.real)**2
    accum_vi = lambda psum,z: psum + (z.imag - mu.imag)**2
    accum_cv = lambda psum,z: psum + (z.imag - mu.imag)*(z.real - mu.real)
    
    cv_11 = reduce(accum_vr,zseq,0.0) / df 
    cv_22 = reduce(accum_vi,zseq,0.0) / df
    cv_12 = reduce(accum_cv,zseq,0.0) / df

    return VarianceCovariance(cv_11,cv_12,cv_12,cv_22)

#-----------------------------------------------------------------------------------------
def multi_estimate_real(seq_of_seq,labels=None):
    """Return a sequence of uncertain real numbers 

    :arg seq_of_seq: a sequence of sequences of data
    :arg labels: a sequence of `str` labels 
    
    :rtype: seq of :class:`~lib.UncertainReal`

    The sequences in ``seq_of_seq`` must all be the same length.
    Each sequence contains 
    a sample of data associated with a particular quantity. 
    An uncertain number will be created for the quantity
    from sample statistics. The covariance 
    between the different quantities will also be evaluated.
    
    A sequence of elementary uncertain numbers is returned. These uncertain numbers 
    are considered to be related, allowing a degrees-of-freedom calculations 
    to be performed on derived quantities. 

    **Example**::
    
        # From Appendix H2 in the GUM
        
        >>> V = [5.007,4.994,5.005,4.990,4.999]
        >>> I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        >>> phi = [1.0456,1.0438,1.0468,1.0428,1.0433]
        >>> v,i,p = type_a.multi_estimate_real((V,I,phi),labels=('V','I','phi'))
        >>> v
        ureal(4.999000...,0.0032093613071761...,4, label='V')
        >>> i
        ureal(0.019661,9.471008394041335...e-06,4, label='I')
        >>> p
        ureal(1.044460...,0.0007520638270785...,4, label='phi')
        
        >>> r = v/i*cos(p)
        >>> r
        ureal(127.732169928102...,0.071071407396995...,4.0)
        
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

        mu_i =  value( sum(seq_i) / N )
        means.append( mu_i )
        dev.append( tuple( value(x_j)-mu_i for x_j in seq_i ) )

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
def multi_estimate_complex(seq_of_seq,labels=None):
    """
    Return a sequence of uncertain complex numbers

    :arg seq_of_seq: a sequence of sequences of data
    :arg labels: a sequence of `str` labels
    
    :rtype: a sequence of :class:`~lib.UncertainComplex`
        
    The sequences in ``seq_of_seq`` must all be the same length.
    Each sequence contains data that is associated with 
    a particular quantity. An uncertain number for that quantity will  
    be created from sample statistics. The covariance 
    between the different quantities will also be evaluated.
    
    A sequence of elementary uncertain complex numbers is returned. These   
    uncertain numbers are considered to be related, allowing a degrees-of-freedom  
    calculations to be performed on derived quantities. 
    
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
        ureal(127.732169928102...,0.071071407396995...,4.0)
        >>> get_correlation(z.real,z.imag)
        -0.588429784423515...
        
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
        x.append( [ value(z_i.real) for z_i in seq_of_seq[i] ] )
        x.append( [ value(z_i.imag) for z_i in seq_of_seq[i] ] )
        if len(x[-1]) != N:
            raise RuntimeError(
                "{:d}th sequence length is incorrect".format(i)
            )

    TWOM = 2*M
    N_1 = N-1
    N_N_1 = N*N_1

    # 2. Evaluate the means and uncertainties (keep the deviation sequences)
    x_mean = [ value( math.fsum(seq_i) / N ) for seq_i in x ]
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
        
#--------------------------------------------------------------------
def merge(a,b,TOL=1E-13):
    """Combine the uncertainty components of ``a`` and ``b``

    :arg a: an uncertain real or complex number
    :arg b: an uncertain real or complex number
    :arg TOL: float

    :returns:   an uncertain number with the value of ``a`` and the 
                uncertainty components of ``a`` and ``b`` combined

    The absolute difference between the values of ``a`` and ``b`` 
    must be less than ``TOL`` and the components of uncertainty 
    associated with ``a`` and ``b`` must be distinct, otherwise
    a :class:`RuntimeError` will be raised.

    Use this function to combine results from
    type-A and type-B uncertainty analyses 
    performed on a common sequence of data.

    .. note::

        Some judgement will be required as to
        when it is appropriate to merge 
        uncertainty components.

        There is a risk of 'double-counting'
        uncertainty if type-B components
        are contributing to the variability
        observed in the data, and therefore
        assessed in a type-A analysis.

    .. versionchanged:: 1.3.3
       Added the `TOL` keyword argument.

    **Example**::

        # From Appendix H3 in the GUM
        
        # Thermometer readings (degrees C)
        t = (21.521,22.012,22.512,23.003,23.507,
            23.999,24.513,25.002,25.503,26.010,26.511)

        # Observed differences with calibration standard (degrees C)
        b = (-0.171,-0.169,-0.166,-0.159,-0.164,
            -0.165,-0.156,-0.157,-0.159,-0.161,-0.160)

        # Arbitrary offset temperature (degrees C)
        t_0 = 20.0
        
        # Calculate the temperature relative to t_0
        t_rel = [ t_k - t_0 for t_k in t ]

        # A common systematic error in all differences
        e_sys = ureal(0,0.01)
        
        b_type_b = [ b_k + e_sys for b_k in b ]

        # Type-A least-squares regression
        y_1_a, y_2_a = type_a.line_fit(t_rel,b_type_b).a_b

        # Type-B least-squares regression
        y_1_b, y_2_b = type_b.line_fit(t_rel,b_type_b)

        # `y_1` and `y_2` have uncertainty components  
        # related to the type-A analysis as well as the 
        # type-B systematic error
        y_1 = type_a.merge(y_1_a,y_1_b)
        y_2 = type_a.merge(y_2_a,y_2_b)

    """
    if abs( value(a) - value(b) ) > TOL:
        raise RuntimeError(
            "|a - b| = {} > {}: {!r} != {!r}".format(
                abs( value(a) - value(b) ),TOL,a,b
            )
        )
    else:
        return a + (b - value(b))
    
#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *    
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

    

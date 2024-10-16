"""
Sample estimates
----------------
    *   :func:`estimate` returns an uncertain number for the mean of a sample of real- or complex-valued data.
    *   :func:`multi_estimate_real` returns a sequence of related 
        uncertain numbers for the multivariate sample means of real-valued data. 
    *   :func:`multi_estimate_complex` returns a sequence of related uncertain complex
        numbers for the multivariate sample means for complex-valued data. 
    *   :func:`estimate_digitized` returns an uncertain number for the sample mean 
        of real-valued data quantised by rounding or truncation.
    *   :func:`mean` evaluates the sample mean (real- or complex-valued).
    *   :func:`standard_uncertainty` evaluates the standard 
        uncertainty associated with the mean of a sample of real- or complex-valued data.
    *   :func:`standard_deviation` evaluates the standard 
        deviation of data (real-valued or complex-valued).
    *   :func:`variance_covariance_complex` evaluates the variances
        and covariance associated with the sample mean of complex-valued data.
    
Least squares regression
------------------------
    *   :func:`line_fit` performs an ordinary least-squares straight 
        line fit to a sample of data.  
    *   :func:`line_fit_rwls` performs a weighted least-squares  
        straight-line fit to a sample of data. The weights
        are only assumed normalise the relative variability of observations.
    *   :func:`line_fit_wls` performs a weighted least-squares straight-line 
        fit to a sample of data. The weights are assumed to be exact.
    *   :func:`line_fit_wtls` performs a weighted total least-squares straight-line
        fit to a sample of data.   

    Fitting results are returned in objects related to the type of regression: 
    :class:`LineFitOLS`, :class:`LineFitRWLS`,
    :class:`LineFitWLS`, and :class:`LineFitWTLS`.
    
    These objects have attributes to access the results and define a few methods 
    that use regression results. 

Orthogonal distance regression
------------------------------
    *   :func:`line_fit_odr` performs an orthogonal distance regression straight 
        line fit to a sample of data. Results are returned in an :class:`LineFitODR` object.
        
Merging uncertain components
----------------------------
    *   :func:`merge` combines the uncertain-number results from a type-A and type-B 
        analysis of the same data. 
 
.. note::

    Most functions in :mod:`type_a` treat data as pure numbers. 
    Sequences of uncertain numbers can be passed to these 
    functions, but only the uncertain-number values are used.
    
    Some functions in the :mod:`type_b` module have the same names as those in :mod:`type_a`. 
    These functions take uncertain-number arguments and propagate uncertainty through 
    the same regression formulae. So, the values obtained will be the same 
    as when using a :mod:`type_a` function with the same name.
    However, the uncertainties will be different and have a different interpretation.

    :func:`merge` is provided to allow the results of type-A 
    and type-B analyses to be combined when it is appropriate. 
    

Module contents
---------------

"""
import sys
import math
import numbers

import numpy as np

from scipy.odr import ODR as spODR
from scipy.odr import Model as spModel
from scipy.odr import RealData as spRealData

from functools import reduce

from GTC import (
    inf,
    type_b,
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
from GTC.type_b import LineFit

from GTC.named_tuples import (
    VarianceCovariance,
    StandardUncertainty,
    StandardDeviation,
)

ureal = UncertainReal._elementary
ucomplex = UncertainComplex._elementary
result = lambda un,label: un._intermediate(label)

EPSILON = sys.float_info.epsilon 
HALF_PI = math.pi/2.0

__all__ = (
    'LineFitOLS','LineFitRWLS','LineFitWLS','LineFitWTLS', 'LineFitODR',
    'estimate',
    'estimate_digitized',
    'line_fit', 'line_fit_wls', 'line_fit_rwls', 'line_fit_wtls', 'line_fit_odr',
    'mean',
    'merge',
    'multi_estimate_real',
    'multi_estimate_complex',
    'standard_deviation',
    'standard_uncertainty',
    'variance_covariance_complex',
)

#-----------------------------------------------------------------------------------------
#
class LineFitOLS(LineFit):
    
    """
    Holds the results of an ordinary least-squares regression to a line.
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Type-A Ordinary Least-Squares Straight-Line:
'''
        return header + LineFit.__str__(self)

    def x_from_y(self,yseq,x_label=None,y_label=None):
        """Predict a stimulus value from the sequence of response values in ``yseq``

        :arg yseq: a sequence of response values 
        :arg x_label: a label for the uncertain-number stimulus 
        :arg y_label: a label for the mean response value.

        The predicted value is evaluated using the slope and intercept
        to transform the mean response value.
        
        The uncertainty in the predicted value is evaluated using the 
        sum of squared residuals (ssr). 
        The number of degrees of freedom associated with uncertainty in
        the predicted value is based on the regression sample size.
        
        .. note::
            The result is declared using :func:`~.result` if ``x_label`` is defined. 

        .. note::
            An uncertain number representing the mean of observations is labelled with ``y_label``. 
            This uncertain number is only used internally but the label may appear in uncertainty budgets.

        **Example** ::
        
            >>> x_data = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5,
            ...                 0.7, 0.7, 0.7, 0.9, 0.9, 0.9]
            >>> y_data = [0.028, 0.029, 0.029, 0.084, 0.083, 0.081, 0.135, 0.131,
            ...                 0.133, 0.180, 0.181, 0.183, 0.215, 0.230, 0.216]

            >>> fit = type_a.line_fit(x_data,y_data,label="ols")
            
            >>> x = fit.x_from_y( [0.0712, 0.0716], y_label="av_obs",x_label="x")            
            >>> print(f"{x.label}: {x}")
            x:  0.260(18)
            
            >>> for cpt in rp.budget(x):
            ...     print(f"{cpt.label}: {cpt.u}")
            ...
            av_obs: 0.016095175127594584
            a_ols: 0.01193650134308067
            b_ols: 0.005405932013155317

        """
        a, b = self._a_b
        df = a.df

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
        
        if abs(b) < 1E-15:
            # When b == 0, the best-fit line is horizontal 
            # so no use of new y data is made
            x = a
        else:
            x = (y - a)/b
            
        if x_label is not None:
            x = result( x, label=x_label )

        return x
        
    def y_from_x(self,x,s_label=None,y_label=None):
        """Predict the response to a stimulus 

        :arg x: stimulus value 
        :arg s_label: a label for the random error attributed to the response estimate 
        :arg y_label: a label for the uncertain-number response
        
        .. versionchanged:: 1.5.1       
            ``x`` must be a pure number
            
        The predicted value is evaluated using the slope and intercept 
        to transform the stimulus ``x``.
        
        The uncertainty in the predicted value is is evaluated using 
        the sum of squared residuals (ssr).
        The number of degrees of freedom associated with uncertainty in
        the predicted value is based on the regression sample size.

        An uncertain number representing the error due to variability in responses is created.         
        The standard uncertainty for this uncertain number
        is calculated using the sum of squared residuals. 
 
        .. note::
            The stimulus is considered a pure number.
            If an uncertain real number is supplied, its uncertainty is ignored.
        
        .. note::
            The result is declared using :func:`~.result` if ``y_label`` is defined.
 
        .. note::
            The uncertain number representing the error due to variability in responses is labelled with ``s_label``. 
            This uncertain number is used internally but the label may appear in uncertainty budgets.

            
        **Example** ::
        
            >>> x_data = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5,
            ...                 0.7, 0.7, 0.7, 0.9, 0.9, 0.9]
            >>> y_data = [0.028, 0.029, 0.029, 0.084, 0.083, 0.081, 0.135, 0.131,
            ...                 0.133, 0.180, 0.181, 0.183, 0.215, 0.230, 0.216]

            >>> fit = type_a.line_fit(x_data,y_data,label="ols")
            
            >>> x = ureal(0.26,0.10,label="x")
            >>> y = fit.y_from_x( 0.26, s_label="E_rnd",y_label="y")            
            >>> print(f"{y.label}: {y}")
            y:  0.0714(58)
            
            >>> for cpt in rp.budget(y):
            ...     print(f"{cpt.label}: {cpt.u}")
            ...
            E_rnd: 0.005485645603965661
            a_ols: 0.0028766968236824406
            b_ols: 0.0013019984639007856

        """
        a, b = self._a_b   
        df = a.df 
        
        u = math.sqrt( self._ssr/df )
        
        E_rnd = ureal(
            0,u,df,label=s_label,independent=False
        )
        
        append_real_ensemble(a,E_rnd)
                  
        if y_label is None:
            y = a + b*value(x) + E_rnd
        else:
            y = result( a + b*value(x) + E_rnd, label=y_label )
        
        return y

#-----------------------------------------------------------------------------------------
#
class LineFitRWLS(LineFit):
    
    """
    Holds the results of a relative weighted least-squares regression.
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Type-A Relative Weighted Least-Squares Straight-Line:
'''
        return header + LineFit.__str__(self)

    def x_from_y(self,yseq,s_y,x_label=None,y_label=None):
        """Predict a stimulus from the sequence of responses in ``yseq``

        :arg yseq: a sequence of response values
        :arg s_y: a scale factor for the variability of response values
        :arg x_label: a label for the uncertain-number stimulus estimate
        :arg y_label: a label for the mean response value

        .. note::
            The result is declared using :func:`~.result` if ``x_label`` is defined.

        .. note::
            An uncertain number representing the mean response is labelled with ``y_label``. 
            This uncertain number is used internally but the label may appear in uncertainty budgets.

        """
        a, b = self._a_b
        df = a.df 
        
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

        if abs(b) < 1E-15:
            # When b == 0, the best-fit line is horizontal 
            # so no use of new y data is made
            x = a
        else:
            x = (y - a)/b
            
        if x_label is not None:
            x = result( x, label=x_label )

        return x
        

    def y_from_x(self,x,s_y,s_label=None,y_label=None):
        """Predict the response to a stimulus

        :arg x: stimulus value
        :arg s_y: scale factor for variability in the response values
        :arg s_label: label for the random error attributed to the response estimate 
        :arg y_label: label for the uncertain-number response

        .. versionchanged:: 1.5.1                  

        Returns the response ``y`` predicted for a stimulus ``x``.

        It is assumed that the standard deviation of variability in ``y`` values is 
        proportional to ``s_y``.  
        
        .. note::
            When ``x`` is an uncertain number, the associated uncertainty 
            is propagated into the response.
        
        .. note::
            When ``y_label`` is defined, the uncertain number returned will be 
            declared an intermediate result (using :func:`~.result`)

        .. note::
            The variability of observations determined during regression is represented by an uncertain number
            labelled with ``s_label``. This uncertain number is used internally 
            but the label may appear in uncertainty budgets.

        """
        a, b = self._a_b   
        df = a.df 
        
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
    Holds the results of a weighted least-squares regression.
    
    Weight factors are assumed to characterise observation variability.
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Type-A Weighted Least-Squares Straight-Line:
'''
        return header + LineFit.__str__(self)

    def x_from_y(self,y_data,u_y_data,x_label=None,y_label=None):
        """Predict the stimulus corresponding to a sequence of responses

        :arg y_data: sequence of responses
        :arg u_y_data: standard uncertainties for the responses
        :arg x_label: label for the uncertain-number stimulus estimate
        :arg y_label: label for the mean response

        The variability in observations is assumed to be from independent random effects.
        
        .. note::
            The result is declared using :func:`~.result` if ``x_label`` is defined.

        .. note::
            An uncertain number representing the mean response is labelled with ``y_label``. 
            This uncertain number is used internally but the label may appear in uncertainty budgets.

        """
        a, b = self._a_b
        df = a.df
        
        p = len(y_data)
        y = math.fsum( y_data ) / p
        
        y = ureal(
            y,
            u_y_data / math.sqrt( p ),
            df,
            label=y_label,
            independent=False
        )            

        append_real_ensemble(a,y)

        if abs(b) < 1E-15:
            # When b == 0, the best-fit line is horizontal 
            # so no use of new y data is made
            x = a
        else:
            x = (y - a)/b
            
        if x_label is not None:
            x = result( x, label=x_label )

        return x

    def y_from_x(self,x,s_y,s_label=None,y_label=None):
        """Predict the response to a stimulus

        :arg x: stimulus
        :arg s_y: response uncertainty
        :arg s_label: label for the random error attributed to response variability  
        :arg y_label: label for the uncertain-number response

        .. versionchanged:: 1.5.1       

        Estimates the response ``y`` to a stimulus ``x``.

        It is assumed that the standard deviation of variability in ``y`` values 
        is proportional to ``s_y``.
        
        .. note::
            When ``x`` is an uncertain number, the associated uncertainty is propagated into ``y``.
        
        .. note::
            When ``y_label`` is defined, the uncertain number returned is 
            declared an intermediate result (using :func:`~.result`)

        """
        a, b = self._a_b   
        df = a.df 
        
        noise = ureal(0,s_y,df,label=s_label)

        append_real_ensemble(a,noise)
                  
        if y_label is None:
            y = a + b*x + noise
        else:
            y = result( a + b*x + noise, label=y_label )
        
        return y
 
#-----------------------------------------------------------------------------------------
class LineFitWTLS(LineFit):
    
    """
    This object holds results from a TLS linear regression to data.
    
    .. versionadded:: 1.2
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Type-A Weighted Total Least-Squares Straight-Line:
'''
        return header + LineFit.__str__(self)
 
#-----------------------------------------------------------------------------------------
class LineFitODR(LineFit):
    
    """
    This object holds results from orthogonal distance regression to data.
    
    .. versionadded:: 2.0
    """
    
    def __init__(self,a,b,ssr,N):
        LineFit.__init__(self,a,b,ssr,N)

    def __str__(self):
        header = '''
Type-A Orthogonal Distance Regression to a Straight-Line:
'''
        return header + LineFit.__str__(self)

#-----------------------------------------------------------------------------------------
def line_fit(x,y,label=None):
    """Least-squares straight-line fit
     
    :arg x:     sequence of stimulus data (independent-variable)  
    :arg y:     sequence of response data (dependent-variable)  
    :arg label: label suffix for slope and intercept

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
 
    .. versionadded:: 1.2    
 
    """
    N = len(x)
    df = N-2
    if df <= 0 or N != len(y):
        raise RuntimeError(
            f"Invalid sequences: len({x!r}), len({y!r})"
        )
    
    x = value_seq(x)
    y = value_seq(y)

    S_x = math.fsum( x )
    S_y = math.fsum( y )
        
    k = S_x / N
    t = [ (x_i - k) for x_i in x ]

    S_tt =  math.fsum( t_i*t_i for t_i in t )

    b_ =  math.fsum( t_i*y_i/S_tt for t_i,y_i in zip(t,y) )
    a_ = (S_y - b_*S_x)/N

    siga = math.sqrt( (1.0 + S_x*S_x/(N*S_tt))/N )
    sigb = math.sqrt( 1.0/S_tt )
    r_ab = -S_x/(N*S_tt*siga*sigb)
    
    # Sum of squared residuals needed to correctly calculate parameter uncertainties
    f = lambda x_i,y_i: (y_i - a_ - b_*x_i)**2 
    ssr =  math.fsum( f(x_i,y_i) for x_i,y_i in zip(x,y) )

    data_u = math.sqrt( ssr/df )
    siga *= data_u
    sigb *= data_u
            
    a = ureal(
        a_,
        siga,
        df=df,
        label=f'a_{label}' if label is not None else None,
        independent=False
    )
    b = ureal(
        b_,
        sigb,
        df=df,
        label=f'b_{label}' if label is not None else None,
        independent=False
    )
    
    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitOLS(a,b,ssr,N)

#-----------------------------------------------------------------------------------------
def _line_fit_wls(x,y,u_y):
    """Utility function
    
    All sequences contain pure numbers
    """
    N = len(x)

    v = [ u_y_i*u_y_i for u_y_i in u_y ]
    S =  math.fsum( 1.0/v_i for v_i in v)

    S_x =  math.fsum( x_i/v_i for x_i,v_i in zip(x,v) )
    S_y =  math.fsum( y_i/v_i for y_i,v_i in zip(y,v) )

    k = S_x / S
    t = [ (x_i - k)/u_y_i for x_i,u_y_i in zip(x,u_y) ]

    S_tt =  math.fsum( t_i*t_i for t_i in t )

    b_ =  math.fsum( t_i*y_i/u_y_i/S_tt for t_i,y_i,u_y_i in zip(t,y,u_y) )
    a_ = (S_y - b_*S_x)/S

    siga = math.sqrt( (1.0 + S_x*S_x/(S*S_tt))/S )
    sigb = math.sqrt( 1.0/S_tt )
    r_ab = -S_x/(S*S_tt*siga*sigb)
    
    f = lambda x_i,y_i,u_y_i: ((y_i - a_ - b_*x_i)/u_y_i)**2 
    ssr =  math.fsum( f(x_i,y_i,u_y_i) for x_i,y_i,u_y_i in zip(x,y,u_y) )

    return a_,b_,siga,sigb,r_ab,ssr,N

#-----------------------------------------------------------------------------------------
def line_fit_wls(x,y,u_y,dof=None,label=None):
    """Return a weighted least-squares straight-line fit
    
    :arg x:     sequence of stimulus data (independent-variable)  
    :arg y:     sequence of response data (dependent-variable)  
    :arg u_y:   sequence of uncertainties in the response data 
    :arg dof:   degrees of freedom
    :arg label: label suffix for slope and intercept

    :rtype:     :class:`.LineFitWLS`
    
    The variability in each observation is characterised by 
    values in ``u_y`` (i.e., infinite degrees of freedom).
    
    However, the optional argument ``dof`` can be used to attribute a 
    finite number of degrees of freedom.  

    **Example**::
    
        >>> x = [1,2,3,4,5,6]
        >>> y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
        >>> u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        
        >>> fit = type_a.line_fit_wls(x,y,u_y)
        >>> fit.a_b     
         InterceptSlope(a=ureal(0.8852320675105...,0.5297081435088...,inf),
         b=ureal(2.056962025316...,0.177892016741...,inf))

    .. versionchanged:: 1.4.1 ``dof`` keyword argument added
    .. versionadded:: 1.2
     
    """
    N = len(x)
    if N-2 <= 0 or N != len(y) or N != len(u_y):
        raise RuntimeError(
            f"Invalid sequences: len({x!r}), len({y!r}), len({u_y!r})"
        )
        
    x = value_seq(x)
    y = value_seq(y)

    a_,b_,siga,sigb,r_ab,ssr,N = _line_fit_wls(x,y,u_y)
    
    if dof is None:
        df = inf 
    else:
        if isinstance(dof,numbers.Number) and dof > 0:
            df = dof
        else:
            raise RuntimeError( 
                f"{dof!r} is an invalid degrees of freedom" 
            )
    
    a = ureal(
        a_,
        siga,
        df,
        label=f'a_{label}' if label is not None else None,
        independent=False
    )
    b = ureal(
        b_,
        sigb,
        df,
        label=f'b_{label}' if label is not None else None,
        independent=False
    )    

    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitWLS(a,b,ssr,N)

#-----------------------------------------------------------------------------------------
def line_fit_rwls(x,y,s_y,dof=None,label=None):
    """Return a relative weighted least-squares straight-line fit
    
    The sequence ``s_y`` contains scale factors for the ``y`` data.
    The standard deviation in the variability of each ``y`` value is 
    proportional to the corresponding ``s_y`` scale factor.
    The unknown common factor in the standard deviations is estimated from the residuals.
    
    :arg x:     sequence of stimulus data (independent-variable)  
    :arg y:     sequence of response data (dependent-variable)  
    :arg s_y:   sequence of scale factors for response data
    :arg dof:   degrees of freedom
    :arg label: label suffix for slope and intercept

    :rtype:     :class:`.LineFitRWLS`

    The variability in each observation is known 
    up to a common scale factor.     
    Residuals are used to estimate this common factor,
    which is then associated with N - 2 degrees of freedom. 
    
    The optional argument ``dof`` can be used to change the default 
    number of degrees of freedom. 
    
    Adjustment of degrees of freedom may be appropriate, because the
    degrees of freedom will tend to be high when the elements of ``s_y`` 
    are not all equal. 
    
    **Example**::

        >>> x = [1,2,3,4,5,6]
        >>> y = [3.014,5.225,7.004,9.061,11.201,12.762]
        >>> s_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        >>> fit = type_a.line_fit_rwls(x,y,s_y)
        >>> a, b = fit.a_b
        >>>
        >>> print(fit)
        <BLANKLINE>
        Type-A Relative Weighted Least-Squares Straight-Line:
        <BLANKLINE>
          Intercept: 1.14(12)
          Slope: 1.973(41)
          Correlation: -0.87
          Sum of the squared residuals: 1.3395217958...
          Number of points: 6
        <BLANKLINE>
 
    .. versionchanged:: 1.4.1 ``dof`` keyword argument added
    .. versionadded:: 1.2
        
    """
    N = len(x)
    if dof is None:
        df = N-2 
    else:
        if isinstance(dof,numbers.Number) and dof > 0:
            df = dof
        else:
            raise RuntimeError( 
                f"{dof!r} is an invalid degrees of freedom" 
            )
    
    if N != len(y) or N != len(s_y):
        raise RuntimeError(
            f"Invalid sequences: len({x!r}), len({y!r}), len({s_y!r})"
        )
        
    x = value_seq(x)
    y = value_seq(y)
    
    a_,b_,siga,sigb,r_ab,ssr,N = _line_fit_wls(x,y,s_y)

    # The sample estimate of sigma is incorporated in the
    # standard uncertainties of the fitted parameters
    sigma_hat = math.sqrt(ssr/df)
    siga *= sigma_hat
    sigb *= sigma_hat
    
    a = ureal(
        a_,
        siga,
        df,
        label=f'a_{label}' if label is not None else None,
        independent=False
    )
    b = ureal(
        b_,
        sigb,
        df,
        label=f'b_{label}' if label is not None else None,
        independent=False
    )
    
    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitRWLS(a,b,ssr,N)
    
#--------------------------------------------------------------------
#
def line_fit_wtls(x,y,u_x,u_y,a0_b0=None,r_xy=None,dof=None,label=None):
    """Return a total least-squares straight-line fit 
    
    :arg x:     sequence of stimulus data (independent-variable)
    :arg y:     sequence of response data (dependent-variable) 
    :arg u_x:   sequence of uncertainties in stimulus data
    :arg u_y:   sequence of uncertainties in response data
    :arg a0_b0: initial estimates for intercept and slope
    :arg r_xy:  correlation coefficients between stimulus-response pairs
    :arg dof:   degrees of freedom
    :arg label: label suffix for intercept and slope

    :rtype:     :class:`.LineFitWTLS`

    The optional argument ``a_b`` provides initial estimates for fitting. 
    
    By default, the degrees of freedom are infinite, 
    because weighting is provided for the stimulus data
    and the response data, suggesting that the amounts of variability are known.  
    However, the optional argument ``dof`` can be used to adjust the number of 
    degrees of freedom attributed to regression results. 

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
        ureal(5.47991018...,0.29193349...,inf)
        >>> slope
        ureal(-0.48053339...,0.057616740...,inf)

    .. versionchanged:: 1.4.1 ``dof`` keyword argument added
    .. versionadded:: 1.2    

    """
    N = len(x)
    if dof is None:
        df = inf 
    else:
        if isinstance(dof,numbers.Number) and dof > 0:
            df = dof
        else:
            raise RuntimeError( 
                f"{dof!r} is an invalid degrees of freedom" 
            )
    
    if N != len(y):
        raise RuntimeError(
            f"Invalid sequences: len({x!r}), len({y!r})"
        )
    if N != len(u_x) or N != len(u_y):
        raise RuntimeError(
            f"Invalid sequences: len({u_x!r}), len({u_y!r})"
        )

    independent = r_xy is None

    x_u = [ ureal( value(x_i),u_i,inf,None,independent=independent)
        for x_i, u_i in zip(x,u_x)
    ]
    y_u = [ ureal( value(y_i),u_i,inf,None,independent=independent)
        for y_i, u_i in zip(y,u_y)
    ]
    if not independent:
        for x_i,y_i,r_i in zip(x_u,y_u,r_xy):
            x_i.set_correlation(r_i,y_i)

    result = type_b.line_fit_wtls(x_u,y_u,a_b=a0_b0)

    a, b = result.a_b
    N = result.N
    ssr = result.ssr
    r_ab = a.get_correlation(b)
    
    a = ureal(
        a.x,
        a.u,
        df,
        label=f'a_{label}' if label is not None else None,
        independent=False
    )
    b = ureal(
        b.x,
        b.u,
        df,
        label=f'b_{label}' if label is not None else None,
        independent=False
    )

    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitWTLS(a,b,ssr,N)
 
#--------------------------------------------------------------------
#
def _ols(x,y):
    """
    A utility function
    """
    S_x = sum( x ) 
    S_y = sum( y )

    k = S_x / N
    t = [ x_i - k for x_i in x ]

    S_tt = sum( t_i*t_i for t_i in t )
    
    b = sum( t_i*y_i/S_tt for t_i,y_i in zip(t,y) )
    a = (S_y - b*S_x)/N
    
    return a,b

def line_fit_odr(x,y,u_x,u_y,a0_b0=[0.,1.],dof=None,label=None):
    """Return an orthogonal distance regression straight-line fit 
    
    :arg x:     sequence of stimulus data (independent-variable)
    :arg y:     sequence of response data (dependent-variable) 
    :arg u_x:   sequence of uncertainties in stimulus data
    :arg u_y:   sequence of uncertainties in response data
    :arg a0_b0: initial estimates for intercept and slope
    :arg dof:   degrees of freedom
    :arg label: label suffix for intercept and slope

    :rtype:     :class:`.LineFitODR`

    The optional argument ``a0_b0`` provides initial estimates for fitting.
    If it is not supplied and ordinary least-squares estimate is evaluated.
    
    By default, the degrees of freedom are infinite, 
    because weighting is provided for the stimulus data
    and the response data, suggesting that the amounts of variability are known.  
    However, the optional argument ``dof`` can be used to adjust the number of 
    degrees of freedom attributed to regression results. 
    
    **Example**::
        >>> x = [1.2, 1.9, 2.9, 4.0, 4.7, 5.9]
        >>> u_x = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        >>> y = [3.4, 4.4, 7.2, 8.5, 10.8, 13.5]
        >>> u_y = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4]
        <BLANKLINE>        
        >>> fit = ta.line_fit_odr(x,y,u_x,u_y)
        >>> print(fit.a_b)
        InterceptSlope(a=ureal(0.5788...,0.4764...,inf), b=ureal(2.1596...,0.1355...,inf))

    .. versionadded:: 2.0    

    """
    N = len(x)
    if dof is None:
        df = inf 
    else:
        if isinstance(dof,numbers.Number) and dof > 0:
            df = dof
        else:
            raise RuntimeError( 
                f"{dof!r} is an invalid degrees of freedom" 
            )
    
    if N != len(y):
        raise RuntimeError(
            f"Invalid sequences: len({x!r}), len({y!r})"
        )
    if N != len(u_x) or N != len(u_y):
        raise RuntimeError(
            f"Invalid sequences: len({u_x!r}), len({u_y!r})"
        )

    # Use numpy arrays
    data = spRealData(
        np.array(x), 
        np.array(y), 
        sx=np.array(u_x), 
        sy=np.array(u_y)
    )

    result = spODR(
        data, 
        spModel(lambda p,x: p[0] + x * p[1]), 
        beta0=_ols(x,y) if a0_b0 is None else a0_b0
    ).run()


    x_a, x_b = result.beta
    ssr = result.sum_square
    u_a, u_b = np.sqrt(np.diag(result.cov_beta))
    r_ab = result.cov_beta[0,1]/(u_a*u_b)
    
    a = ureal(
        x_a,
        u_a,
        df,
        label=f'a_{label}' if label is not None else None,
        independent=False
    )
    b = ureal(
        x_b,
        u_b,
        df,
        label=f'b_{label}' if label is not None else None,
        independent=False
    )

    real_ensemble( (a,b), df )
    a.set_correlation(r_ab,b)

    return LineFitODR(a,b,ssr,N) 
    
#-----------------------------------------------------------------------------------------
def estimate_digitized(seq,delta,label=None,truncate=False):
    """
    Return an uncertain number for the mean of digitized data in ``seq``

    :arg seq: data
    :type seq: float, :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`
    :arg float delta: digitization step size 
    :arg str label: label for the returned uncertain number 
    :arg bool truncate: ``True`` when data were truncated rather than rounded
    :rtype: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`

    A sequence of data rounded or truncated to a fixed precision  
    can conceal or obscure small amounts of variability.  
    
    This function evaluates the mean and the associated uncertainty
    while taking fixed precision effects into account.   
            
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
    """Return an uncertain number for the mean of ``seq``

    :arg seq: a sequence of data
    :arg str label: a label for the returned uncertain number
    
    :rtype:   :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`
                
    The elements of ``seq`` may be real numbers, complex numbers. 
    If uncertain real or uncertain complex numbers are supplied in ``seq``
    only the values will be used.
    
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
        ucomplex((1.059187840567141+0.9574410497332932j), u=[0.28881665310241805,0.2655555630050262], r=-0.3137404512459525, df=9)

    """
    df = len(seq)-1
    if 0 >= df:
        raise RuntimeError(
            f"require: 0 >= len({seq!r})"
        )
        
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
    
    If ``seq`` contains real or uncertain real numbers, a real number is returned.

    If ``seq`` contains complex or uncertain complex
    numbers, a complex number is returned.
    
    **Example**::

        >>> data = range(15)
        >>> type_a.mean(data)
        7.0
            
    .. note::
        When ``seq`` is an empty :class:`~numpy.ndarray` or 
        a :class:`~numpy.ndarray` containing any ``NaN`` elements
        ``NaN`` is returned. 
        
        In other cases, a :class:`ZeroDivisionError` is raised when there are no elements in ``seq``.

    """
    return value( type_b.mean(seq,*args,**kwargs) )
    
#-----------------------------------------------------------------------------------------
def standard_deviation(seq,mu=None):
    """Return the sample standard deviation of data in ``seq``
    
    :arg seq: sequence of data
    :arg mu: the arithmetic mean of ``seq``

    If ``mu`` is ``None`` the mean will be evaluated by :func:`~type_a.mean`.

    If ``seq`` contains real or uncertain real numbers, 
    the sample standard deviation is returned.
    
    If ``seq`` contains complex or uncertain complex
    numbers, the standard deviation in the real and
    imaginary components is evaluated, as well as
    the correlation coefficient between the components.
    The results are returned in a pair of objects: a
    :obj:`~named_tuples.StandardDeviation` namedtuple 
    and a correlation coefficient. 

    If ``seq`` contains uncertain numbers, only the values are used. 
    
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
    if N == 0:
        raise RuntimeError(
            f"empty sequence: {seq!r}"
        )
    
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
            f"unexpected type for mean value: {mu!r}"
        )

#-----------------------------------------------------------------------------------------
def standard_uncertainty(seq,mu=None):
    """Return the standard uncertainty associated with the sample mean

    :arg seq: sequence of data
    :arg mu: the arithmetic mean of ``seq``
    
    :rtype: float or :obj:`~named_tuples.StandardUncertainty`
    
    If ``mu`` is ``None`` the mean will be evaluated by :func:`~type_a.mean`.

    If ``seq`` contains real or uncertain real numbers,
    the standard uncertainty of the sample mean 
    is returned.

    If ``seq`` contains complex or uncertain complex
    numbers, the standard uncertainties of the real and
    imaginary components are evaluated, as well as the
    sample correlation coefficient are returned in a
    :obj:`~named_tuples.StandardUncertainty` namedtuple

    If ``seq`` contains uncertain numbers, only the values are used. 

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
    N = len(seq)
    if N == 0:
        raise RuntimeError(
            f"empty sequence: {seq!r}"
        )

    ROOT_N = math.sqrt(N)
    
    if mu is None:
        mu = mean(seq)

    if isinstance(mu,numbers.Real):
        sd = standard_deviation(seq,mu)
        return sd / ROOT_N  
        
    elif isinstance(mu,numbers.Complex):
        sd,r = standard_deviation(seq,mu)
        return StandardUncertainty(sd.real/ROOT_N,sd.imag/ROOT_N),r
        
    else:
        assert False, f"unexpected, mu={mu!r}"

#-----------------------------------------------------------------------------------------
def variance_covariance_complex(seq,mu=None):
    """Return the sample variance-covariance matrix

    :arg seq: sequence of data   
    :arg mu: the arithmetic mean of ``seq``

    :returns: a 4-element sequence

    If ``mu`` is ``None`` the mean will be evaluated by :func:`~type_a.mean`.

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
    df = len(seq)-1
    if 0 >= df:
        raise RuntimeError(
            f"require: 0 >= len({seq!r})"
        )
    
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

    :arg seq_of_seq: a sequence of data sequences  
    :arg labels: a sequence of labels 
    
    :rtype: seq of :class:`~lib.UncertainReal`

    Sequences in ``seq_of_seq`` must all be the same length.
    Each sequence contains a sample of data. 
    An uncertain number will be created from sample statistics for each sequence. 
    The covariance between the sample means is evaluated.
    
    A sequence of elementary uncertain numbers is returned. These 
    are considered to be 'related', allowing a degrees-of-freedom calculations 
    to be performed on derived quantities. 

    .. note::
        The term 'related' here means the data are considered samples
        from a multivariate distribution with a certain covariance structure.    
        This assumption, and the fact that the sample sizes are equal,
        enables downstream effective degrees of freedom calculation even 
        when sample means are correlated.
        
    **Example**::
    
        # From Appendix H2 in the GUM
        
        >>> V = [5.007,4.994,5.005,4.990,4.999]
        >>> I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        >>> phi = [1.0456,1.0438,1.0468,1.0428,1.0433]
        >>> v,i,p = type_a.multi_estimate_real((V,I,phi),labels=('V','I','phi'))
        >>> v
        ureal(4.999...,0.0032093613071761...,4, label='V')
        >>> i
        ureal(0.019661,9.471008394041335...e-06,4, label='I')
        >>> p
        ureal(1.04446...,0.0007520638270785...,4, label='phi')
        
        >>> r = v/i*cos(p)
        >>> r
        ureal(127.732169928102...,0.071071407396995...,4.0)
        
    """
    M = len(seq_of_seq)
    N = len(seq_of_seq[0])
    
    if labels is not None and len(labels) != M:
        raise RuntimeError(
            f"Incorrect number of labels: '{labels!r}'" 
        ) 
        
    # Calculate the deviations from the mean for each sequence
    means = [ ]
    dev = []
    for i,seq_i in enumerate(seq_of_seq):
        if len(seq_i) != N:
            raise RuntimeError( f"{i:d}th sequence length inconsistent" )

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
                        for d_i,d_j in zip(seq_i,seq_j)
                )/N_N_1
            )

    # Create a list of elementary uncertain numbers
    # to return a list of standard uncertainties
    # to normalise the CV matrix.
    df = N-1
    rtn = []
    for i in range(M):
        mu_i = means[i]
        u_i = u[i]
        l_i = labels[i] if labels is not None else ""
        rtn.append( ureal(mu_i,u_i,df,l_i,independent=False) )

    # Create the list of ensemble id's,
    # assign it to the register in the context,
    # set the correlation between nodes
    real_ensemble( rtn, df )
    
    for i in range(M):
        u_i = u[i]
        un_i = rtn[i]
        
        for j in range(M-1-i):
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

    :arg seq_of_seq: a sequence of data sequences 
    :arg labels: a sequence of labels
    
    :rtype: a sequence of :class:`~lib.UncertainComplex`
        
    Sequences in ``seq_of_seq`` must all be the same length.
    Each sequence contains a sample of data. 
    An uncertain number is created using the sample statistics for each sequence. 
    The covariance between the sample means is evaluated.
    
    A sequence of elementary uncertain complex numbers is returned. 
    These are considered to be ‘related’, allowing a degrees-of-freedom 
    calculations to be performed on derived quantities.

    .. note::
        The term 'related' here means the data are considered samples
        from a multivariate distribution with a certain covariance structure.    
        This assumption, and the fact that the sample sizes are equal,
        enables downstream effective degrees of freedom calculation even 
        when sample means are correlated.

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
            f"Incorrect number of labels: '{labels!r}'" 
        ) 

    # 1. Create a 2M sequence of sequences of real values
    x = []
    for i in range(M):
        x.append( [ value(z_i.real) for z_i in seq_of_seq[i] ] )
        x.append( [ value(z_i.imag) for z_i in seq_of_seq[i] ] )
        if len(x[-1]) != N:
            raise RuntimeError(
                f"{i:d}th sequence length is incorrect"
            )

    TWOM = 2*M
    N_1 = N-1
    N_N_1 = N*N_1

    # 2. Evaluate the means and uncertainties (keep the deviation sequences)
    x_mean = [ value( math.fsum(seq_i) / N ) for seq_i in x ]
    x_u = []
    for i in range(TWOM):
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
    for i in range(M):
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
    for i in range(TWOM-1):
        x_i = x[i]
        un_i = x_influences[i]
        for j in range(i+1,TWOM):
            x_j = x[j]
            cv = math.fsum( 
                d_i*d_j for d_i,d_j in zip(x_i,x_j)
            )/N_N_1
            if cv != 0.0:
                r = cv/(x_u[i]*x_u[j]) 
                set_correlation_real(un_i,x_influences[j],r)

    complex_ensemble( rtn, N_1 )
    
    return rtn
        
#--------------------------------------------------------------------
def merge(a,b,TOL=1E-13):
    """Combine two uncertain numbers with the same value

    :arg a: an uncertain real or uncertain complex number
    :arg b: an uncertain real or uncertain complex number
    :arg TOL: float

    :returns:   an uncertain number having the value of ``a`` and the 
                combined uncertainty components of ``a`` and ``b``

    Use this function to combine results from
    type-A and type-B uncertainty analyses 
    performed on a common sequence of data.

    The absolute difference between the values of ``a`` and ``b`` 
    must be less than ``TOL`` and the components of uncertainty 
    associated with ``a`` and ``b`` must be distinct, otherwise
    a :class:`RuntimeError` is raised.

    .. note::

        Some judgement is required as to
        when it is appropriate to use :func:`merge`. 

        There is a risk of 'double-counting'
        uncertainty components if type-B components
        are also contributing to the variability
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
            f"|a - b| = {abs(value(a) - value(b))} > {TOL}: {a!r} != {b!r}"
        )
    else:
        return a + (b - value(b))
    
#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *    
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

    

.. _regression_functions:

=============================================
Some comments about ``GTC`` fitting functions
=============================================

.. contents::
   :local:

Overview
========
   
``GTC`` provides straight-line regression functions in both the :mod:`type_a` and :mod:`type_b` modules. 

Functions in :mod:`type_a` implement a variety of least-squares straight-line fitting algorithms that provide results in the form of uncertain numbers. When used, the input data (sequences ``x`` and ``y``) are treated as pure numbers (if sequences of uncertain numbers are provided, only the values are used in calculations).

Functions defined in :mod:`type_b`, on the other hand, expect input sequences of uncertain numbers. These functions estimate the slope and intercept of a line by applying the same type of regression algorithms, but uncertainties are propagated through the equations using the GUM LPU and esiduals are ignored.  

The distinction between functions that evaluate the uncertainty of estimates from residuals (:mod:`type_a`) and functions that evaluate uncertainty using uncertain numbers (:mod:`type_b`) is useful. There will be circumstances that require the use of a function in :mod:`type_b`, such as when systematic errors contribute to uncertainty but cannot be estimated properly using conventional type-A regression. Without the methods available in :mod:`type_b`, such components of uncertainty could not be propagated. On the other hand, functions in :mod:`type_a` implement conventional statistical regression methods.

Discretion will be needed if it is believed that variability in a sample of data is due, in part, to errors not fully accounted for in an uncertain-number description of the data. The question is then: just how much variability can be explained by components of uncertainty already defined as uncertain number influences? If the answer is 'very little' then it will be appropriate to use a function from :mod:`type_a` to estimate the additional contribution to uncertainty from the observable variability. At the same time, components of uncertainty associated with the uncertain-number data should be propagated using a function from :mod:`type_b` for the same type of regression. The two results will be identical (the estimates of the slope and intercept will be the same) but the uncertainties will differ. :func:`type_a.merge` can then be used to merge the results. 

Clearly, this approach could over-estimate the importance of some influences and inflate the uncertainty of results. It is a matter of judgement as to whether to merge type-A and type-B results in a particular procedure. 

The ``type_a`` module regression functions
==========================================

Ordinary least-squares
----------------------

:func:`type_a.line_fit` implements a conventional ordinary least-squares straight-line regression. The residuals are used to estimate the underlying variance of the `y` data. The resulting uncertain numbers for the slope and intercept have finite degrees of freedom and are usually correlated.

Weighted least-squares
----------------------

:func:`type_a.line_fit_wls` implements a so-called weighted least-squares straight-line regression. Weighting implies that variability in observations of input data is known exactly (i.e., infinite degrees of freedom). The uncertainties in the slope and intercept are therefore calculated without considering residuals.

This approach to linear regression is described in two well-known references [#Bevington]_ [#NR]_ , but it may not be what many statisticians associate with the term 'weighted least-squares'.

Relative weighted least-squares
-------------------------------

:func:`type_a.line_fit_rwls` implements a form of weighted least-squares straight-line regression that we refer to as 'relative weighted least-squares'. (Statisticians may regard this as conventional weighted least-squares.)

:func:`type_a.line_fit_rwls` takes a sequence of numbers associated with the observations, which are used as weighting factors. For an observation :math:`y`, it is assumed that the uncertainty :math:`u(y) = \sigma\cdot s_y`, where :math:`\sigma` is an unknown common factor and :math:`s_y` is the weighting value provided.  
 
The procedure estimates :math:`\sigma` from the residuals, so the results for slope and intercept have finite degrees of freedom. 

Note, because the relative weighting of different observations is specified, the ordinary least-squares function :func:`type_a.line_fit` and :func:`type_a.line_fit_rwls` would return equivalent results if all `y` observations were given the same weighting.

Weighted total least-squares
----------------------------

:func:`type_a.line_fit_wtls` implements a form of least-squares straight-line regression that takes account of errors in both the stimulus and response data [#Krystek]_.

As in :func:`type_a.line_fit_wls`, the sequences of uncertainties provided for the `x` and `y` data are assumed exact. 
When calculating uncertainties in the slope and intercept, residuals are ignored and the uncertain numbers returned have infinite degrees of freedom.

Degrees of freedom
------------------
In the GUM, the number of `degrees of freedom` is used as a measure of how accurately a standard uncertainty is known. This relates to the standard treatment of sample data in classical statistics. 

The methods of straight-line regression in GTC are based on theory for Gaussian errors. This is often be a good approximation in metrology. However, only the simple linear fit to data (:func:`type_a.line_fit`) has a clear prescription for evaluating degrees of freedom (in classical statistics). For other types of fit, additional information is incorporated so uncertainties in parameter estimates depend on more than just the number of observations. Also, no allowance is given for the accuracy of the model assumed (the choice of a straight line). 

For these reasons, the default values of degrees of freedom attributed to the uncertain numbers returned by GTC regression functions may be overridden. See the respective function docstrings for details. 


The ``type_b`` module regression functions
==========================================

Ordinary least-squares
----------------------
:func:`type_b.line_fit` estimates the slope and intercept of a line through the data using conventional ordinary least-squares straight-line regression. The response data is a sequence of uncertain numbers. The uncertainty of the fitted parameters is found by propagating uncertainty in the response data---residuals are ignored.

Weighted least-squares
----------------------
:func:`type_b.line_fit_wls` estimates the slope and intercept of a line through the data using a weighted least-squares straight-line regression. The response data is a sequence of uncertain numbers.  By default, the uncertainty of each response is used as a weighting factor for regression. However, an optional sequence of values for uncertainty may be supplied to weight the response data.  
In either case, uncertainty in the estimates of slope and intercept is obtained by propagating the uncertainty associated with the input data through the regression equations (residuals are ignored).

.. note::

    :func:`type_a.line_fit_wls` and :func:`type_b.line_fit_wls` yield the same result when a sequence of elementary uncertain numbers is defined for `y` and used with :func:`type_a.line_fit_wls` and the values and uncertainties of that sequence are used with :func:`type_a.line_fit_wls`.

.. note::

    There is no need for a 'relative weighted least-squares' function in the :mod:`type_b` module. Using a sequence of ``u_y`` values with :func:`type_b.line_fit_wls` will perform this calculation.

Weighted total least-squares
----------------------------

:func:`type_b.line_fit_wtls` implements a form of least-squares straight-line regression that takes account of errors in both the `x` and `y` data. [#Krystek]_.

By default, the input data uncertainties are used as weights during the regression. However, as with :func:`type_b.line_fit_wls`, sequences of uncertainty values may be supplied for stimulus and response data. In either case, uncertainty in the estimates of slope and intercept is calculated by propagating uncertainty from the input data through the regression equations (residuals are ignored).

.. rubric:: Footnotes

.. [#Bevington] Philip Bevington and D. Keith Robinson, *Data Reduction and Error Analysis for the Physical Sciences*
.. [#NR] William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, *Numerical Recipes: The Art of Scientific Computing*
.. [#Krystek] M Krystek and M Anton, Meas. Sci. Technol. 22 (2011) 035101 (9pp)
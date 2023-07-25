.. _regression_functions:

=============================================
Some comments about ``GTC`` fitting functions
=============================================

.. contents::
   :local:

Overview
========
   
``GTC`` has straight-line regression functions in both the :mod:`type_a` and :mod:`type_b` modules. 

Functions in :mod:`type_a` implement a variety of least-squares straight line fitting algorithms that provide results in the form of uncertain numbers. When used, the input data (the sequences ``x`` and ``y``) are treated as pure numbers (if sequences of uncertain numbers are provided, only the values are used in calculations).

Functions defined in :mod:`type_b`, on the other hand, expect input sequences of uncertain numbers. These functions estimate the slope and intercept of a line by applying the same type of regression algorithms, but uncertainties are propagated through the equations and the residuals are ignored.  

The distinction between functions that evaluate the uncertainty of estimates from residuals (:mod:`type_a`) and functions that evaluate uncertainty using uncertain numbers (:mod:`type_b`) is useful. There will be circumstances that require the use of a function in :mod:`type_b`, such as when systematic errors contribute to uncertainty but cannot be estimated properly using conventional regression. Without the methods available in :mod:`type_b`, such components of uncertainty could not be propagated. On the other hand, functions in :mod:`type_a` implement conventional regression methods.

Discretion will be needed if it is believed that variability in a sample of data is due, in part, to errors not fully accounted for in an uncertain-number description of the data. The question is then: just how much of that variability can be explained by components of uncertainty already defined as uncertain number influences? If the answer is 'very little' then it will be appropriate to use a function from :mod:`type_a` to estimate the additional contribution to uncertainty from the sample variability. At the same time, components of uncertainty associated with the uncertain-number data should be propagated using a function from :mod:`type_b` that performs the same type of regression. The two result values will be identical (the estimates of the slope and intercept will be the same) but the uncertainties will differ. :func:`type_a.merge` can then be used to merge the results. 

Clearly, this approach could potentially over-estimate the effect of some influences and inflate the combined uncertainty of results. It is a matter of judgement as to whether to merge type-A and type-B results in a particular procedure. 

The ``type_a`` module regression functions
==========================================

Ordinary least-squares
----------------------

:func:`type_a.line_fit` implements a conventional ordinary least-squares straight-line regression. The residuals are used to estimate the underlying variance of the `y` data. The resulting uncertain numbers for the slope and intercept have finite degrees of freedom and are correlated.

Weighted least-squares
----------------------

:func:`type_a.line_fit_wls` implements a so-called weighted least-squares straight-line regression. This assumes that the uncertainties provided with input data are known exactly (i.e., with infinite degrees of freedom). The uncertainties in the slope and intercept are calculated without considering the residuals.

This approach to linear regression is described in two well-known references [#Bevington]_ [#NR]_ , but it may not be what many statisticians associate with the term 'weighted least-squares'.

Relative weighted least-squares
-------------------------------

:func:`type_a.line_fit_rwls` implements a form of weighted least-squares straight-line regression that we refer to here as 'relative weighted least-squares'. (Statisticians may regard this as conventional weighted least-squares.)

:func:`type_a.line_fit_rwls` accepts a sequence of scale factors associated with the observations `y`, which are used as weighting factors. For an observation :math:`y`, it is assumed that the uncertainty :math:`u(y) = \sigma s_y`, where :math:`\sigma` is an unknown factor common to all the `y` data and :math:`s_y` is the weight factor provided.  
 
The procedure estimates :math:`\sigma` from the residuals, so the uncertain numbers returned for the slope and intercept have finite degrees of freedom. 

Note, because the scale factors describe the relative weighting of different observations, the ordinary least-squares function :func:`type_a.line_fit` and :func:`type_a.line_fit_rwls` would return equivalent results if all `y` observations are given the same weighting.

Weighted total least-squares
----------------------------

:func:`type_a.line_fit_wtls` implements a form of least-squares straight-line regression that takes account of errors in both the `x` and `y` data [#Krystek]_.

As in the case of :func:`type_a.line_fit_wls`, the uncertainties provided for the `x` and `y` data are assumed exact. When calculating the uncertainty in the slope and intercept, the residuals are ignored and the uncertain numbers returned have infinite degrees of freedom.

Degrees of freedom
------------------
The number of `degrees of freedom` is used as a measure of how accurately a standard uncertainty is known in the GUM. This relates to the standard treatment of sample statistics for a Gaussian error. 

The methods of straight-line regression in GTC are based on theory for Gaussian errors. This will often be a good approximation in metrology. However, only the simple linear fit to data (:func:`type_a.line_fit`) has a clear prescription for evaluating degrees of freedom. In other types of fit, additional information is incorporated in the form of weighting coefficients. So, estimates of parameter uncertainty depend on more than just the number of observations. Also, no allowance is given for the accuracy of the model assumed (the choice of a straight line). 

For these reasons, default values of degrees of freedom are attributed to the uncertain numbers returned by regression functions, but these may be overridden. The function docstrings give more details. 


The ``type_b`` module regression functions
==========================================

Ordinary least-squares
----------------------
:func:`type_b.line_fit` implements the conventional ordinary least-squares straight-line regression to obtain estimates of the slope and intercept of a line through the data. The `y` data is a sequence of uncertain numbers. The uncertainty of the slope and intercept is found by propagating uncertainty from the input data; the residuals are ignored.

Weighted least-squares
----------------------
:func:`type_b.line_fit_wls` implements a weighted least-squares straight-line regression to estimate the slope and intercept of a line through the data. The `y` data is a sequence of uncertain numbers. An explicit sequence of uncertainties for the data points may also be provided. If so, these uncertainties are used as weights in the algorithm when estimating the slope and intercept. Otherwise, the uncertainty of each uncertain number for `y` is used. In either case, uncertainty in the estimates of slope and intercept is obtained by propagating the uncertainty associated with the input data through the estimate equations (the residuals are ignored).

.. note::

    :func:`type_a.line_fit_wls` and :func:`type_b.line_fit_wls` yield the same results when a sequence of elementary uncertain numbers is defined for `y` and used with :func:`type_a.line_fit_wls` and the values and uncertainties of that sequence are used with :func:`type_a.line_fit_wls`.

.. note::

    There is no need for a 'relative weighted least-squares' function in the :mod:`type_b` module. Using a sequence of ``u_y`` values with :func:`type_b.line_fit_wls` will perform this calculation.

Weighted total least-squares
----------------------------

:func:`type_b.line_fit_wtls` implements a form of least-squares straight-line regression that takes account of errors in both the `x` and `y` data. [#Krystek]_.

As with :func:`type_b.line_fit_wls`, sequences of uncertainties for the `x` and `y` data may be supplied in addition to sequences of the `x` and `y` data. When the optional uncertainty sequences are provided, estimates of the slope and intercept use those uncertainties as weights in the regression process. Otherwise, the input data uncertainties are used as weights in the regression process.  In either case, uncertainty in the estimates of slope and intercept is calculated by propagating uncertainty from the input data through the regression equations (residuals are ignored).

.. rubric:: Footnotes

.. [#Bevington] Philip Bevington and D. Keith Robinson, *Data Reduction and Error Analysis for the Physical Sciences*
.. [#NR] William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, *Numerical Recipes: The Art of Scientific Computing*
.. [#Krystek] M Krystek and M Anton, Meas. Sci. Technol. 22 (2011) 035101 (9pp)
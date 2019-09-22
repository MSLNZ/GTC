.. _linear_regression:

*************************
Linear Regression Results
*************************

.. contents::
   :local:

Conventional least-squares regression of a line to a set of data estimates the parameters of a linear model (slope and intercept) The best-fit line is sometimes called a *calibration line*.  

Example
=======

Linear regression is performed with ``x`` data, that are considered to be error-free stimuli, and ``y`` data that are observations subject to noise (random errors) ::

    >>> x = [3, 7, 11, 15, 18, 27, 29, 30, 30, 31, 31, 32, 33, 33, 34, 36, 
    ...     36, 36, 37, 38, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 50]
    >>> y = [5, 11, 21, 16, 16, 28, 27, 25, 35, 30, 40, 32, 34, 32, 34, 37, 
    ...     38, 34, 36, 38, 37, 36, 45, 39, 41, 40, 44, 37, 44, 46, 46, 49, 51]

    >>> fit = type_a.line_fit(x,y)
 
The object :class:`~type_a.LineFitOLS` (returned by :func:`type_a.line_fit`) contains the results of the regression and can be used in different ways. 

Estimates of the slope and intercept
------------------------------------
Least-squares regression assumes that a model of the system is

.. math::

    Y = \alpha + \beta \, x + E \;,
    
where :math:`\alpha` and :math:`\beta` are unknown parameters, :math:`E` is a random error with zero mean and unknown variance :math:`\sigma^2`, :math:`x` is the independent (stimulus) variable and :math:`Y` is the response. 

Least-squares regression returns the ``fit`` object, which holds uncertain numbers representing :math:`\alpha` and :math:`\beta`: ::

    >>> a = fit.intercept
    >>> a
    ureal(3.829633197588695, 1.7684473272506525, 31)
    >>> b = fit.slope
    >>> b
    ureal(0.9036432105793234, 0.050118973559182003, 31)
    >>> get_correlation(a,b)
    -0.9481240708919155
 
The response
------------
The uncertain numbers ``a`` and ``b`` can be used to estimate the response to a particular stimulus, say :math:`x = 21.5`, in the absence of noise::

    >>> y = a + 21.5 * b
    >>> y
    ureal(23.25796222504415, 0.8216070588885063, 31.0)
    
The result ``y`` is an estimate of :math:`\alpha + 21.5 \times \beta`. It is subject to uncertainty because the regression used a sample of data.

A predicted future response
---------------------------
A single future indication in response to a given stimulus may also be of interest. Again, say :math:`x = 21.5`, :: 

    >>> y0 = fit.y_from_x(21.5)
    >>> y0
    ureal(23.25796222504415, 3.3324092579571105, 31.0)

The value here is the same as above (because the stimulus is the same), but the uncertainty is much larger, reflecting the variability of single indications as well as the underlying uncertainty in the intercept and slope. 

Estimating the stimulus from observations of the response
---------------------------------------------------------
Another possibility is that several indications of the response to a steady stimulus are collected. This sample of data may be used to estimate the stimulus [#]_. 

Suppose three observations were collected ``[31.4, 29.3, 27.1]`` ::

    >>> x0 = fit.x_from_y( [31.4, 29.3, 27.1] )
    >>> x0
    ureal(28.149421332751846, 2.1751408733425195, 31.0)

``x0`` is an estimate of the stimulus based on the observations, but also taking into account the variability in the ``y`` data used earlier in the regression.

.. rubric:: Footnotes

.. [#] This scenario is sometimes called `calibration`. The response of an instrument to a number of different reference stimuli is observed and a calibration curve is calculated. The curve is then used in the opposite sense, to convert observations of the instrument response into estimates of the stimulus applied.
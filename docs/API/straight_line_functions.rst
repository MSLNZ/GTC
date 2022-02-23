.. _ISO_28037:

***********************************
Straight-line calibration functions
***********************************

.. contents::
   :local:

This section uses straight-line least-squares regression algorithms to obtain calibration functions and then shows some uses of those functions. Each example uses a small sample of *x-y* observation pairs for regression to obtain a calibration function, together with information about the variability of the data. No context is given about the measurement; these examples have been selected from a draft British standard on the use of straight-line calibration functions [#BSI]_. 

In some examples, we show how results can be used to estimate a stimulus value ``x``, when given additional observations of the response ``y``, or a future response ``y`` to a given stimulus ``x``.

Example 1: equal weights
========================

A series of six pairs of ``x-y`` observations have been collected. 

The data sequences ``x`` and ``y`` and a sequence of uncertainties in the :math:`y` values are ::

    x = [1,2,3,4,5,6]
    y = [3.3,5.6,7.1,9.3,10.7,12.1]
    u_y = [0.5] * 6

We apply weighted least-squares regression to the data, which assumes that the values in ``u_y`` are known standard deviations for noise in ``y`` data (i.e., the data have infinite degrees of freedom) ::

    fit = type_a.line_fit_wls(x,y,u_y)
    print(fit)

This displays ::

Weighted Least-Squares Results:

    Intercept:  1.87(47)
    Slope:  1.76(12)
    Correlation: -0.9
    Sum of the squared residuals: 1.664761904761908
    Number of points: 6   

More significant figures could be obtained with ::

    a = fit.intercept
    print( "a={:.15G}, u={:.15G}".format(value(a),uncertainty(a)) )
    b = fit.slope
    print( "b={:.15G}, u={:.15G}".format(value(b),uncertainty(b)) )
    print( "cov(a,b)={:.15G}".format(a.u*b.u*get_correlation(a,b)) )

giving ::

    a=1.86666666666667, u=0.465474668125631
    b=1.75714285714286, u=0.119522860933439
    cov(a,b)=-0.05   

These results agree with published values [#]_ ::

    a = 1.867, u(a) = 0.465
    b = 1.757, u(b) = 0.120
    cov(a,b) = -0.05
    chi-squared = 1.665, with 4 degrees of freedom

The value of ``chi-squared`` can be compared with the ``Sum of the squared residuals`` above and the degrees of freedom is the ``Number of points`` less 2.    
    
Application: an additional `y` observation after regression
-----------------------------------------------------------

The results may be used to find a value for ``x`` that corresponds to another observation ``y`` made following the regression. This is a typical application of a calibration curve.

For example, if an additional observation :math:`y_1 = 10.5` has been made, with :math:`u(y_1) = 0.5`, we can evaluate an uncertain number for the corresponding stimulus :math:`x_1`::

    y1 = ureal(10.5,0.5)
    x1 = (y1-a)/b
    print( "x1={:.15G}, u={:.15G}".format(value(x1),uncertainty(x1)) )

giving ::

    x1=4.91327913279133, u=0.32203556012891

This uncertain number has components of uncertainty for the estimates of slope and intercept. So the combined uncertainty takes account of uncertainty in the parameter estimates for the calibration curve, and correlation between them.
  
Forward evaluation: an additional `x` value
-------------------------------------------

The results can also be used to estimate the response :math:`y_2` to a stimulus :math:`x_2`. 

For example, if  :math:`x_2 = 3.5`, and :math:`u(x_2) = 0.2`, we can evaluate an uncertain number for :math:`y_2` as follows ::

    x2 = ureal(3.5,0.2)
    y2 = a + b*x2
    print( "y2={:.15G}, u={:.15G}".format(value(y2),uncertainty(y2)) )

giving ::

    y2=8.01666666666667, u=0.406409531732455

This is an uncertain number representing the mean, or underlying true, response to :math:`x_2`.  Again, the uncertain number for :math:`y_2` has components of uncertainty for the estimates of slope and intercept.
    
Example 2: unequal weights
==========================
A series of six pairs of ``x-y`` observations have been collected. 

The data sequences for ``x`` and ``y``, with uncertainties in ``y``, are ::

    x = [1,2,3,4,5,6]
    y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
    u_y = [0.5,0.5,0.5,1.0,1.0,1.0]

Again, a weighted least-squares regression can be used, which assumes that the uncertainties in ``y`` values are exactly known (i.e., infinite degrees of freedom) ::

    fit = type_a.line_fit_wls(x,y,u_y)
    print( fit )

This generates ::

    Weighted Least-Squares Results:

      Number of points: 6
      Intercept: 0.89, u=0.53, df=inf
      Slope: 2.06, u=0.18, df=inf
      Correlation: -0.87
      Sum of the squared residuals: 4.1308   

More significant figures can be obtained by the same commands used in Example 1::

    a=0.885232067510549, u=0.529708143508836
    b=2.05696202531646, u=0.177892016741205
    cov(1,b)=-0.0822784810126582

These results agree with published values [#]_ ::

    a = 0.885, u(a) = 0.530
    b = 2.057, u(b) = 0.178
    cov(a,b) = -0.082
    chi-squared = 4.131, with 4 degrees of freedom
      
Application: an additional `y` observation after regression
-----------------------------------------------------------

After regression, the uncertain numbers for the intercept and slope can be used to estimate the stimulus :math:`x_1` for a further observation :math:`y_1`. For example, if :math:`y_1 = 10.5` and :math:`u(y_1) = 1.0`, :math:`x_1` is obtained in the same way as Example 1 ::

    y1 = ureal(10.5,1)
    x1 = (y1-a)/b
    print( "x1={:.15G}, u={:.15G}".format(value(x1),uncertainty(x1))

giving ::
  
    x1=4.67425641025641, u=0.533180902231294
  
Example 3: uncertainty in `x` and `y`
=====================================
A series of six pairs of observations have been collected.  

The data sequences for ``x``, ``y``, each with uncertainties are ::

    x = [1.2,1.9,2.9,4.0,4.7,5.9]
    u_x = [0.2] * 6
    y = [3.4,4.4,7.2,8.5,10.8,13.5]
    u_y = [0.2,0.2,0.2,0.4,0.4,0.4]

We use total least-squares regression in this case, because there is uncertainty in both the dependent and independent variablest ::

    fit = type_a.line_fit_wtls(x,y,u_x,u_y,fit_i.a_b)
    print( fit )

which gives ::

    Weighted Total Least-Squares Results:

      Intercept: 0.58(48)
      Slope: 2.16(14)
      Correlation: -0.9
      Sum of the squared residuals: 2.74267678973
  Number of points: 6
 
Again, more figures can be obtained using the same commands as in Example 1 ::

    a=0.578822122145264, u=0.480359046511757
    b=2.15965656740064, u=0.136246483136605
    cov(1,b)=-0.0586143419560877

These results agree with the published values [#]_ ::

    a = 0.5788, u(a) = 0.0.4764
    b = 2.159, u(b) = 0.1355
    cov(a,b) = -0.0577
    chi-squared = 2.743, with 4 degrees of freedom
 
(There are slight differences due to a different number of iterations in the TLS calculation.)

Example 4: relative uncertainty in *y*
======================================
A series of six pairs of ``x-y`` observations are used. The uncertainties in the :math:`y` values are not known. However, a scale factor :math:`s_y` is given and it is assumed that, for every observation :math:`y`, the associated uncertainty :math:`u(y) = s_y \sigma`. The common factor :math:`\sigma` is not known, but can be estimated from the residuals. This is done by the function :func:`type_a.line_fit_rwls`.

We proceed as above ::

    x = [1,2,3,4,5,6]
    y = [3.014,5.225,7.004,9.061,11.201,12.762]
    u_y = [1] * 6
    fit = type_a.line_fit_rwls(x,y,u_y)

    print( fit )

which displays ::

    Relative Weighted Least-Squares Results:

      Intercept: 1.17(16)
      Slope: 1.964(41)
      Correlation: -0.9
      Sum of the squared residuals: 0.116498285714
      Number of points: 6

More precise values of the fitted parameters are ::

    a=1.172, u=0.158875093196181
    b=1.96357142857143, u=0.0407953578791729
    cov(a,b)=-0.00582491428571429

These results agree with the published values [#]_ ::

    a = 1.172, u(a) = 0.159
    b = 1.964, u(b) = 0.041
    cov(a,b) = -0.006
    chi-squared = 0.171, with 4 degrees of freedom

.. note::

    In our solution, 4 degrees of freedom are associated with estimates of the intercept and slope. This is the usual statistical treatment. However, a trend in recent uncertainty guidelines is to dispense with the notion of degrees of freedom. So, in a final step, reference [#BSI]_ multiplies :math:`u(a)` and :math:`u(b)` by an additional factor of 2. We do not agree with this last step. ``GTC`` uses the finite degrees of freedom associated with the intercept and slope to calculate the coverage factor required for an expanded uncertainty.

Example 5: unknown uncertainty in `y`
=====================================
The data in previous example could also have been processed by an 'ordinary' least-squares regression algorithm, because the scale factor for each observation of `y` was unity. In effect, a series of six values for the dependent and independent variables were collected, and the variance associated with each observation was assumed to be the same.
    
We proceed as follows. The data sequences are defined and the ordinary least-squares function is applied ::

    x = [1,2,3,4,5,6]
    y = [3.014,5.225,7.004,9.061,11.201,12.762]
    fit = type_a.line_fit(x,y)

    print( fit )

which displays ::

    Ordinary Least-Squares Results:

      Intercept: 1.17(16)
      Slope: 1.964(41)
      Correlation: -0.9
      Sum of the squared residuals: 0.116498285714
      Number of points: 6

More precise values of the fitted parameters are ::

    a=1.172, u=0.158875093196181
    b=1.96357142857143, u=0.0407953578791729
    cov(a,b)=-0.00582491428571429

The same results were obtained in Example 4.
  
Application: an additional response
-----------------------------------
After regression, if a further observation of :math:`y` becomes available, or a set of observations, then the corresponding stimulus can be estimated. 

For example, if we wish to know the stimulus :math:`x_1` that gave rise to a response :math:`y_1 = 10.5`, we can use the object ``fit`` returned by the regression (note that :meth:`~type_a.LineFitOLS.x_from_y` takes a sequence of `y` values) ::

    y1 = 10.5
    x1 = fit.x_from_y( [y1] )
    print( x1 )

which displays ::

    4.751(97)
    
Forward evaluation: an additional stimulus
------------------------------------------

The regression results can also be used to predict a single future response :math:`y` for a given stimulus :math:`x`.  

For example, if  :math:`x_2 = 3.5` we can find :math:`y_2` as follows ::

    x2 = 3.5
    y2 = fit.y_from_x(x2)
    print( y2 )

giving ::

    8.04(18)

In this case, the uncertainty reported for :math:`y_2` includes a component for the variability of individual responses. The method :meth:`~type_a.LineFitOLS.y_from_x` incorporates this information from the regression analysis. 

Alternatively, the mean response to a stimulus :math:`x_2` can be obtained directly from the fitted parameters ::

    x2 = 3.5
    a, b = fit.a_b 
    y2 = a + b*x2 
    print( y2 )
    
which gives ::

    8.044(70)
   
.. rubric:: Footnotes

.. [#BSI]  These examples also appear in BS DD ISO/TS 28037:2010 *Determination and use of straight-line calibration functions*, (British Standards Institute, 2010). 
.. [#]  Section 6.3, page 13, in BS DD ISO/TS 28037:2010.
.. [#]  Section 6.3, page 15, in BS DD ISO/TS 28037:2010.  
.. [#]  Section 7.4, page 21, in BS DD ISO/TS 28037:2010.
.. [#]  Appendix E, pages 58-59, in BS DD ISO/TS 28037:2010. 
.. _linear_cal:

****************************
Linear Calibration Equations
****************************

This section applies ``GTC`` to a simple calibration problem [#Kessel]_.

.. contents::
   :local:

Calibration
===========

A pressure sensor with an approximately linear response is to be calibrated.

Eleven reference pressures are accurately generated and the corresponding sensor indications are recorded. The standard pressure values are entered in the`y_data` sequence and sensor readings in `x_data` (data from Table 4 in [#Kessel]_)::

    >>> y_data = (0.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0)
    >>> x_data = (0.0000,0.2039,0.4080,0.6120,0.8160,1.0201,1.2242,1.4283,1.6325,1.8367,2.0410)

The sensor indication does not change when observations are repeated at the same reference pressure values, which suggests that the digital resolution of the sensor is much less than any repeatability errors associated with calibration. So we ignore random noise as a source of error.

Measurement model
-----------------

A linear model of the sensor's behaviour is

.. math::

    Y = \alpha + \beta\, X\;,

where :math:`Y` represents the applied pressure and :math:`X` the sensor response.

In operation, the sensor indication, :math:`x` is taken as an estimate of :math:`X`. The relationship between an applied pressure :math:`Y_i` and the indication :math:`x_i` may be expressed as

.. math::

    Y_i = \alpha + \beta\, (x_i - E_{\mathrm{res} \cdot i}) + E_{\mathrm{lin} \cdot i}

where :math:`E_{\mathrm{res} \cdot i}` and :math:`E_{\mathrm{lin} \cdot i}` are errors.

:math:`E_{\mathrm{res} \cdot i}` is a round-off error due to the finite number of digits displayed (i.e., instead of :math:`X_i`, the number displayed is :math:`x_i = X_i + E_{\mathrm{res} \cdot i}`).

:math:`E_{\mathrm{lin} \cdot i}` is the difference between an actual applied pressure :math:`Y_i` and the pressure predicted by the linear model :math:`\alpha + \beta\, X_i`.

..
    :math:`E_{\mathrm{lin} \cdot i}` is not considered while estimating :math:`\alpha` and :math:`\beta` [#]_.

During calibration, the applied reference pressure :math:`Y_{\mathrm{cal} \cdot i}` is not known exactly. The nominal reference pressure is

.. math::

    y_{\mathrm{cal} \cdot i} = Y_{\mathrm{cal} \cdot i} + E_{\mathrm{cal} \cdot i} \;,

where :math:`E_{\mathrm{cal} \cdot i}` is a measurement error in the reference. The uncertainty of :math:`y_{\mathrm{cal} \cdot i}` as an estimate of :math:`Y_{\mathrm{cal} \cdot i}` is given as a relative standard uncertainty

.. math ::

    \frac{u(y_{\mathrm{cal} \cdot i})}{y_{\mathrm{cal} \cdot i}} = 0.000115 \; .

A 2-point calibration curve
---------------------------

A calibration procedure estimates :math:`\alpha` and :math:`\beta`. The actual slope, :math:`\beta`, is

.. math::

    \beta = \frac{Y_{\mathrm{cal} \cdot 10} - Y_{\mathrm{cal} \cdot 0}}{X_{\mathrm{cal} \cdot 10}-X_{\mathrm{cal} \cdot 0}} \;.

Points near the ends of the range of data available are most influential when estimating the slope and intercept of a linear calibration function, So, an estimate of the slope is

.. math::

   b = \frac{y_{\mathrm{cal} \cdot 10} - y_{\mathrm{cal} \cdot 0}}{x_{\mathrm{cal} \cdot 10}-x_{\mathrm{cal} \cdot 0}} \;.

Using uncertain numbers, this can be calculated ::

    >>> u_ycal_rel = 0.000115
    >>> u_res = type_b.uniform(0.00005)

    >>> x_0 = x_data[0] - ureal(0,u_res,label='e_res_0')
    >>> x_10 = x_data[10] - ureal(0,u_res,label='e_res_10')

    >>> y_0 = ureal(y_data[0],y_data[0]*u_ycal_rel,label='y_0')
    >>> y_10 = ureal(y_data[10],y_data[10]*u_ycal_rel,label='y_10')

    >>> b = (y_10 - y_0)/(x_10 - x_0)
    >>> a = y_10 - b * x_10

The results for ``a`` and ``b``, as well as the correlation coefficient, are ::

    >>> a
    ureal(0.0,0.0002828761730473424,inf)
    >>> b
    ureal(9.799118079372857,0.0011438175474686209,inf)
    >>> get_correlation(a,b)
    -0.12117041864179227


The non-linearity error
-----------------------
Using the remainder of the calibration data, we can compare the calibration line with the calibration data points and thereby assess the importance of non-linear sensor response across the range. The following will display a table of differences between the data and the model ::

    >>> for x_i,y_i in zip(x_data,y_data):
    ...     dif = y_i - (x_i * b + a)
    ...     print("x = {:G}, dif ={}".format(x_i,dif))
    ...
    x = 0, dif = ...0.00000(28)
    x = 0.2039, dif = 0.00196(34)
    x = 0.408, dif = 0.00196(52)
    x = 0.612, dif = 0.00294(72)
    x = 0.816, dif = 0.00392(94)
    x = 1.0201, dif = 0.0039(12)
    x = 1.2242, dif = 0.0039(14)
    x = 1.4283, dif = 0.0039(16)
    x = 1.6325, dif = 0.0029(19)
    x = 1.8367, dif = 0.0020(21)
    x = 2.041, dif = 0.0000(23)

A maximum deviation (worst case error) is taken to be 0.005.[#Kessel]_ This amount of deviation is assumed to cover departures from linearity of the sensor [#]_.

The calibration equation
------------------------

We now have sufficient information to define a calibration function that takes a sensor indication and returns an uncertain number for applied pressure. For instance, ::

    >>> u_lin = type_b.uniform(0.005)
    >>> u_res = type_b.uniform(0.00005)

    >>> a = ureal(0.0,0.00028,label='a',independent=False)
    >>> b = ureal(9.79912, 0.00114,label='b',independent=False)
    >>> set_correlation(-0.1212,a,b)

    >>> def cal_fn(x):
    ...     """-> pressure estimate
    ...     :arg x: sensor reading (a number)
    ...     :returns: an uncertain number representing the applied pressure
    ...     """
    ...     e_res_i = ureal(0,u_res,label='e_res_i')
    ...     e_lin_i = ureal(0,u_lin,label='e_lin_i')
    ...     return a + b * (x + e_res_i) + e_lin_i
    ...

With this function, we can calculate pressures and expanded uncertainties (:math:`k=2`) for the calibration data, which can be compared with Table 7 in the reference [#Kessel]_ ::

    >>> for i,x_i in enumerate(x_data):
    ...    y_i = cal_fn(x_i)
    ...    print("{}: p={:G},  U(p)={:G}".format(i,y_i.x,2*y_i.u))
    ...
    0: p=0,  U(p)=0.00582812
    1: p=1.99804,  U(p)=0.00584124
    2: p=3.99804,  U(p)=0.00589119
    3: p=5.99706,  U(p)=0.00597701
    4: p=7.99608,  U(p)=0.0060972
    5: p=9.99608,  U(p)=0.00624986
    6: p=11.9961,  U(p)=0.00643263
    7: p=13.9961,  U(p)=0.00664303
    8: p=15.9971,  U(p)=0.00687865
    9: p=17.998,  U(p)=0.00713689
    10: p=20,  U(p)=0.00741554

Linearising the sensor response
===============================

With additional information about the typical behaviour of this type of sensor, we can pre-process readings and improve the linearity of the response. The following equation takes a raw indication :math:`x` and returns a value that will vary more linearly with applied pressure than :math:`x`. The effect of :math:`f_\mathrm{lin}` is to reduce the difference between the pressure estimates and actual pressures.

.. math::

    f_\mathrm{lin}(x) = c_0 + c_1x + c_2x^2 + c_3x^3

The coefficients :math:`c_i` apply to the type of sensor; they are **not** determined as part of the calibration procedure. No uncertainty need be associated with these numbers.

The pre-processing function can be implemented as ::

    >>> def f_lin(x):
    ...    """improve sensor linearity"""
    ...    c0 = 0.0
    ...    c1 = 9.806
    ...    c2 = -2.251E-3
    ...    c3 = -5.753E-4
    ...    return c0 + (c1 + (c2 + c3*x)*x)*x
    ...

Our model of the measurement is now

.. math::

    Y_i = \alpha + \beta\, f_\mathrm{lin}(x_i - E_{\mathrm{res} \cdot i}) + E_{\mathrm{lin} \cdot i} \;

To calibrate this 'linearised' sensor, the original indications :math:`x_{\mathrm{cal} \cdot 10}` and :math:`x_{\mathrm{cal} \cdot 0}` are transformed by :math:`f_\mathrm{lin}(X)` before calculating the slope and intercept (this transformation also takes account of the reading error). ::

    >>> u_ycal_rel = 0.000115
    >>> u_res = type_b.uniform(0.00005)

    >>> x_0 = f_lin( x_data[0] - ureal(0,u_res,label='e_res_0') )
    >>> x_10 = f_lin( x_data[10] - ureal(0,u_res,label='e_res_10') )

    >>> y_0 = ureal(y_data[0],y_data[0]*u_ycal_rel,label='y_0')
    >>> y_10 = ureal(y_data[10],y_data[10]*u_ycal_rel,label='y_10')

    >>> b = (y_10 - y_0)/(x_10 - x_0)
    >>> a = y_10 - b * x_10

The results are ::

    >>> a
    ureal(0.0,0.00028307798251305335,inf)
    >>> b
    ureal(1.000011112006328,0.00011672745986082041,inf)
    >>> get_correlation(a,b)
    -0.12125729816056871

The differences between nominal standard values and the sensor estimates can be displayed by  ::

    >>> for x_i,y_i in zip(x_data,y_data):
    ...    dif = y_i - (f_lin(x_i) * b + a)
    ...    print("x = {: G}, dif ={}".format(x_i,dif))
    ...
    x = 0, dif =...0.00000(28)
    x = 0.2039, dif = 0.00063(34)
    x = 0.408, dif =-0.00048(52)
    x = 0.612, dif =-0.00036(72)
    x = 0.816, dif = 0.00003(94)
    x = 1.0201, dif =-0.0003(12)
    x = 1.2242, dif =-0.0002(14)
    x = 1.4283, dif = 0.0002(16)
    x = 1.6325, dif = 0.0000(19)
    x = 1.8367, dif = 0.0003(21)
    x = 2.041, dif = 0.0000(23)

Which shows that the differences are much smaller than before.

The worst-case error is now about :math:`\pm 0.0007`.

The new calibration equation
----------------------------

A new calibration function that takes a sensor indication and returns the applied pressure can be defined ::

    >>> u_lin = type_b.uniform(0.0007)
    >>> u_res = type_b.uniform(0.00005)

    >>> a = ureal(0.0,0.00028,label='a',independent=False)
    >>> b = ureal(1.000011, 0.000117,label='b',independent=False)
    >>> set_correlation(-0.1215,a,b)

    >>> def lin_cal_fn(x):
    ...     """-> linearised pressure estimate
    ...     :arg x: sensor reading (a number)
    ...     :returns: an uncertain number representing the applied pressure
    ...     """
    ...     e_res_i = ureal(0,u_res,label='e_res_i')
    ...     e_lin_i = ureal(0,u_lin,label='e_lin_i')
    ...     return a + b * f_lin(x + e_res_i) + e_lin_i
    ...

The improvement to accuracy can be seen by applying this function to the calibration data ::

    >>> for i,x_i in enumerate(x_data):
    ...     y_i = lin_cal_fn(x_i)
    ...     print("{}: p={:0.5G},  U(p)={:.2G}".format(i,y_i.x,2*y_i.u))
    ...
    0: p=0,  U(p)=0.0011
    1: p=1.9994,  U(p)=0.0012
    2: p=4.0005,  U(p)=0.0014
    3: p=6.0004,  U(p)=0.0018
    4: p=8,  U(p)=0.0021
    5: p=10,  U(p)=0.0025
    6: p=12,  U(p)=0.003
    7: p=14,  U(p)=0.0034
    8: p=16,  U(p)=0.0038
    9: p=18,  U(p)=0.0043
    10: p=20,  U(p)=0.0047

.. rubric:: Footnotes

.. [#Kessel]

    R Kessel, R N Kacker and K-D Sommer,
    *Uncertainty budget for range calibration*,
    Measurement **45** (2012) 1661 -- 1669.

.. [#] The uncertainty due to linearity errors can be estimated later by comparing the calibration data with the pressure predicted by the linear calibration curve.

.. [#] A linear model is chosen for simplicity of use by the client. There is an obvious bias in the residuals that is ignored at this stage.


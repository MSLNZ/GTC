.. _CITAC_A5:

***************************************
Cadmium released from ceramic-ware (A5)
***************************************

.. contents::
   :local:

This section is based on a measurement described in Appendix 5 of the 3rd edition of the EURACHEM / CITAC Guide [#]_.

The CITAC Guide gives a careful discussion of the uncertainty analysis leading to particular numerical values. The following shows only how the data processing could be preformed using ``GTC``.

The measurement
===============

The experiment determines the amount of cadmium released from ceramic ware.  

The measurand can be expressed as

.. math::

    r = \frac{c_0 \cdot V_\mathrm{L}}{a_\mathrm{v}} 
    \cdot d \cdot f_\mathrm{acid} \cdot f_\mathrm{time} \cdot f_\mathrm{temp}
    \; ,
    
where 
    *   :math:`r` is the mass of cadmium leached per unit area :math:`(\mathrm{mg}\, \mathrm{dm}^{-2})`, 
    *   :math:`c_0` cadmium content in the extraction solution (:math:`\mathrm{mg}\, \mathrm{L}^{-1}`), 
    *   :math:`V_\mathrm{L}` is the volume of leachate (L), 
    *   :math:`d` is the dilution factor, 
    *   :math:`a_\mathrm{v}` is the surface area of the liquid (:math:`\mathrm{dm}^2`).
    *   :math:`f_\mathrm{acid}` is the influence of acid concentration.
    *   :math:`f_\mathrm{time}` is the influence of the duration,
    *   :math:`f_\mathrm{temp}` is the influence of temperature.

The uncertainty contributions
=============================

Section A5.4 of the CITAC Guide provides numerical estimates of these quantities that can be used to define uncertain numbers for the calculation. 

Dilution factor
---------------
In this example there was no dilution.

Leachate volume
---------------
Several factors contribute to the uncertainty of  :math:`V_\mathrm{L}`:

    * :math:`V_{\mathrm{L}-\mathrm{fill}}` the relative accuracy with which the vessel can be filled
    * :math:`V_{\mathrm{L}-\mathrm{temp}}` temperature variation affects the determined volume
    * :math:`V_{\mathrm{L}-\mathrm{read}}` the accuracy with which the volume reading can be made
    * :math:`V_{\mathrm{L}-\mathrm{cal}}` the accuracy with which the manufacturer can calibrate a 500 mL vessel

Uncertain numbers for each contribution can be defined and combined to obtain an uncertain number for the volume. In this case, the volume of leachate is 332 mL. ::

    >>> v_leachate = 332 # mL
    >>> a_liquid = 2.1E-4   # liquid volume expansion per degree
    >>> v_fill = ureal(0.995,tb.triangular(0.005),label='v_fill')
    >>> v_temp = ureal(0,tb.uniform(v_leachate*a_liquid*2),label='v_temp')
    >>> v_reading = ureal(1,tb.triangular(0.01),label='v_reading')
    >>> v_cal = ureal(0,tb.triangular(2.5),label='v_cal')
    
    # Change units to liters now
    >>> V_L = result( 
    ...     (v_leachate * v_fill * v_reading + v_temp + v_cal)/1000, 
    ...     label='V_L') # L
    ...
    >>> print( "V leachate: {}".format(V_L) )
    V leachate: 0.3303(18) 
    
A calibration curve for cadmium concentration
---------------------------------------------
The amount of leached cadmium is calculated using a calibration curve. A linear relationship is assumed between observed absorbance and cadmium concentration. 

.. math:: A_i = c_i \cdot B_1 + B_0 + E_i \; ,

where :math:`B_1` and :math:`B_0` are the slope and intercept, respectively, of the line, :math:`A_i` is the observed absorbance, :math:`c_i` is the concentration of the :math:`i^\mathrm{th}` calibration standard and :math:`E_i` is the unknown measurement error incurred during the :math:`i^\mathrm{th}` observation.

Three repeat observations are made for each of five calibration standards and the parameters of the calibration line are estimated by ordinary least-squares regression.

The ``GTC`` calculation uses the :func:`~type_a.line_fit` function ::
 
    >>> x_data = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9]
    >>> y_data = [0.028, 0.029, 0.029, 0.084, 0.083, 0.081, 0.135, 
    ...             0.131, 0.133, 0.180, 0.181, 0.183, 0.215, 0.230, 0.216]
    ... 
    >>> fit = ta.line_fit(x_data,y_data,label='regression')

    >>> B_0 = fit.intercept
    >>> B_1 = fit.slope

    >>> print( "Intercept: {}".format(B_0) )
    Intercept: 0.0087(29)
    >>> print( "Slope: {}".format(B_1) )
    Slope: 0.2410(50)
    
There is correlation between these uncertain numbers (the estimates are correlated) ::

    >>> print( get_correlation(B_0, B_1) )
    -0.87038...
    
The object ``fit`` contains information about the regression that can be used to make predictions about cadmium concentration from subsequent observations of absorbance. In this case, two further values of absorbance are used to estimate the concentration :math:`c_0`. 

Using the function :meth:`~type_a.LineFitOLS.x_from_y` we write (the label 'absorbance' will be attached to the mean of the observations and identify this influence in the uncertainty budget below) ::

    >>> c_0 = fit.x_from_y( [0.0712,0.0716], x_label='absorbance',y_label='noise' )
    >>> print( "absorbance: {}".format(c_0) )
    absorbance: 0.260(18)
    
Liquid surface area 
-------------------
There are two contributions to the uncertainty of :math:`a_\mathrm{V}`:

    * :math:`a_{\mathrm{V}-\mathrm{dia}}` uncertainty due to the diameter measurement
    * :math:`a_{\mathrm{V}-\mathrm{shape}}` uncertainty due to imperfect shape

Uncertain numbers for each contribution can be combined to obtain an estimate of the surface area  ::

    >>> dia = ureal(2.70,0.01,label='dia')
    >>> a_dia = math.pi*(dia/2)**2
    >>> a_shape = ureal(1,0.05/1.96,label='a_shape')
    >>> a_V = result( a_dia * a_shape, label='a_V' )
    >>> print( "a_V: {}".format(a_V) ) 
    a_V: 5.73(15)
    

Temperature effect
------------------
The temperature factor is given as :math:`f_\mathrm{temp} = 1 \pm 0.1`. Assuming a uniform distribution we define ::

    >>> f_temp = ureal(1,tb.uniform(0.1),label='f_temp')
    
Time effect
-----------
The time factor is given as :math:`f_\mathrm{time} = 1 \pm 0.0015`. Assuming a uniform distribution we define ::

    >>> f_time = ureal(1,tb.uniform(0.0015),label='f_time')
   
Acid concentration
------------------
The acid concentration factor is given as :math:`f_\mathrm{acid} = 1 \pm 0.0008`. This is already in the form of a standard uncertainty so we define ::

    >>> f_acid = ureal(1,0.0008,label='f_acid')
  
The uncertainty calculation
===========================

To estimate :math:`r` we now evaluate ::

    >>> r = c_0 * V_L / a_V * f_acid * f_time * f_temp
    >>> print( "r: {}".format(r) ) 
    r: 0.0150(14)

The contribution from the different influences can be examined ::

    >>> for cpt,u in rp.budget(r,[c_0,V_L,a_V,f_acid,f_time,f_temp]):
    ...     print( " {}: {:G}".format(cpt,u) )
    ... 
     absorbance: 0.00102956
     f_temp: 0.00086663
     a_V: 0.000398736
     V_L: 8.28714E-05
     f_time: 1.29994E-05
     f_acid: 1.20084E-05
 
   
The results (which can be compared with Figure A5.8 in the Guide) show that the content of cadmium in the extraction solution is the dominant component of uncertainty.

The full uncertainty budget can be obtained by writing ::

    >>> for cpt,u in rp.budget(r,trim=0):
    ...     print( " {}: {:G}".format(cpt,u) )
    ...
     noise: 0.000928623
     f_temp: 0.00086663
     a_regression: 0.000688685
     a_shape: 0.00038292
     b_regression: 0.000311899
     dia: 0.000111189
     v_reading: 6.128E-05
     v_cal: 4.63764E-05
     v_fill: 3.0794E-05
     f_time: 1.29994E-05
     f_acid: 1.20084E-05
     v_temp: 3.65814E-06

This reveals that the additional observations of absorbance have contributed most to the uncertainty (so perhaps a few more observations would help)

.. rubric:: Footnotes

.. [#] On-line: http://www.citac.cc/QUAM2012_P1.pdf
.. [#error] Note there is a mistake in the standard uncertainty quoted in the CITAC Guide :math:`u(a_\mathrm{V})=0.19`, as can be verified by evaluating :math:`\sqrt{(0.042^2 + 0.146^2)}`.
.. [#error2] The mistake in :math:`u(a_\mathrm{V})`, mentioned above , leads to give a slightly different value :math:`u(r)=0.0015\; \mathrm{mg}\,\mathrm{dm}^{-2}` in the CITAC Guide.   

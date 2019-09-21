.. _GUM_H2:

*********************************************
Resistance and reactance measurement (GUM H2)
*********************************************
An example from Appendix H2 of the GUM [#GUM]_. 

.. contents::
    :local:
    
Code
====

.. literalinclude:: ../../examples/GUM_H2.py

Explanation
===========
Several quantities associated with an electrical component in an AC electrical circuit are of interest here. Measurements of the resistance :math:`R`, the reactance :math:`X` and the magnitude of the impedance :math:`|Z|` are required. These can be obtained by measuring voltage :math:`V`, current :math:`I` and phase angle :math:`\phi` and then using the measurement equations:

.. math:: 

    R = V I \cos \phi

    X = V I  \sin \phi

    |Z| = V  I

Five repeat measurements of each quantity are performed. The mean values, and associated uncertainties (type-A analysis) provide estimates of voltage, current and phase angle. The correlation coefficients between pairs of estimates is also calculated. 

This information is used to define three inputs to the calculation and assign correlation coefficients (the additional argument ``independent=False`` is required for ``set_correlation`` to be used).  ::

    V = ureal(4.999,3.2E-3,independent=False)      # volt
    I = ureal(19.661E-3,9.5E-6,independent=False)  # amp
    phi = ureal(1.04446,7.5E-4,independent=False)  # radian

    set_correlation(-0.36,V,I)
    set_correlation(0.86,V,phi)
    set_correlation(-0.65,I,phi)

Estimates of the three required quantities are then ::

    R = result( V * cos(phi) / I )
    X = result( V * sin(phi) / I )
    Z = result( V / I )

Results are displayed by ::

    print 'R = {}'.format(R) 
    print 'X = {}'.format(X)
    print 'Z = {}'.format(Z) 
    print
    print 'Correlation between R and X = {:+.2G}'.format( get_correlation(R,X) )
    print 'Correlation between R and Z = {:+.2G}'.format( get_correlation(R,Z) )
    print 'Correlation between X and Z = {:+.2G}'.format( get_correlation(X,Z) )
  
The output is ::

    R = 127.732(70)
    X = 219.85(30)
    Z = 254.26(24)

    Correlation between R and X = -0.59
    Correlation between R and Z = -0.49
    Correlation between X and Z = +0.99

Calculating the expanded uncertainty
------------------------------------

The expanded uncertainties for ``R``, ``X`` and ``Z`` are not evaluated in the GUM, because the Welch-Satterthwaite equation for the effective degrees of freedom is invalid when input estimates are correlated. We created ``V``, ``I`` and ``phi`` with infinite degrees of freedom, the default).

However, an alternative calculation is applicable in this case [#Willink2007]_. There are two different ways to carry out the calculation in ``GTC``. One uses :func:`type_a.multi_estimate_real`, the other uses :func:`~core.multiple_ureal`.

:func:`~core.multiple_ureal` creates several elementary uncertain real numbers that are associated with each other (called an *ensemble* in ``GTC``). The documentation shows this applied to the GUM H2 example.

:func:`type_a.multi_estimate_real`, performs a type-A analysis on raw data (three sets of five readings) and returns an *ensemble* of elementary uncertain real numbers. The documentation shows this applied to the GUM H2 example.

.. note::

    The impedance calculation can also be treated as a complex-valued problem, so there are other functions that can do data processing of uncertain complex numbers. The documentation for :func:`type_a.multi_estimate_complex` and :func:`~core.multiple_ucomplex` both use GUM H2 as an example.

.. [#GUM]

    BIPM and IEC and IFCC and ISO and IUPAC and IUPAP and OIML, 
    *Evaluation of measurement data - Guide to the expression of uncertainty in measurement JCGM 100:2008 (GUM 1995 with minor corrections)*, (2008) `http://www.bipm.org/en/publications/guides/gum <http://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e.html?csnumber=50461>`_

..  [#Willink2007] R Willink, *'A generalization of the Welch-Satterthwaite formula for use with correlated uncertainty components'*, Metrologia **44** (2007) 340-349, Sec. 4.1
.. _CITAC_A1:

******************************************
Preparation of a Calibration Standard (A1)
******************************************

This section is based on a measurement described in Appendix 1 of the 3rd edition of the EURACHEM / CITAC Guide [#]_.

The CITAC Guide gives a careful discussion of the uncertainty analysis leading to particular numerical values. The following shows only how the subsequent calculation can be preformed using ``GTC``.

The measurement
===============

The concentration of Cd in a standard solution is to be determined. 

This can be expressed by the equation

.. math::

    c_\mathrm{Cd} = \frac{1000 \cdot m \cdot P}{V} \; ,
    
where 
    *   :math:`c_\mathrm{Cd}` is the concentration expressed (mg/L), 
    *   :math:`1000` is a conversion factor from mL to L, 
    *   :math:`m` is the mass of high purity metal (mg), 
    *   :math:`P` is the purity of the metal as a mass fraction, 
    *   :math:`V` is the volume of liquid of the standard (mL).

The uncertainty contributions
=============================

In section A1.4 of the CITAC Guide the numerical estimates of influence quantities are described. These can be used to define uncertain numbers for the mass, purity and volume. The mass and purity are defined directly as elementary uncertain numbers [#]_::

    >>> P = ureal(0.9999,type_b.uniform(0.0001),label='P')
    >>> m = ureal(100.28,0.05,label='m') # mg

The volume has three influences that contribute to the overall uncertainty: the manufacturing tolerances of the measuring flask, the repeatability of filling and the variability of temperature during the experiment. Each is represented by an elementary uncertain number ::

    >>> V_flask = ureal(100,type_b.triangular(0.1),label='V_flask')
    >>> V_rep = ureal(0,0.02,label='V_rep')
    >>> V_T = ureal(0,type_b.uniform(0.084),label='V_T')

Note that the value assigned to ``V_rep`` and ``V_T`` is zero. These represent repeatability error and the temperature error incurred during the experiment. The best estimate of these errors is zero but the uncertainty is given in the second argument to ``ureal``.
    
After these definitions an uncertain number representing the volume of fluid is (we label the uncertain number for convenience when reporting the uncertainty budget later) ::

    >>> V = result( V_flask + V_rep + V_T,label = 'V')
   
The uncertainty calculation
===========================

The concentration calculation is then simply [#]_ ::

    >>> c_Cd = 1000 * m * P / V
    >>> print( "c_Cd={:G}, u={:G}".format(c_Cd.x,c_Cd.u) )
    c_Cd=1002.7, u=0.835199 
 
The contributions to the standard uncertainty can be itemised using :func:`reporting.budget`::

    >>> for cp,u in rp.budget(c_Cd):
    ...     print( " {}: {:G}".format(cp,u) )
    ...
     m: 0.49995
     V_T: 0.486284
     V_flask: 0.40935
     V_rep: 0.20054
     P: 0.0578967
  
The contribution from the overall uncertainty in the volume of fluid, rather than the individual terms can also be compared with other contributions by using a list of influences :: 

    >>> for cp,u in rp.budget(c_Cd,influences=[m,P,V]):
    ...     print( " {}: {:G}".format(cp,u) )
    ...
     V: 0.666525
     m: 0.49995
     P: 0.0578967

These results can be compared with Figure A1.5 in the CITAC Guide.

.. rubric:: Footnotes

.. [#] On-line: http://www.citac.cc/QUAM2012_P1.pdf
.. [#] Functions from the :mod:`type_b` module are used to scale the uncertainty parameter of a non-Gaussian error to obtain the standard deviation.
.. [#] The numbers differ slightly because numbers in the CITAC Guide calculations have been rounded
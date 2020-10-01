.. _CITAC_A2:

**********************************************
Standardising a Sodium Hydroxide Solution (A2)
**********************************************

This section is based on a measurement described in Appendix 2 of the 3rd edition of the EURACHEM / CITAC Guide [#]_.

The CITAC Guide gives a careful discussion of the uncertainty analysis leading to particular numerical values. The following shows only how the subsequent calculation can be preformed using ``GTC``.

The measurement
===============

The concentration of a solution of NaOH is to be determined. The NaOH is titrated against the titrimetric standard potassium hydrogen phthalate (KHP). 

The measurand can be expressed as

.. math::

    c_\mathrm{NaOH} = \frac{1000 \cdot m_\mathrm{KHP} \cdot P_\mathrm{KHP}}{M_\mathrm{KHP} \cdot V_\mathrm{T}} \; ,
    
where 
    *   :math:`c_\mathrm{NaOH}` is the concentration expressed in mol/L, 
    *   :math:`1000` is a volume conversion factor from mL to L, 
    *   :math:`m_\mathrm{KHP}` is the mass of the titrimetric standard in g, 
    *   :math:`P_\mathrm{KHP}` is the purity of the titrimetric standard as a mass fraction, 
    *   :math:`M_\mathrm{KHP}` is the molar mass of KHP in g/mol,
    *   :math:`V_\mathrm{T}` is the titration volume of NaOH solution in mL.

The uncertainty contributions
=============================

Section A2.4 of the CITAC Guide provides numerical estimates of influence quantities, which can be used to define uncertain numbers for the calculation. 

The mass :math:`m_\mathrm{KHP}` is determined from the difference of two weighings with balance linearity as the only source of measurement error considered. However, a linearity error occurs twice: once in the tare weighing and once in the gross weighing. So in the calculations we introduce the nett weight as a number (0.3888) and the uncertainty contribution is found by taking the difference of uncertain numbers representing the errors that occur during the weighings (if the raw observations were available, they might have been used to define ``u_lin_tare`` and ``u_lin_gross``)  [#]_. 

    >>> u_lin_tare = ureal(0,type_b.uniform(0.15E-3),label='u_lin_tare')
    >>> u_lin_gross = ureal(0,type_b.uniform(0.15E-3),label='u_lin_gross')
    >>> u_m_KHP = u_lin_gross - u_lin_tare
    >>> m_KHP = result( 0.3888 + u_m_KHP, label='m_KHP' )
    
The purity :math:`P_\mathrm{KHP}` is [#]_ ::

    >>> P_KHP = ureal(1.0,type_b.uniform(0.0005),label='P_KHP')

The molar mass :math:`m_\mathrm{KHP}` is calculated from IUPAC data and the number of each constituent element in the KHP molecule :math:`\mathrm{C}_8\mathrm{H}_5\mathrm{O}_4\mathrm{K}`. ::

    >>> M_C = ureal(12.0107,type_b.uniform(0.0008),label='M_C')
    >>> M_H = ureal(1.00794,type_b.uniform(0.00007),label='M_H')
    >>> M_O = ureal(15.9994,type_b.uniform(0.0003),label='M_O')
    >>> M_K = ureal(39.0983,type_b.uniform(0.0001),label='M_K')

    >>> M_KHP = result( 8*M_C + 5*M_H + 4*M_O + M_K, label='M_KHP' )

The volume term :math:`V_\mathrm{T2}` is affected by contributions from calibration error and temperature.  ::

    >>> uV_T_cal = ureal(0,type_b.triangular(0.03),label='V_T_cal')
    >>> uV_T_temp = ureal(0,0.006,label='V_T_temp')

    >>> V_T = result( 18.64 + uV_T_cal + uV_T_temp, label='V_T' )

The CITAC Guide introduces a further multiplicative term :math:`R` to represent repeatability errors (:math:`R \approx 1`)

.. math::
    c_\mathrm{NaOH} = R\,\frac{1000 \cdot m_\mathrm{KHP} \cdot P_\mathrm{KHP}}{M_\mathrm{KHP} \cdot V_\mathrm{T}} \; ,

In the ``GTC`` calculation this is represented by another uncertain number ::

    >>> R = ureal(1.0,0.0005,label='R')

The uncertainty calculation
===========================

The calculation of :math:`c_\mathrm{NaOH}` is now [#]_::

    >>> c_NaOH = R * (1000 * m_KHP * P_KHP)/(M_KHP * V_T)
    >>> c_NaOH
    ureal(0.102136159706...,0.000100500722124...,inf)

The contribution from different influences can be examined (and compared with Fig. A2.9 in the Guide) ::

    >>> for cpt,u in rp.budget(c_NaOH,influences=[m_KHP,P_KHP,M_KHP,V_T,R]):
    ... 	print( " {}: {:G}".format(cpt,u) )
    ... 
     V_T: 7.47292E-05
     R: 5.10681E-05
     m_KHP: 3.21735E-05
     P_KHP: 2.94842E-05
     M_KHP: 1.88312E-06

The full uncertainty budget is ::

    >>> for cpt,u in rp.budget(c_NaOH):
    ... 	print( " {}: {:G}".format(cpt,u) )
    ... 	
     V_T_cal: 6.71088E-05
     R: 5.10681E-05
     V_T_temp: 3.28764E-05
     P_KHP: 2.94842E-05
     u_lin_tare: 2.27501E-05
     u_lin_gross: 2.27501E-05
     M_C: 1.84798E-06
 
 
.. rubric:: Footnotes

.. [#] On-line: http://www.citac.cc/QUAM2012_P1.pdf
.. [#] If the balance indications for the tare and gross weighings were known they could have been used to define the values of these uncertain numbers, however the Guide does not provide this raw data. Instead, the zero value used here represents an estimate of the linearity *error*.  
.. [#] Functions from the :mod:`type_b` module are used here to scale the uncertainty parameters, as described in the CITAC Guide
.. [#] The numbers differ slightly because numbers in the the CITAC Guide calculations have been rounded
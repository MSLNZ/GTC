.. _CITAC_A3:

***************************
An Acid/Base Titration (A3)
***************************

This section is based on a measurement described in Appendix Appendix 3 of the 3rd edition of the EURACHEM / CITAC Guide [#]_.

The CITAC Guide gives a careful discussion of the uncertainty analysis leading to particular numerical values. The following shows only how the subsequent calculation can be preformed using ``GTC``.

The measurement
===============

The method determines the concentration of an HCl solution by a sequence of experiments. This is a longer calculation than the previous examples, so the code shown below should be considered as lines of text in a file that can be executed by ``GTC``.   

The measurand can be expressed by 

.. math::

    c_\mathrm{HCl} = \frac{1000 \cdot m_\mathrm{KHP} \cdot P_\mathrm{KHP} \cdot V_\mathrm{T2}}
    {V_\mathrm{T1} \cdot M_\mathrm{KHP} \cdot V_\mathrm{HCl}} \; ,
    
where 
    *   :math:`c_\mathrm{HCl}` is the concentration expressed (mol/L), 
    *   :math:`1000` is a volume conversion factor from mL to L, 
    *   :math:`m_\mathrm{KHP}` is the mass of KHP taken (g), 
    *   :math:`P_\mathrm{KHP}` is the purity of KHP as a mass fraction, 
    *   :math:`V_\mathrm{T1}` is the volume of NaOH to titrate KHP (mL).
    *   :math:`V_\mathrm{T2}` is the volume of NaOH to titrate HCl (mL).
    *   :math:`M_\mathrm{KHP}` is the molar mass of KHP (g/mol),
    *   :math:`V_\mathrm{T}` is the titration volume of NaOH solution (mL).

The uncertainty contributions
=============================

Section A3.4 of the CITAC Guide provides numerical estimates of influence quantities, which can be used to define uncertain numbers for the uncertainty calculation. 

The mass :math:`m_\mathrm{KHP}` is determined from the difference of two weighings with balance linearity as the only source of measurement error. However, a linearity error arises twice: once in the tare weighing and once in the gross weighing. So, in the calculations we introduce the nett weight as a number (0.3888) and the uncertainty contribution is found by taking the difference of uncertain numbers representing estimates of the errors that occur during the weighings (if the raw observations were available, they might have been used to define ``u_lin_tare`` and ``u_lin_gross``) [#]_.  ::

    >>> u_lin_tare = ureal(0,type_b.uniform(0.15E-3),label='u_lin_tare')
    >>> u_lin_gross = ureal(0,type_b.uniform(0.15E-3),label='u_lin_gross')
    >>> m_KHP = result( 0.3888 + u_lin_gross - u_lin_tare, label = 'm_KHP' )
    
The purity :math:`P_\mathrm{KHP}` is [#]_ ::

    >>> P_KHP = ureal(1.0,type_b.uniform(0.0005),label='P_KHP')

The volume term :math:`V_\mathrm{T2}` is affected by contributions from calibration error and temperature. In calculating the uncertainty contribution due to temperature, the volume expansion coefficient for water  :math:`2.1 \times 10^{-4} \, ^\circ\mathrm{C}^{-1}` is used, the volume of the pipette is 15 mL and the temperature range is :math:`\pm 4\, ^\circ\mathrm{C}`. ::

    >>> uV_T2_cal = ureal(0,type_b.triangular(0.03),label='V_T2_cal')
    >>> uV_T2_temp = ureal(0,type_b.uniform(15 * 2.1E-4 * 4),label='V_T2_temp')

    >>> V_T2 = result( 14.89 + uV_T2_cal + uV_T2_temp, label='V_T2' )

The influences of the volume term :math:`V_\mathrm{T1}` are almost the same as :math:`V_\mathrm{T2}`, only the temperature contribution is different because a 19 mL volume of NaOH was used. ::

    >>> uV_T1_cal = ureal(0,type_b.triangular(0.03),label='V_T1_cal')
    >>> uV_T1_temp = ureal(0,type_b.uniform(19 * 2.1E-4 * 4),label='V_T1_temp')

    >>> V_T1 = result( 18.64 + uV_T1_cal + uV_T1_temp, label = 'V_T1' )

The molar mass :math:`m_\mathrm{KHP}` is calculated from IUPAC data and the number of each constituent element in the KHP molecule :math:`\mathrm{C}_8\mathrm{H}_5\mathrm{O}_4\mathrm{K}`. This can be done as follows ::

    >>> M_C = ureal(12.0107,type_b.uniform(0.0008),label='M_C')
    >>> M_H = ureal(1.00794,type_b.uniform(0.00007),label='M_H')
    >>> M_O = ureal(15.9994,type_b.uniform(0.0003),label='M_O')
    >>> M_K = ureal(39.0983,type_b.uniform(0.0001),label='M_K')

    >>> M_KHP = result( 8*M_C + 5*M_H + 4*M_O + M_K, label='M_KHP' )

The influences on the volume term :math:`V_\mathrm{HCl}` are similar to the :math:`V_\mathrm{T1}` and :math:`V_\mathrm{T2}`. A 15 mL pipette was used with a stated uncertainty tolerance of 0.02. The range of temperature variation in the laboratory is :math:`4\, ^\circ\mathrm{C}`. ::

    >>> uV_HCl_cal = ureal(0,type_b.triangular(0.02),label='uV_HCl_cal')
    >>> uV_HCl_temp = ureal(0,type_b.uniform(15 * 2.1E-4 * 4),label='uV_HCl_temp')

    >>> V_HCl = result( 15 + uV_HCl_cal + uV_HCl_temp, label='V_HCl' )

The CITAC Guide introduces a further multiplicative term :math:`R` to represent repeatability error (:math:`R \approx 1`)

.. math::
    c_\mathrm{NaOH} = R\,\frac{1000 \cdot m_\mathrm{KHP} \cdot P_\mathrm{KHP}}{M_\mathrm{KHP} \cdot V_\mathrm{T}} \; ,

Another uncertain number is defined to represent this ::

    >>> R = ureal(1.0,0.001,label='R')

The uncertainty calculation
===========================

The calculation of :math:`c_\mathrm{NaOH}` is now ::

    >>> c_HCl = R * (1000 * m_KHP * P_KHP * V_T2)/(M_KHP * V_T1 * V_HCl)
    >>> print(c_HCl)
    0.10139(18)
    >>> print( uncertainty(c_HCl) )  
    0.0001843...
    
The contribution from different influences can be examined ::

    >>> for i in rp.budget(c_HCl,influences=[m_KHP,P_KHP,M_KHP,V_T1,V_T2,V_HCl,R]):
    ...     print( " {}: {:G}".format(i.label,i.u) )
    ...
    R: 0.000101387
    V_T2: 9.69953E-05
    V_T1: 8.33653E-05
    V_HCl: 7.39151E-05
    m_KHP: 3.19376E-05
    P_KHP: 2.9268E-05
    M_KHP: 1.86931E-06
    
The results (which can be compared with Figure A3.6 in the Guide) show that repeatability is the dominant component of uncertainty
 
The full uncertainty budget is obtained by  ::

    >>> for i in rp.budget(c_HCl):
    ...    print( " {}: {:G}".format(i.label,i.u) )
    ...
     R: 0.000101387
     V_T2_cal: 8.33938E-05
     V_T1_cal: 6.66166E-05
     uV_HCl_cal: 5.51882E-05
     V_T1_temp: 5.01198E-05
     V_T2_temp: 4.95334E-05
     uV_HCl_temp: 4.91702E-05
     P_KHP: 2.9268E-05
     u_lin_tare: 2.25833E-05
     u_lin_gross: 2.25833E-05
     M_C: 1.83443E-06
    
This shows that calibration error in the volume titrated is also an important component of uncertainty .
 
Special aspects of this measurement
===================================

The CITAC Guide discusses some aspects of this measurement in section A3.6. Two in particular are: the uncertainty associated with repeatability and bias in titration volumes.

A reduction in the uncertainty attributed to repeatability, by a factor of :math:`\sqrt{3}`, has a small effect on the final combined uncertainty. This may be seen in the ``GTC`` calculation by changing the definition of the uncertain number ``R`` ::

    >>> R = ureal(1.0,0.001/math.sqrt(3),label='R')
 
    >>> c_HCl = R * (1000 * m_KHP * P_KHP * V_T2)/(M_KHP * V_T1 * V_HCl)
    >>> print('c_HCl ={}'.format(c_HCl))
    c_HCl = 0.10139(16)

 
    >>> for i in rp.budget(c_HCl,influences=[m_KHP,P_KHP,M_KHP,V_T1,V_T2,V_HCl,R]):
    ...     print( " {}: {:G}".format(i.label,i.u) )
    ... 
     V_T2: 9.69953E-05
     V_T1: 8.33653E-05
     V_HCl: 7.39151E-05
     R: 5.85359E-05
     m_KHP: 3.19376E-05
     P_KHP: 2.9268E-05
     M_KHP: 1.86931E-06
    
The new results show that the combined uncertainty is not much changed when the repeatability is improved.
 
Another consideration is that a bias may be introduced by the use of phenolphthalein as an indicator. The excess volume in this case is about 0.05 mL with a standard uncertainty of 0.03 mL.

We can adapt our calculations above by defining two elementary uncertain numbers to represent the bias. These can be subtracted from the previous estimates [#]_::

    >>> V_T1_excess = ureal(0.05,0.03,label='V_T1_excess')
    >>> V_T1 = V_T1 - V_T1_excess

    >>> V_T2_excess = ureal(0.05,0.03,label='V_T2_excess')
    >>> V_T2 = V_T2 - V_T2_excess

    >>> print( uncertainty(V_T1) )
    0.033688...

    >>> print( uncertainty(V_T2) )
    0.033210...

The uncertainties are roughly twice the previous values.

The concentration of HCl can then be re-calculated using the same measurement equation ::

    >>> c_HCl = R * (1000 * m_KHP * P_KHP * V_T2)/(M_KHP * V_T1 * V_HCl)
    >>> print( c_HCl )
    0.10132(31)
    >>> print( uncertainty(c_HCl) )  
    0.0003096...

The combined uncertainty is now about twice as large (in mol/L).

.. rubric:: Footnotes

.. [#] On-line: http://www.citac.cc/QUAM2012_P1.pdf
.. [#] If the balance indications for the tare weighing and gross weighing were known they could have been used to define the values of these uncertain numbers, however the CITAC Guide does not provide this raw data. Instead, the zero value used here represents the linearity *error*.  
.. [#] Functions from the :mod:`type_b` module are used here to scale the uncertainty parameters, as described in the CITAC Guide
.. [#] The CITAC Guide does not provide different raw titration results for this case. However, the numerical values of ``V_T1`` and ``V_T2`` will not be the same, because there are now two different parts to the experiment.
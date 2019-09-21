.. _GUM_H1:

********************************
Gauge block measurement (GUM H1)
********************************

An example from Appendix H1 of the GUM [#GUM]_. 

.. contents::
    :local:
    

Code
====

.. literalinclude:: ../../examples/GUM_H1.py

Explanation
===========

The measurand is the length of an end-gauge at :math:`20\,^\circ\mathrm{C}`. The measurement equation is [#]_

.. math::

    l = l_\mathrm{s} + d - l_\mathrm{s}(\delta_\alpha \theta + \alpha_\mathrm{s} \delta_\theta) \;,
    
where 

    * :math:`l_\mathrm{s}` - the length of the standard
    * :math:`d` - the difference in length between the standard and the end-gauge
    * :math:`\delta_\alpha` - the difference between coefficients of thermal expansion for the standard and the end-gauge 
    * :math:`\theta` - the deviation in temperature from :math:`20\,^\circ\mathrm{C}`
    * :math:`\alpha_\mathrm{s}` - the coefficient of thermal expansion for the standard
    * :math:`\delta_\theta` - the temperature difference between the standard and the end-gauge

The calculation proceeds in stages. First, three inputs are defined: 

    * the length difference measurement(``d0``) using the comparator, which is the arithmetic mean of several indications; 
    * an estimate of comparator random errors (``d1``) and 
    * an estimate of comparator systematic errors (``d2``). 
    
These are used to define the intermediate result ``d`` :: 

    d0 = ureal(215,5.8,24,label='d0')  
    d1 = ureal(0.0,3.9,5,label='d1')  
    d2 = ureal(0.0,6.7,8,label='d2')
    
    # Intermediate quantity 'd'
    d = d0 + d1 + d2

Then terms are introduced to account for temperature variability and thermal properties of the gauge blocks. 

In particular, the quantity :math:`\theta` is defined in terms of two other input quantities 

.. math:: \theta = \bar{\theta} + \Delta

where 

    * :math:`\bar{\theta}` is the mean deviation of the test-bed temperature from :math:`20\,^\circ\mathrm{C}` 
    * :math:`\Delta` is a cyclical error in the test-bed temperature
    
In defining these inputs, functions :func:`type_b.uniform` and :func:`type_b.arcsine` convert the widths of particular error distributions into standard uncertainties [#Gaussian]_. ::

    alpha_s = ureal( 11.5E-6, type_b.uniform(2E-6), label='alpha_s' )
    d_alpha = ureal(0.0, type_b.uniform(1E-6), 50, label='d_alpha')
    d_theta = ureal(0.0, type_b.uniform(0.05), 2, label='d_theta')

    theta_bar = ureal(-0.1,0.2, label='theta_bar')
    Delta = ureal(0.0, type_b.arcsine(0.5), label='Delta' )

    # Intermediate quantity 'theta'
    theta = theta_bar + Delta

The length of the standard gauge block is given in a calibration report :: 

    l_s = ureal(5.0000623E7,25,18,label='ls')     

two more intermediate results, representing thermal errors, are then ::

    # two more intermediate steps
    tmp1 = l_s * d_alpha * theta
    tmp2 = l_s * alpha_s * d_theta

Finally, the length of the gauge block is evaluated :: 

    # Final equation for the measurement result
    l = result( l_s + d - (tmp1 + tmp2), label='l')


The script then evaluates the measurement result ::

    print( "Measurement result for l={}".format(l) ) 
    
which displays ::

    Measurement result for l=50000838(32)
    
and the following commands display the components of uncertainty for ``l``, due to each influence::

    print("""
    Components of uncertainty in l (nm)
    -----------------------------------""")

    for l_i,u_i in reporting.budget(l):
        print( "  {!s}: {:G}".format(l_i,u_i) )
        
The output is ::

    Components of uncertainty in l (nm)
    -----------------------------------
      ls: 25
      d_theta: 16.599
      d2: 6.7
      d0: 5.8
      d1: 3.9
      d_alpha: 2.88679
      alpha_s: 0
      theta_bar: 0
      Delta: 0
 
..
    Second-order contributions to the uncertainty
    ---------------------------------------------

    The GUM, in H.1.7, notes that the uncertainty associated with the products :math:`\alpha_\mathrm{s}\,\delta_\theta` and :math:`\delta_\alpha\, \theta` may be underestimated, because in each case one of the factors is estimated as zero.

    The ``GTC`` calculation can include second-order terms associated with a zero product. This can be done by modifying the definitions of ``tmp`` and ``tmp2`` as follows (:func:`function.mul2` includes second-order contributions to the product uncertainty)::

        # two more intermediate steps
        tmp1 = l_s * function.mul2(d_alpha, theta)
        tmp2 = l_s * function.mul2(alpha_s, d_theta)
     
    The result is now ::

        Measurement result for l: 50000838., u=34., df=21.6

    which agrees with the GUM (although no value is given in the GUM for the degrees of freedom). 

    We note, however, that :math:`\alpha_\mathrm{s}` and :math:`\theta` represent estimates based on measured data. In that case, a different second-order calculation is preferred ::

        tmp1 = l_s * function.mul2(d_alpha, theta, estimate=True)
        tmp2 = l_s * function.mul2(alpha_s, d_theta estimate=True)

    which gives ::

        Measurement result for l: 50000838., u=32., df=16.5
        
    This result is better than the GUM's treatment of this problem, which over-estimates the uncertainty [#Hall2011]_.
    
    [#Hall2011] B D Hall, *Using simulation to check uncertainty calculations*, Meas. Sci. Technol., **22** (2011) 025105 (10pp) 

.. rubric:: Footnotes
 
.. [#GUM]

    BIPM and IEC and IFCC and ISO and IUPAC and IUPAP and OIML, 
    *Evaluation of measurement data - Guide to the expression of uncertainty in measurement JCGM 100:2008 (GUM 1995 with minor corrections)*, (2008) `http://www.bipm.org/en/publications/guides/gum <http://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e.html?csnumber=50461>`_

.. [#] In fact, the GUM uses more terms to calculate the uncertainty than are defined: quantities :math:`d` and :math:`\theta` depend on more than one influence quantity.
         
.. [#Gaussian] ``ureal`` creates a new uncertain real number. It takes a standard uncertainty as its second argument.  

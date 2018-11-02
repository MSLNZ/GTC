.. _intro_uncertain_numbers: 

############
Introduction
############

.. contents::
   :local:

The GUM Tree calculator (``GTC``) is a data processing tool that uses `uncertain numbers` to represent measurements of quantities, and automates the evaluation of uncertainty in calculations of quantities derived from measured data. 


Measurement error
=================

Measurement obtains information about physical quantities; but the quantity of interest (the *measurand*) can never be determined exactly, it can only be estimated. There will always be some *measurement error* involved. Writing this as a mathematical equation, where the unknown measurand is :math:`Y` and the measurement result is :math:`y`, we have

.. math::

    y = Y + E_y\; ,
    
where :math:`E_y` represents the measurement error. So, when we talk about 'uncertainty', it may be the 'uncertainty of :math:`y` as an estimate of :math:`Y`' that is intended. In other words, the 'uncertainty' of using, in some way, the measured value :math:`y` as an estimate of the target value of the measurand :math:`Y`.

Clearly, the error :math:`E_y` gives rise to the uncertainty; but this value is never known, we may only describe it in statistical terms. So, a related use of the word *uncertainty* is to refer to the extent of a statistical distribution associated with :math:`E_y`. For example, the term 'standard uncertainty' refers to the standard deviation of a distribution associated with the results of an unpredictable quantity.

Measurement models
------------------
It is generally possible to enumerate a number of factors that influence the outcome of a measurement process, thereby contributing to the final measurement error. In an analysis of the measurement, these factors must be included in a measurement model, which defines the measurand in terms of all other significant influence quantities. Mathematically, we may write   

.. math::

    Y = f(X_1, X_2, \cdots) \;,
 
where the :math:`X_i` are influence quantities. 

Nevertheless, the actual quantities :math:`X_1, X_2, \cdots` are not known; only estimates :math:`x_1, x_2, \cdots` are available. These estimates are used to calculate a measured value that is approximately equal to the measurand 

.. math::

        y = f(x_1, x_2, \cdots) \;.

     
Uncertain Numbers
=================

An uncertain number is a data type designed to represent a quantity that has been measured. It encapsulates information about the measurement: the value obtained and the uncertainty of the measurement process associated with that estimate. There are two different types of uncertain number: one for real-valued quantities and one for complex-valued quantities.

Uncertain real numbers
----------------------

To define an uncertain real number, at least two pieces of information are needed: 

    #. A *value* (an estimate, or approximate value of, the quantity represented) 
    #. A *standard uncertainty* (the standard deviation of the distribution associated with error in the estimate). 
    
Example: an electrical circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, suppose the current flowing in an electrical circuit :math:`I` and the voltage across a circuit element :math:`V` have been measured. 

The measured values are :math:`x_V = 0.1\, \mathrm{V}` and :math:`x_I = 15\,\mathrm{mA}`, with standard uncertainties :math:`u(x_V) = 1\, \mathrm{mV}` and :math:`u(x_I) = 0.5\,\mathrm{mA}`, respectively. 

Uncertain numbers for :math:`V` and :math:`I` are defined using :func:`~core.ureal` ::

	>>> V = ureal(0.1,1E-3)
	>>> I = ureal(15E-3,0.5E-3)

The resistance can be calculated from these uncertain numbers directly using Ohm's law ::

    >>> R = V/I
    >>> print(R)
    6.67(23)
    
We obtain a measured value of resistance :math:`x_R = 6.67 \,\Omega`, which is an estimate (or approximation) for :math:`R`, the measurand. The standard uncertainty in :math:`x_R` as an estimate of :math:`R` is :math:`0.23 \,\Omega`.

Example: height of a flag pole
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose a flag is flying from a pole that has been measured to be 15 metres away from an observer (with an uncertainty of 3 cm). The angle between horizontal and line-of-sight to the top of the pole is measured as 38 degrees (with an uncertainty of 2 degrees). The question is: how high is the flag? 

A measurement model expresses the relationship between the quantities involved: the height of the pole :math:`H`, the distance to the base of the pole :math:`B` and the line-of-sight angle :math:`\Phi`,

.. math::

    H = B \tan\Phi \;.

To calculate the height, we create uncertain numbers representing the measured quantities and use the model to derive the result. ::

    >>> B = ureal(15,3E-2)
    >>> Phi = ureal(math.radians(38),math.radians(2))
    >>> H = B * tan(Phi)
    >>> print(H)
    11.72(84)
    
The measured value of 11.7 metres is our best estimate of the height :math:`H`. The standard uncertainty of this value, as an estimate of the actual height, is 0.8 metres. 
    
It is important to note that these calculations are open ended. We can continue the calculation above and evaluate what the observer angle would be at 20 metres from the pole (the uncertainty in the base distance remains 3 cm) ::

    >>> B_20 = ureal(20,3E-2)
    >>> Phi_20 = atan( H/B_20 ) 
    >>> print(Phi_20)
    0.530(31)
    >>> Phi_20_deg= Phi_20 * 180./math.pi
    >>> print(Phi_20_deg)
    30.4(1.8)

The value of 30.4 degrees for the angle at 20 metres from the pole has a standard uncertainty of 1.8 degrees.

Uncertain complex numbers
-------------------------

To define an uncertain number for a complex quantity, at least two pieces of information are needed: 

    #. A *value* (an estimate, or approximate value of, the quantity represented) 
    #. The extent of the distribution associated with error in the estimate. (For complex quantities, there are different ways to characterise the extent of the distribution.) 

Example: AC electric circuit 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
For example, suppose measurements have been made of: the alternating current :math:`i` flowing in an electrical circuit, the voltage :math:`v` across a circuit element and the phase :math:`\phi` of the voltage with respect to the current. The measured values are: :math:`x_v \approx 4.999\, \mathrm{V}`, :math:`x_i \approx 19.661\,\mathrm{mA}` and :math:`x_\phi \approx 1.04446\,\mathrm{rad}`, with standard uncertainties :math:`u(x_v) = 0.0032\, \mathrm{V}`, :math:`u(x_i) = 0.0095\,\mathrm{mA}` and :math:`u(x_\phi) = 0.00075\,\mathrm{rad}`. 

Uncertain numbers for the quantities :math:`v`, :math:`i` and :math:`\phi` can be defined using :func:`~core.ucomplex`::

    >>> v = ucomplex(complex(4.999,0),(0.0032,0))
    >>> i = ucomplex(complex(19.661E-3,0),(0.0095E-3,0))
    >>> phi = ucomplex(complex(0,1.04446),(0,0.00075))
    
Note, in these definitions, the second argument is a pair of numbers representing the standard uncertainties associated with measured values of the real and imaginary components.

The complex impedance is ::

    >>> z = v * exp(phi) / i
    >>> print(z)
    (127.73(19)+219.85(20)j)
    
We see that an estimate of the impedance is the complex value :math:`(127.73 +\mathrm{j}219.85) \,\Omega`. The standard uncertainty in the real component is :math:`0.19 \,\Omega` and the standard uncertainty in the imaginary component is :math:`0.20 \,\Omega`. There is also correlation between the real and imaginary components ::

    >>> get_correlation(z)
    0.05820381031583993
    
If a polar representation of the impedance is preferred, ::

    >>> print(magnitude(z))
    254.26(20)
    >>> print(phase(z))
    1.04446(75)

Elementary uncertain numbers
----------------------------
We use the term `elementary uncertain number` to describe uncertain numbers associated with problem inputs (e.g., ``B`` and ``Phi`` above). Elementary uncertain numbers are defined by functions like :func:`~core.ureal` and :func:`~core.ucomplex`.    

Uncertain Number Attributes
---------------------------

Uncertain numbers use attributes to provide access to the value (the estimate), the uncertainty (of the estimate) and the degrees of freedom (associated with the uncertainty), as well as some other properties (see :class:`~library_real.UncertainReal`).

Continuing with the flagpole example, the attributes ``x``, ``u``, ``df`` can be used to see the estimate, the uncertainty and the degrees-of-freedom (which is infinity), respectively ::

    >>> H.x
    11.719284397600761
    >>> H.u
    0.84353295110757898
    >>> H.df
    inf

Alternatively, there are ``GTC`` functions that return the same numbers ::

    >>> value(H)
    11.719284397600761
    >>> uncertainty(H)
    0.84353295110757898
    >>> dof(H)
    inf

Uncertain numbers and measurement errors
----------------------------------------

To make the best use of ``GTC`` it is helpful to think in terms of the actual quantities that appear in measurement equations. These quantities are not known exactly and many will be residual errors with estimates of zero or unity. 

In the context of the example above, :math:`B` and :math:`\Phi` are quantities in the measurement equation. When measured, there will be errors, which can be written as :math:`E_b` and :math:`E_\phi`. So the measured values :math:`b=15\,\mathrm{m}` and :math:`\phi=38 \, \mathrm{deg}` are related to the quantities of interest as

.. math :: 
  
        b = B + E_b 
        
        \phi = \Phi + E_\phi


Our best estimates of the errors are :math:`E_b \approx 0` and :math:`E_\phi \approx 0`, with uncertainties in these estimates of :math:`u(E_b)=3\times 10^{2}\, \mathrm{m}` and  :math:`u(E_\phi)=2\, \mathrm{deg}`. 

The ``GTC`` calculation now looks like this ::

    >>> b = 15
    >>> E_b = ureal(0,3E-2)
    >>> B = b - E_b
    >>> phi = math.radians(38)
    >>> E_phi = ureal(0,math.radians(2))
    >>> Phi = phi - E_phi
    >>> H = B * tan(Phi)
    >>> H
    ureal(11.719284397600761, 0.843532951107579, inf)

This way of expressing the calculation reflects our understanding of the problem: :math:`b=15` and :math:`\phi=38` are precisely known numbers, there is nothing 'uncertain' about their values. However, when we use those number as estimates of :math:`B` and :math:`\Phi` the unknown errors :math:`E_b` and :math:`E_\phi` give rise to uncertainty.

Measurements are usually easier to analyse by making the errors explicit in this way. 


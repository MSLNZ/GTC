.. _intro_uncertain_numbers: 

############
Introduction
############

.. contents::
   :local:

The GUM Tree calculator (``GTC``) is a data processing tool that uses `uncertain numbers` to represent measured quantities, and automates the evaluation of uncertainty when derived quantities are calculated from measured data. 

.. _measurement_error:

Measurement error
=================

A measurement obtains information about a quantity; but the quantity itself (the *measurand*) is never determined exactly, it can only be estimated. There is always some *measurement error* involved. Writing this as an equation, where the unknown measurand is :math:`Y` and the measurement result is :math:`y`, we have

.. math::

    y = Y + E_y\; ,
    
where :math:`E_y` represents the measurement error. So, the measurement result, :math:`y`, is not quite what is wanted; it only provides an approximate value for :math:`Y`. 

This is how 'uncertainty' arises. After any measurement, we are faced with uncertainty about what will happen if we take the measured value :math:`y` and use it instead of the (unknown) value of the measurand, :math:`Y`. 

For example, suppose the speed of a car is measured by a law enforcement officer. The measurement is made to decide whether, in fact, a car was travelling faster than the legal limit. However, this simple fact cannot be determined perfectly, because the actual speed :math:`Y` remains unknown. It is possible that the measured value :math:`y` will indicate that the car was speeding when in fact it was not, or that it was not speeding when in fact it was. The difficulty in making the correct decision is inevitable. So, in practice, a decision rule must be used that takes account of the measurement uncertainty. In this example, the rule will probably err on the side of caution (a few speeding drivers will escape rather than unfairly accusing good drivers of speeding).

Like the measurand, the measurement error :math:`E_y` is not known. At best its behaviour can be described in statistical terms. This leads to more technical uses of the word 'uncertainty'. For instance, the term 'standard uncertainty' refers to the standard deviation of a distribution associated with an unpredictable quantity.

.. _measurement_models:

Measurement models
------------------
It is generally possible to identify the most important factors that influence the outcome of a measurement process, thereby contributing to the final measurement error. In a formal analysis, these factors must be included in a measurement model, which defines the measurand in terms of all other significant influence quantities. In mathematical terms, we write   

.. math::

    Y = f(X_1, X_2, \cdots) \;,
 
where the :math:`X_i` are influence quantities. 

Once again, the actual quantities :math:`X_1, X_2, \cdots` are not known; only estimates :math:`x_1, x_2, \cdots` are available. These are used to calculate a measured value that is approximately equal to the measurand 

.. math::

        y = f(x_1, x_2, \cdots) \;.

     
Uncertain Numbers
=================

Uncertain numbers are data-types designed to represent measured quantities. They encapsulate information about the measurement, including the measured value and the uncertainty of the measurement process. 

Uncertain numbers are are intended to be used to process measurement data; that is, to evaluate measurement models. The inputs to a measurement model (like :math:`X_1, X_2, \cdots` above) are defined as uncertain numbers. Calculations then obtain an uncertain number for the measurand (:math:`Y`). 

There are two types of uncertain number: one for real-valued quantities and one for complex-valued quantities. At the very least, two pieces of information are needed to define an uncertain number: a value (that is a measured, or approximate, value of the quantity) and an uncertainty that describes the extent of a distribution associated with error in the measured value.

Uncertain real numbers
----------------------
    
The function :func:`~core.ureal` is often used to define uncertain real number inputs. 
    
Example: an electrical circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose the current flowing in an electrical circuit :math:`I` and the voltage across a circuit element :math:`V` have been measured. 

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

To calculate the height, we create uncertain numbers representing the measured quantities and use the model. ::

    >>> B = ureal(15,3E-2)
    >>> Phi = ureal(math.radians(38),math.radians(2))
    >>> H = B * tan(Phi)
    >>> print(H)
    11.72(84)
    
The result 11.7 metres is our best estimate of the height :math:`H`. The standard uncertainty of this value, as an estimate of the actual height, is 0.8 metres. 
    
It is important to note that uncertain-number calculations are open ended. In this case, for example, we can keep going and evaluate what the observer angle would be at 20 metres from the pole (the uncertainty in the base distance remains 3 cm) ::

    >>> B_20 = ureal(20,3E-2)
    >>> Phi_20 = atan( H/B_20 ) 
    >>> print(Phi_20)
    0.530(31)
    >>> Phi_20_deg= Phi_20 * 180./math.pi
    >>> print(Phi_20_deg)
    30.4(1.8)

The value of 30.4 degrees obtained for the angle, at 20 metres from the pole, has a standard uncertainty of 1.8 degrees.

Uncertain complex numbers
-------------------------

The function :func:`~core.ucomplex` is often used to define uncertain complex number inputs. 

Example: AC electric circuit 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
Suppose measurements have been made of: the alternating current :math:`i` flowing in an electrical circuit, the voltage :math:`v` across a circuit element and the phase :math:`\phi` of the voltage with respect to the current. The measured values are: :math:`x_v \approx 4.999\, \mathrm{V}`, :math:`x_i \approx 19.661\,\mathrm{mA}` and :math:`x_\phi \approx 1.04446\,\mathrm{rad}`, with standard uncertainties :math:`u(x_v) = 0.0032\, \mathrm{V}`, :math:`u(x_i) = 0.0095\,\mathrm{mA}` and :math:`u(x_\phi) = 0.00075\,\mathrm{rad}`. 

Uncertain numbers for the quantities :math:`v`, :math:`i` and :math:`\phi` can be defined using :func:`~core.ucomplex`::

    >>> v = ucomplex(complex(4.999,0),(0.0032,0))
    >>> i = ucomplex(complex(19.661E-3,0),(0.0095E-3,0))
    >>> phi = ucomplex(complex(0,1.04446),(0,0.00075))
    
Note, in these definitions, the uncertainty argument is a pair of numbers. These represent the standard uncertainties associated with measured values of the real and imaginary components.

The complex impedance is ::

    >>> z = v * exp(phi) / i
    >>> print(z)
    (127.73(19)+219.85(20)j)
    
We see that our best estimate of the impedance is the complex value :math:`(127.73 +\mathrm{j}219.85) \,\Omega`. The standard uncertainty in the real component is :math:`0.19 \,\Omega` and the standard uncertainty in the imaginary component is :math:`0.20 \,\Omega`. There is also some correlation between the real and imaginary components ::

    >>> get_correlation(z)
    0.05820381031583993
    
If a polar representation of the impedance is preferred, ::

    >>> print(magnitude(z))
    254.26(20)
    >>> print(phase(z))
    1.04446(75)


Uncertain Number Attributes
---------------------------

Uncertain number objects have attributes that provide access to: the measured value (the estimate), the uncertainty (of the estimate) and the degrees of freedom (associated with the uncertainty) (see :class:`~lib.UncertainReal`).

Continuing with the flagpole example, the attributes ``x``, ``u``, ``df`` can be used to show the value, the uncertainty and the degrees-of-freedom (which is infinity), respectively ::

    >>> H.x
    11.719284397600761
    >>> H.u
    0.843532951107579
    >>> H.df
    inf

Alternatively, there are functions that return the same attributes ::

    >>> value(H)
    11.719284397600761
    >>> uncertainty(H)
    0.843532951107579
    >>> dof(H)
    inf

Uncertain numbers and measurement errors
----------------------------------------

It is often is helpful to to formulate problems by explicitly acknowledging measurement errors. As we have said above, these errors are not known exactly; many will be residual quantities with estimates of zero or unity. However, errors usually have a physical meaning in the model that can be identified and it is often useful to do so. 

In the context of the example above, the errors associated with measured values of :math:`B` and :math:`\Phi` were not identified. We can introduce these terms as :math:`E_b` and :math:`E_\phi`. Then, the measured values :math:`b=15\,\mathrm{m}` and :math:`\phi=38 \, \mathrm{deg}` are related to the quantities of interest as

.. math :: 
  
        B = b - E_b 
        
        \Phi = \phi - E_\phi


Our best estimates of these errors :math:`E_b \approx 0` and :math:`E_\phi \approx 0` are trivial, but uncertainties can now be correctly associated with these unpredictable errors :math:`u(E_b)=3\times 10^{2}\, \mathrm{m}` and  :math:`u(E_\phi)=2\, \mathrm{deg}`, not with the invariant quantities :math:`B` and :math:`\Phi`. 

The calculation can be carried out simply as ::

    >>> B = 15 - ureal(0,3E-2,label='E_b')
    >>> Phi = math.radians(38) - ureal(0,math.radians(2),label='E_phi')
    >>> H = B*tan(Phi)
    >>> print(H)
    11.72(84)

This calculation reflects our understanding of the problem better: the numbers :math:`b=15` and :math:`\phi=38` are known, there is nothing 'uncertain' about their values. What is uncertain, however, is how to correct for the unknown errors :math:`E_b` and :math:`E_\phi`. 

The use of labels, when defining the uncertain numbers, allows us to display an uncertainty budget (see :func:`~reporting.budget`) ::

    >>> for cpt in rp.budget(H):
    ...     print("{0.label}: {0.u:.3f}".format(cpt))
    ...
    E_phi: 0.843
    E_b: 0.023

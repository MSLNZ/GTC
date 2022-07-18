from collections import namedtuple

VarianceAndDof = namedtuple('VarianceAndDof','cv, df')
""":obj:`~collections.namedtuple`: Values of the variance and degrees of freedom.
 
.. attribute:: cv
  
   Variance.
  
.. attribute:: df
  
   :class:`float`: Degrees of freedom.
 
"""

VarianceCovariance = namedtuple('VarianceCovariance','rr, ri, ir, ii')
""":obj:`~collections.namedtuple`: Values of variance-covariance for a complex quantity
 
.. attribute:: rr
  
   :class:`float`: variance in the real component
  
.. attribute:: ri
  
   :class:`float`: covariance between the real and imaginary components
 
.. attribute:: ir
  
   :class:`float`: covariance between the real and imaginary components
   
.. attribute:: ii
  
   :class:`float`: variance in the imaginary component

   """

StandardUncertainty = namedtuple('StandardUncertainty','real,imag')
""":obj:`~collections.namedtuple`: Standard uncertainty values of a complex quantity
 
.. attribute:: real
  
   :class:`float`: standard uncertainty in the real component
   
.. attribute:: imag
  
   :class:`float`: standard uncertainty in the imaginary component
   
"""

StandardDeviation = namedtuple('StandardDeviation','real,imag')
""":obj:`~collections.namedtuple`: Standard deviation values of a complex quantity
 
.. attribute:: real
  
   :class:`float`: standard deviation in the real component
   
.. attribute:: imag
  
   :class:`float`: standard deviation in the imaginary component
   
"""

ComponentOfUncertainty = namedtuple('ComponentOfUncertainty','rr, ri, ir, ii')
""":obj:`~collections.namedtuple`: Component of uncertainty values for a complex quantity
 
.. attribute:: rr
  
   :class:`float`: real component with respect to real component
  
.. attribute:: ri
  
   :class:`float`: real component with respect to imaginary component
 
.. attribute:: ir
  
   :class:`float`: imaginary component with respect to real component
   
.. attribute:: ii
  
   :class:`float`: imaginary component with respect to imaginary component

   """

JacobianMatrix = namedtuple('JacobianMatrix','rr, ri, ir, ii')
""":obj:`~collections.namedtuple`: Partial derivatives of one complex with respect to another
 
.. attribute:: rr
  
   :class:`float`: real component with respect to real component
  
.. attribute:: ri
  
   :class:`float`: real component with respect to imaginary component
 
.. attribute:: ir
  
   :class:`float`: imaginary component with respect to real component
   
.. attribute:: ii
  
   :class:`float`: imaginary component with respect to imaginary component

"""

Influence = namedtuple('Influence','label, u, uid')
""":obj:`~collections.namedtuple`: label, value, and identifier of a component of uncertainty
 
.. attribute:: label
  
   :class:`str`: influence quantity label
   
.. attribute:: u
  
   :class:`float`: component of uncertainty
   
.. attribute:: uid
  
   :class:`tuple`: unique identifier of the uncertain number
   
..  versionadded:: 1.3.7
    The attribute `uid` has been added
        
"""

Component = namedtuple('Component','uid, u')
""":obj:`~collections.namedtuple`: unique identifier and value of a component of uncertainty
 
.. attribute:: uid
  
   :class:`tuple`: unique identifier tuple
   
.. attribute:: u
  
   :class:`float`: component of uncertainty
   
"""

CorrelationMatrix = namedtuple("CorrelationMatrix","rr,ri,ir,ii")
""":obj:`~collections.namedtuple`: Correlation coefficients for a pair of quantities ``x`` and ``y``
 
.. attribute:: rr
  
   :class:`float`: correlation between ``x.real`` and ``y.real``
  
.. attribute:: ri
  
   :class:`float`: correlation between ``x.real`` and ``y.imag``
 
.. attribute:: ir
  
   :class:`float`: correlation between ``x.imag`` and ``y.real``
   
.. attribute:: ii
  
   :class:`float`: correlation between ``x.imag`` and ``y.imag``

"""

CovarianceMatrix = namedtuple("CovarianceMatrix","rr,ri,ir,ii")
""":obj:`~collections.namedtuple`: Values of covariance for a pair of quantities ``x`` and ``y``
 
.. attribute:: rr
  
   :class:`float`: covariance between ``x.real`` and ``y.real``
  
.. attribute:: ri
  
   :class:`float`: covariance between ``x.real`` and ``y.imag``
 
.. attribute:: ir
  
   :class:`float`: covariance between ``x.imag`` and ``y.real``
   
.. attribute:: ii
  
   :class:`float`: covariance between ``x.imag`` and ``y.imag``

"""

InterceptSlope = namedtuple('InterceptSlope','a b')
""":obj:`~collections.namedtuple`: Values for intercept ``a`` and slope ``b``
 
.. attribute:: a
  
   :class:`~.lib.UncertainReal`: intercept
  
.. attribute:: b
  
   :class:`~.lib.UncertainReal`: slope
"""

# TODO should the r attribute be added for symmetry with the complex case?
FormattedUncertainReal = namedtuple('FormattedUncertainReal', 'x u df label')
""":obj:`~collections.namedtuple`: The formatted representation of an :class:`~.lib.UncertainReal` number.

.. versionadded:: 1.4.0

.. attribute:: x

   :class:`float`: value 

.. attribute:: u

   :class:`float`: standard uncertainty

.. attribute:: df

   :class:`float`: degrees of freedom

.. attribute:: label

   :class:`str`: the label of the :class:`~.lib.UncertainReal` number

"""

FormattedUncertainComplex = namedtuple('FormattedUncertainComplex', 'x u r df label')
""":obj:`~collections.namedtuple`: The formatted representation of an :class:`~.lib.UncertainComplex` number.

.. versionadded:: 1.4.0

.. attribute:: x

   :class:`complex`: value 

.. attribute:: u

   :class:`StandardUncertainty`: standard uncertainty

.. attribute:: r

   :class:`float`: correlation coefficient between real and imaginary components

.. attribute:: df

   :class:`float`: degrees of freedom

.. attribute:: label

   :class:`str`: the label of the :class:`~.lib.UncertainComplex` number

"""

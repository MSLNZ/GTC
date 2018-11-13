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
  
   :class:`float`: covariance between th real and imaginary components
 
.. attribute:: ir
  
   :class:`float`: covariance between th real and imaginary components
   
.. attribute:: ii
  
   :class:`float`:  variance in the imaginary component

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

Influence = namedtuple('Influence','label, u')
""":obj:`~collections.namedtuple`: label and value of a component of uncertainty
 
.. attribute:: label
  
   :class:`str`: influence quantity label
   
.. attribute:: u
  
   :class:`float`: component of uncertainty due to influence quantity
   
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
   
GroomedUncertainReal = namedtuple('ureal','x u df label precision df_decimals u_digits')
GroomedUncertainComplex = namedtuple(
    'ucomplex','x u r df label precision df_decimals re_u_digits im_u_digits'
)

# TypeA = namedtuple('type_a','x u df r')
# ExpandedUncertainty = namedtuple('expanded_uncertainty','lower upper')
# RadialTangentialUncertainty = namedtuple('rt_uncertainty','radial, tangent')
# PolarCoordinates = namedtuple('polar','magnitude, phase')

# SimulatedRealInput = namedtuple('simulated_real_input','x,u,df')

# SimulatedComplexInput = namedtuple(
        # 'simulated_complex_input',
        # 'z cv df'
    # )
# SimulatedVectorInput = namedtuple(
        # 'simulated_vector_input',
        # 'x cv df'
    # )
# InterceptSlope = namedtuple('intercept_slope','a b')

# JacobianMatrix = namedtuple('jacobian_matrix','rr ri ir ii')


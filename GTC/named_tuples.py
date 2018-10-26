from collections import namedtuple

VarianceAndDof = namedtuple('variance_and_dof','cv, df')
VarianceCovariance = namedtuple('variance_covariance','rr, ri, ir, ii')
StandardUncertainty = namedtuple('standard_uncertainty','real,imag')
StandardDeviation = namedtuple('standard_deviation','real,imag')
ComponentOfUncertainty = namedtuple('u_components','rr, ri, ir, ii')
Influence = namedtuple('influence','label, u')
CorrelationMatrix = namedtuple("correlation_matrix","rr,ri,ir,ii")
CovarianceMatrix = namedtuple("covariance_matrix","rr,ri,ir,ii")

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


"""
A Python package for evaluating measurement uncertainty
in real and complex quantities.

The method of uncertainty propagation is compatible with the approach described
in the 'Guide to the Expression of Uncertainty in Measurement' - the GUM.

"""
from __future__ import division

import math
import cmath
import sys 

try:
    basestring  # Python 2
    from collections import Sequence
except NameError:
    basestring = (str, bytes)
    from collections.abc import Sequence

#----------------------------------------------------------------------------
# Global constants, etc

# The degrees of freedom is considered infinite above `inf_dof`
inf_dof = 1E5

inf = float('inf')
nan = float('nan')

EPSILON = sys.float_info.epsilon 

# Do not consider strings as sequences
# Note numpy arrays are not detected as sequences
# by this function.
def is_sequence(obj):
    if isinstance(obj, basestring):
        return False
    else:
        return isinstance(obj, Sequence)

#----------------------------------------------------------------------------

__all__ = (
        'ureal'
    ,   'multiple_ureal'
    ,   'multiple_ucomplex'
    ,   'ucomplex'
    ,   'constant'
    ,   'value'
    ,   'uncertainty'
    ,   'variance'
    ,   'dof'
    ,   'label'
    ,   'uid'
    ,   'component'
    ,   'inf'
    ,   'nan'
    ,   'get_correlation'
    ,   'set_correlation'
    ,   'result'
    ,   'get_covariance'
    ,   'cos'
    ,   'sin'
    ,   'tan'
    ,   'acos'
    ,   'asin'
    ,   'atan'
    ,   'atan2'
    ,   'exp'
    ,   'pow'
    ,   'log'
    ,   'log10'
    ,   'sqrt'
    ,   'sinh'
    ,   'cosh'
    ,   'tanh'
    ,   'acosh'
    ,   'asinh'
    ,   'atanh'
    ,   'mag_squared'
    ,   'magnitude'
    ,   'phase'
    ,   'fmod'
    ,   'copyright'
    ,   'version'
    ,   'reporting',    'rp'
    ,   'function',     'fn'
    ,   'type_b',       'tb'
    ,   'type_a',       'ta'
    ,   'persistence',  'pr'
    ,   'linear_algebra', 'la'
    ,   'math'
    ,   'cmath'
    ,   'inf'
    ,   'nan'
    ,   'EPSILON'
    ,   'create_format'
    ,   'apply_format'
    ,   'to_string'
)

#----------------------------------------------------------------------------
if sys.version_info[:2] < (3, 8):
    import warnings
    warnings.simplefilter('once', DeprecationWarning)
    warnings.warn(
        'GTC will stop supporting Python %d.%d in GTC version 2.0. '
        'GTC 2.0 will require Python 3.8, or above. '
        'Please update your version of Python.' % sys.version_info[:2],
        DeprecationWarning,
        stacklevel=2
    )
    del warnings
#----------------------------------------------------------------------------
version = "1.5.2.dev0"

copyright = """Copyright (c) 2024, \
Measurement Standards Laboratory of New Zealand"""

from .core import *
from .formatting import *

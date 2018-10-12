"""
GTC is a Python package for evaluating measurement uncertainty 
in real and complex quantities.

Calculations involving uncertain numbers propagate uncertainty
according to methods described in the
'Guide to the Expression of Uncertainty in Measurement' - the GUM.

    
Copyright (c) 2018, Callaghan Innovation, All rights reserved.

"""
from __future__ import division

import math
import cmath
import collections

# GTC global constants, etc  
inf_dof = 1E5               # DoF is considered infinite

inf = float('inf')
nan = float('nan') 

is_infinity = math.isinf 
is_undefined = math.isnan

def is_sequence(obj):
    if isinstance(obj, basestring):
        return False
    return isinstance(obj, collections.Sequence)
    
from core import *

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
    ,   'copyright', 'version'
    ,   'reporting',   'rp'
    ,   'type_b', 'tb'
    ,   'type_a', 'ta'
    ,   'math'
    ,   'cmath'
    ,   'is_infinity'
    ,   'is_undefined'
    ,   'inf'
    ,   'nan'
)
    
#----------------------------------------------------------------------------
version = "2.0.1"
copyright = "Copyright (c) 2018, Callaghan Innovation. All rights reserved."




       
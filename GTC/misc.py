import numpy as np
from collections.abc import Sequence

#----------------------------------------------------------------------------
# Do not consider strings as sequences
def is_sequence(array):
    if isinstance(array, (str, bytes)): 
        return False
    else:
        return isinstance(array, Sequence)

#----------------------------------------------------------------------------
def _dtype_float(a):
    """Promote integer arrays to float 
    
    Use this to avoid creating an array that might truncate values when 
    you do not know the dtype.
    
    """
    try:
        if np.issubdtype(a.dtype, np.integer):
            return np.float64
        else:
            return a.dtype
    except AttributeError:  
            return np.float64
            
#----------------------------------------------------------------------------
def is_numeric_non_complex(value):
    if isinstance(value, (int, float)):
        return True
    elif np.issubdtype(type(value), np.number):
        return not np.issubdtype(type(value), np.complexfloating)
    else:
        return False

#----------------------------------------------------------------------------
def is_numeric_complex(value):
    if isinstance(value, complex):
        return True
    elif np.issubdtype(type(value), np.number):
        return np.issubdtype(type(value), np.complexfloating)
    else:
        return False

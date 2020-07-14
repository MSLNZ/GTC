import json

from archive import (
    FrozenLeaf, 
    TaggedElementaryReal,
    TaggedIntermediateReal,
    TaggedElementaryComplex,
    TaggedIntermediateComplex
)

def fr_leaf_json(x):
    tags = [
        'uid','label','u','df','independent',
        'complex','correlation','ensemble'
    ]
    return {
        t_i : getatrr(x,t_i)
            for t_i in tags if hasattr(x,t_i) 
    }
    
def el_real_json(x):
    tags = [ 'x','uid' ]
    return {
        t_i : getatrr(x,t_i) for t_i in tags 
    }

def int_real_json(x):
    tags = [ 
        'value','u_components','d_components','i_components',
        'label','uid'
    ]
    return {
        t_i : getatrr(x,t_i) for t_i in tags 
    }
    
def el_complex_json(x):
    return dict()
def int_complex_json(x):
    return dict()

class ArchiveEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x,FrozenLeaf):
            return fr_leaf_json(x)
        elif isinstance(z,TaggedElementaryReal):
            return el_real_json(x)
        elif isinstance(z,TaggedIntermediateReal):
            return int_real_json(x)
        elif isinstance(z,TaggedElementaryComplex):
            return el_complex_json(x)
        elif isinstance(z,TaggedIntermediateComplex):
            return int_complex_json(x)
        else:
            return super().default(z)
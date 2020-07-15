import json

from archive import (
    Archive,
    LeafNode, 
    ElementaryReal,
    IntermediateReal,
    ElementaryComplex,
    IntermediateComplex
)

from vector import Vector

__all__ = ( 'JSONArchiveEncoder' )

#----------------------------------------------------------------------------
# 
def vector_to_json(x): 
    return {
        str(i) : float(x_i) 
            for (i,x_i) in x.iteritems()
    }
    
#----------------------------------------------------------------------------
# 
def leaf_to_json(x):
    j = dict(
        CLASS = x.__class__.__name__,
        uid = tuple(x.uid),
        label = str(x.label),
        u = float(x.u),
        df = float(x.df),
        independent = bool(x.independent)
    )
    # The last 3 attributes may not be assigned 
    # TODO: need to change these object types
    if hasattr(x,'complex'):
        j['complex'] = x.complex
    if hasattr(x,'correlation'):
        j['correlation'] = x.correlation
    if hasattr(x,'ensemble'):
        j['ensemble'] = x.ensemble
    
    return j
 
#----------------------------------------------------------------------------
# 
def tagged_to_json(x):
    if isinstance(x,ElementaryReal):
        return el_real_to_json(x)
    elif isinstance(x,IntermediateReal):
        return int_real_to_json(x)
    elif isinstance(x,ElementaryComplex):
        return el_complex_to_json(x)
    elif isinstance(x,IntermediateComplex):
        return int_complex_to_json(x)
 
#----------------------------------------------------------------------------
# 
def el_real_to_json(x):
    return dict( 
        CLASS = x.__class__.__name__, 
        x = float(x.x), 
        uid = tuple(x.uid) 
    ) 

#----------------------------------------------------------------------------
# 
def int_real_to_json(x):
    j =  dict(
        CLASS = x.__class__.__name__, 
        value= float(x.value), 
        label = str(x.label), 
        uid= tuple(x.uid) 
    ) 
    j['u_components'] = vector_to_json(x.u_components)
    j['d_components'] = vector_to_json(x.d_components)
    j['i_components'] = vector_to_json(x.i_components)

    return j
    
#----------------------------------------------------------------------------
# 
def el_complex_to_json(x):
    return dict()
#----------------------------------------------------------------------------
# 
def int_complex_to_json(x):
    return dict()

#----------------------------------------------------------------------------
# 
def archive_to_json(a): 
    
    j = dict( CLASS = a.__class__.__name__ ) 
    
    j['leaf_nodes'] = {
        str(i) : leaf_to_json(o_i)
            for (i, o_i) in a._leaf_nodes.iteritems()
    }
    
    j['tagged'] = {
        str(i) : tagged_to_json(o_i)
            for (i, o_i) in a._tagged.iteritems()
    }
    
    j['tagged_reals'] = {
        str(i) : tagged_to_json(o_i)
            for (i, o_i) in a._tagged_reals.iteritems()
    }
    
    j['intermediate_uids'] = {
        str(i) : (o_i)
            for (i, o_i) in a._intermediate_uids.iteritems()
    }
   
    return j
    
#----------------------------------------------------------------------------
class JSONArchiveEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x,Archive):
            return archive_to_json(x)
        else:
            return super(JSONArchiveEncoder,self).default(x)
            

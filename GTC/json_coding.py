import json

from GTC.archive import (
    Archive,
    LeafNode, 
    ElementaryReal,
    IntermediateReal,
    Complex
)

from GTC.vector import Vector

__all__ = ( 
    'JSONArchiveEncoder',
    'json_to_archive'
)

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
    if hasattr(x,'complex'):
        j['complex'] = [ 
            tuple(x_i) for x_i in x.complex
        ]
    if hasattr(x,'correlation'):
        j['correlation'] = { 
            tuple(uid) : x_i for (uid,x_i) in x.correlation
        }
    if hasattr(x,'ensemble'):
        j['ensemble'] = [ 
            tuple(x_i) for x_i in x.ensemble
        ]
    
    return j
 
#----------------------------------------------------------------------------
# 
def tagged_to_json(x):
    if isinstance( x, ElementaryReal ):
        return el_real_to_json(x)
    elif isinstance( x, IntermediateReal ):
        return int_real_to_json(x)
    elif isinstance(x, Complex ):
        return complex_to_json(x)
    else:
        raise TypeError( "Unrecognised: {}".format( type(x) ) )
 
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
def complex_to_json(x):
    return dict(
        CLASS = x.__class__.__name__, 
        n_re = str(x.n_re), 
        n_im = str(x.n_im), 
        label = tuple(x.label) 
    ) 

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
#
class JSONArchiveEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o,Archive):
            return archive_to_json(o)
        else:
            return super(JSONArchiveEncoder,self).default(o)
            
#----------------------------------------------------------------------------
# 
def json_to_archive(js): 

    if 'CLASS' in js and js['CLASS'] == 'Archive':
        ar = Archive() 

        ar._leaf_nodes = {
            eval(i) : LeafNode(d)
                for (i,d) in js['leaf_nodes'].iteritems()
        }

        return ar 
    else:
        # Don't touch the JSON object
        return js
        
    

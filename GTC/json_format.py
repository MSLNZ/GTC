"""
This module handles conversion of an archive object to a JSON format 
and then restoration of an archive from JSON.
"""
import json
import math
import ast

from GTC import inf
from GTC.archive import (
    Archive,
    LeafNode, 
    ElementaryReal,
    IntermediateReal,
    Complex,
    PY2,
)

from GTC.vector import Vector
from GTC.nodes import Leaf

__all__ = ( 
    'JSONArchiveEncoder',
    'json_to_archive'
)

# This is the same as $id in the schema and will also be
# the value of the 'version' property in a JSON Archive file.
JSON_SCHEMA = r"https://measurement.govt.nz/gtc/json_1.5.0"

# math.inf cannot be represented in JSON so we adopt null (None)
to_dof_json = lambda df: None if math.isinf(df) else df
from_dof_json = lambda s: inf if s is None else s
#     
to_uid_string = lambda uid: repr(uid)
from_uid_string = lambda s: ast.literal_eval(s) 

#----------------------------------------------------------------------------
# 
def jason_to_leaf(j):
    """
    Return a Leaf node to initialise archive.LeafNode 
    
    """
    n = Leaf(
        uid = from_uid_string( j['uid'] ),
        label = j['label'],
        u = j['u'],
        df = from_dof_json( j['df'] ),
        independent = j['independent']
    )
    
    if 'complex' in j:
        n.complex = [
            from_uid_string( j['complex'][0] ),
            from_uid_string( j['complex'][1] )
        ]
    if 'correlation' in j:
        items = j['correlation'].iteritems() if PY2 else j['correlation'].items()
        n.correlation = {
            from_uid_string(x_i) : r_i 
                for x_i,r_i in items
        }
    if 'ensemble' in j:
        n.ensemble = frozenset( 
            from_uid_string(i) for i in j['ensemble'] 
        )                               

    return n
    
#----------------------------------------------------------------------------
# 
def vector_to_json(x): 
    return dict(
        CLASS = x.__class__.__name__,
        index = [ to_uid_string(k_i) for k_i in x.keys() ],
        value = x.values()
    )
    
    # # Alternative representation as key-value pairs?
    # return dict( 
        # CLASS = x.__class__.__name__,
        # items=list( zip(
            # [to_uid_string(k_i) for k_i in x.keys()],
            # x.values()
        # ) )
    # )
    
#----------------------------------------------------------------------------
# 
def leaf_to_json(x):
    j = dict(
        CLASS = x.__class__.__name__,
        uid = to_uid_string( tuple(x.uid) ),
        label = x.label,
        u = x.u,
        df = to_dof_json( float(x.df) ),
        independent = x.independent
    )
    
    # These 3 attributes may not have been assigned 
    if hasattr(x,'complex'):
        # The `complex` attribute is a tuple of UIDs for the 
        # real and imaginary components.
        j['complex'] = [ 
            to_uid_string(cpt_i) for cpt_i in x.complex
        ]
    if hasattr(x,'correlation'):
        # `correlation` is a dict in the GTC Leaf node, but this 
        # does not pickle, so LeafNode holds a list of item pairs.
        j['correlation'] = { 
            to_uid_string(uid_i) : x_i for (uid_i,x_i) in x.correlation
        }
    if hasattr(x,'ensemble'):
        # `ensemble` is a set in the GTC Leaf node, but JSON will use an array
        j['ensemble'] = [ 
            to_uid_string(uid_i) for uid_i in x.ensemble
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
        x = x.x, 
        uid = to_uid_string(x.uid) 
    ) 

#----------------------------------------------------------------------------
# 
def int_real_to_json(x):
    j =  dict(
        CLASS = x.__class__.__name__, 
        value= x.value, 
        label = x.label, 
        uid= to_uid_string(x.uid) 
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
        n_re = x.n_re, 
        n_im = x.n_im, 
        label = x.label
    ) 

#----------------------------------------------------------------------------
# 
def archive_to_json(a): 
    
    j = dict( 
        CLASS = a.__class__.__name__, 
        version = JSON_SCHEMA
    )

    if PY2:
        leaf_nodes_items = a._leaf_nodes.iteritems
        tagged_real_items = a._tagged_real.iteritems
        tagged_complex_items = a._tagged_complex.iteritems
        untagged_real_items = a._untagged_real.iteritems
        intermediate_uids_items = a._intermediate_uids.iteritems
    else:
        leaf_nodes_items = a._leaf_nodes.items
        tagged_real_items = a._tagged_real.items
        tagged_complex_items = a._tagged_complex.items
        untagged_real_items = a._untagged_real.items
        intermediate_uids_items = a._intermediate_uids.items
    
    j['leaf_nodes'] = {
        to_uid_string(i) : leaf_to_json(o_i)
            for (i, o_i) in leaf_nodes_items()
    }
    
    j['tagged_real'] = {
        tag_i : tagged_to_json(o_i)
            for (tag_i, o_i) in tagged_real_items()
    }
    
    j['tagged_complex'] = {
        tag_i : tagged_to_json(o_i)
            for (tag_i, o_i) in tagged_complex_items()
    }
    
    j['untagged_real'] = {
        tag_i : tagged_to_json(o_i)
            for (tag_i, o_i) in untagged_real_items()
    }
    
    j['intermediate_uids'] = {
        to_uid_string(i) : [ o_i[0], o_i[1], to_dof_json(o_i[2]) ]
            for (i, o_i) in intermediate_uids_items()
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
def json_to_archive(j): 
    """
    Called during retrieval of an archive in JSON format by `json.loads()`
    The function is called every time an object is not a recognised JSON type
    during the `loads()` parsing of a JSON record. `j` is always a dict. 
    By transforming `j` into an appropriate object we can reconstruct the 
    elements of the archive and finally assemble the archive object.
    
    """
    if 'CLASS' in j and (j['CLASS'] == Vector.__name__):  
        return Vector( 
            index=[ from_uid_string(i) for i in j['index'] ], 
            value=j['value'] 
        )

        # # Alternative representation as key-value pairs?
        # index, value = [],[]
        # for i,v in j['items']:
            # index.append( from_uid_string(i) )
            # value.append( v )
        # return Vector( index=index, value=value )
        
    elif 'CLASS' in j and (j['CLASS'] == LeafNode.__name__):
        return LeafNode( jason_to_leaf(j) ) 
        
    elif 'CLASS' in j and (j['CLASS'] == ElementaryReal.__name__):
        return ElementaryReal(
            j['x'],
            from_uid_string( j['uid'] )     
        )
        
    elif 'CLASS' in j and (j['CLASS'] == IntermediateReal.__name__):
        return IntermediateReal(
            j['value'],
            j['u_components'],
            j['d_components'],
            j['i_components'],
            j['label'],
            from_uid_string( j['uid'] )     
        )
        
    elif 'CLASS' in j and (j['CLASS'] == Complex.__name__):
        return Complex(
            j['n_re'],
            j['n_im'],
            j['label']
        )
        
    elif 'CLASS' in j and (j['CLASS'] == Archive.__name__):
        ar = Archive() 
        ar._dump = ar._ready = False
        
        # Load the data
        if PY2:
            leaf_nodes_items = j['leaf_nodes'].iteritems
            intermediate_uids_items = j['intermediate_uids'].iteritems
        else:
            leaf_nodes_items = j['leaf_nodes'].items
            intermediate_uids_items = j['intermediate_uids'].items

        ar._leaf_nodes = {
            from_uid_string(i) : o
                for (i,o) in leaf_nodes_items()
        }

        # Mapping uid -> (label, u)
        ar._intermediate_uids = {
            from_uid_string(i) : (
                args[0],
                args[1],
                from_dof_json( args[2] )
            ) for (i,args) in intermediate_uids_items()
        }
 
        ar._tagged_real = j['tagged_real']
        ar._tagged_complex = j['tagged_complex']
        ar._untagged_real = j['untagged_real']

        return ar 
        
    else:
        # Allow parsing to continue at the next level
        return j 
        
  

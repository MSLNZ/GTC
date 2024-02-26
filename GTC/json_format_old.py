"""
This module handles conversion of an archive object to a JSON format 
and then restoration of an archive from JSON.
"""
import json

from GTC.archive_old import (
    Archive,
    LeafNode, 
    ElementaryReal,
    IntermediateReal,
    Complex,
    PY2,
)

from GTC.vector import Vector

__all__ = ( 
    'JSONArchiveEncoder',
    'json_to_archive'
)

#----------------------------------------------------------------------------
# 
def vector_to_json(x): 
    return dict(
        CLASS = x.__class__.__name__,
        index = x.keys(),
        value = x.values()
    )
    
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
            str(x_i) for x_i in x.complex
        ]
    if hasattr(x,'correlation'):
        j['correlation'] = { 
            str(uid) : x_i for (uid,x_i) in x.correlation
        }
    if hasattr(x,'ensemble'):
        j['ensemble'] = [ 
            str(x_i) for x_i in x.ensemble
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
        label = x.label
    ) 

#----------------------------------------------------------------------------
# 
def archive_to_json(a): 
    
    j = dict( CLASS = a.__class__.__name__ )

    if PY2:
        leaf_nodes_items = a._leaf_nodes.iteritems
        tagged_items = a._tagged.iteritems
        tagged_reals_items = a._tagged_reals.iteritems
        intermediate_uids_items = a._intermediate_uids.iteritems
    else:
        leaf_nodes_items = a._leaf_nodes.items
        tagged_items = a._tagged.items
        tagged_reals_items = a._tagged_reals.items
        intermediate_uids_items = a._intermediate_uids.items
    
    j['leaf_nodes'] = {
        str(i) : leaf_to_json(o_i)
            for (i, o_i) in leaf_nodes_items()
    }
    
    j['tagged'] = {
        str(i) : tagged_to_json(o_i)
            for (i, o_i) in tagged_items()
    }
    
    j['tagged_reals'] = {
        str(i) : tagged_to_json(o_i)
            for (i, o_i) in tagged_reals_items()
    }
    
    j['intermediate_uids'] = {
        # o_i is the pair (label,u)
        str(i) : (o_i)
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
    The function is called, when `loads()` parses the JSON record, every time 
    an object is not a recognised JSON type. `j` is always a dict. 
    By transforming `j` into an appropriate object we can reconstruct the 
    elements of the archive and finally assemble the archive object.
    
    """
    if 'CLASS' in j and (j['CLASS'] == Vector.__name__):  
        # uid must be hashable, so we apply `tuple()`
        return Vector( 
            index=[ tuple(i) for i in j['index'] ], 
            value=j['value'] 
        )
        
    elif 'CLASS' in j and (j['CLASS'] == LeafNode.__name__):
        return LeafNode(j) 
        
    elif 'CLASS' in j and (j['CLASS'] == ElementaryReal.__name__):
        return ElementaryReal(
            j['x'],
            tuple(j['uid'])     # Must be hashable
        )
        
    elif 'CLASS' in j and (j['CLASS'] == IntermediateReal.__name__):
        label = j['label']
        return IntermediateReal(
            j['value'],
            j['u_components'],
            j['d_components'],
            j['i_components'],
            label if label != "None" else None,
            tuple(j['uid'])     # Must be hashable
        )
        
    elif 'CLASS' in j and (j['CLASS'] == Complex.__name__):
        label = j['label']
        return Complex(
            j['n_re'],
            j['n_im'],
            label if label != "None" else None
        )
        
    elif 'CLASS' in j and (j['CLASS'] == Archive.__name__):
        ar = Archive(dump=False)  # Still need to thaw the data

        if PY2:
            leaf_nodes_items = j['leaf_nodes'].iteritems
            intermediate_uids_items = j['intermediate_uids'].iteritems
        else:
            leaf_nodes_items = j['leaf_nodes'].items
            intermediate_uids_items = j['intermediate_uids'].items

        ar._leaf_nodes = {
            # eval(i) transforms the string repr of a UID into a tuple
            eval(i) : o
                for (i,o) in leaf_nodes_items()
        }

        # Mapping uid -> (label, u)
        ar._intermediate_uids = {
            # eval(i) transforms the string repr of a UID into a tuple
            eval(i) : tuple(args) 
                for (i,args) in intermediate_uids_items()
        }
 
        ar._tagged = j['tagged']
        ar._tagged_reals = j['tagged_reals']

        return ar 
        
    else:
        # Allow parsing to continue at the next level
        return j 
        
  

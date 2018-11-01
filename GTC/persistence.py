"""
Class
-----

    An :class:`Archive` object can be used to marshal a set of uncertain numbers 
    for storage, or restore a set of uncertain numbers from storage. 
    
    Python pickle is used for the storage mechanism.
    
Functions
---------

    An archive can be pickled and stored in a file, or a string. 

    Functions for storing and retrieving a pickled archive file are
    
        * :func:`load`
        * :func:`dump`
        
    Functions for storing and retrieving a pickled archive string are
    
        * :func:`dumps`
        * :func:`loads`

Module contents
---------------

"""
import itertools
try:
    import cPickle as pickle  # Python 2
    PY2 = True
except ImportError:
    import pickle
    PY2 = False

from GTC.lib import (
    UncertainComplex,
    UncertainReal
)
from GTC.vector import Vector 
from GTC import context

__all__ = (
    'Archive',
    'load',
    'dump',
    'dumps',
    'loads',
)
    
#============================================================================
# When an archive is prepared for storage, uncertain number objects 
# are translated into the following simple representations that
# Python will pickle.
#
class FrozenLeaf(object):
    def __init__(self,node):
        self.uid = node.uid
        self.label = node.label 
        self.u = node.u 
        self.df = node.df 
        self.independent = node.independent 
        if hasattr(node,'complex'):
            self.complex = node.complex
        if hasattr(node,'correlation'):
            self.correlation = node.correlation.items()
        if hasattr(node,'ensemble'):
            self.ensemble = frozenset( node.ensemble )
        
class TaggedElementaryReal(object):
    def __init__(self,x,uid):
        self.x = x
        self.uid = uid

class TaggedIntermediateReal(object):
    def __init__(self,value,u_components,d_components,i_components,label,uid):
        self.value = value
        self.u_components = u_components    
        self.d_components = d_components     
        self.i_components = i_components    
        self.label = label
        self.uid = uid
    
class TaggedElementaryComplex(object):
    def __init__(self,n_re,n_im,label):
        self.n_re = n_re
        self.n_im = n_im
        self.label = label
    
class TaggedIntermediateComplex(object):
    def __init__(self,n_re,n_im,label):
        self.n_re = n_re
        self.n_im = n_im
        self.label = label

#----------------------------------------------------------------------------
class Archive(object):
    """
    An :class:`Archive` object can be used to marshal a set of uncertain numbers 
    for storage, or restore a set of uncertain numbers from storage.
    """
    def __init__(self):

        # `self._tagged_reals` contains information about every
        # real uncertain number destined for storage: a pair 
        # of entries is included when a complex UN is tagged.
        #
        self._tagged = {}           # name->object-ref pairs
        self._tagged_reals = {}     # name->object-ref pairs
        
        # Filled by add() and then used when freezing to
        # associate the uid's of intermediate components with UNs.
        self._uid_to_intermediate = {}
        
        self._extract = False   # initially in write-only mode

    def keys(self):
        """Return a list of names 
        """
        return self._tagged.keys()

    def iterkeys(self):
        """Return an iterator for names 
        """
        if PY2:
            return self._tagged.iterkeys()
        return self._tagged.keys()

    def values(self):
        """Return a list of uncertain numbers 
        """
        return self._tagged.values()
        
    def itervalues(self):
        """Return an iterator for uncertain numbers 
        """
        if PY2:
            return self._tagged.itervalues()
        return self._tagged.values()

    def items(self):
        """Return a list of name -to- uncertain-number pairs 
        """
        return self._tagged.items()
        
    def iteritems(self):
        """Return an iterator of name -to- uncertain-number pairs 
        """
        if PY2:
            return self._tagged.iteritems()
        return self._tagged.items()

    def __len__(self):
        """Return the number of entries 
        """
        return len(self._tagged)

    def _setitem(self,key,value):
        """Add a name -to- uncertain-number pair
        
        """
        if key in self._tagged:
            raise RuntimeError(
                "'{!s}' is already in use".format(key)
            )
        else:
            if isinstance(value,UncertainReal):
                if key in self._tagged_reals:
                    raise RuntimeError(
                        "'{!s}' is being used as a name-tag".format(key)
                    )
                    
                if not value.is_elementary:
                    self._uid_to_intermediate[value.real._node.uid] = value.real
                
                self._tagged_reals[key] = value

            elif isinstance(value,UncertainComplex):
            
                n_re = "{!s}_re".format(key)
                if n_re in self._tagged_reals:
                    raise RuntimeError(
                        "'{!s}' is being used as a name-tag".format(n_re)
                    )

                n_im = "{!s}_im".format(key)
                if n_im in self._tagged_reals:
                    raise RuntimeError(
                        "'{!s}' is being used as a name-tag".format(n_im)
                    )
                    
                self._tagged_reals[n_re] = value.real
                self._tagged_reals[n_im] = value.imag
                
                if not value.is_elementary:
                    self._uid_to_intermediate[value.real._node.uid] = value.real
                    self._uid_to_intermediate[value.imag._node.uid] = value.imag
                        
            else:
                raise RuntimeError(
                    "'{!r}' cannot be archived: wrong type"
                )
                
            self._tagged[key] = value

    def __setitem__(self,key,value):
        """
        Add an uncertain number to the archive
        
        **Example**::
        
            >>> a = Archive()
            >>> x = ureal(1,1)
            >>> y = ureal(2,1)
            >>> a['x'] = x
            >>> a['fred'] = y
            
        """
        if self._extract:
            raise RuntimeError('This archive is read-only!')
        else:
            self._setitem(key,value)


    def add(self,**kwargs):
        """Add entries ``name = uncertain-number`` to the archive

        **Example**::
        
            >>> a = Archive()
            >>> x = ureal(1,1)
            >>> y = ureal(2,1)
            >>> a.add(x=x,fred=y)
            
        """
        if self._extract:
            raise RuntimeError('This archive is write-only!')

        items = kwargs.iteritems() if PY2 else kwargs.items()
        for key,value in items:
            self._setitem(key,value)

    def _getitem(self,key):
        """
        """
        try:
            value = self._tagged[key]
        except KeyError:
            raise RuntimeError(
                "'{!s}' not found".format(key)
            )

        return value

    def __getitem__(self,key):
        """Extract an uncertain number

        `key` - the name of the archived number
        
        """
        if not self._extract:
            raise RuntimeError('This archive is write-only!')
        else:
            return self._getitem(key)

    def extract(self,*args):
        """
        Extract one or more uncertain numbers

        :arg args: names of archived uncertain numbers
        
        If just one name is given, a single uncertain 
        number is returned, otherwise a sequence of
        uncertain numbers is returned.
        
        # **Example**::

            # >>> x, fred = a.extract('x','fred')
            # >>> harry = a.extract('harry')
            
        """        
        if not self._extract:
            raise RuntimeError('This archive is read-only!')
        
        lst = [ self._getitem(n) for n in args ]
            
        return lst if len(lst) > 1 else lst[0]

    # -----------------------------------------------------------------------
    def _freeze(self):
        """Prepare archive for for storage
        
        NB after freezing, the archive object is immutable.
        
        """        
        values = self._tagged_reals.itervalues() if PY2 else self._tagged_reals.values()
        self._leaf_nodes = {
            n_i.uid  : FrozenLeaf(n_i)
                for un in values
                    for n_i in itertools.chain(
                        un._u_components.iterkeys(),
                        un._d_components.iterkeys()
                    )
        }                      
                        
        # -------------------------------------
        # Intermediate real uncertain numbers
        #
        # All elementary influences of intermediate nodes 
        # have been found above and will be archived. 
        # However, intermediate influences may not have 
        # been tagged, in which case they are not archived.
        values = self._tagged_reals.itervalues() if PY2 else self._tagged_reals.values()
        _intermediate_node_to_uid = {
            v._node: v._node.uid 
            for v in values
                if not v.is_elementary
        }
                
        # Use this to recreate intermediate nodes in _thaw
        self._intermediate_uids = {
            n_i.uid : (n_i.label,n_i.u)
            for n_i in _intermediate_node_to_uid
        }
        
        # -------------------------------------------------------------------
        # Convert tagged objects into a standard form for storage 
        #
        items = self._tagged.iteritems() if PY2 else self._tagged.items()
        for n,obj in items:
            if obj.is_elementary:
                if isinstance(obj,UncertainReal):
                    tagged = TaggedElementaryReal(
                        x=obj.x,
                        uid=obj._node.uid
                    )
                    self._tagged[n] = tagged
                    self._tagged_reals[n] = tagged
                    
                elif isinstance(obj,UncertainComplex):
                    re = TaggedElementaryReal(
                        x=obj.real.x,
                        uid=obj.real._node.uid
                    )
                    im = TaggedElementaryReal(
                        x=obj.imag.x,
                        uid=obj.imag._node.uid
                    )
                    n_re = "{}_re".format(n)
                    self._tagged_reals[n_re] = re
                    
                    n_im = "{}_im".format(n)
                    self._tagged_reals[n_im] = im
                    
                    self._tagged[n] = TaggedElementaryComplex(
                        n_re=n_re,
                        n_im=n_im,
                        label = obj.label
                    )
                else:
                    assert False, 'unexpected'
            else:
                if isinstance(obj,UncertainReal):
                    un = TaggedIntermediateReal(
                        value = obj.x,
                        u_components = _vector_index_to_uid( 
                            obj._u_components ),
                        d_components = _vector_index_to_uid( 
                            obj._d_components ),
                        i_components = _ivector_index_to_uid(
                            obj._i_components,
                            _intermediate_node_to_uid
                        ),
                        label = obj.label,
                        uid = obj._node.uid
                    )
                    self._tagged[n] = un
                    self._tagged_reals[n] = un
                    
                elif isinstance(obj,UncertainComplex):
                    re = TaggedIntermediateReal(
                        value = obj.real.x,
                        u_components = _vector_index_to_uid(
                            obj.real._u_components ),
                        d_components = _vector_index_to_uid( 
                            obj.real._d_components ),
                        i_components = _ivector_index_to_uid(
                            obj.real._i_components,
                            _intermediate_node_to_uid
                        ),
                        label = obj.real.label,
                        uid = obj.real._node.uid,
                    )
                        
                    n_re = "{}_re".format(n)
                    self._tagged_reals[n_re] = re
                    
                    im = TaggedIntermediateReal(
                        value = obj.imag._x,
                        u_components = _vector_index_to_uid( 
                            obj.imag._u_components ),
                        d_components = _vector_index_to_uid( 
                            obj.imag._d_components ),
                        i_components = _ivector_index_to_uid(
                            obj.imag._i_components,
                            _intermediate_node_to_uid
                        ),
                        label = obj.imag.label,
                        uid = obj.imag._node.uid,
                    )

                    n_im = "{}_im".format(n)
                    self._tagged_reals[n_im] = im
                    
                    self._tagged[n] = TaggedIntermediateComplex(
                        n_re=n_re,
                        n_im=n_im,
                        label=obj.label
                    )
                else:
                    assert False,"unexpected"
                    
        # Python cannot pickle this
        del self._uid_to_intermediate 
        
    # -----------------------------------------------------------------------
    def _thaw(self):
        _leaf_nodes = dict()
        items = self._leaf_nodes.iteritems() if PY2 else self._leaf_nodes.items()
        for uid_i,fl_i in items:
            l = context._context.new_leaf(
                uid_i, 
                fl_i.label, 
                fl_i.u, 
                fl_i.df, 
                fl_i.independent,
            )            
            if hasattr(fl_i,'complex'):
                l.complex = fl_i.complex 
            if hasattr(fl_i,'correlation'):
                l.correlation = dict( fl_i.correlation )
            if hasattr(fl_i,'ensemble'):
                l.ensemble = set( fl_i.ensemble )
            _leaf_nodes[uid_i] = l

        # Create the nodes associated for intermediate 
        # uncertain numbers. This must be done before the 
        # intermediate uncertain numbers are recreated.
        items = self._intermediate_uids.iteritems() if PY2 else self._intermediate_uids.items()
        _nodes = {
            uid: context._context.new_node(uid, *args)
                for uid, args in items
        }
            
        # When reconstructing, `_tagged` needs to be updated with 
        # the new uncertain numbers.
        #
        items = self._tagged.iteritems() if PY2 else self._tagged.items()
        for name,obj in items:
            if isinstance(obj,TaggedElementaryReal):
                un = _builder(
                    name,
                    _leaf_nodes,
                    self._tagged_reals
                )                    
                self._tagged[name] = un

            elif isinstance(obj,TaggedIntermediateReal):
                un = _builder(
                    name,
                    _nodes,
                    self._tagged_reals
                )                    
                self._tagged[name] = un
                
            elif isinstance(
                    obj,
                    (TaggedElementaryComplex,TaggedIntermediateComplex)
                ):
                name_re = obj.n_re
                name_im = obj.n_im
                
                un_re = _builder(
                    name_re,
                    _nodes,
                    self._tagged_reals
                )
                un_im = _builder(
                    name_im,
                    _nodes,
                    self._tagged_reals
                )

                assert un_re.is_elementary == un_im.is_elementary
                unc = UncertainComplex(un_re,un_im)
                self._tagged[name] = unc
            else:
                assert False
                        
        # Change the archive status
        self._extract = True
 
#----------------------------------------------------------------------------
def _vector_index_to_uid(v):
    """
    Change the vector index from a node to a uid 
    
    """
    return Vector(
        index = [
            n_i.uid for n_i in v._index
        ],
        value = v._value
    )

#----------------------------------------------------------------------------
def _vector_index_to_node(v):
    """
    Change the vector index from a uid to a node
    
    """
    _nodes = context._context._registered_leaf_nodes
    return Vector(
        index = [
            _nodes[uid_i] for uid_i in v._index
        ],
        value = v._value
    )    

#----------------------------------------------------------------------------
def _ivector_index_to_uid(i_components,_intermediate_node_to_uid):
    """
    Return a Vector containing the uids of tagged intermediate UNs 
    
    """
    i_sequence = []
    v_sequence = []
    
    for i,v in i_components.iteritems():
        if i in _intermediate_node_to_uid:
            i_sequence.append( _intermediate_node_to_uid[i] )
            v_sequence.append(v)
            
    return Vector(index=i_sequence,value=v_sequence)
    
#----------------------------------------------------------------------------
def _ivector_index_to_node(i_components,_intermediate_node):
    """
    Return a Vector containing intermediate nodes as indices 
    
    """            
    return Vector(
        index= [
            _intermediate_node[uid_i]
            for uid_i in i_components.iterkeys()
            
        ],
        value=i_components._value
    )
    
#----------------------------------------------------------------------------
"""
Notes
-----

`tagged_real` is indexed by name and maps to an object to be constructed.

"""
def _builder(o_name,_nodes,_tagged_reals):
    """
    Construct an intermediate un object for `o_name`.
    
    """
    obj = _tagged_reals[o_name]
    
    if isinstance(obj,TaggedElementaryReal):
        un = UncertainReal._archived_elementary(
            uid = obj.uid,
            x = obj.x
        )
        _tagged_reals[o_name] = un    
                
    elif isinstance(obj,TaggedIntermediateReal):                
            
        un = UncertainReal(
            obj.value,
            _vector_index_to_node( obj.u_components ),
            _vector_index_to_node( obj.d_components ),
            _ivector_index_to_node( obj.i_components, _nodes ),
            _nodes[obj.uid],
            )
        
        _tagged_reals[o_name] = un

    else:
        assert False, "unexpected: {!r}".format(obj)

    return un


#------------------------------------------------------------------     
def dump(file,ar):
    """Save an archive in a file

    :arg file:  a file object opened in binary
                write mode (with 'wb')
                
    :arg ar: an :class:`Archive` object
      
    Several archives can be saved in a file 
    by repeated use of this function.
    
    """
    ar._freeze()
    pickle.dump(ar,file,protocol=pickle.HIGHEST_PROTOCOL)

#------------------------------------------------------------------     
def load(file):
    """Load an archive from a file

    :arg file:  a file object opened in binary
                read mode (with 'rb')

    Several archives can be extracted from 
    one file by repeatedly calling this function.
    
    """
    ar = pickle.load(file)
    ar.context = context._context
    ar._thaw()
    
    return ar

#------------------------------------------------------------------     
def dumps(ar,protocol=pickle.HIGHEST_PROTOCOL):
    """
    Return a string representation of the archive 

    :arg ar: an :class:`Archive` object
    :arg protocol: encoding type 

    Possible values for ``protocol`` are described in the 
    Python documentation for the 'pickle' module.

    ``protocol=0`` creates an ASCII string, but note
    that many (special) linefeed characters are embedded.
    
    """
    # Can save one of these strings in a single binary file,
    # using write(), when protocol=pickle.HIGHEST_PROTOCOL is used. 
    # A corresponding read() is required to extract the string. 
    # Alternatively, when protocol=0 is used a text file can be 
    # used, but again write() and read() have to be used, 
    # because otherwise the embedded `\n` characters are 
    # interpreted incorrectly.
    
    ar._freeze()
    s = pickle.dumps(ar,protocol)
    
    return s
    
#------------------------------------------------------------------     
def loads(s):
    """
    Return an archive object restored from a string representation

    :arg s: a string created by :func:`dumps`
    
    """
    ar = pickle.loads(s)
    ar.context = context._context
    ar._thaw()
    
    return ar

#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *  
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

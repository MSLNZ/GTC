"""
Class
-----

    An :class:`Archive` is used to store and retrieve uncertain numbers. 
    
Module contents
---------------

"""
import itertools
import sys
import pickle 

PY2 = bool( sys.version_info[0] == 2 )

from GTC.lib import (
    UncertainComplex,
    UncertainReal
)
from GTC.vector import Vector 
from GTC import context

__all__ = (
    'Archive',
)

#------------------------------------------------------------------     
class _Unpickler(pickle.Unpickler, object):
    """
    Prefer classes defined in this module over any others
    """
    def find_class(self, module, name):
        if hasattr(sys.modules[__name__],name):
            return getattr(sys.modules[__name__],name)
        else:
            return super(_Unpickler,self).find_class(module, name)

def load(file):
    """
    To unpickle an older-style Archive 
    """
    return _Unpickler(file).load()

#============================================================================
# When an archive is prepared for storage, uncertain number objects 
# are translated into the following simple representations that
# Python will pickle.
#
class LeafNode(object):
    def __init__(self,node):
        
        # If called during freezing, a
        # Leaf object will be provided.
        # If called after a JSON record has 
        # been restored, a dict will be provided.
        
        if isinstance( node, dict ): 
            # JSON record
            
            # A uid must be hashable
            self.uid = tuple(node['uid'])
            
            # 'None' must be restored to an object
            label = node['label']
            if label != "None" : 
                self.label = label 
            else:
                self.label = None
                
            self.u = float( node['u'] )
            self.df = float( node['df'] )
            
            self.independent = bool( node['independent'] ) 
            
            if 'complex' in node:
                self.complex = ( 
                    eval(node['complex'][0]), 
                    eval(node['complex'][1]) 
                )
            if 'correlation' in node:
                items = node['correlation'].iteritems() if PY2 else node['correlation'].items()
                self.correlation = {
                    eval(x_i) : r_i 
                        for x_i,r_i in items
                }
            if 'ensemble' in node:
                self.ensemble = frozenset( 
                    eval(i) for i in node['ensemble'] 
                )                               
        else:    
            # Leaf object        
            self.uid = node.uid
            self.label = node.label 
            self.u = node.u 
            self.df = node.df 
            self.independent = node.independent 
            if hasattr(node,'complex'):
                self.complex = node.complex
            if hasattr(node,'correlation'):
                self.correlation = node.correlation.items()
                if not PY2:
                    # In Python 2 items() returns a list but
                    # in Python 3 a dict_items object is returned.
                    # Since it is not possible to pickle a
                    # dict_items object we convert dict_items
                    # to be a list.
                    self.correlation = list(self.correlation)
            if hasattr(node,'ensemble'):
                self.ensemble = frozenset( node.ensemble )
        
class ElementaryReal(object):
    def __init__(self,x,uid):
        self.x = x
        self.uid = uid

class IntermediateReal(object):
    def __init__(self,value,u_components,d_components,i_components,label,uid):
        self.value = value
        self.u_components = u_components    
        self.d_components = d_components     
        self.i_components = i_components    
        self.label = label
        self.uid = uid
    
class Complex(object):
    def __init__(self,n_re,n_im,label):
        self.n_re = n_re
        self.n_im = n_im
        self.label = label
            
#----------------------------------------------------------------------------
# """
# Archive plays several roles related to storage of uncertain numbers.

# When uncertain numbers are stored, an archive is used to marshal
# information related to the selected uncertain numbers. This 
# ensures that when they are retrieved from storage, later in another Python  
# process, their behaviour will be the same as it was in the original process.

# This information is collected after specific uncertain numbers have been 
# identified. It is referred to as 'freezing' the archive. Freezing also 
# involves a transformation of the archive object contents into a form that 
# is suitable for generic storage in different formats. In this way, the 
# archive represents an intermediate staging point, which can be 
# followed by different format-specific storage operations.

# When a stored archive is to be retrieved, the reverse process occurs.
# First, format-specific operations must be used to recreate an archive 
# in its frozen state. Then the archive will be 'thawed', which will 
# create various GTC objects and restore the necessary environment for 
# the specific uncertain numbers that were saved initially. Finally, 
# these uncertain numbers may be restored (on demand).
# """
class Archive(object):
    """
    An :class:`Archive` helps to store and retrieve uncertain numbers,
    so that they can be used in later calculations. 
    
    A particular :class:`Archive` object can either be used to prepare 
    a record of uncertain numbers for storage, or to retrieve a stored record. 
        
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
        else:
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
        else:
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
        else:
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
                    try:
                        uid = value.real._node.uid
                    except AttributeError:
                        raise RuntimeError(
                            "uncertain number labelled '{}' is not declared intermediate".format(key)
                        )
                        
                    self._uid_to_intermediate[uid] = value.real            
                
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
                    try:
                        uid_r = value.real._node.uid
                        uid_i = value.imag._node.uid
                    except AttributeError:
                        raise RuntimeError(
                            "uncertain number labelled '{}' is not declared intermediate".format(key)
                        )
                        
                    self._uid_to_intermediate[uid_r] = value.real
                    self._uid_to_intermediate[uid_i] = value.imag
                        
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
        """Add entries to an archive.
        
        Each entry is given a name that identifies it within the archive.

        **Example**
        
        .. code-block:: pycon   
        
            >>> a = pr.Archive()
            >>> x = ureal(1,1)
            >>> y = ureal(2,1)
            >>> z = ureal(20,1)
            >>> a.add(x=x,fred=y)

            # Entries can also be added using the name as a key       
            >>> a['z'] = z

        .. invisible-code-block: pycon
        
            >>> import tempfile
            >>> f = open(tempfile.gettempdir() + '/GTC-archive-test.gar', 'wb')

        Here ``f`` is a file stream opened in mode 'wb':
        
        .. code-block:: pycon   
        
            >>> pr.dump(f, a)
            >>> f.close()
   

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
        Extract uncertain numbers by name

        :arg args: names of uncertain numbers stored in the archive
        
        If just one name is provided, a single uncertain number is returned.
        Otherwise a sequence of uncertain numbers is returned.
        
        **Example**
        
        Continuing the example in :meth:`~.Archive.add`, but in a different 
        Python session, ``f`` is now a file stream opened in 'rb' mode:

        .. invisible-code-block: pycon

            >>> import tempfile
            >>> f = open(tempfile.gettempdir() + '/GTC-archive-test.gar', 'rb')
            
        .. code-block:: pycon
        
            >>> a = pr.load(f)
            >>> f.close()
            
            >>> a.extract('fred')
            ureal(2.0,1.0,inf)
            >>> x, fred = a.extract('x','fred')
            >>> x
            ureal(1.0,1.0,inf)            
 
            # Entries can also be extracted using the name as a key
            >>> a['z']
            ureal(20.0,1.0,inf)

        .. invisible-code-block: pycon
            
            >>> import os, tempfile
            >>> os.remove(tempfile.gettempdir() + '/GTC-archive-test.gar')
  
        """        
        if not self._extract:
            raise RuntimeError('This archive is read-only!')
        
        lst = [ self._getitem(n) for n in args ]
            
        return lst if len(lst) > 1 else lst[0]

    # -----------------------------------------------------------------------
    def _freeze(self):
        """Prepare archive for storage
        
        NB after freezing, the archive object is immutable.
        
        """    
        if not len(self):
            raise RuntimeError(
                "There is nothing in the archive!"
            )

        values = self._tagged_reals.itervalues() if PY2 else self._tagged_reals.values()
        
        self._leaf_nodes = {
            n_i.uid  : LeafNode(n_i)
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
            n_i.uid : (n_i.label,n_i.u,n_i.df)
            for n_i in _intermediate_node_to_uid
        }
        
        # -------------------------------------------------------------------
        # Convert tagged objects into a standard form for storage 
        #
        items = self._tagged.iteritems() if PY2 else self._tagged.items()
        for n,obj in items:
            if obj.is_elementary:
                if isinstance(obj,UncertainReal):
                    tagged = ElementaryReal(
                        x=obj.x,
                        uid=obj._node.uid
                    )
                    self._tagged[n] = tagged
                    self._tagged_reals[n] = tagged
                    
                elif isinstance(obj,UncertainComplex):
                    re = ElementaryReal(
                        x=obj.real.x,
                        uid=obj.real._node.uid
                    )
                    im = ElementaryReal(
                        x=obj.imag.x,
                        uid=obj.imag._node.uid
                    )
                    n_re = "{}_re".format(n)
                    self._tagged_reals[n_re] = re
                    
                    n_im = "{}_im".format(n)
                    self._tagged_reals[n_im] = im
                    
                    self._tagged[n] = Complex(
                        n_re=n_re,
                        n_im=n_im,
                        label = obj.label
                    )
                else:
                    assert False, 'unexpected'
            else:
                if isinstance(obj,UncertainReal):
                    un = IntermediateReal(
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
                    re = IntermediateReal(
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
                    
                    im = IntermediateReal(
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
                    
                    self._tagged[n] = Complex(
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

        # Create the nodes associated with intermediate 
        # uncertain numbers. This must be done before the 
        # intermediate uncertain numbers are recreated.
        items = self._intermediate_uids.iteritems() if PY2 else self._intermediate_uids.items()
        
        # In v1.3.5, a df field was added to intermediate nodes. 
        # In previous versions it was absent. This shim function
        # will add `None` to the df attribute, which can then be
        # fixed once the uncertain number object is formed.
        # This is a temporary feature that will be removed after a few releases.
        # See also lib.py property df  
        shim_1_3_3 = lambda args: args + (None,) if len(args) == 2 else args
            
        _nodes = {
            uid: context._context.new_node(uid, *shim_1_3_3(args) )
                for uid, args in items
        }
            
        # When reconstructing, `_tagged` needs to be updated with 
        # the new uncertain numbers.
        #
        items = self._tagged.iteritems() if PY2 else self._tagged.items()
        for name,obj in items:
            if isinstance(obj,ElementaryReal):
                un = _builder(
                    name,
                    _leaf_nodes,
                    self._tagged_reals
                )                    
                self._tagged[name] = un

            elif isinstance(obj,IntermediateReal):
                un = _builder(
                    name,
                    _nodes,
                    self._tagged_reals
                )                    
                self._tagged[name] = un
                
            elif isinstance(obj,Complex):
                # This is an intermediate uncertain complex
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
                
                # An intermediate complex needs to 
                # link the nodes of its components 
                # (same as in `UncertainComplex._intermediate`)
                complex_id = (un_re._node.uid,un_im._node.uid)
                unc.real._node.complex = complex_id 
                unc.imag._node.complex = complex_id
        
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
    
    if isinstance(obj,ElementaryReal):
        un = UncertainReal._archived_elementary(
            uid = obj.uid,
            x = obj.x
        )
        _tagged_reals[o_name] = un    
                
    elif isinstance(obj,IntermediateReal):                
            
        _node = _nodes[obj.uid] 
        
        un = UncertainReal(
            obj.value,
            _vector_index_to_node( obj.u_components ),
            _vector_index_to_node( obj.d_components ),
            _ivector_index_to_node( obj.i_components, _nodes ),
            _node,
            )
        
        _tagged_reals[o_name] = un

    else:
        assert False, "unexpected: {!r}".format(obj)

    return un

#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *  
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )

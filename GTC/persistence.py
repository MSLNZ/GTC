"""
Class
-----

    An :class:`Archive` object will handle the marshaling 
    of a set of uncertain numbers for storage, as well as 
    the recovery of uncertain numbers from storage. 
    
    The default method of storage is to use Python pickle.
    However, the :class:`Archive` class can also be used 
    as a base class to implement other storage formats.

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
import collections
import itertools
import numbers
import weakref
import cPickle as pickle

from GTC.weak_symmetric_matrix import WeakSymmetricMatrix
from GTC.symmetric_matrix import SymmetricMatrix
from GTC.lib_complex import UncertainComplex
from GTC.lib_real import UncertainReal
from GTC.vector import Vector, is_ordered, merge_vectors
from GTC.context import context
from GTC.nodes import Leaf, Node

inf = float('inf')

__all__ = (
    'Archive',
    'load',
    'dump',
    'dumps',
    'loads',
)
    
#============================================================================
# When an archive is prepared for storage, uncertain number objects 
# are translated into the following simple representations.
#
class TaggedElementaryReal(object):
    def __init__(self,x,u,df,label,uid,independent):
        self.x = x
        self.u = u
        self.df = df
        self.label = label
        self.uid = uid
        self.independent = independent

class TaggedIntermediateReal(object):
    def __init__(self,value,u_components,d_components,i_components,label,uid):
        self.value = value
        self.u_components = u_components    # Vector of uid: u 
        self.d_components = d_components    # Vector of uid: u 
        self.i_components = i_components    # Vector of uid: u 
        self.label = label
        self.uid = uid
    
class TaggedElementaryComplex(object):
    def __init__(self,n_re,n_im,label):
        self.n_re = un_re
        self.n_im = un_im
        self.label = label
    
class TaggedIntermediateComplex(object):
    def __init__(self,n_re,n_im,label):
        self.n_re = n_re
        self.n_im = n_im
        self.label = label

#------------------------------------------------------------------     
class Archive(object):

    """
    """

    def __init__(self):

        self._context_id = context._id

        # `self._tagged_reals` contains information about every
        # real uncertain number: a pair of entries is included
        # when a complex un is tagged.
        #
        self._tagged = {}           # name->object-ref pairs
        self._tagged_reals = {}     # name->object-ref pairs
        
        # Filled by add() and used when freezing to
        # associate the uids in i_components with UNs.
        self._uid_to_intermediate = {}
        
        self._extract = False   # initially in write-only mode

    def keys(self):
        """Return a list of name-tags 
        """
        return self._tagged.keys()

    def iterkeys(self):
        """Return an iterator for name-tags 
        """
        return self._tagged.iterkeys()

    def values(self):
        """Return a list of uncertain numbers 
        """
        return self._tagged.values()
        
    def itervalues(self):
        """Return an iterator for uncertain numbers 
        """
        return self._tagged.itervalues()

    def items(self):
        """Return a list of name-tag -to- uncertain-number pairs 
        """
        return self._tagged.items()
        
    def iteritems(self):
        """Return an iterator of name-tag -to- uncertain-number pairs 
        """
        return self._tagged.iteritems()

    def __len__(self):
        """Return the number of entries 
        """
        return len(self._tagged)

    def _setitem(self,key,value):
        """Add a name-tag -to- uncertain-number pair
        
        """
        if key in self._tagged:
            raise RuntimeError(
                "'{!s}' is already in use".format(key)
            )
        else:
            # Fill `self._tagged_reals`            
            if isinstance(value,UncertainReal):
                if key in self._tagged_reals:
                    raise RuntimeError(
                        "'{!s}' is being used as a name-tag".format(key)
                    )

                if not value.is_elementary:
                    # NB,`result()` assigns the uid attribute
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
                    # NB,`result()` assigns the uid attribute
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
        
            >>> a = ar.Archive()
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
        
            >>> a = ar.Archive()
            >>> x = ureal(1,1)
            >>> y = ureal(2,1)
            >>> a.add(x=x,fred=y)
            
        """
        if self._extract:
            raise RuntimeError('This archive is write-only!')
        
        for key,value in kwargs.iteritems(): 
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
        
        .. note::

            When an uncertain number is restored,
            it is removed from the archive. 

        **Example**::

            >>> x = a['x']
            >>> harry = a['harry']

        """
        if not self._extract:
            raise RuntimeError('This archive is write-only!')
        else:
            return self._getitem(key)

    def extract(self,*args):
        """
        Extract one or more uncertain numbers

        :arg args: archived names of the uncertain numbers
        
        If just one name is given a single uncertain 
        number is returned, otherwise a sequence of
        uncertain numbers is returned.
        
        **Example**::

            >>> x, fred = a.extract('x','fred')
            >>> harry = a.extract('harry')
            
        """        
        if not self._extract:
            raise RuntimeError('This archive is read-only!')
        
        lst = [ self._getitem(n) for n in args ]
            
        return lst if len(lst) > 1 else lst[0]

    def _correlations_submatrix(self,uids):
        """
        Return a SymmetricMatrix correlation matrix 
        that uses uids, instead of nodes, as indices
        
        """        
        _registered_leaf_nodes = context._registered_leaf_nodes
        
        R = SymmetricMatrix()

        mat = context._correlations._mat
        uid_lst = list(uids)
        
        for i,uid_i in enumerate(uid_lst):
            key_i = _registered_leaf_nodes[ uid_i ]
            if key_i in mat:
                row_i = mat[key_i]
                for j,uid_j in enumerate(uid_lst[i+1:]):
                    key_j = _registered_leaf_nodes[ uid_j ]
                    if key_j in row_i:
                        R[key_i._uid,key_j._uid] = row_i[key_j]
                        
        return R

##"""
##Notes about _freeze
##===================
##
##The initial stages do the following:
##    - find all elementary UNs that influence tagged objects, 
##      use these to make records (indexed by uid) of:
##        - leaf nodes 
##        - dof 
##        - complex_uid 
##        - labels
##        - ensembles
##        - correlation matrix
##
##    - find all tagged intermediate real nodes
##
##    - convert all tagged object data into a generic form that can 
##      stored in different formats.  
##"""

    # -----------------------------------------------------------------------
    def _freeze(self):
        """Prepare for storage
        
        NB after freezing, the archive object is immutable.
        
        """        
        
        # _registered_leaf_nodes = context._registered_leaf_nodes
            
        _leaf_nodes = frozenset(
            n_i 
            for un in self._tagged_reals.itervalues()
                for n_i in itertools.chain(
                    un._u_components.iterkeys(),
                    un._d_components.iterkeys()
                )
        )
        
        self._uid_to_u_leaves = {
            n_i.uid: n_i.u
            for n_i in _leaf_nodes
        }
         
        self._dof_record = {
            n_i.uid : n_i.df
            for n_i in _leaf_nodes
        }                

        self._labels = {
            n_i.uid: n_i.tag 
            for n_i in _leaf_nodes
        }

        # Record ensemble groupings.
        # We need to restrict the ensemble in the archive to tagged objects 
        # or their influences.
        
        # `uids` contains all elementary uids that influence tagged objects.        
        uids = frozenset(
            self._uid_to_u_leaves.iterkeys()
        )
        
        # Check all elementary uids, when one is found to belong to an
        # ensemble, trim the ensemble of any other uids that are not
        # included in the archive. 
        _ensemble = dict()
        for n_i in _leaf_nodes:
            if n_i in context._ensemble:
                _ensemble[ n_i.uid ] = [
                    (i.uid[0],i.uid[1]) 
                        for i in context._ensemble[n_i]
                            if i.uid in uids
                ]
        self._ensemble = _ensemble

        # The context maintains a register of `_complex_ids`. 
        # The keys are are pairs of uids, one for the
        # real component and one for the imaginary component.
        # If a complex uid is identified in the context,
        # the key is loaded into the archive `_complex_ids`.
        
        # Note, although a uid may be one component of a complex
        # number, the other component may not be tagged,
        # in which case we don't record a complex number.
        
        _complex_ids = dict()
        for uid in uids: 
            if uid in context._complex_ids:
                uid_re, uid_im = context._complex_ids[uid]
                if uid_re in uids and uid_im in uids:
                    _complex_ids[uid] = ( uid_re, uid_im )
                
        self._complex_ids = _complex_ids
                 
        # A correlation matrix indexed by uid's 
        self._correlations = context._correlations.submatrix( uids )
        
        # -------------------------------------
        # Intermediate real uncertain numbers
        #
        # All elementary influences of intermediate nodes 
        # have been recorded above and will be archived. 
        # However, intermediate influences may not have 
        # been tagged, in which case they are not archived.
        
        _intermediate_node_to_uid = {
            v._node: v._node.uid 
            for v in self._tagged_reals.itervalues()
                if not v.is_elementary
        }
                
        # Use this to recreate intermediate nodes in _thaw
        self._intermediate_uids = {
            n_i.uid : (n_i.tag,n_i.u)
            for n_i in _intermediate_node_to_uid
        }
        
        # -------------------------------------------------------------------
        # Convert tagged objects into a standard form for storage 
        #
        for n,obj in self._tagged.iteritems():
            if obj.is_elementary:
                if isinstance(obj,UncertainReal):
                    tagged = TaggedElementaryReal(
                        x=obj.x,
                        u=obj.u,
                        df=obj.df,
                        label=obj.label,
                        uid=obj._node.uid,
                        independent=obj._node.independent
                    )
                    self._tagged[n] = tagged
                    self._tagged_reals[n] = tagged
                    
                elif isinstance(obj,UncertainComplex):
                    re = TaggedElementaryReal(
                        x=obj.real.x,
                        u=obj.real.u,
                        df=obj.df,
                        label=obj.real.label,
                        uid=obj.real._node.uid,
                        independent=obj._node.independent
                    )
                    im = TaggedElementaryReal(
                        x=obj.imag.x,
                        u=obj.imag.u,
                        df=obj.df,
                        label=obj.imag.label,
                        uid=obj.imag._node.uid,
                        independent=obj._node.independent
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
                    assert False
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
                    assert False,"should never occur"
                
    # -----------------------------------------------------------------------
    def _thaw(self,context):
        """

        """
        context._complex_ids.update(self._complex_ids)
                
        _mat = self._correlations._mat  
                
        _leaf_nodes = dict()
        for uid,u in self._uid_to_u_leaves.iteritems():
            # NB a new node is not created if the uid is already
            # associated with a node
            _leaf_nodes[uid] = context.new_leaf(
                uid, 
                self._labels[uid], 
                u, 
                self._dof_record.get(uid,inf), 
                # Assume that the existence of a row 
                # in the correlation matrix implies 
                # correlation with something.
                independent=(uid not in _mat)  
            )

        # Update context ensemble register
        _new_ensemble = weakref.WeakKeyDictionary()
        for uid, ensemble in self._ensemble.iteritems():
            # context ensembles are indexed by node
            nid = context._registered_leaf_nodes[uid]
            
            # The members of an ensemble are all indices 
            # of  the ensemble, so we don't want to repeat 
            # this step unnecessarily.
            if nid not in _new_ensemble:            
                # Make a set of nodes for this uid 
                _ensemble = weakref.WeakSet()
                for uid_i in ensemble:
                    _ensemble.add(
                        context._registered_leaf_nodes[uid_i]
                    )
                
                # Update the _ensemble mapping for each  
                # element of the set
                for nid_i in _ensemble:
                    assert nid_i not in _new_ensemble
                    _new_ensemble[nid_i] = _ensemble
        
        # The ensembles associated with these nodes             
        # can be merged with current context ensembles.
        context._ensemble.update(_new_ensemble)

        # Create the nodes associated with intermediate 
        # uncertain numbers. This must be done before these 
        # uncertain numbers are recreated.
        _nodes = {
            uid: context.new_node(uid, *args)
            for uid, args in self._intermediate_uids.iteritems()
        }
            
        # When reconstructing, `_tagged` needs to be updated with 
        # the new uncertain numbers.
        #
        # real_buf = {}
        # complex_buf = {}
        for name,obj in self._tagged.iteritems():
            if isinstance(obj,TaggedElementaryReal):
                un = _builder(
                    name,
                    _leaf_nodes,
                    self._tagged_reals,
                    context
                )
                # real_buf[obj.uid] = un
                    
                self._tagged[name] = un

            elif isinstance(obj,TaggedIntermediateReal):
                # if obj.uid in real_buf:
                    # un = real_buf[obj.uid]
                # else:
                un = _builder(
                    name,
                    _nodes,
                    self._tagged_reals,
                    context
                )
                # real_buf[obj.uid] = un
                    
                self._tagged[name] = un
                
            # elif isinstance(
                    # obj,
                    # (TaggedElementaryComplex,TaggedIntermediateComplex)
                # ):
                # # Caching is more complicated for complex. The components
                # # should use the same cache as reals (because just one component
                # # may be relabeled in the archive). The complex container does not
                # # have a node, so it cannot be identified so easily. Need to
                # # have a complex buffer with nid pairs as keys.
                # name_re = obj.n_re
                # name_im = obj.n_im
                
                # obj_re_nid = self._tagged_reals[name_re].nid
                # obj_im_nid = self._tagged_reals[name_im].nid

                # # Complex caching
                # complex_key  = (obj_re_nid,obj_im_nid)
                # if complex_key in complex_buf:
                    # unc = complex_buf[complex_key]
                # else:
                    # if obj_re_uid in real_buf:
                        # un_re = real_buf[obj_re_uid]
                    # else:
                        # un_re = _builder(
                            # name_re,
                            # _nodes,
                            # self._tagged_reals
                        # )
                        # real_buf[obj_re_uid] = un_re
                        
                    # if obj_im_uid in real_buf:
                        # un_im = real_buf[obj_im_uid]
                    # else:
                        # un_im = _builder(
                            # name_im,
                            # _nodes,
                            # self._tagged_reals
                        # )
                        # real_buf[obj_im_uid] = un_im

                    # assert un_re.is_elementary == un_im.is_elementary
                    # unc = UncertainComplex(un_re,un_im)
                    # complex_buf[complex_key] = unc                    
                    
                    # # # TODO: do I need to let the context have these uid's?
                    # # # The _builder will register the real uids, but does
                    # # # not see a complex pair!
                    # # if unc.is_elementary:
                        # # unc._uid = (unc.real._uid,unc.imag._uid)
                        
                    # # unc.label = obj.label
                
                # self._tagged[name] = unc
            else:
                assert False
                        
        # Add correlations between all elementary
        # uncertain numbers to the existing record
        R = context._correlations
                
        # _mat = self._correlations._mat (above)
        _uids = _mat.keys()             
        for i,uid_i in enumerate( _uids ):
            row_i = _mat[uid_i]
            for uid_j in _uids[i+1:]:
                if uid_j in row_i:
                    R[
                        context._registered_leaf_nodes[uid_i],
                        context._registered_leaf_nodes[uid_j]
                    ] = row_i[uid_j]

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
    _nodes = context._registered_leaf_nodes
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

If an elementary real UN is needed, this is constructed and the new node
is added to _nodes.

"""
def _builder(o_name,_nodes,_tagged_reals,context):
    """
    Construct an intermediate un object for `o_name`.
    
    """
    obj = _tagged_reals[o_name]
    
    if isinstance(obj,TaggedElementaryReal):
        un = context._archived_elementary_real(
            uid = obj.uid,
            x = obj.x,
            u = obj.u,
            df = obj.df,
            label = obj.label,
            independent = obj.independent
        )
        _tagged_reals[o_name] = un    # tag now maps to an object
                
    elif isinstance(obj,TaggedIntermediateReal):                

        # For older archives, there were no `i_components` or `d_components`
        if not hasattr(obj,'i_components'):
            obj.i_components = Vector()
        if not hasattr(obj,'d_components'):
            obj.d_components = Vector()
            
        un = UncertainReal(
            context,
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
def load(file,context):
    """Load an archive from a file

    :arg file:  a file object opened in binary
                read mode (with 'rb')

    Several archives can be extracted from 
    one file by repeatedly calling this function.
    
    """
    ar = pickle.load(file)
    ar._thaw(context)
    
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
def loads(s,context):
    """
    Return an archive object restored from a string representation

    :arg s: a string created by :func:`dumps`
    
    """
    ar = pickle.loads(s)
    ar._thaw(context)
    
    return ar


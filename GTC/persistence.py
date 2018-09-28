"""
Class
-----

    An :class:`Archive` object marshals a set of
    uncertain numbers for storage and recreates
    uncertain numbers, when restoring an archive.

Functions
---------

    An archive can be stored as a computer file, or in a string. 

    Functions for storing and retrieving an archive file are
    
        * :func:`load`
        * :func:`dump`
        
    Functions for storing and retrieving string archives are
    
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
from GTC.library_complex import UncertainComplex
from GTC.library_real import UncertainReal
from GTC.vector import Vector, is_ordered, merge_vectors
from GTC.context import context
from GTC.node import LeafNode, Node

inf = float('inf')

__all__ = (
    'Archive',
    'load',
    'dump',
    'dumps',
    'loads',
)
    
#====================================================================

# Use the Python `id` to keep track of 
# objects when preparing an archive. 
# This will help to prevent using the same 
# name for different objects.
#

class TaggedElementaryReal(object):
    def __init__(self,x,u,df,label,uid,nid):
        self.x = x
        self.u = u
        self.df = df
        self.label = label
        self.uid = uid
        self.nid = nid

class TaggedIntermediateReal(object):
    def __init__(self,value,u_components,i_components,pds,label,nid):
        self.value = value
        self.u_components = u_components
        self.d_components = d_components
        self.i_components = i_components
        self.pds = pds
        self.label = label
        self.uid = None
        self.nid = nid
    
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

#------------------------------------------------------------------     
class Archive(object):

    """
    """

    def __init__(self):

        self._context_id = context._id

        # Context registers
        # Some information from registers will be copied when the 
        # archive is frozen. The thawed archive can transfer 
        # this information later to the new context.

        # `self._tagged_reals` contains information about every
        # real uncertain number: a pair of entries is included
        # when a complex un is tagged.
        #
        self._tagged = {}           # name->object-ref pairs
        self._tagged_reals = {}     # name->object-ref pairs
        
        # A mapping of name: name-sequence
        # The user assigns a name to an uncertain number tagged for archiving
        # and the names of a sequence of other archived uncertain numbers
        # for which the intermediate uncertainty components are required.
        # All names will be related to tagged-intermediate-reals
        #
        # self._intermediate_sensitivities = {}

        # Maps a uid to a UN
        # Filled when add() is called and used when freezing to
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
        """Return a list of pairs of name-tags and uncertain-numbers 
        """
        return self._tagged.items()
        
    def iteritems(self):
        """Return an iterator for pairs of name-tags and uncertain-numbers 
        """
        return self._tagged.iteritems()

    def __len__(self):
        """Return the number of entries in the archive
        """
        return len(self._tagged)

    def _setitem(self,key,value):
        """Add an object to the archive
        
        """
        # Special cases to watch for:
        #   1) the un is complex but some of its components are already tagged
        #   2) the un is real but it is the component of a tagged complex un

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
                    # NB, unless `result()` was called on `value` there 
                    # will not be a uid attribute
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
                    
                # error checking is over: it is safe to register the complex UN 
                self._tagged_reals[n_re] = value.real
                self._tagged_reals[n_im] = value.imag
                
                # We need a register intermediate results 
                # to use later when freezing the archive 
                if not value.is_elementary:
                    # NB, unless `result()` was called on `value` there 
                    # will not be a uid attribute
                    self._uid_to_intermediate[value.real._node.uid] = value.real
                    self._uid_to_intermediate[value.imag._node.uid] = value.imag
                        
            else:
                raise RuntimeError(
                    "'{!r}' cannot be archived: wrong type"
                )
                
            # Tag the object  
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
        
        # Add object references to the `tagged` register
        # `name` must be unique in the archive.
        #
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

        # del self._tagged[key]
        
        # if isinstance(value,UncertainComplex):
            # del self._tagged_reals["%s_re" % key]
            # del self._tagged_reals["%s_im" % key]
        # else:
            # del self._tagged_reals[key]

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

    def _freeze(self):
        """Prepare for storage
        
        NB after freezing, the archive object is immutable.
        
        """        
        # `uids` contains all elementary uids that influence tagged objects.        
        uids = set(
            uid 
                for un in self._tagged_reals.itervalues()
                    for uid in un._u_components.iterkeys()
        )
        
        _registered_leaf_nodes = context._registered_leaf_nodes

        _leaf_nodes = set(
            _registered_leaf_nodes[uid_i] for uid_i in uids
        )
        
        # To reconstruct these nodes, we need the object 
        # uid, the standard uncertainty and the nid.
        self._leaf_nodes = {
            id(n): [n.uid,n.u] 
            for n in _leaf_nodes
        }
         
        # The dofs for all influences.
        # This mapping will be transferred to the new context
        # when the archive is thawed
        self._dof_record = = {
            n_i.uid : n_i.df
            for n_i in _leaf_nodes
        }                

        # A mapping of uid: label
        # _labels = dict()
        # for uid_i in uids:
            # ln = _registered_leaf_nodes[uid_i]
            # if ln.tag:
                # _labels[uid_i] = ln.tag
        self._labels = {
            n_i.uid_i: n_i.tag 
            for n_i in _leaf_nodes
        }

        # Record ensemble groupings.
        # A context ensemble entry is an elementary node mapped to a set of nodes.
        # We need to convert the nodes to uids. We also need to restrict the 
        # ensemble in the archive to tagged objects or their influences.
        
        # Check all elementary uids, when one is found to belong to an
        # ensemble, trim the ensemble of any other uids that are not
        # included in the archive. 
        _ensembles = dict()
        for n_i in _leaf_nodes:
            if n_i in context._ensembles:
                _ensembles[ n_i.uid ] = [
                    (i.uid[0],i.uid[1]) 
                        for i in context._ensembles[n_i]
                            if i.uid in uids
                ]
        self._ensembles = _ensembles

        # The context maintains a register of `_complex_ids`. 
        # The keys in this register are 2-tuples of uids, one for the
        # real component and one for the imaginary component.
        # If a complex uid is identified in the context,
        # the key is loaded into the archive `_complex_ids`.
        # Note, although a uid may be a component of a complex
        # number, the other component may not be tagged,
        # in which case we don't store a complex number.
        
        _complex_ids = dict()
        for uid in uids: 
            if uid in context._complex_ids:
                uid_re, uid_im = context._complex_ids[uid]
                if uid_re in uids and uid_im in uids:
                    _complex_ids[uid] = ( uid_re, uid_im )
                
        self._complex_ids = _complex_ids
                 
        # Obtain a correlation matrix that uses uid's instead of nodes 
        self._correlations = context._correlations_submatrix( uids )
        
        # -------------------------------------
        # Intermediate real uncertain numbers
        #
        # All elementary influences of an intermediate node 
        # have been recorded above and will be archived. 
        # However, some intermediate influences may not have 
        # been tagged. In that case we will not archive them.
        
        _intermediate_node_to_uid = {
            v._node: v._node.uid 
            for v in self._tagged_reals.itervalues()
                if not v.is_elementary
        }
        # _tagged_intermediates = {
            # n : v 
            # for n,v in self._tagged_reals.iteritems()
                # if not v.is_elementary
        # }
                
        # -------------------------------------                    
        # Convert object references to data that can be pickled. 
        for n,obj in self._tagged.iteritems():
            if obj.is_elementary:
                # Complete the information about values
                if isinstance(obj,UncertainReal):
                    tagged = TaggedElementaryReal(
                        x=obj.x,
                        u=obj.u,
                        df=obj.df,
                        label=obj.label,
                        uid=obj._node.uid,
                        nid=id(obj._node)
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
                        nid=id(obj.real._node)
                    )
                    im = TaggedElementaryReal(
                        x=obj.imag.x,
                        u=obj.imag.u,
                        df=obj.df,
                        label=obj.imag.label,
                        uid=obj.imag._node.uid,
                        nid=id(obj.imag._node)
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
                # Intermediate uns are defined by their type,
                # value, their sensitivity to elementary UNs, 
                # and their sensitivity to intermediate  
                # UNs and their label(s).
                
                if isinstance(obj,UncertainReal):
                    un = TaggedIntermediateReal(
                        value = obj.x,
                        u_components = obj._u_components,
                        d_components = obj._d_components,
                        i_components = _tagged_intermediate_node_to_uid(
                            obj._i_components,
                            _intermediate_node_to_uid
                        ),
                        label = obj.label,
                        uid = obj._node.uid
                        nid = id(obj._node),
                    )
                    self._tagged[n] = un
                    self._tagged_reals[n] = un
                    
                elif isinstance(obj,UncertainComplex):
                    # A complex is just a pair of reals
                    re = TaggedIntermediateReal(
                        value = obj.real.x,
                        u_components = obj.real._u_components,
                        d_components = obj.real._d_components,
                        i_components = _tagged_intermediate_node_to_uid(
                            obj.real._i_components,
                            _intermediate_node_to_uid
                        ),
                        label = obj.real.label,
                        uid = obj.real._node.uid,
                        nid = id(obj.real._node)
                    )
                        
                    n_re = "{}_re".format(n)
                    self._tagged_reals[n_re] = re
                    
                    im = TaggedIntermediateReal(
                        value = obj.imag._x,
                        u_components = obj.imag._u_components,
                        d_components = obj.imag._d_components,
                        i_components = _tagged_intermediate_node_to_uid(
                            obj.imag._i_components,
                            _intermediate_node_to_uid
                        ),
                        label = obj.imag.label,
                        uid = obj.imag._node.uid,
                        nid = id(obj.imag._node)
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
                
    def _thaw(self,context):
        """
        Convert a pickled archive into one containing uncertain numbers.

        """        
        # Update context registers
                                
        context._complex_ids.update(self._complex_ids)
        
        _mat = self._correlations._mat  
        
        _nodes = dict()
        for nid,data in self._leaf_nodes.iteritems():
            uid, u = data
                        
            # NB a new node is not created if the uid is already
            # associated with a node
            _nodes[nid] = context.new_leaf(
                uid, 
                self._labels[uid], 
                u, 
                self._dof_record.get(uid,inf), 
                # Assume that the existence of a row 
                # in the correlation matrix implies 
                # correlation with something.
                independent=(uid not in _mat)  
            )
                            
        # Update ensembles
        _ensembles = weakref.WeakKeyDictionary()
        for uid, ensemble in self._ensembles.iteritems():
            # context ensembles are indexed by node
            nid = context._registered_leaf_nodes[uid]
            
            # The members of an ensemble are all indices 
            # of _ensembles, so we don't want to repeat 
            # this step unnecessarily.
            if nid not in _ensembles:
            
                # Make a set of nodes for this uid 
                _ensemble = weakref.WeakSet()
                for uid_i in ensemble:
                    _ensemble.add(
                        context._registered_leaf_nodes[uid_i]
                    )
                
                # Update the _ensembles mapping for each  
                # element of the set
                for nid_i in _ensemble:
                    assert nid_i not in _ensembles
                    _ensembles[nid_i] = _ensemble
                
        # The ensembles associated with these nodes 
        # can be merged with current context ensembles.
        context._ensemble.update(_ensembles)
                                              
        # When reconstructing, `_tagged` needs to be updated with 
        # new uncertain numbers.
        #
        # Every UN has a unique node, even when different names
        # were used to archive the same uncertain number.
        # This is implemented here by caching.
        # For real's, build a mapping of (old)nid -> (new)UN.
        real_buf = {}
        complex_buf = {}
        for name,obj in self._tagged.iteritems():
            if isinstance(obj,(TaggedElementaryReal,TaggedIntermediateReal)):
                if obj.nid in real_buf:
                    un = real_buf[obj.nid]
                else:
                    # TODO: builder has to change
                    un = _builder(
                        context,
                        name,
                        _nodes,
                        self._tagged_reals
                    )
                    real_buf[obj.nid] = un
                    
                self._tagged[name] = un
                
                
            elif isinstance(
                    obj,
                    (TaggedElementaryComplex,TaggedIntermediateComplex)
                ):
                # Caching is more complicated for complex. The components
                # should use the same cache as reals (because just one component
                # may be relabeled in the archive). The complex container does not
                # have a node, so it cannot be identified so easily. Need to
                # have a complex buffer with nid pairs as keys.
                name_re = obj.n_re
                name_im = obj.n_im
                
                obj_re_nid = self._tagged_reals[name_re].nid
                obj_im_nid = self._tagged_reals[name_im].nid

                # Complex caching
                complex_key  = (obj_re_nid,obj_im_nid)
                if complex_key in complex_buf:
                    unc = complex_buf[complex_key]
                else:
                    if obj_re_nid in real_buf:
                        un_re = real_buf[obj_re_nid]
                    else:
                        # TODO: builder has to change
                        un_re = _builder(
                            context,
                            name_re,
                            _nodes,
                            self._tagged_reals
                        )
                        real_buf[obj_re_nid] = un_re
                        
                    if obj_im_nid in real_buf:
                        un_im = real_buf[obj_im_nid]
                    else:
                        # TODO: builder has to change
                        un_im = _builder(
                            context,
                            name_im,
                            _nodes,
                            self._tagged_reals
                        )
                        real_buf[obj_im_nid] = un_im

                    unc = UncertainComplex(un_re,un_im)
                    complex_buf[complex_key] = unc
                    
                    assert un_re._is_elementary == un_im._is_elementary
                    unc._is_elementary = un_re._is_elementary
                    
                    # TODO: do I need to let the context have these uid's?
                    # The _builder will register the real uids, but does
                    # not see a complex pair!
                    if unc._is_elementary:
                        unc._uid = (unc.real._uid,unc.imag._uid)
                        
                    unc.label = obj.label
                
                self._tagged[name] = unc
            else:
                assert False
                
        # Add correlations between all elementary
        # uncertain numbers to the existing record
        R = context._correlations
        
        # _mat = self._correlations._mat  # A mapping of mappings
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
def _tagged_intermediate_node_to_uid(i_components,_intermediate_node_to_uid):
    """
    Return a Vector containing only tagged intermediate UNs 
    
    """
    i_sequence = []
    v_sequence = []
    for i,v in i_components.iteritems():
        if i in _intermediate_node_to_uid:
            i_sequence.append(i)
            v_sequence.append(v)
            
     return Vector(index=i_sequence,value=v_sequence)
    

"""
Notes
-----

`tagged_real` is indexed by name and maps to an object to be constructed.

If an elementary real UN is needed, this is constructed and the new node
is added to _nodes using the node ID as index.

"""
def _builder(context,obj_name,_nodes,_tagged_reals):
    """
    Construct an intermediate un object for `obj_name`.
    
    """
    # NB: the name gives access to the object
    obj = _tagged_reals[obj_name]
    
    # There are 2 possible types of `obj`:
    #   TaggedElementaryReal
    #   TaggedIntermediateReal
    if isinstance(obj,TaggedElementaryReal):
        un = context._archived_elementary_real(
            uid = obj.uid,
            x = obj.x,
            u = obj.u,
            df = obj.df,
            label = obj.label
        )
        _nodes[obj.nid] = un._node      # map old nid to new node       
        _tagged_reals[obj_name] = un    # tag now maps to an object
        
    elif isinstance(obj,TaggedIntermediateReal):                
        # All nodes in `obj.pds` have been reconstructed. 
        # ` _nodes` can be used to map the old ID to the new node.

        # For older archives, there were no `i_components`
        # This for compatibility with older GTC archives
        if not hasattr(obj,'i_components'):
            obj.i_components = Vector()
            
        un = UncertainReal(
            obj.value,
            obj.u_components,
            obj.i_components,
            _nodes[obj.nid],
            context
            )

        if len(obj.u_components) == 0:
            # Restore constants
            un._is_number = True 
            
        if hasattr(obj,'uid') and obj.uid is not None:
            # Older archives will not have the uid attribute
            un._uid = obj.uid
            un._is_intermediate = True
            
        un.label = obj.label
        
        _tagged_reals[obj_name] = un

    else:
        assert False, "unexpected: %r" % obj

    return un


#------------------------------------------------------------------     
def dump(file,ar):
    """Save an archive in a file

    :arg file:  a file object opened in binary
                write mode (with 'wb')
                
    :arg ar: an :class:`Archive` object
      
    Several archives can be saved in a file 
    by repeated use of this function.
    
    .. note::

        This function can only be called once on a
        particular archive.
    
    """
    ar._freeze()
    pickle.dump(ar,file,protocol=pickle.HIGHEST_PROTOCOL)
    del ar

#------------------------------------------------------------------     
def load(file):
    """Load an archive from a file

    :arg file:  a file object opened in binary
                read mode (with 'rb')

    Several archives can be extracted from 
    one file by repeatedly calling this function.
    
    """
    ar = pickle.load(file)
    ar._thaw(default.context)
    
    return ar

#------------------------------------------------------------------     
def dumps(ar,protocol=-1):
    """
    Return a string representation of the archive 

    :arg ar: an :class:`Archive` object
    :arg protocol: encoding type 

    Possible values for ``protocol`` are described in the 
    Python documentation for the 'pickle' module.

    ``protocol=0`` creates an ASCII string, but note
    that many (special) linefeed characters are embedded.
    
    .. note::

        This function can only be called once on a
        particular archive.

    """
    # Can save one of these strings in a single binary file,
    # using write(), when protocol=-1 is used. A corresponding
    # read() is required to extract the string. Alternatively,
    # when protocol=0 is used a text file can be used, but again
    # write() and read() have to be used, because otherwise the
    # embedded `\n` characters are interpreted incorrectly.
    #
    # It is possible to save these strings in CSV files too, 
    # provided protocol=0 is used.
    ar._freeze()
    s = pickle.dumps(ar,protocol)
    del ar
    
    return s
    
#------------------------------------------------------------------     
def loads(s):
    """
    Return an archive object restored from a string representation

    :arg s: a string created by :func:`dumps`
    
    """
    ar = pickle.loads(s)
    ar._thaw(default.context)
    
    return ar


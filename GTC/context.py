"""
"""
import os 
import weakref

from GTC2.GTC.weak_symmetric_matrix import WeakSymmetricMatrix
from GTC2.GTC.lib_real import UncertainReal
from GTC2.GTC.lib_complex import UncertainComplex
from GTC2.GTC.vector import *
from GTC2.GTC.nodes import *

inf = float('inf')
nan = float('nan')

__all__ = ('context',)

#----------------------------------------------------------------------------
class Context(object):

    """
    A ``Context`` object keeps track of uncertain-number details 
    """
    
    def __init__(self,id=None):
        
        # A UUID4 identifier is the default.
        # However, the user can enter a specific ID if 
        # required (there is no guarantee then that 
        # Context IDs will be unique - user beware!)
        self._id = _uuid() if id is None else id
    
        self._elementary_id_counter = long(0)
        self._intermediate_id_counter = long(0)

        # The value will be the (real._node,imag._node) pair, when
        # either the real or imag node is used as key.
        self._complex_ids = weakref.WeakKeyDictionary()

        # Correlation coefficients between elementary UNs 
        # Keys are Leaf objects 
        self._correlations = WeakSymmetricMatrix()

        # A register of elementary uncertain numbers that are 
        # considered simultaneous samples from a multivariate parent. 
        # The key is any Leaf, the value is the set of objects 
        # in the ensemble.
        self._ensemble = weakref.WeakKeyDictionary()
  
        # Leaf cache - used to avoid duplicating UNs when unpacking archives 
        self._registered_leaf_nodes = weakref.WeakValueDictionary()
  
    #------------------------------------------------------------------------
    def _next_elementary_id(self):
        """
        Return an ID number

        Each elementary uncertain number has a 
        unique ID. This is composed of the 
        context's ID and an object instance ID.

        Returns
        -------
        ID : tuple(context_ID,integer)
        
        """
        self._elementary_id_counter += 1
        return (
            self._id,
            self._elementary_id_counter
        )

    #------------------------------------------------------------------------
    def _next_intermediate_id(self):
        """
        Return an ID number

        Each intermediate uncertain number has
        a unique ID composed of the context's ID
        and an object instance index.

        Returns
        -------
        ID : tuple(context_ID,integer)
        
        
        ..Note:
        
            The ID generated here has the same  
            format as elementary UIDs, but they  
            are independent: values can overlap!
            
        """
        self._intermediate_id_counter += 1
        return (
            self._id,
            self._intermediate_id_counter
        )
 
    #------------------------------------------------------------------------
    def new_leaf(self, uid, tag, u, df, independent):
        """
        Return a new ``Leaf`` node unless one with the same uid exists
        
        """
        try:
            return self._registered_leaf_nodes[uid]
        except KeyError:
            l = Leaf(self,uid,tag,u,df,independent=independent)
            self._registered_leaf_nodes[uid] = l
            return l 

    #------------------------------------------------------------------------
    def complex_ensemble(self,seq,df):
        """
        Register this group of elementary uncertain numbers as
        belonging to an ensemble.

        The members of an ensemble may be correlated and have finite 
        degrees of freedom without causing problems with the Welch-
        Satterthwaite or the Willink-Hall calculations. See: 
        
        R Willink, Metrologia 44 (2007) 340-349, section 4.1.1

        The uncertain numbers in ``seq`` must be elementary
        and have the same numbers of degrees of freedom. 
        
        Effectively, members of an ensemble are treated 
        as simultaneous independent measurements of 
        a multivariate distribution. 
        
        """
        # NB, complex_ensemble() simply assigns ``dof`` 
        # without checking for previous values. This 
        # avoids overhead and it won't matter because 
        # ordinary users call this via the core methods.
        
        if len(seq):
            # All UNs passed to this function will
            # have been declared independent=False 
            try:
                assert all( 
                    x._node.independent == False 
                        for pair in seq 
                            for x in (pair.real,pair.imag) 
                )
            except AttributeError:
                raise RuntimeError(
                    "members of an ensemble must be elementary and dependent"
                )

            # ensemble members must have the same degrees of freedom
            assert all( s_i.df == df for s_i in seq )

            # ensemble members must be elementary
            assert all( s_i.is_elementary for s_i in seq )
            
            ensemble = weakref.WeakSet( 
                x._node 
                    for pair in seq 
                        for x in (pair.real,pair.imag) 
            )

            for pair in seq:
                for x in (pair.real,pair.imag):
                    nid =x._node
                    assert nid not in self._ensemble
                    self._ensemble[nid] = ensemble     
                
    #------------------------------------------------------------------------
    def real_ensemble(self,seq,df):
        """Declare the ``seq`` of uncertain numbers as an ensemble.

        Uncertain numbers in ``seq`` must be elementary. 
        
        Members of an ensemble may be correlated without causing  
        problems with the Welch-Satterthwaite calculation. 
        
        See: 
        
        R Willink, Metrologia 44 (2007) 340-349, section 4.1.1
        
        Effectively, members of an ensemble are treated 
        as simultaneous independent measurements of 
        a multivariate distribution. 
        
        """
        if len(seq):
            # All UNs passed to this function 
            # have been declared independent=False 
            assert all( s_i._node.independent == False for s_i in seq )

            # ensemble members must have the same degrees of freedom
            assert all( s_i.df == df for s_i in seq )

            # ensemble members must be elementary
            assert all( s_i.is_elementary for s_i in seq )
                    
            ensemble = weakref.WeakSet( x._node for x in seq )

            for s_i in seq:
                nid = s_i._node
                # Index keys are any member node
                assert nid not in self._ensemble,\
                    "found '{!r}' already".format(nid)
                self._ensemble[nid] = ensemble     
                
    #------------------------------------------------------------------------
    def append_real_ensemble(self,member,x):
        """
        Append an element to the an existing ensemble

        The uncertain number must be elementary and have the 
        numbers of degrees of freedom as the ensemble. 
        
        """
        nid = member._node
        try:
            ensemble = self._ensemble[nid]
        except KeyError:
            raise RuntimeError(
                "cannot find an ensemble for '{!r}'".format(member)
            )
            
        if x.df != member.df:
            raise RuntimeError(
                "members of an ensemble must have the same dof!"
            )

        if not x.is_elementary:
            raise RuntimeError(
                "members of an ensemble must be elementary!"
            )

        x._node.independent = False

        ensemble.add(x._node)
            
        # Expect different keys to reference the same object 
        # TODO: could be removed in release versions
        assert all( self._ensemble[nid] is ensemble for nid in ensemble )
            
    #------------------------------------------------------------------------
    def constant_real(self,x,label):
        """
        Return a constant uncertain real number with value 'x' 
        
        A constant uncertain real number has no uncertainty
        and infinite degrees of freedom.        
        
        Parameters
        ----------
        x : float

        Returns
        -------
        UncertainReal
        
        """
        # A constant does not need a UID, 
        # because it does not need to be archived.
        return UncertainReal(
                self
            ,   x
            ,   Vector( )
            ,   Vector( )
            ,   Vector( )
            ,   Leaf(self,uid=None,tag=label,u=0.0,df=inf)
        )
        
    #------------------------------------------------------------------------
    def elementary_real(self,x,u,df,label,independent):
        """
        Return an elementary uncertain real number.

        Creates an uncertain number with value 'x', standard
        uncertainty 'u' and degrees of freedom 'df'.

        A ``RuntimeError`` is raised if the value of 
        `u` is less than zero or the value of `df` is less than 1.

        Parameters
        ----------
        x : float
        u : float
        df : float
        label : string, or None
        independent : this UN cannot be correlated with another

        Returns
        -------
        UncertainReal
        
        """
        if df < 1:
            raise RuntimeError(
                "invalid degrees of freedom: {!r}".format(df) 
            )
        if u < 0:
            # u == 0 can occur in complex UNs.
            raise RuntimeError(
                "invalid uncertainty: {!r}".format(u)
            )
        
        uid = self._next_elementary_id()

        # Needed for archiving?
        # self._uid_to_u[id] = u
        
        ln = self.new_leaf(uid,label,u,df,independent=independent)
        
        if independent:
            return UncertainReal(
                    self
                ,   x
                ,   Vector( index=[ln],value=[u] )
                ,   Vector( )
                ,   Vector( )
                ,   ln
                )
        else:
            # The node must go in the dependent vector
            return UncertainReal(
                    self
                ,   x
                ,   Vector( )
                ,   Vector( index=[ln],value=[u] )
                ,   Vector( )
                ,   ln
                )
   
    #------------------------------------------------------------------------
    def real_intermediate(self,un,tag):
        """
        Create an intermediate uncertain number
        
        .. note::
        
            If called more than once on the same object, 
            later calls have no effect!

        Parameters
        ----------
        un : uncertain real number
        
        """
        if not un.is_elementary and not un.is_intermediate:
            
            # This is a new registration 
            uid = self._next_intermediate_id()
            u = un.u
            
            # A Node must be assigned
            un._node = Node(
                self,
                uid,
                tag,
                u
            )

            # The i_component wrt itself!
            un._i_components = merge_vectors(
                un._i_components,
                Vector( index=[un._node], value=[u] )
            )
            un.is_intermediate = True
                
            # else:
                # Assume that everything has been registered, perhaps 
                # the user has repeated the registration process.
                # pass

        # else:
            # # There should be no harm in ignoring elementary UNs.
            # # They will be archived properly and they are not dependent
            # # on anything. It is convenient for the user not to worry
            # # whether or not something is elementary our intermediate 
            # pass            
  
    #------------------------------------------------------------------------
    def constant_complex(self,z,label):
        """
        Return a constant uncertain complex number.
        
        Creates a constant with value 'z' with an
        optional label.

        When a label is supplied, the encapsulated real
        and imaginary components are given the label with added
        suffixes '_re' and '_im'.
        
        A constant uncertain complex number has no uncertainty
        and infinite degrees of freedom.        
        
        Parameters
        ----------
        z : complex
        label : string, or None

        Returns
        -------
        UncertainComplex
        
        """
        if label is None:
            label_r,label_i = None,None
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
        real = self.constant_real(z.real,label_r)
        imag = self.constant_real(z.imag,label_i)

        ucomplex = UncertainComplex(real,imag)
        
        ucomplex._label = label
            
        return ucomplex        

    #------------------------------------------------------------------------
    def elementary_complex(self,z,u_r,u_i,r,df,label,independent):
        """
        Return an elementary uncertain complex number.

        Parameters
        ----------
        x : complex
        u_r, u_i : standard uncertainties 
        r : correlation coefficient
        df : float
        label : string, or None

        Returns
        -------
        UncertainComplex
        
        When a label is supplied, the encapsulated real
        and imaginary components are given the label with added
        suffixes '_re' and '_im'.

        """
        if label is None:
            label_r,label_i = None,None
            
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
        # We assume that the IDs assigned are consecutive.
        real = self.elementary_real(z.real,u_r,df,label_r,independent)
        imag = self.elementary_real(z.imag,u_i,df,label_i,independent)

        # We need to be able to look up complex pairs
        # TODO: is this only needed in dof calculations?
        # If so, we might be able to relax this later.
        complex_id = (real._node,imag._node)
        self._complex_ids[real._node] = complex_id        
        self._complex_ids[imag._node] = complex_id

        if r is not None and r != 0.0:
            self._correlations[ real._node,imag._node ] = r
            
        ucomplex = UncertainComplex(real,imag)
        ucomplex.is_elementary = True
        
        ucomplex._label = label
            
        return ucomplex        
 
    #------------------------------------------------------------------------
    def complex_intermediate(self,z,label):
        """
        Return an intermediate uncertain complex number

        :arg z: the uncertain complex number
        :type z: :class:`UncertainComplex`

        :arg label: a label

        If ``label is not None`` the label will be applied
        to the uncertain complex number and labels with
        a suitable suffix will also be applied to the
        real and imaginary component uncertain real numbers.
        
        """
        # TODO: does the fact that this is a complex 
        # number need to be taken into account for archiving?
        
        if label is None:
            self.real_intermediate(z.real,None)
            self.real_intermediate(z.imag,None) 
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
            self.real_intermediate(z.real,label_r)
            self.real_intermediate(z.imag,label_i) 
            
        z._label = label
            

    #------------------------------------------------------------------------
    def uncertain_complex(self,r,i,label):
        """
        Return an uncertain complex number

        :arg r: the real component
        :type r: :class:`UncertainReal`

        :arg i: the imaginary component
        :type i: :class:`UncertainReal`

        :arg label: a label

        When a label is supplied, the encapsulated real
        and imaginary components are given the label with added
        suffixes '_re' and '_im'.
        
        """
        if label is None:
            return UncertainComplex(r,i)
        else:
            r.label = "{}_re".format(label)
            i.label = "{}_im".format(label)
            
            uc = UncertainComplex(r,i)
            uc.label = label
            
            return uc         
 
    #------------------------------------------------------------------------
    def set_correlation(self,x1,x2,r):
        """
        Assign a correlation coefficient to a pair of uncertain numbers

        Requires |r| <= 1, otherwise a RuntimeError is raised.
        
        If a definition exists, it will be quietly overwritten.
        
        Raises `RuntimeError` if either `x1` or `x2` was not 
        declared with `independent=False`.
        
        Parameters
        ----------
        x1, x2 : `UncertainReal`
        r : float

        Returns
        -------
        None
        
        """
        if abs(r) > 1.0:
            raise RuntimeError,"invalid value: '%s'" % r
        

        if (
            x1.is_elementary and 
            x2.is_elementary
        ):        
        
            # The leaf nodes 'ln1' and 'ln2' are keys to
            # the correlation coefficient stored by the context.
            ln1 = x1._node
            ln2 = x2._node
            
            if (
                not ln1.independent and
                not ln2.independent
            ):
                if ln1 is ln2 and r != 1.0:
                    raise RuntimeError(
                        "value should be 1.0, got: '{}'".format(r)
                    )
                else:
                    self._correlations[ln1,ln2] = r
            else:
                raise RuntimeError( 
                    "`set_correlation` called on independent node"
                )
            
        else:
            raise RuntimeError(
                "Arguments must be elementary uncertain numbers, \
                got: {!r} and {!r}".format(x1,x2)
            )

    #------------------------------------------------------------------------
    def get_correlation(self,x1,x2):
        """
        Return the correlation coefficient for a pair of uncertain numbers

        Parameters
        ----------
        id1, id2 : integer

        Returns
        -------
        float
        
        """
        ln1 = x1._node
        ln2 = x2._node
        
        if ln1 is ln2:
            return 1.0
        else:
            return self._correlations.get( (ln1,ln2) )
        
#----------------------------------------------------------------------------
# TODO: use Python library for this in Python 3 version
# did not do so in Python 2 version because there were 
# issues with DLLs and the VC++ complier used for Python 
#
def _uuid():
    """
    Return a UUID (version 4) as 128-byte integer 
    
    """
    # Obtain 16 integers each between 0 and 255   
    byte = bytearray( os.urandom(16) )

    # clock_seq_hi_and_reserved
    byte[7] &= 0xBF  # b6=0 
    byte[7] |= 0x80  # b7=1
    
    # MSB of time_hi_and_version
    byte[9] &= 15   # upper nibble = 0
    byte[9] |= 64   # upper nibble = 4
         
    # Create a 128-byte integer 
    u = reduce(
        lambda u,i: u + (byte[i] << i*8),
        xrange(16),
        0
    )
    
    # # Can verify that the UUID conforms to 
    # # RFC-4122 here, expect version == 4 
    # import uuid
    # x = uuid.UUID( int=u ) 
    # print x 
    # print x.version
    
    return u
    
#----------------------------------------------------------------------------
context = Context()


"""
Copyright (c) 2018, Measurement Standards Laboratory of New Zealand.

"""
import os 
import weakref

from GTC import inf, nan
from GTC.lib_real import UncertainReal
from GTC.lib_complex import UncertainComplex
from GTC.vector import *
from GTC.nodes import *

__all__ = ('Context',)

#----------------------------------------------------------------------------
class Context(object):

    """
    A ``Context`` object the creation of uncertain numbers 
    """
    
    def __init__(self,id=None):
        
        # A UUID4 identifier is the default.
        # However, the user can enter a specific ID if 
        # required (there is no guarantee then that 
        # Context IDs will be unique - user beware!)
        self._id = _uuid() if id is None else id
    
        self._elementary_id_counter = long(0)
        self._intermediate_id_counter = long(0)
  
        # Caching to avoid duplication when unpacking archives 
        self._registered_leaf_nodes = weakref.WeakValueDictionary()
        self._registered_intermediate_nodes = weakref.WeakValueDictionary()
          
    #------------------------------------------------------------------------
    def _next_elementary_id(self):
        """
        Return an ID number

        Each elementary uncertain number has a 
        unique ID. This is composed of the 
        context's ID and an integer.

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
        and an integer.

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
    def new_leaf(self,uid,label,u,df,independent):
        """
        Return a new ``Leaf`` node unless one with the same uid exists
        
        """
        try:
            l =  self._registered_leaf_nodes[uid]
        except KeyError:          
            l = Leaf(uid,label,u,df,independent)
            self._registered_leaf_nodes[uid] = l
            
        return l 
            
    #------------------------------------------------------------------------
    def new_node(self,uid,label,u):
        """
        Return a new ``Node`` unless one with the same uid exists
        
        """
        try:
            n = self._registered_intermediate_nodes[uid]
        except KeyError:
            n = Node(uid,label,u)
            self._registered_intermediate_nodes[uid] = n

        return n
        
    #------------------------------------------------------------------------
    def archived_elementary_real(self,uid,x):
        """
        Return an elementary uncertain real number.

        Restore an uncertain number that has been archived. 
        Most properties will be associated with a Leaf node 
        that can be obtained using `uid` as key. The value 
        'x', is not stored in the Leaf. 

        Parameters
        ----------
        uid : unique identifier
        x : float

        Returns
        -------
        UncertainReal
        
        """
        # Use the cache `self._registered_leaf_nodes` 
        # to avoid creating multiple Leaf objects.
        l = self._registered_leaf_nodes[uid]

        # The Leaf object is used to seed one 
        # Vector component, so that uncertainty 
        # will be propagated
        if l.independent:
            un = UncertainReal(
                    self
                ,   x
                ,   Vector( index=[l],value=[l.u] )
                ,   Vector( )
                ,   Vector( )
                ,   l
                )
        else:
            un = UncertainReal(
                    self
                ,   x
                ,   Vector( )
                ,   Vector( index=[l],value=[l.u] )
                ,   Vector( )
                ,   l
                )
        
        return un  
        
    #------------------------------------------------------------------------
    def complex_ensemble(self,seq,df):
        """
        Declare the uncertain numbers in ``seq`` to be an ensemble.

        The uncertain numbers in ``seq`` must be elementary
        and have the same numbers of degrees of freedom. 
        
        It is permissible for members of an ensemble to be correlated 
        and have finite degrees of freedom without causing problems 
        when evaluating the effective degrees of freedom. See: 
        
        R Willink, Metrologia 44 (2007) 340-349, section 4.1.1

        Effectively, members of an ensemble are treated 
        as simultaneous independent measurements of 
        a multivariate distribution. 
        
        """
        # NB, we simply assign ``dof`` without checking for previous values. 
        # This avoids overhead and should not be a risk, because 
        # users call this method via functions in the ``core`` module.
        
        if len(seq):
            # TODO: assertions not required in release version
            # ensemble members must have the same degrees of freedom
            assert all( s_i.df == df for s_i in seq )

            # ensemble members must be elementary
            assert all( s_i.is_elementary for s_i in seq )
            
            # All UNs will have been declared with ``independent=False`` 
            if not all( 
                x._node.independent == False 
                    for pair in seq 
                        for x in (pair.real,pair.imag) 
            ):
                raise RuntimeError(
                    "members of an ensemble must be elementary and dependent"
                )
                
            ensemble = set( 
                x._node.uid 
                    for pair in seq 
                        for x in (pair.real,pair.imag) 
            )
            # This object is referenced from the Leaf node of each member
            for pair in seq:
                for x in (pair.real,pair.imag):
                    x._node.ensemble = ensemble
            
    #------------------------------------------------------------------------
    def real_ensemble(self,seq,df):
        """
        Declare the uncertain numbers in ``seq`` to be an ensemble.

        The uncertain numbers in ``seq`` must be elementary
        and have the same numbers of degrees of freedom. 
        
        It is permissible for members of an ensemble to be correlated 
        and have finite degrees of freedom without causing problems 
        when evaluating the effective degrees of freedom. See: 
        
        R Willink, Metrologia 44 (2007) 340-349, section 4.1.1

        Effectively, members of an ensemble are treated 
        as simultaneous independent measurements of 
        a multivariate distribution. 
        
        """
        if len(seq):
            # TODO: assertions not required in release version
            # have been declared independent=False 
            assert all( s_i._node.independent == False for s_i in seq )

            # ensemble members must have the same degrees of freedom
            assert all( s_i.df == df for s_i in seq )

            # ensemble members must be elementary
            assert all( s_i.is_elementary for s_i in seq )
                    
            ensemble = set( x._node.uid for x in seq )
            # This object is referenced from the Leaf node of each member
            for s_i in seq:
                s_i._node.ensemble = ensemble     
                
    #------------------------------------------------------------------------
    def append_real_ensemble(self,member,x):
        """
        Append an element to the an existing ensemble

        The uncertain number must be elementary and have the 
        numbers of degrees of freedom as other members 
        of the ensemble (not checked). 
        
        """
        # TODO: remove assertions in release version, because 
        # this function is only called from within GTC modules. 
        assert x.df == member._node.df
        assert x.is_elementary
        assert x._node.independent == False
        
        # All Leaf nodes refer to the same ensemble object 
        # So by adding a member here, all the other Leaf nodes 
        # see the change.
        x._node.ensemble.add(x._node)
                        
    #------------------------------------------------------------------------
    def constant_real(self,x,label):
        """
        Return a constant uncertain real number with value ``x`` 
        
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
            ,   Leaf(uid=None,label=label,u=0.0,df=inf)
        )
        
    #------------------------------------------------------------------------
    def elementary_real(self,x,u,df,label,independent):
        """
        Return an elementary uncertain real number.

        Creates an uncertain number with value ``x``, standard
        uncertainty ``u`` and degrees of freedom ``df``.

        A ``RuntimeError`` is raised if the value of 
        `u` is less than zero or the value of `df` is less than 1.

        The ``independent`` argument controls whether this
        uncertain number may be correlated with others.
        
        Parameters
        ----------
        x : float
        u : float
        df : float
        label : string, or None
        independent : Boolean

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
            
        # NB, we may create an uncertain number with no uncertainty 
        # that is not recognised as a 'constant' by setting u=0. 
        # It may be desirable to allow this. In the case of a complex UN,
        # for example, we would still see a zero-valued component in the 
        # uncertainty budget. That might be less confusing than to make 
        # the constant component disappear quietly.      
        
        uid = self._next_elementary_id()
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
            return UncertainReal(
                    self
                ,   x
                ,   Vector( )
                ,   Vector( index=[ln],value=[u] )
                ,   Vector( )
                ,   ln
                )
   
    #------------------------------------------------------------------------
    def real_intermediate(self,un,label):
        """
        Create an intermediate uncertain number
        
        To investigate the sensitivity of subsequent results,
        an intermediate UN must be declared.
        
        Parameters
        ----------
        un : uncertain real number
        
        """
        if not un.is_elementary:
            if not un.is_intermediate:                     
                # This is a new registration 
                uid = self._next_intermediate_id()
                
                u = un.u
                un._node = Node(
                    uid,
                    label,
                    u
                )

                # Seed the Vector of intermediate components 
                # with this new Node object, so that uncertainty 
                # will be propagated.
                un._i_components = merge_vectors(
                    un._i_components,
                    Vector( index=[un._node], value=[u] )
                )
                un.is_intermediate = True
                
                self._registered_intermediate_nodes[uid] = un._node
            
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
        
        A constant uncertain complex number has no uncertainty
        and infinite degrees of freedom.        

        The real and imaginary components are given labels 
        with the suffixes '_re' and '_im' to added ``label``.
        
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
        
        The real and imaginary components are given labels 
        with the suffixes '_re' and '_im' to added ``label``.

        The ``independent`` argument controls whether this
        uncertain number may be correlated with others.
        
        """
        if label is None:
            label_r,label_i = None,None
            
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
        # `independent` will be False if `r != 0`
        real = self.elementary_real(z.real,u_r,df,label_r,independent)
        imag = self.elementary_real(z.imag,u_i,df,label_i,independent)

        # We need to be able to look up complex pairs
        # The integer part of the IDs are consecutive.
        complex_id = (real._node.uid,imag._node.uid)
        real._node.complex = complex_id 
        imag._node.complex = complex_id
        
        if r is not None:
            real._node.correlation[imag._node.uid] = r 
            imag._node.correlation[real._node.uid] = r 
            
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
        a suitable suffix will be applied to the
        real and imaginary components.
        
        """
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
        
#----------------------------------------------------------------------------
# TODO: use Python library for this in Python 3 version
# did not do so in Python 2 version because there were 
# issues with DLLs and the VC++ complier Python 2 used.  
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
# A default context
#
_context = Context()

"""
Copyright (c) 2018, Measurement Standards Laboratory of New Zealand.

"""
import os 
import weakref

from GTC.vector import *
from GTC.nodes import *

from GTC import inf, nan

__all__ = ('Context','_context')

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
        if uid in self._registered_leaf_nodes:
            raise RuntimeError(
                "the Leaf node uid({}) is in use already".format(uid)
            )
        else:          
            l = Leaf(uid,label,u,df,independent)
            self._registered_leaf_nodes[uid] = l
            
        return l 
            
    #------------------------------------------------------------------------
    def new_node(self,uid,label,u):
        """
        Return a new ``Node`` unless one with the same uid exists
        
        """
        if uid in self._registered_intermediate_nodes:
            raise RuntimeError(
                "the intermediate node uid({}) is in use already".format(uid)
            )
        else:          
            n = Node(uid,label,u)
            self._registered_intermediate_nodes[uid] = n

        return n
        
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

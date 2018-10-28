"""
Copyright (c) 2018, Measurement Standards Laboratory of New Zealand.

"""
import uuid
import weakref
try:
    long  # Python 2
except NameError:
    long = int

from GTC.nodes import (
    Leaf,
    Node
)

__all__ = (
    'Context',
    '_context'
)

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
        self._id = uuid.uuid4().int if id is None else id
    
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
# A default context
#
_context = Context()

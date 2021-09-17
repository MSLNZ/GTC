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
        # Note the format of intermediate uncertain-number UIDs 
        # differs from elementary uncertain-number UIDs (changed in v1.3.7).
        # GTC code only tests UIDs for order and equality, it does not make 
        # use of the internal structure of a UID. However, previously an elementary 
        # UID and an intermediate UID could be equal, although they were intended 
        # to identify different types of node object. Now intermediate UIDs are 
        # 3-tuples and elementary UIDs 2-tuples, so there can be no confusion.
        self._intermediate_id_counter += 1
        return (
            self._id,
            self._intermediate_id_counter,
            0
        )
 
    #------------------------------------------------------------------------
    def new_leaf(self,uid,label,u,df,independent):
        """
        Return a new ``Leaf`` node unless one with the same uid exists
        
        """        
        if uid in self._registered_leaf_nodes:
            # If the node found is indistinguishable from the new node 
            # then quietly ignore the request 
            l = self._registered_leaf_nodes[uid]
            OK = (
                label == l.label and 
                u == l.u and 
                df == l.df and 
                independent == l.independent
            )
            if not OK:
                raise RuntimeError(
                    "the Leaf node uid({}) is in use already".format(uid)
                )
        else:          
            l = Leaf(uid,label,u,df,independent)
            self._registered_leaf_nodes[uid] = l
            
        return l 
            
    #------------------------------------------------------------------------
    def new_node(self,uid,label,u,df):
        """
        Return a new ``Node`` unless one with the same uid exists
        
        """
        
        # Prior to v1.3.5, degrees of freedom was not buffered in a Node.
        # Now, to maintain backward comparability, a value ``None`` 
        # will be inserted for backward compatibility when needed.
        # See the ``shim_1_3_3`` function in archive.py.        

        if uid in self._registered_intermediate_nodes:
            # If the node found is indistinguishable from the new node 
            # then quietly ignore the request 
            n = self._registered_intermediate_nodes[uid]
            
            OK = (
                label == n.label and 
                u == n.u and 
                df == n.df 
            )
            if not OK:
                raise RuntimeError(
                    "intermediate node uid({}), '{}', u={}, df={} is used".format(
                        uid,label,u,df
                    )
                )
        else:          
            n = Node(uid,label,u,df)
            self._registered_intermediate_nodes[uid] = n

        return n


#----------------------------------------------------------------------------
# A default context
#
_context = Context()

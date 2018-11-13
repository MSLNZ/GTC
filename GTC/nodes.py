"""
Copyright (c) 2018, Measurement Standards Laboratory of New Zealand.

"""

__all__ = (
    'Leaf',
    'Node'
)

#----------------------------------------------------------------------------
class Node(object):
    
    """
    A `Node` holds information about an intermediate uncertain real number
    """

    __slots__ = [
        'uid',
        'label',
        'u',
        '__weakref__'
    ]
 
    def __init__(self,uid,label,u):    
        self.uid = uid
        self.label = label
        self.u = u
 
#----------------------------------------------------------------------------
class Leaf(Node):

    """
    A `Leaf` holds information about an elementary uncertain real number
    """
    
    __slots__ = [
        'df',
        'independent',
        'complex',
        'correlation',
        'ensemble'
    ]
    
    def __init__(self,uid,label,u,df,independent=True):
        Node.__init__(self,uid,label,u)
    
        self.df = df
        self.independent = independent
        if not independent:
            self.correlation = {uid: 1.0}
            self.ensemble = set()
        
    
    
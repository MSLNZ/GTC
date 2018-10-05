"""
"""

__all__ = [ 'Leaf', 'Node' ]

#----------------------------------------------------------------------------
class Node(object):
    
    """
    A `Node` holds information about an uncertain real number
    """

    __slots__ = [
        'uid',
        'tag',
        'u',
        '__weakref__'
    ]
 
    def __init__(self,uid,tag,u):    
        self.uid = uid
        self.tag = tag
        self.u = u
 
#----------------------------------------------------------------------------
class Leaf(Node):

    """
    A `Leaf` holds information about an elementary uncertain real number
    """
    
    __slots__ = [
        'df',
        'independent'
    ]
    
    def __init__(self,uid,tag,u,df,independent=True):
        Node.__init__(self,uid,tag,u)
    
        self.df = df
        self.independent = independent
        
    
    
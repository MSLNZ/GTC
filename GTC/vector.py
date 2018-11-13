"""
The Vector class is used in the propagation of uncertainty in uncertain numbers 
by forward automatic differentiation. A Vector holds a sequence of components 
of uncertainty
 
The class is like a mapping of key-value pairs. The key is associated with 
the identity of the uncertainty component and the value is the component of
uncertainty.

Propagation of uncertainty is a time-critical process, so elements are 
ordered to help with optimisation. Ordering uses the built-in ``cmp`` 
function applied to the `uid` attribute of a key. 

The function ``is_ordered()`` checks that all reference-value pairs 
in a vector are in order. When ``__debug__`` is ``True``, the order 
of elements is checked when appending (but not during construction).

Copyright (c) 2018, Measurement Standards Laboratory of New Zealand.

"""
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    cmp = lambda a, b: (a > b) - (a < b)


__all__ = (
    'Vector',
    'scale_vector',
    'scale_vector_twice',
    'merge_vectors',
    'merge_weighted_vectors',
    'merge_weighted_vectors_twice',
    'extend_vector'
)

# The `cmp` function works as follows.
# Tuples and lists are compared lexicographically by comparing 
# corresponding elements. This means that for sequences to compare equal, 
# each element must compare equal, the sequences must be of the same  
# type and have the same length.  
#
# If not equal, the sequences are ordered by comparing the first differing 
# elements. For example, cmp([1,2,x], [1,2,y]) returns the same as cmp(x,y). 
# If the corresponding element does not exist, the shorter sequence is ordered 
# first (for example, [1,2] < [1,2,3]).  
#

# The value of `uid` is a pair of integers used for ordering.
# The end point can be marked by a pair of infinities,
# so any other pair compares less than this marker.
# The `END` object is used in the iteration control structures to
# mark an end point. 
INF = ( float('inf'), float('inf') )
class INF_UID(object): uid = INF

#--------------------------------------------------------------
# TODO:
# Look at the initialisation overhead. It might be better to
# call Vector.copy instead of Vector(copy=v)
#
# Look at using the bisect module to improve performance
#
class Vector(object):
    
    """
    A Vector is a collection of ordered index-value pairs.
    
    """
    
    def __init__(self,**kwargs):
        """
        Vector()                # construct an empty vector
        Vector(copy=v)          # construct a copy of vector 'v'
        Vector(index=i,value=v) # construct a vector with 
                                  values 'v' indexed by 'i'
        
        Index order is preserved when a Vector is constructed,
        but not checked.
        
        """
        if len(kwargs) == 0:
            self._index = []
            self._value = []
            
        elif 'copy' in kwargs:
            copy = kwargs['copy']
            self._index = list(copy._index)
            self._value = list(copy._value)
            
        elif 'index' in kwargs and 'value' in kwargs:
        
            # References to the index and value sequences are stored,
            # instead of making copies. Because of this we need to ensure 
            # the mutability of input sequences 
            idx = kwargs['index']
            if getattr(idx,'append',None) is None:
                raise RuntimeError(
                    "mutable sequence required, got {!r}".format(type(idx))
                )
                
            val = kwargs['value']
            if getattr(val,'append',None) is None:
                raise RuntimeError(
                    "mutable sequence required, got {!r}".format(type(val))
                )
                
            self._index = idx
            self._value = val
            
        elif len(kwargs) != 0:
            raise RuntimeError(
                "unidentified keywords '%s'" % kwargs.keys()
            )

### There appears to be a large overhead associated with this            
##        if __debug__: 
##            # Vector is ordered initially   
##            assert is_ordered(self), self._index
                
    def __len__(self):
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def __str__(self):
        return str( dict( izip(self._index,self._value) ) )

    def __contains__(self,i):
        return i in self._index
        
    def __getitem__(self,i):
        return self._value[ self._index.index(i) ]

    def get(self,i,default=None):
        # TODO: use bisect to speed this up, because
        # it impacts on the calculation of covariances
        try:
            return self._value[ self._index.index(i) ]
        except ValueError:
            return default
        
    def items(self):
        return zip(self._index,self._value)
    
    # NB: `keys` and `values` are exposing the 
    # internal structures, which would allow a 
    # client to change, for e.g., an element of 
    # the sequence returned by `keys` and thereby
    # change the underlying Vector. 
    # It would be better to return a copy of the index 
    # and value sequences. 
    # TODO: look at whether there is a significant 
    # time penalty associated with this change.
    def keys(self):
        return self._index

    def values(self):
        return self._value

    def iteritems(self):
        return izip(self._index,self._value)
    
    def iterkeys(self):
        return iter(self._index)

    def itervalues(self):
        return iter(self._value)

    def insert(self,k,v):
        """
        Insert `v` at the correct position for `k` 
        
        """
        assert False, "Not implemented"
        
        # TODO: use bisect to locate the insertion point `i`
        
        self._index.insert(i,k)
        self._value.insert(i,v)
        
    def extend(self,it):
        """
        Append the elements in the iterator `it`
        
        """
        for i,v in it:
            self.append(i,v)
        
    def append(self,i,v):
        """
        Append one (i,v) pair to the vector.

        Note that ordering is not checked unless `__debug__` 
        is `True`. It the responsibility of the client to 
        append elements in the correct order. 
        
        """
        if __debug__:
            # Make sure that order is preserved
            try:
                assert self._index[-1].uid < i.uid, (self._index[-1].uid, i.uid)
            except IndexError: 
                # Don't care about an empty list
                pass
            
        self._index.append(i)
        self._value.append(v)
 
    def pop(self):
        """
        Remove and return the last item 
        
        """
        i = self._index.pop()
        v = self._value.pop()
                
        return i,v 
        
#--------------------------------------------------------------
def is_ordered(v):
    """
    The invariant for all Vector operations.
    
    Returns `True` when the elements are correctly ordered.
    
    """
    try:
        it = v.iterkeys()
        x = next(it)
        while True:
            last, x = x, next(it)
            if x.uid < last.uid: return False
    except StopIteration:
        return True    
            
#--------------------------------------------------------------
def scale_vector(v,w):
    """
    Return a Vector containing the values in `v` multiplied by 'w'

    Parameters
    ----------
    v : Vector
    w : float

    Returns
    -------
    Vector
    
    """
    if w == 1:
        return Vector( copy=v )  
    elif len(v._index):
        return Vector(
            index = v._index,
            value = [ w*x for x in v._value ]
        )
    else:
        # `v` is empty
        return Vector()

        #--------------------------------------------------------------
def merge_vectors(v1,v2):
    """
    Return a vector obtained by merging the vector arguments
    
    Parameters
    ----------
    v1, v2 : Vector

    Returns
    -------
    Vector
    
    """
    if (len(v1) == 0): return v2
    if (len(v2) == 0): return v1
        
    value = []
    index = []

    if v1._index == v2._index:
        # No need to check indices => much faster
        value = [ x1 + x2 for (x1,x2) in izip(v1._value,v2._value) ]
        index = v1._index
    else:
        # A marker at the ends: must remove afterwards!
        v1.append(INF_UID,0)
        v2.append(INF_UID,0)

        # Merge vectors by iteration
        it1 = v1.iteritems()
        it2 = v2.iteritems()

        i1,x1 = next(it1)
        i2,x2 = next(it2)
            
        #------------------------------
        while True:
            
            if i1.uid==INF and i2.uid==INF: break

            case = cmp(i1.uid,i2.uid)
            
            if case == 0:
                # i1 == i2
                index.append(i1)
                value.append( x1+x2 )           
                i1,x1 = next(it1)
                i2,x2 = next(it2)
                    
            elif case < 0:
                # i1 < i2
                index.append(i1)
                value.append( x1 )           
                i1,x1 = next(it1)
            else:
                # i1 > i2
                index.append(i2)
                value.append( x2 )           
                i2,x2 = next(it2)
                
        #------------------------------
        v1.pop() 
        v2.pop()

    return Vector( index=index, value=value )

#--------------------------------------------------------------
def merge_weighted_vectors(v1,w1,v2,w2):
    """
    Return a Vector formed by merging and weighting `v1` and `v2`.

    Parameters
    ----------
    v1, v2 : Vector
    w1, w2 : float
    
    Returns
    -------
    Vector
    
    """
    # If w1==0 or w2==0 the other vector is scaled.
    if (len(v1) == 0) and len(v2):
        return scale_vector(v2,w2)
        
    if (len(v2) == 0) and len(v1):
        return scale_vector(v1,w1)
        
    if (w1 == 1.0) and (w2 == 1.0):
        return merge_vectors(v1,v2)
        
    value = []
    index = []

    if v1._index == v2._index:
        # No need to check indices => much faster
        value = [ w1*x1 + w2*x2 for (x1,x2) in izip(v1._value,v2._value) ]
        index = v1._index
    else:
        # A marker at the ends: must remove afterwards!
        v1.append(INF_UID,0)
        v2.append(INF_UID,0)
        
        # Merge vectors by iteration
        it1 = v1.iteritems()
        it2 = v2.iteritems()

        i1,x1 = next(it1)
        i2,x2 = next(it2)
                    
        #------------------------------
        while True:
            
            if i1.uid==INF and i2.uid==INF: break

            case = cmp(i1.uid,i2.uid)
            
            if case == 0:
                # i1 == i2
                index.append(i1)
                value.append(w1*x1+w2*x2)
                i1,x1 = next(it1)
                i2,x2 = next(it2)
                    
            elif case < 0:
                # i1 < i2
                index.append(i1)
                value.append(w1*x1)
                i1,x1 = next(it1)
            else:
                # i1 > i2
                index.append(i2)
                value.append(w2*x2)
                i2,x2 = next(it2)

        #------------------------------
        v1.pop() 
        v2.pop()
        
    return Vector( value=value, index=index )

#--------------------------------------------------------------
def scale_vector_twice(v,w):
    """
    Return a pair of Vectors weighted by different amounts.
    
    'w' is a pair of weightings.

    Parameters
    ----------
    v : Vector
    w : a pair of weightings

    Returns
    -------
    (Vector,Vector)
    
    """
    w1,w2 = w
    if w1 == 1 and w2 == 1:
        return Vector( copy=v ), Vector( copy=v )
    else:
        return (
            Vector(
                index = v._index,
                value = [ w1*x for x in v._value ]
            ),
            Vector(
                index = v._index,
                value = [ w2*x for x in v._value ]
            )
        )

#--------------------------------------------------------------
def merge_weighted_vectors_twice(v1,w1,v2,w2):
    """
    Return a pair of Vectors obtained by merging `v1` and `v2` 
    with different weightings

    'w1' and 'w2' are weighting pairs (ie, v1 and v2
    are merged two times using different weightings).
    
    Parameters
    ----------
    v1, v2 : Vector
    w1, w2 : weighting pairs

    Returns
    -------
    (Vector,Vector)
    
    """
    if (len(v1) == 0):
        return scale_vector_twice(v2,w2)
        
    if (len(v2) == 0):
        return scale_vector_twice(v1,w1)
        
    value1 = []
    value2 = []
    index = []

    w11,w21 = w1
    w12,w22 = w2
    if v1._index == v2._index:
        # No need to check indices => much faster
        value1 = [ w11*x1 + w12*x2 for (x1,x2) in izip(v1._value,v2._value) ]
        value2 = [ w21*x1 + w22*x2 for (x1,x2) in izip(v1._value,v2._value) ]
        index = v1._index
    else:
        # A marker at the ends: must remove afterwards!
        v1.append(INF_UID,0)
        v2.append(INF_UID,0)

        # Merge vectors by iteration
        it1 = v1.iteritems()
        it2 = v2.iteritems()

        i1,x1 = next(it1)
        i2,x2 = next(it2)
        
        #------------------------------
        while True:
            
            if i1.uid==INF and i2.uid==INF: break

            case = cmp(i1.uid,i2.uid)
            
            if case == 0:
                # i1 == i2
                index.append(i1)
                value1.append(w11*x1+w12*x2)
                value2.append(w21*x1+w22*x2)
                
                i1,x1 = next(it1)
                i2,x2 = next(it2)
                    
            elif case < 0:
                # i1 < i2
                index.append(i1)
                value1.append(w11*x1)
                value2.append(w21*x1)
                i1,x1 = next(it1)

            else:
                # i1 > i2
                index.append(i2)
                value1.append(w12*x2)
                value2.append(w22*x2)
                i2,x2 = next(it2)

        #------------------------------
        v1.pop() 
        v2.pop()

    return (
        Vector( value=value1, index=index )
    ,   Vector( value=value2, index=index )
    )

#--------------------------------------------------------------
def extend_vector(v1,v2):
    """
    Return a Vector containing all the indices in `v1` and `v2`
    
    The values from `v1` are retained, but those from `v2` are 
    not copied; instead those values are zero.

    Parameters
    ----------
    v1, v2 : Vector

    Returns
    -------
    Vector
    
    """
    return merge_weighted_vectors(v1,1.0,v2,0.0)            

    
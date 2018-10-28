import unittest
import copy
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

TOL = 1E-13
DIGITS = 13

from GTC.vector import (
    Vector,
    scale_vector,
    extend_vector,
    merge_weighted_vectors,
    merge_vectors,
    is_ordered
)

from testing_tools import *

#----------------------------------------------------------------------------
# The index of a vector is an object that must have a `uid` attribute
#
class Dummy(object):
    def __init__(self,uid):
        self.uid = (uid, 0)
        
#----------------------------------------------------------------------------
class TestVector(unittest.TestCase):
           
    # Vector objects are a core component. The speed of execution is 
    # critical to GTC. 
    
    # A Vector contains index-object reference pairs, where
    # the index values can be compared for order.
    
    # The implementation does not enforce ordering.
    # There is a utility function `is_ordered` available 
    # to check that Vector elements are correctly ordered
    
    def test_construct(self):
        """
        A Vector can be constructed empty and then updated
        or it can be constructed with named sequences, or it
        can be copy constructed.
        
        """
        v = Vector()
        self.assertEqual(len(v),0)
        
        uids = (1,2,3)
        indices = [ Dummy(i) for i in uids ]
        values = [12,34,45]
        
        v.extend(zip(indices,values))
        
        d = dict(zip(indices,values))
        for i,x in v.iteritems():
            self.assertTrue( equivalent(x,d[i]) )

        v = Vector(index=indices,value=values)
        for i,x in v.iteritems():
            self.assertEqual(x,d[i])

        vc = Vector(copy=v)
        for i,x in v.iteritems():
            self.assertEqual(x,d[i])
        
    def test_append(self):
        """
        Elements can be appended 
        
        """
        v = Vector()
        lst = [( Dummy(1),2),( Dummy(3),2)] 
        v.append( *lst[0] )     # First pair
        self.assertEqual(len(v),1)
        
        for p in v.items():
            self.assertEqual(lst[0][0],p[0])
            self.assertEqual(lst[0][1],p[1])

        v.append( *lst[1] )     # 2nd pair
        self.assertEqual(len(v),2)
        for i,p in enumerate(v.items()):
            self.assertEqual(lst[i][0],p[0])
            self.assertEqual(lst[i][1],p[1])

        self.assertTrue(is_ordered(v))
        
        lst[0] = ( Dummy(5),6)
        self.assertNotEqual(lst[0][0],p[0])
        self.assertNotEqual(lst[0][1],p[1])

    def test_scale(self):
        """
        A new vector can be created by scaling the
        values of another vector.
        
        """
        indices = [ Dummy(i) for i in range(9) ]
        values = list(range(9))
        indices_cp = copy.copy(indices)
        values_cp = copy.copy(values)
        
        v1 = Vector( index=indices,value=list(range(9)) )
        self.assertTrue(is_ordered(v1))
        
        k = 5.
        v2 = scale_vector(v1,k)
        
        # Make sure the arguments do not change 
        self.assertTrue( indices == indices_cp )
        self.assertTrue( values == values_cp )
        
        for i,x in v2.iteritems():
            self.assertEqual(i.uid[0]*k,x)

        self.assertTrue(is_ordered(v2))

        for i1,i2 in izip(v1.iterkeys(),v2.iterkeys()):
            self.assertEqual(i1.uid,i2.uid)

    def test_extend(self):
        """
        Extension is the process of making sure that one Vector has all the indices 
        in another Vector, as well as its own. 
        The initial values of any new indices are zero.
        
        """
        index1 = [ Dummy(i) for i in range(9) ]
        v1 = Vector( index=index1,value=list(range(9)) )
        index2 = [ Dummy(i) for i in range(5,15) ]
        v2 = Vector( index=index2,value=list(range(10)) )
        self.assertTrue(is_ordered(v1))
        self.assertTrue(is_ordered(v2))
        
        v = extend_vector(v1,v2)  
        
        v_range = [ i.uid[0] for i in v.keys() ]
        self.assertEqual(v_range,list(range(15)))
        self.assertTrue(is_ordered(v))
        
        for i,x in v.iteritems():
            # The values from v1 should remain,
            # those from v2 should be zero.
            if i.uid[0] < 9:
                self.assertEqual(x,i.uid[0])
            else:
                self.assertEqual(0,x)

    def test_merge_weighted_vectors(self):
        # Merge two overlapping continuous ranges
        rng1 = list(range(10))
        index1 = [ Dummy(i) for i in rng1 ]
        v1 = Vector( index=index1,value=rng1 )
        self.assertTrue(is_ordered(v1))
        k1 = 3.
        
        idx1_cp = copy.copy(index1)
        v1_cp = copy.copy(rng1)
        
        index2 = [ Dummy(i) for i in range(5,15) ]
        v2 = Vector( index=index2,value=rng1 )   # Note that the values are the same as v1
        self.assertTrue(is_ordered(v2))
        k2 = -4.
 
        idx2_cp = copy.copy(index2)
 
        v = merge_weighted_vectors(v1,k1,v2,k2)
        
        self.assertTrue(is_ordered(v))
        
        self.assertTrue( rng1 == v1_cp )
        self.assertTrue( index1 == idx1_cp )
        self.assertTrue( index2 == idx2_cp )
        
        rng = list(range(15))
        rng_merged = [ i.uid[0] for i in v.keys() ]
        self.assertEqual(rng_merged,rng)
        
        for i,p in enumerate( v.items() ):
            if i < 5:
                self.assertEqual(p[1],k1*rng[i])
            elif i > 4 and i < 10:
                self.assertEqual(p[1],k1*rng[i] + k2*rng[i-5])
            elif i > 9 and i < 15:
                self.assertEqual(p[1],k2*rng[i-5])
            else:
                assert False

    def test_merge_vectors(self):
        # Merge two overlapping continuous ranges
        rng1 = list(range(10))
        index1 = [ Dummy(i) for i in rng1 ]
        v1 = Vector( index=index1,value=rng1 )
        self.assertTrue(is_ordered(v1))
        
        idx1_cp = copy.copy(index1)
        v1_cp = copy.copy(rng1)
        
        index2 = [ Dummy(i) for i in range(5,15) ]
        v2 = Vector( index=index2,value=rng1 )   # Note that the values are the same as v1
        self.assertTrue(is_ordered(v2))
 
        idx2_cp = copy.copy(index2)
 
        v = merge_vectors(v1,v2)
        
        self.assertTrue(is_ordered(v))
        
        self.assertTrue( rng1 == v1_cp )
        self.assertTrue( index1 == idx1_cp )
        self.assertTrue( index2 == idx2_cp )
        
        rng = list(range(15))
        rng_merged = [ i.uid[0] for i in v.keys() ]
        self.assertEqual(rng_merged,rng)
        
        for i,p in enumerate( v.items() ):
            if i < 5:
                self.assertEqual(p[1],rng[i])
            elif i > 4 and i < 10:
                self.assertEqual(p[1],rng[i] + rng[i-5])
            elif i > 9 and i < 15:
                self.assertEqual(p[1],rng[i-5])
            else:
                assert False
                
    def test_merge_elementary(self):
        v1 = Vector( index=[Dummy(1)], value=[1] )
        v2 = Vector( index=[Dummy(2)], value=[1] )
        v = merge_weighted_vectors(v1,2,v2,3)
        self.assertEqual( len(v), 2)
        self.assertTrue(is_ordered(v))


#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
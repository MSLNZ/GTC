import unittest
import sys
import math
import cmath
import numpy
import operator

TOL = 1E-13
DIGITS = 13

from GTC2.GTC.weak_symmetric_matrix import WeakSymmetricMatrix 
from testing_tools import *

class wrapper(object):
    # A class of objects that can be used to create  
    # index pairs for WeakSymmetricMatrix objects. 
    
    __slots__ = ('x','__weakref__',)
    
    def __init__(self,x):
        self.x = x
        
    def __str__(self):
        return "<{}>".format(self.x)
        
#-----------------------------------------------------
class TestSymmetricMatrix(unittest.TestCase):

    # WSM objects use pairs of object references
    # as keys to a real-number value.
    
    # The diagonal value of the matrix is always unity.
    # The default value, when there is no entry in the 
    # matrix is always zero.
    
    # The order of the references in the key 
    # is not important.
    
    # It is possible to delete a single entry 
    # It is possible to remove all entries  
    # associated with one particular reference
    
    # A subset of object references may be used 
    # to extract a matrix (a WSM) containing all 
    # entries associated with those references.

    def test_matrix_read_write_del(self):
        c = WeakSymmetricMatrix()
        
        id1 = wrapper(1)
        id2 = wrapper(2)
        c[id1,id2] = 3
        self.assertEqual(c[id1,id2],3)
        self.assertEqual(c[id2,id1],3)

        id3 = wrapper(3)
        c[id1,id3] = 5
        self.assertEqual(c[id1,id3],5)
        self.assertEqual(c[id3,id1],5)

        del c[id1,id2]
        self.assertEqual(c.get((id1,id2)),0.0)
        self.assertEqual(c[id1,id3],5)
        self.assertEqual(c[id3,id1],5)

    def test_remove(self):
        c = WeakSymmetricMatrix()
        id1 = wrapper(1)
        id2 = wrapper(10)
        j_range = [ wrapper(i) for i in range(10) ]
        i_range= [ wrapper(i) for i in range(2,6)]
        
        for i in j_range: c[id1,i] = 1
        for i in i_range: c[i,id2] = 0.5

        for i in j_range:
            self.assertEqual(c[id1,i],1)
        for i in i_range:
            self.assertEqual(c[i,id2],0.5)

        c.remove(id2)        
        for j in j_range:
            self.assertEqual(c[id1,j],1)
        for i in i_range:
            self.assertEqual(c.get((i,id2)),0.0)

        c.remove(id1)            
        for i in j_range:
            self.assertEqual(c.get((id1,i)),0.0)

        c.clear()
        
        for i in j_range: c[i,id1] = 1
        for i in i_range: c[id2,i] = 0.5

        for i in j_range:
            self.assertEqual(c[i,id1],1)
        for i in i_range:
            self.assertEqual(c[id2,i],0.5)

        c.remove(id2)        
        for i in j_range:
            self.assertEqual(c[i,id1],1)
        for i in i_range:
            self.assertEqual(c.get((id2,i)),0.0)

        c.remove(id1)            
        for i in j_range:
            self.assertEqual(c.get((i,id1)),0.0)
        for i in i_range:
            self.assertEqual(c.get((id2,i)),0.0)

    def testSubmatrix(self):
        c = WeakSymmetricMatrix()

        i_range= [ wrapper(i) for i in range(10)]

        for i in range(10):
            for j in range(i+1,10):
                id_i = i_range[i]
                id_j = i_range[j]
                c[id_i,id_j] = i+j

        sub_c = c.submatrix( i_range[:2] )
        self.assertEqual(len(sub_c),2)
        for i in range(2):
            id_i = i_range[i]
            self.assertEqual(sub_c[id_i,id_i],1)
            for j in range(i+1,2):
                id_j = i_range[j]
                self.assertEqual(sub_c[id_i,id_j],i+j)
                self.assertEqual(sub_c[id_j,id_i],i+j)
        
        sub_c = c.submatrix( i_range[2:5])
        self.assertEqual(len(sub_c),3)
        for i in range(2,5):
            id_i = i_range[i]
            self.assertEqual(sub_c[id_i,id_i],1)
            for j in range(i+1,5):
                id_j = i_range[j]
                self.assertEqual(sub_c[id_i,id_j],i+j)
                self.assertEqual(sub_c[id_j,id_i],i+j)

        idx = (0,3,7)
        sub_c = c.submatrix( [ i_range[i] for i in idx ] )
        self.assertEqual(len(sub_c),3)
        for i,ix in enumerate(idx):
            id_i = i_range[ix]
            self.assertEqual(sub_c[id_i,id_i],1)
            for jx in idx[i+1:]:
                id_j = i_range[jx]
                self.assertEqual(sub_c[id_i,id_j],ix+jx)
                self.assertEqual(sub_c[id_j,id_i],ix+jx)

        for i in range(10):
            id_i = i_range[i]
            if i in idx:
                self.assertEqual(sub_c[id_i,id_i],1)
            else:
                self.assertEqual(sub_c.get((id_i,id_i)),0)               
            for j in range(10):
                if i not in idx:
                    id_j = i_range[j]
                    self.assertEqual(sub_c.get((id_i,id_j)),0)

#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
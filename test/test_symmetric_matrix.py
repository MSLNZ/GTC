import unittest
import sys
import math
import cmath
import itertools

import numpy

from GTC import *
from GTC.symmetric_matrix import SymmetricMatrix

from testing_tools import *

TOL = 1E-13 
                
#-----------------------------------------------------
class TestSymmetricMatrix(unittest.TestCase):

    def test_matrix_read_write_del(self):
        c = SymmetricMatrix()
        
        id1 = 1
        id2 = 2
        c[id1,id2] = 3
        self.assertEqual(c[id1,id2],3)
        self.assertEqual(c[id2,id1],3)

        id3 = 3
        c[id1,id3] = 5
        self.assertEqual(c[id1,id3],5)
        self.assertEqual(c[id3,id1],5)

        del c[id1,id2]
        self.assertEqual(c[id1,id2],0.0)
        self.assertEqual(c[id1,id3],5)
        self.assertEqual(c[id3,id1],5)

    def test_remove(self):
        c = SymmetricMatrix()
        id1 = 1
        id2 = 10
        for i in range(10): c[id1,i] = 1
        for i in range(2,6): c[i,id2] = 0.5

        for i in range(10):
            self.assertEqual(c[id1,i],1)
        for i in range(2,6):
            self.assertEqual(c[i,id2],0.5)

        c.remove(id2)        
        for i in range(10):
            self.assertEqual(c[id1,i],1)
        for i in range(2,6):
            self.assertEqual(c[i,id2],0.0)

        c.remove(id1)            
        for i in range(10):
            self.assertEqual(c[id1,i],0.0)

        c.clear()
        for i in range(10): c[i,id1] = 1
        for i in range(2,6): c[id2,i] = 0.5

        for i in range(10):
            self.assertEqual(c[i,id1],1)
        for i in range(2,6):
            self.assertEqual(c[id2,i],0.5)

        c.remove(id2)        
        for i in range(10):
            self.assertEqual(c[i,id1],1)
        for i in range(2,6):
            self.assertEqual(c[id2,i],0.0)

        c.remove(id1)            
        for i in range(10):
            self.assertEqual(c[i,id1],0.0)
        for i in range(2,6):
            self.assertEqual(c[id2,i],0.0)
       

    def testSubmatrix(self):
        c = SymmetricMatrix()

        for i in range(10):
            for j in range(i+1,10):
                c[i,j] = i+j

        sub_c = c.submatrix( (0,1) )
        for i in range(2):
            self.assertEqual(sub_c[i,i],1)
            for j in range(i+1,2):
                self.assertEqual(sub_c[i,j],i+j)
                self.assertEqual(sub_c[j,i],i+j)
        
        sub_c = c.submatrix( (2,3,4) )
        for i in range(2,5):
            self.assertEqual(sub_c[i,i],1)
            for j in range(i+1,5):
                self.assertEqual(sub_c[i,j],i+j)
                self.assertEqual(sub_c[j,i],i+j)

        idx = (0,3,7)
        sub_c = c.submatrix( idx )
        for i,ix in enumerate(idx):
            self.assertEqual(sub_c[ix,ix],1)
            for jx in idx[i+1:]:
                self.assertEqual(sub_c[ix,jx],ix+jx)
                self.assertEqual(sub_c[jx,ix],ix+jx)

        for i in range(10):
            if i in idx:
                self.assertEqual(sub_c[i,i],1)
            else:
                self.assertEqual(sub_c[i,i],0)               
            for j in range(10):
                if i not in idx:
                    self.assertEqual(sub_c[i,j],0) 
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
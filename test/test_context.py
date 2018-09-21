import unittest
import sys
import math
import cmath
import itertools

from GTC import *
from GTC.context import Context

from testing_tools import *

TOL = 1E-13 

#----------------------------------------------------------------------------
class TestContext(unittest.TestCase):

    def test_construction(self):
        c = Context()
        
        x_value = 1.2
        x_u = 0.5
        x_dof = 6
        x1 = c.elementary_real(x_value,x_u,x_dof,None,independent=True)
        x2 = c.elementary_real(x_value,x_u,x_dof,None,independent=True)

        # uid's must be in order
        self.assert_( x1._node.uid < x2._node.uid )  
        self.assertEqual(x_dof, x1._node.df )
        self.assertEqual(x_dof, x2._node.df )

        c._registered_leaf_nodes[x1._node.uid].df = x_dof
        c._registered_leaf_nodes[x2._node.uid].df = x_dof

        # illegal dof is checked when the object is created
        self.assertRaises(
            RuntimeError,
            c.elementary_real,x_value,x_u,0,None,False
        )

    def test_simple_correlation(self):
        c = Context()
        
        x1 = c.elementary_real(0,1,5,None,independent=True)
        x2 = c.elementary_real(0,1,5,None,independent=True)

        self.assertRaises(
            RuntimeError,
            c.set_correlation,x1,x2,.1
        )

        x1 = c.elementary_real(0,1,5,None,independent=False)
        x2 = c.elementary_real(0,1,5,None,independent=False)
        
        # Self correlation
        self.assertEqual( 1, c.get_correlation(x1,x1) )
        
        r = 0.65
        c.set_correlation(x1,x2,r)
        
        self.assert_( equivalent( r, c.get_correlation(x1,x2) ) )

        # When we delete one node the other remains
        del x1
        self.assertEqual(1, len(c._registered_leaf_nodes) )
        self.assert_( x2._node.uid in c._registered_leaf_nodes )
        
        # The diagonal element remains in place
        self.assertEqual(1, len(c._correlations) )
        self.assert_( x2._node in c._correlations._mat )
        self.assertEqual(1, len(c._correlations._mat[x2._node]) )
        self.assertEqual(1.0, c._correlations._mat[x2._node][x2._node] )
        
        del x2 
        self.assertEqual(0, len(c._registered_leaf_nodes) )
        self.assertEqual(0, len(c._correlations) )
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
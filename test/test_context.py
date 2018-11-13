import unittest

from GTC import ucomplex
from GTC.lib import UncertainReal

TOL = 1E-13 

#----------------------------------------------------------------------------
class TestContext(unittest.TestCase):

    def test_construction(self):        
        # import here to get the tests to work with Python 2.7 on linux
        from GTC.context import _context

        x_value = 1.2
        x_u = 0.5
        x_dof = 6
        x1 = UncertainReal._elementary(x_value,x_u,x_dof,None,independent=True)
        x2 = UncertainReal._elementary(x_value,x_u,x_dof,None,independent=True)

        # uid's must be in order
        self.assertTrue( x1._node.uid < x2._node.uid )
        self.assertEqual(x_dof, x1._node.df )
        self.assertEqual(x_dof, x2._node.df )

        self.assertTrue( _context._registered_leaf_nodes[x1._node.uid].df == x_dof )
        self.assertTrue( _context._registered_leaf_nodes[x2._node.uid].df == x_dof )

        # illegal dof is checked when the object is created
        self.assertRaises(
            ValueError,
            UncertainReal._elementary,x_value,x_u,0,None,False
        )

    def test_invalid_ucomplex_creation(self):
        # RuntimeError: covariance elements not equal: None and 1
        self.assertRaises(TypeError,ucomplex,1 + 0j, (1, None, 1, 1))
        # RuntimeError: covariance elements not equal: 999999j and 1
        self.assertRaises(TypeError,ucomplex,1 + 0j, (1, 999999j, 1, 1))

        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
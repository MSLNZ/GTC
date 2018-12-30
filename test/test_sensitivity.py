import unittest
import os
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

from GTC import *
from GTC.context import Context
from GTC import context
from GTC import persistence
from GTC.vector import is_ordered

from testing_tools import *

TOL = 1E-13

#-----------------------------------------------------
class TestSensitivity(unittest.TestCase):

    """
    reporting.sensitivity is a function that evaluates
    the linear partial derivatives. 

    Here we can use the u_component calculations to
    test sensitivity because they are independent
    when elementary UNs are concerned.
    """

    def test1(self):
        """Reals
        """
        x1 = ureal(2,1)
        x2 = ureal(5,1)
        x3 = ureal(7,1)

        y = x3 ** x1 * x2

        self.assert_(
            equivalent(
                rp.u_component(y,x1),
                rp.sensitivity(y,x1)
        ))
        
        self.assert_(
            equivalent(
                rp.u_component(y,x2),
                rp.sensitivity(y,x2)
        ))

        self.assert_(
            equivalent(
                rp.u_component(y,x3),
                rp.sensitivity(y,x3)
        ))

        x4 = result( y )
        y = sin(x4)
        # Do the differentiation by hand
        self.assert_(
            equivalent(
                math.cos(x4.x),
                rp.sensitivity(y,x4)
        ))


    def test2(self):
        """Complex
        """

        z1 = ucomplex(1+2j,1)
        z2 = ucomplex(2-1.5j,1)
        z3 = ucomplex(.1+1.8j,1)

        z = z1**z2 * z3
        
        self.assert_(
            equivalent_sequence(
                rp.u_component(z,z1),
                rp.sensitivity(z,z1)
        ))

        self.assert_(
            equivalent_sequence(
                rp.u_component(z,z2),
                rp.sensitivity(z,z2)
        ))

        self.assert_(
            equivalent_sequence(
                rp.u_component(z,z3),
                rp.sensitivity(z,z3)
        ))

        z4 = result( z )
        z = sqrt(z4)
        # Do the differentiation by hand
        dz_dz4 = fn.complex_to_seq(
            1.0/(2 * cmath.sqrt(z4.x))
        )

        self.assert_(
            equivalent_sequence(
                dz_dz4,
                rp.sensitivity(z,z4)
        ))


#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
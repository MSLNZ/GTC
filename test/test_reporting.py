import unittest
import sys
import math
import cmath
import itertools

import numpy

from GTC2.GTC import *
from GTC2.GTC.context import context 
from GTC2.GTC.vector import *
from GTC2.GTC.nodes import *
from GTC2.GTC.vector import is_ordered
from GTC2.GTC.reporting import u_component
from GTC2.GTC.lib_real import (UncertainReal,welch_satterthwaite)
from GTC2.GTC import reporting

from testing_tools import *

TOL = 1E-13 
                
#-----------------------------------------------------
class TestBudget(unittest.TestCase):
    def test_empty_budget(self):
        """
        When there are no influence quantities, an empty 
        sequence should be returned. 
        
        """
        seq = rp.budget(3.0)
        self.assertEqual(len(seq),0)
 
        x = constant(6)
        seq = rp.budget(x)
        self.assertEqual(len(seq),0)
 
        x = constant(6+4j)
        seq = rp.budget(x)
        self.assertEqual(len(seq),0)
        x = constant(6)
        seq = rp.budget(x)
        self.assertEqual(len(seq),0)
 
        x = constant(6+4j)
        seq = rp.budget(x)
        self.assertEqual(len(seq),0)

    def test_real(self):
        """
        The budget of a real quantity will consist of a
        list of named tuples with elements for the
        labels and values (the magnitude)
        of the components of uncertainty.

        When a complex quantity has been involved, expect to
        see the components of uncertainty in terms of the
        real and imaginary components.

        A sequuence of influences may be given, which may include
        real or complex uncertain numbers.
        
        """
        x1 = dict(x=1,u=.1,label='x1')
        x2 = dict(x=2,u=.2,label='x2')

        # defined as a complex but it has 0 imaginary        
        z1 = dict(z=3,u=1,label='z1')    
        
        ux1 = ureal(**x1)
        ux2 = ureal(**x2)
        uz1 = ucomplex(**z1)

        y = -ux1 + ux2 * magnitude(-uz1)

        # Trim should remove the zero element b[3]
        b = reporting.budget(y,trim=0.01)
        self.assertEqual( len(b), 3  )

        b = reporting.budget(y,trim=0)
        self.assertEqual( len(b), 4  )
        
        # The default order is in terms of the biggest uncertainty
        self.assertEqual(b[0].label,'z1_re')
        self.assertEqual(b[1].label,'x2')
        self.assertEqual(b[2].label,'x1')
        self.assertEqual(b[3].label,'z1_im')

        self.assert_( equivalent(b[0].u,2.0,TOL) )
        self.assert_( equivalent(b[1].u,3*.2,TOL) )
        self.assert_( equivalent(b[2].u,0.1,TOL) )
        self.assert_( equivalent(b[3].u,0,TOL) )

        # Sorting in different ways
        b = reporting.budget(y,reverse=False,trim=0)
        self.assertEqual(b[0].label,'z1_im')
        self.assertEqual(b[1].label,'x1')
        self.assertEqual(b[2].label,'x2')
        self.assertEqual(b[3].label,'z1_re')
        
        b = reporting.budget(y,key='label',reverse=False,trim=0)
        self.assertEqual(b[0].label,'x1')
        self.assertEqual(b[1].label,'x2')
        self.assertEqual(b[2].label,'z1_im')
        self.assertEqual(b[3].label,'z1_re')

        # With triming
        b = reporting.budget(y,key='label',reverse=False,trim=0.01)
        self.assertEqual(b[0].label,'x1')
        self.assertEqual(b[1].label,'x2')
        self.assertEqual(b[2].label,'z1_re')

        b = reporting.budget(y,[ux1,uz1.real,uz1.imag],trim=0)
        self.assertEqual( len(b), 3  )
        self.assertEqual(b[0].label,'z1_re')
        self.assertEqual(b[1].label,'x1')
        self.assertEqual(b[2].label,'z1_im')

        # A complex quantity may be passed as an
        # influence but the budget reports real
        # and imaginary components
        b = reporting.budget(y,[ux1,uz1],trim=0)
        self.assertEqual( len(b), 3  )
        self.assertEqual(b[0].label,'z1_re')
        self.assertEqual(b[1].label,'x1')
        self.assertEqual(b[2].label,'z1_im')

    def test_complex(self):
        """
        The budget of a complex quantity will consist of a
        list of named tuples with elements for the
        labels and values (the u_bar magnitude)
        of the components of uncertainty.

        Real quantities can be involved.

        Sorting is tested in the real case above.        
        
        """
        z1 = dict(z=1+1j,u=(1,1),label='z1')
        uz1 = ucomplex(**z1)
        z2 = dict(z=2-1j,u=(.5,.5),label='z2')
        uz2 = ucomplex(**z2)
        x1 = dict(x=1,u=.1,label='x1')
        ux1 = ureal(**x1)

        y = -uz1 + uz2* ux1
        
        b = reporting.budget(y)
        self.assertEqual( len(b), 3)

        self.assert_( equivalent(b[0].u,1.0,TOL) )
        self.assert_( equivalent(b[1].u,0.5,TOL) )
        self.assert_( equivalent(b[2].u,math.sqrt((.1**2 + .2**2)/2),TOL) )

        b = reporting.budget(y,[ux1,uz1])
        self.assertEqual( len(b), 2)
        
        self.assert_( equivalent(b[0].u,1.0,TOL) )
        self.assert_( equivalent(b[1].u,math.sqrt((.1**2 + .2**2)/2),TOL) )   


#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
import unittest
import os
import numpy

try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

from GTC import *
from GTC.context import Context
from GTC import context
from GTC import persistence
from GTC.vector import is_ordered
from GTC.linear_algebra import uarray, UncertainArray

from testing_tools import *

TOL = 1E-13

#----------------------------------------------------------------------------
# Note these tests simultaneously check `sensitivity` and `u_component` 
#
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

        self.assertTrue(
            equivalent(
                rp.u_component(y,x1),
                rp.sensitivity(y,x1)
        ))
        
        self.assertTrue(
            equivalent(
                rp.u_component(y,x2),
                rp.sensitivity(y,x2)
        ))

        self.assertTrue(
            equivalent(
                rp.u_component(y,x3),
                rp.sensitivity(y,x3)
        ))

        x4 = result( y )
        y = sin(x4)
        # Do the differentiation by hand
        self.assertTrue(
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
        
        self.assertTrue(
            equivalent_sequence(
                rp.u_component(z,z1),
                rp.sensitivity(z,z1)
        ))

        self.assertTrue(
            equivalent_sequence(
                rp.u_component(z,z2),
                rp.sensitivity(z,z2)
        ))

        self.assertTrue(
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

        self.assertTrue(
            equivalent_sequence(
                dz_dz4,
                rp.sensitivity(z,z4)
        ))

    # def test_uarray_singles_ureal(self):
        # """
        # When uarray has just one element 
        
        # """
        # x1 = uarray(ureal(2,1))
        # x2 = uarray(ureal(5,1))
        # x3 = uarray(ureal(7,1))

        # y = x3 ** x1 * x2
        
        # dy_dx = rp.sensitivity(y,x1)
        # uy_x = rp.u_component(y,x1)       
        # self.assertTrue( equivalent(dy_dx,uy_x) ) 

        # dy_dx = rp.sensitivity(y,x2)
        # uy_x = rp.u_component(y,x2)        
        # self.assertTrue( equivalent(dy_dx,uy_x) ) 

        # dy_dx = rp.sensitivity(y,x3)
        # uy_x = rp.u_component(y,x3)        
        # self.assertTrue( equivalent(dy_dx,uy_x) ) 
        
        # x4 = result( y )
        # y = sin(x4)
        # # Do the differentiation by hand
        # self.assertTrue(
            # equivalent(math.cos(x4.x),rp.sensitivity(y,x4))
        # )
        
    # def test_uarray_singles_ucomplex(self):
        # """
        # When uarray has just one element 
        
        # """
        # x1 = uarray(ucomplex(1+2j,1))
        # x2 = uarray(ucomplex(2-1.5j,1))
        # x3 = uarray(ucomplex(.1+1.8j,1))

        # z = x1**x2 * x3
        
        # dz_dx = rp.sensitivity(z,x1)
        # uz_x = rp.u_component(z,x1)       
        # self.assertTrue( equivalent_sequence(dz_dx,uz_x) )
        
        # dz_dx = rp.sensitivity(z,x2)
        # uz_x = rp.u_component(z,x2)       
        # self.assertTrue( equivalent_sequence(dz_dx,uz_x) )
        
        # dz_dx = rp.sensitivity(z,x3)
        # uz_x = rp.u_component(z,x3)       
        # self.assertTrue( equivalent_sequence(dz_dx,uz_x) )
      
        # x4 = result( z )
        # z = sqrt(x4)
        # # Do the differentiation by hand
        # dz_dx4 = fn.complex_to_seq(
            # 1.0/(2 * cmath.sqrt(x4.x))
        # )

        # self.assertTrue(
            # equivalent_sequence(
                # dz_dx4,
                # rp.sensitivity(z,x4)
        # ))
        
    def test_uarray_ureal(self):
        """
        When uarray has more than one element 
        
        """
        x = uarray( [ureal(2,1), ureal(5,1), ureal(7,1) ] )

        y = sin(x)
        
        dy_dx = rp.sensitivity(y,x)
        uy_x = rp.u_component(y,x)    

        self.assertTrue( isinstance(dy_dx,UncertainArray) )
        self.assertTrue( isinstance(uy_x,UncertainArray) )
        
        for dy_dx_i, uy_x_i in zip( dy_dx, uy_x ):
            self.assertTrue( equivalent(dy_dx_i,uy_x_i) ) 

        # Do the differentiation by hand
        dy_dx_calc = cos( value(x) )
        for dy_dx_i, dy_dx_calc_i in zip( dy_dx, dy_dx_calc ):
            self.assertTrue( equivalent(dy_dx_i,dy_dx_calc_i) ) 
        
    def test_uarray_ucomplex(self):
        """
        When uarray has more than one element 
        
        """
        x = uarray( [ucomplex(1+2j,1),ucomplex(2-1.5j,1),ucomplex(.1+1.8j,1)] )

        z = sin(x)
        
        dz_dx = rp.sensitivity(z,x)
        uz_x = rp.u_component(z,x)       
        for dz_dx_i, uz_x_i in zip( dz_dx, uz_x ):
            self.assertTrue( equivalent_sequence(dz_dx_i,uz_x_i) ) 

        # Do the differentiation by hand
        dx_dx_calc = cos( value(x) )
        for dz_dx_i, uz_x_i in zip( dz_dx, uz_x ):
            self.assertTrue( equivalent_sequence(dz_dx_i,uz_x_i) ) 

    def test_uarray_mixed(self):
        """
        When uarray has more than one element of different types
        
        """
        x = uarray( [ ureal(2,1), ucomplex(2-1.5j,1) ] )
        y = sin(x)
        
        dy_dx = rp.sensitivity(y,x)
        uy_x = rp.u_component(y,x)    

        self.assertTrue( isinstance(dy_dx,UncertainArray) )
        self.assertTrue( isinstance(uy_x,UncertainArray) )
        
        self.assertTrue( equivalent(dy_dx[0],uy_x[0]) ) 
        self.assertTrue( equivalent_sequence(dy_dx[1],uy_x[1]) ) 

        # Do the differentiation by hand
        dy_dx_calc = cos( value(x) ) 
        self.assertTrue( equivalent(dy_dx[0],dy_dx_calc[0]) )
        self.assertTrue( equivalent(dy_dx[0],dy_dx_calc[0]) )
        self.assertTrue( 
            equivalent_sequence(
                dy_dx[1],
                fn.complex_to_seq( dy_dx_calc[1])
            ) 
        )             
        
    def test_intermediate_ureal(self):
        """
        When uarray has more than one element of different types
        
        """
        x1 = uarray( [ureal(2,1), ureal(1.5,1)] )
        x2 = uarray( [ureal(5,1), ureal(.2,1)] )
        x3 = uarray( [ureal(7,1), ureal(3.2,1)] )
        
        x4 = result(x3 ** x1 * x2)        
        y = log10(x4) 
        
        dy_dx = rp.sensitivity(y,x4)
        uy_x = rp.u_component(y,x4)       
        self.assertTrue( equivalent_sequence(dy_dx,uy_x/x4.u) ) 
          
        x4 = result(x3 ** x1 * x2, label='x4')        
        y = log10(x4) 
     
        for i,x4_i in enumerate(x4):
            self.assertEqual('x4[{0}]'.format(i),x4_i.label)
            
        dy_dx = rp.sensitivity(y,x4)
        uy_x = rp.u_component(y,x4)       
        self.assertTrue( equivalent_sequence(dy_dx,uy_x/x4.u) ) 
  
        label = [ "x4[{}]".format(i) for i in range(x4.size) ]
        x4 = result(x3 ** x1 * x2, label=label)        
        y = log10(x4) 
     
        for i,x4_i in enumerate(x4):
            self.assertEqual('x4[{0}]'.format(i),x4_i.label)
 
    def test_intermediate_ucomplex(self):
        """
        When uarray has more than one element of different types
        
        """
        z1 = uarray( [ucomplex(1+2j,1), ucomplex(1.5+2.1j,1)] )
        z2 = uarray( [ucomplex(2-1.5j,1), ucomplex(.5-0.5j,1)] )
        z3 = uarray( [ucomplex(.1+1.8j,1), ucomplex(.4+1.1j,1)] )
        
        z4 = result(z1**z2 * z3)        
        z = log(z4) 
       
        for z_i, z4_i in zip(z,z4):
        
            uz_i_z4_i = rp.u_component(z_i,z4_i)
            dz_i_dz4_i = rp.sensitivity(z_i,z4_i)
            
            self.assertTrue(
                equivalent(dz_i_dz4_i.rr,uz_i_z4_i.rr/z4_i.real.u)
            )
            self.assertTrue(
                equivalent(dz_i_dz4_i.ir,uz_i_z4_i.ir/z4_i.real.u)
            )
            self.assertTrue(
                equivalent(dz_i_dz4_i.ri,uz_i_z4_i.ri/z4_i.imag.u)
            )
            self.assertTrue(
                equivalent(dz_i_dz4_i.ii,uz_i_z4_i.ii/z4_i.imag.u)
            )
        
        z4 = result(z1**z2 * z3, label='z4')        
        z = log(z4) 
     
        for i,z4_i in enumerate(z4):
            self.assertEqual('z4[{0}]'.format(i),z4_i.label)
            self.assertEqual('z4[{0}]_re'.format(i),z4_i.real.label)
            self.assertEqual('z4[{0}]_im'.format(i),z4_i.imag.label)
            
        for z_i, z4_i in zip(z,z4):
        
            uz_i_z4_i = rp.u_component(z_i,z4_i)
            dz_i_dz4_i = rp.sensitivity(z_i,z4_i)
            
            self.assertTrue(
                equivalent(dz_i_dz4_i.rr,uz_i_z4_i.rr/z4_i.real.u)
            )
            self.assertTrue(
                equivalent(dz_i_dz4_i.ir,uz_i_z4_i.ir/z4_i.real.u)
            )
            self.assertTrue(
                equivalent(dz_i_dz4_i.ri,uz_i_z4_i.ri/z4_i.imag.u)
            )
            self.assertTrue(
                equivalent(dz_i_dz4_i.ii,uz_i_z4_i.ii/z4_i.imag.u)
            )

        label = [ "z4[{}]".format(i) for i in range(z4.size) ]
        z4 = result(z1**z2 * z3, label=label)        
        z = log(z4) 
     
        for i,z4_i in enumerate(z4):
            self.assertEqual('z4[{0}]'.format(i),z4_i.label)
            self.assertEqual('z4[{0}]_re'.format(i),z4_i.real.label)
            self.assertEqual('z4[{0}]_im'.format(i),z4_i.imag.label) 
#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
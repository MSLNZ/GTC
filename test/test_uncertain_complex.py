import re
import unittest
try:
    from operator import div  # Python 2
except ImportError:
    from operator import truediv as div

import numpy

from GTC import *
from GTC.context import _context 
from GTC.reporting import *
from GTC.lib import (
    UncertainComplex,
    willink_hall,
    _is_uncertain_complex_constant
)

from testing_tools import *

TOL = 1E-13 
LOG10_E = cmath.log10(cmath.e)

#----------------------------------------------------------------------------
def is_positive_definite(x):
    """
    Test whether 'x' is positive definite, or, if one of the
    diagonal elements is zero, then the off-diagonal elements
    must also be zero.
    
    """
    
    if (x[0,0] == 0.0) or (x[1,1] == 0.0):
        # If one diagonal element is zero, both
        # off-diagonal elements must be zero.
        return (x[1,0] == 0.0) and (x[0,1] == 0.0)
    elif abs(x[1,0]-x[0,1]) > 1E-16:
        # Numpy.linalg.cholesky does not test matrix symmetry!!
        return False
    else:
        try:
            numpy.linalg.cholesky(x)
            return True
        except numpy.linalg.LinAlgError:
            return False

#----------------------------------------------------------------------------
def to_std_uncertainty(x):
    """
    Return a 2x2 std_uncertainty matrix
    and a correlation coefficient 'r' from x (a real number,
    a 2-element numeric sequence, or a 4-element sequence)
    
    """
    d = numpy.size(x)
    if( d == 1 ):
        return ( numpy.asarray( [[x,0.],[0.,x]] ), 0.0 )
    elif( d == 2 ):
        # Assume that these are the diagonal standard uncertainies
        x0,x1 = x
        return ( numpy.asarray( [[x0,0.],[0.,x1]] ), 0.0 )
    elif ( d == 4 ):       
        tmp = numpy.asarray( x, numpy.float )
        tmp.shape = ( 2, 2 )
        assert is_positive_definite( tmp ),"not +ve definite: '%s'"
        
        u1 = math.sqrt(tmp[0,0])
        u2 = math.sqrt(tmp[1,1])
        return ( numpy.asarray( [[u1,0.],[0.,u2]] ), tmp[0,1] /(u1*u2) )
    else:
        raise TypeError('cast failed')

#----------------------------------------------------------------------------
def number_to_matrix(*args):
    """
    Create a (2,2N) numpy.ndarray from a series of complex numbers.
    
    """
    mat = numpy.empty( (2,2*len(args)),float )
    for i,a in enumerate(args):
        c = complex( a )
        mat[:,2*i:2*i+2] = ( (c.real, -c.imag) , (c.imag, c.real) ) 

    return mat

#----------------------------------------------------------------------------
def array_to_sequence(m):
    """
    -> a sequence from a numpy array or matrix
    
    A 2-by-2 matrix [ (1,2), (3,4) ] is returned [ 1,2,3,4 ]
    
    """
    return list(m.flat)

#----------------------------------------------------------------------------
class NamesUncertainComplex(unittest.TestCase):

    def test_is_ucomplex(self):
        x = ureal(1,1)
        y = 1+3j
        z = ucomplex(1,1)
        
        self.assertTrue( not is_ucomplex(x) )
        self.assertTrue( not is_ucomplex(y) )
        self.assertTrue( is_ucomplex(z) )
    
#----------------------------------------------------------------------------
class NamesComplex(unittest.TestCase):
    """
    - labels can be assigned to any elementary UN at construction
    - labels can be assigned to intermediate UNs by a special naming function
    - when a label is assigned to a complex quantity, labels are automatically
      assigned to the real and imaginary components, of the form 'label_re' and 'label_im'.
    - the context maintains a register mapping uids to labels of elementary UNs
    - when a UN is garbage-collected, the register entry is removed
    - labels are read by a special function that will return the uid of an elementary if
      no label has been assigned, or a generic 'anon' otherwise
    """
    def setUp(self):
        self._context = _context

        z1 = ucomplex(1,(1,1),label='z1')
        self.assertEqual(label(z1),'z1')
        self.assertEqual(label(z1.real),'z1_re')
        self.assertEqual(label(z1.imag),'z1_im')
        
        z2 = ucomplex(1,(1,1))
        self.assertEqual(label(z2),None)
        self.assertEqual(label(z2.real),None)
        self.assertEqual(label(z2.imag),None)
        
        del z1
        del z2

    def testIntermediateNaming(self):

        z1 = ucomplex(1,(1,1))
        z2 = ucomplex(1,(1,1))
        y1 = result(z1 + z2)
        self.assertEqual(label(y1),None)
        self.assertEqual(label(y1.real),None)
        self.assertEqual(label(y1.imag),None)
        
        y1 = result(z1 + z2, label='y1')
        self.assertEqual(label(y1),'y1')
        self.assertEqual(label(y1.real),'y1_re')
        self.assertEqual(label(y1.imag),'y1_im')
        
        del z1,z2,y1

#----------------------------------------------------------------------------
class ArithmeticTestsComplex(unittest.TestCase):
    def setUp(self):
        self.x1 = 1+2j
        self.u1 = (.5,0,0,.2)
        self.un1 = ucomplex(self.x1,self.u1)
        
        # Added to extend coverage to _d_components
        self.un1b = ucomplex(self.x1,self.u1,independent=False)
        
        self.x2 = 2+2j
        self.u2 = (1.5,.7)
        self.un2 = ucomplex(self.x2,self.u2)
        
        self.x3 = 3+3j
        self.u3 = (.05,1.2)
        self.un3 = ucomplex(self.x3,self.u3)

        self.x4 = 16
        self.u4 = 4.5
        self.un4 = ureal(self.x4,self.u4)
        
        # Added to extend coverage to _d_components 
        self.un4b = ureal(self.x4,self.u4,independent=False)
 
        # Use these as arguments for intermediate testing
        self.un5 = result( (self.un1 + self.un2) * self.un3 )
        self.u5 = uncertainty(self.un5)

        # Added to extend coverage to _d_components 
        self.un5b = result( (self.un1b + self.un2) * self.un3 )

        self.un6 = result( self.un4 * ureal(-45,4) )
        self.u6 = uncertainty(self.un6)
 
        self.un6b = result( self.un4b * ureal(-45,4) )
        
    def testAddition(self):
        # Cases to consider: uc-uc, uc-numb, numb-uc, uc-ur, ur-uc
        # Need to check u_component in each case.

        # uc-uc
        y = self.un1 + self.un2
        equivalent_complex(value(y),self.x1+self.x2,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un2),
            [self.u2[0],0.,0.,self.u2[1]],
            TOL)
        
        y += self.un3
        equivalent_complex(value(y),self.x1+self.x2+self.x3,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(u_component(y,self.un2),[self.u2[0],0.,0.,self.u2[1]],TOL)
        equivalent_sequence(u_component(y,self.un3),[self.u3[0],0.,0.,self.u3[1]],TOL)

        # Test _d_components
        # uc-uc
        y = self.un1b + self.un2
        equivalent_complex(value(y),self.x1+self.x2,TOL)
        equivalent_sequence(
            u_component(y,self.un1b)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un2),
            [self.u2[0],0.,0.,self.u2[1]],
            TOL)
        
        y += self.un3
        equivalent_complex(value(y),self.x1+self.x2+self.x3,TOL)
        equivalent_sequence(
            u_component(y,self.un1b)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )

        # Intermediate u_component test
        y = self.un2 + self.un5        
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )

        # Add to number on left and 0 on left
        num = 5.5
        y = num + self.un2
        equivalent_complex(value(y),num+self.x2,TOL)
        equivalent_sequence(u_component(y,self.un2),[self.u2[0],0.,0.,self.u2[1]],TOL)

        # Intermediate u_component test
        y = num + self.un5        
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )

        num = 0.0
        y = num + self.un2
        self.assertTrue(y is self.un2)
        equivalent_complex(value(y),num+self.x2,TOL)
        equivalent_sequence(u_component(y,self.un2),[self.u2[0],0.,0.,self.u2[1]],TOL)
        

        # Add to number on right and 0 on right
        num = 2.7
        y = self.un1 + num
        equivalent_complex(value(y),self.x1+num,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        
        # Intermediate u_component test
        y = self.un5 + num     
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )

        num = 0.0
        y = self.un1 + num
        self.assertTrue(y is self.un1)
        equivalent_complex(value(y),self.x1+num,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        
        num = 0+2j
        y = self.un1 + num
        self.assertTrue(y.real is not self.un1)
        self.assertTrue(not y.is_elementary)
        equivalent_complex(value(y),self.x1+num,TOL)
        
        # uc-ur and ur-uc
        y = self.un1 + self.un4
        equivalent_complex(value(y),self.x1+self.x4,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4),
            [self.u4,0.,0.,0.0],
            TOL)

        y = self.un4 + self.un1 
        equivalent_complex(value(y),self.x1+self.x4,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4),
            [self.u4,0.,0.,0.0],
            TOL)
        
        # Intermediate u_component test
        y = self.un6 + self.un5        
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   [self.u6,0,0,0]
        ,   TOL
        )
        y = self.un5 + self.un6         
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   [self.u6,0,0,0]
        ,   TOL
        )

        # Extra tests with _d_components
        # uc-ur and ur-uc
        y = self.un1 + self.un4
        y = self.un4b + self.un1 
        equivalent_complex(value(y),self.x1+self.x4,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4b),
            [self.u4,0.,0.,0.0],
            TOL)
        
        # Intermediate u_component test
        y = self.un6b + self.un5        
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6b)
        ,   [self.u6,0,0,0]
        ,   TOL
        )
        y = self.un5 + self.un6b         
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6b)
        ,   [self.u6,0,0,0]
        ,   TOL
        )        
    def testSubtraction(self):
        # Cases to consider: uc-uc, uc-numb, numb-uc, uc-ur, ur-uc
        # Need to check u_component in each case.

        y = self.un1 - self.un2
        equivalent_complex(value(y),self.x1-self.x2,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(u_component(y,self.un2),[-self.u2[0],0.,0.,-self.u2[1]],TOL)
        
        y -= self.un3
        equivalent_complex(value(y),self.x1-self.x2-self.x3,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(u_component(y,self.un2),[-self.u2[0],0.,0.,-self.u2[1]],TOL)
        equivalent_sequence(u_component(y,self.un3),[-self.u3[0],0.,0.,-self.u3[1]],TOL)

        # Test _d_components 
        y = self.un1b - self.un2
        equivalent_complex(value(y),self.x1-self.x2,TOL)
        equivalent_sequence(
            u_component(y,self.un1b)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(u_component(y,self.un2),[-self.u2[0],0.,0.,-self.u2[1]],TOL)
        
        y -= self.un3
        equivalent_complex(value(y),self.x1-self.x2-self.x3,TOL)
        equivalent_sequence(
            u_component(y,self.un1b)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        
        # Intermediate u_component test
        y = self.un2 - self.un5        
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [-self.u5[0],0,0,-self.u5[1]]
        ,   TOL
        )

        # Sub from number on left
        num = 3.5
        y = num - self.un2
        equivalent_complex(value(y),num-self.x2,TOL)
        equivalent_sequence(
            u_component(y,self.un2)
        ,   [-self.u2[0],0.,0.,-self.u2[1]]
        ,   TOL
        )
        
        # Sub from number number on right
        num = 3.5
        y = self.un1 - num
        equivalent_complex(value(y),self.x1-num,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        
        # Intermediate u_component test
        y = self.un5 - num     
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )

        # Sub from 0 on right
        y = self.un1 - 0.0+0.0j
        self.assertTrue(y is self.un1)
        equivalent_complex(value(y),self.x1,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )

        y = self.un1 - 0.0
        self.assertTrue(y is self.un1)
        equivalent_complex(value(y),self.x1,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )

        num = 0+2j
        y = self.un1 - num
        self.assertTrue(y.real is not self.un1)
        self.assertTrue(not y.is_elementary)
        equivalent_complex(value(y),self.x1-num,TOL)

        # Sub from 0 on left
        y = 0+0j - self.un1
        self.assertTrue(y is not self.un1)
        equivalent_complex(value(y),-self.x1,TOL)

        num = 0+2j
        y = num - self.un1
        self.assertTrue(y.real is not self.un1)
        self.assertTrue(not y.is_elementary)
        equivalent_complex(value(y),num-self.x1,TOL)

        # uc-ur and ur-uc
        y = self.un1 - self.un4
        equivalent_complex(value(y),self.x1-self.x4,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4),
            [-self.u4,0.,0.,0.0],
            TOL)

        y = self.un4 - self.un1 
        equivalent_complex(value(y),self.x4-self.x1,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ -math.sqrt(self.u1[0]),0.,0.,-math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4),
            [self.u4,0.,0.,0.0],
            TOL)

        # Extra tests with _d_component 
        # uc-ur and ur-uc
        y = self.un1 - self.un4b
        equivalent_complex(value(y),self.x1-self.x4,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ math.sqrt(self.u1[0]),0.,0.,math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4b),
            [-self.u4,0.,0.,0.0],
            TOL)

        y = self.un4b - self.un1 
        equivalent_complex(value(y),self.x4-self.x1,TOL)
        equivalent_sequence(
            u_component(y,self.un1)
        ,   [ -math.sqrt(self.u1[0]),0.,0.,-math.sqrt(self.u1[3])]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un4b),
            [self.u4,0.,0.,0.0],
            TOL)   
            
        # Intermediate u_component test
        y = self.un6 - self.un5        
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [-self.u5[0],0,0,-self.u5[1]]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   [self.u6,0,0,0]
        ,   TOL
        )
        y = self.un5 - self.un6         
        equivalent_sequence(
            u_component(y,self.un5)
        ,   [self.u5[0],0,0,self.u5[1]]
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   [-self.u6,0,0,0]
        ,   TOL
        )

    def testNegation(self):
        x = self.un1
        w = self.un2
        z = result(x + w)
        y = -(z)
        equivalent_complex(value(y),-value(value(self.un1)+value(self.un2)),TOL)
        u = uncertainty(x)
        equivalent_sequence(
            u_component(y,x),
            [-u[0],0,0,-u[1]],
            TOL)
        u = uncertainty(w)
        equivalent_sequence(
            u_component(y,w),
            [-u[0],0,0,-u[1]],
            TOL)

        # intermediate
        u = uncertainty(z)
        equivalent_sequence(
            u_component(y,z),
            [-u[0],0,0,-u[1]],
            TOL)

    def testMultiplication(self):
        # Cases to consider: uc-uc, uc-numb, numb-uc, uc-ur, ur-uc
        # Need to check u_component in each case.

        y = self.un1 * self.un2
        equivalent_complex(value(y),self.x1*self.x2,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x2), u1)),
            TOL)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un2),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1), u2)),
            TOL)
        
        y *= self.un3
        equivalent_complex(value(y),self.x1*self.x2*self.x3,TOL)
        u3,r = to_std_uncertainty(self.u3)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x2*self.x3), u1)),
            TOL)
        equivalent_sequence(
            u_component(y,self.un2),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1*self.x3), u2)),
            TOL)
        equivalent_sequence(
            u_component(y,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x2*self.x1), u3)),
            TOL)

        # Test _d_components
        y = self.un1b * self.un2
        equivalent_complex(value(y),self.x1*self.x2,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1b),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x2), u1)),
            TOL)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un2),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1), u2)),
            TOL)
        
        y *= self.un3
        equivalent_complex(value(y),self.x1*self.x2*self.x3,TOL)
        u3,r = to_std_uncertainty(self.u3)
        equivalent_sequence(
            u_component(y,self.un1b),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x2*self.x3), u1)),
            TOL)
        equivalent_sequence(
            u_component(y,self.un2),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1*self.x3), u2)),
            TOL)
        equivalent_sequence(
            u_component(y,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x2*self.x1), u3)),
            TOL)
            
        # Intermediate u_component test
        y = self.un2 * self.un5
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix(self.x2), u5))
        ,   TOL
        )

        # Mul to number on left, mul to 1 on left  
        num = -3.2-5j
        y = num * self.un2
        equivalent_complex(value(y),num*self.x2,TOL)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un2),
            array_to_sequence( numpy.matmul(number_to_matrix(num), u2)),
            TOL)

        # Intermediate u_component test
        y = num * self.un5
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix(num), u5))
        ,   TOL
        )

        num = 1.0
        y = num * self.un2
        self.assertTrue(y is self.un2)
        equivalent_complex(value(y),num*self.x2,TOL)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un2),
            array_to_sequence( numpy.matmul(number_to_matrix(num), u2)),
            TOL)

        # Mul to number on right, mul and 1 on right
        num = -3.2+8j
        y = self.un1 * num
        equivalent_complex(value(y),self.x1*num,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(num), u1)),
            TOL)
        
        # Intermediate u_component test
        y = self.un5 * num
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix(num), u5))
        ,   TOL
        )

        num = 1.0
        y = self.un1 * num
        self.assertTrue(y is self.un1)
        equivalent_complex(value(y),self.x1*num,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(num), u1)),
            TOL)

        # uc-ur and ur-uc
        y = self.un1 *self.un4
        equivalent_complex(value(y),self.x1*self.x4,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x4), u1))
        ,   TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1), u4)),
            TOL)

        y = self.un4 * self.un1 
        equivalent_complex(value(y),self.x4*self.x1,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x4), u1)),
            TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1), u4)),
            TOL
        )
        
        # Additional test with _d_components
        # uc-ur and ur-uc
        y = self.un1 *self.un4b
        equivalent_complex(value(y),self.x1*self.x4,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x4), u1))
        ,   TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4b),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1), u4)),
            TOL)

        y = self.un4b * self.un1 
        equivalent_complex(value(y),self.x4*self.x1,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x4), u1)),
            TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4b),
            array_to_sequence( numpy.matmul(number_to_matrix(self.x1), u4)),
            TOL
        )
        
        # Intermediate u_component test
        y = self.un6 * self.un5
        u6 = self.u6
        u5,r = to_std_uncertainty(self.u5)    
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(self.un6) ), u5))
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(self.un5) ), numpy.asarray( [[u6,0],[0,0]] )))
        ,   TOL
        )
        y = self.un5 * self.un6         
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(self.un6) ), u5))
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(self.un5) ), numpy.asarray( [[u6,0],[0,0]] )))
        ,   TOL
        )

    def testDivision(self):
        # Cases to consider: uc-uc, uc-numb, numb-uc, uc-ur, ur-uc
        # Need to check u_component in each case.

        y = self.un1 / self.un2
        equivalent_complex(value(y),self.x1/self.x2,TOL)
        u1,r = to_std_uncertainty(self.u1)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un1) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x1 ), u1 ))
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un2) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x2 ), u2 ))
        ,   TOL
        )

        y /= self.un3        
        u3,r = to_std_uncertainty(self.u3)
        equivalent_complex(value(y),self.x1/self.x2/self.x3,TOL)
        equivalent_sequence(
            u_component(y,self.un3) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x3 ), u3 ))
        ,   TOL
        )

        self.assertRaises(ZeroDivisionError,div,y,0)
 
        # Test for _d_component 
        y = self.un1b / self.un2
        equivalent_complex(value(y),self.x1/self.x2,TOL)
        u1,r = to_std_uncertainty(self.u1)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un1b) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x1 ), u1 ))
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un2) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x2 ), u2 ))
        ,   TOL
        )

        y /= self.un3        
        u3,r = to_std_uncertainty(self.u3)
        equivalent_complex(value(y),self.x1/self.x2/self.x3,TOL)
        equivalent_sequence(
            u_component(y,self.un3) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x3 ), u3 ))
        ,   TOL
        )
        
        # Intermediate u_component test
        y = self.un2 / self.un5
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y)/value(self.un5) ), u5))
        ,   TOL
        )

        y = self.un5 / self.un2
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(y)/value(self.un5) ), u5))
        ,   TOL
        )

        # Div from number on left
        num = 3 - 5j
        y = num / self.un2
        equivalent_complex(value(y),num/self.x2,TOL)
        u2,r = to_std_uncertainty(self.u2)
        equivalent_sequence(
            u_component(y,self.un2) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x2 ), u2 ))
        ,   TOL
        )

        y = num / self.un5
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix(-value(y)/value(self.un5)), u5))
        ,   TOL
        )

        # Div from number on right and 1.0 on right
        num = 2.1 + 7.3j
        y = self.un1 / num
        equivalent_complex(value(y),self.x1/num,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x1 ), u1 ))
        ,   TOL
        )

        y = self.un1 / 1.0
        self.assertTrue(y is self.un1)
        equivalent_complex(value(y),self.x1,TOL)
        equivalent_sequence(
            u_component(y,self.un1) 
        ,   array_to_sequence( numpy.matmul(number_to_matrix( 1.0 ), u1 ))
        ,   TOL
        )

        y = self.un5 / num
        u5,r = to_std_uncertainty(self.u5)
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix(value(y)/value(self.un5)), u5))
        ,   TOL
        )

        # uc-ur and ur-uc
        y = self.un1 / self.un4
        equivalent_complex(value(y),self.x1/self.x4,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x1 ), u1))
        ,   TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix(-value(y) / self.x4), u4)),
            TOL)

        y = self.un4 / self.un1 
        equivalent_complex(value(y),self.x4/self.x1,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x1 ), u1)),
            TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x4), u4)),
            TOL
        )

        # Additional tests with _d_components 
        # uc-ur and ur-uc
        y = self.un1 / self.un4b
        equivalent_complex(value(y),self.x1/self.x4,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x1 ), u1))
        ,   TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4b),
            array_to_sequence( numpy.matmul(number_to_matrix(-value(y) / self.x4), u4)),
            TOL)

        y = self.un4b / self.un1 
        equivalent_complex(value(y),self.x4/self.x1,TOL)
        u1,r = to_std_uncertainty(self.u1)
        equivalent_sequence(
            u_component(y,self.un1),
            array_to_sequence( numpy.matmul(number_to_matrix( -value(y) / self.x1 ), u1)),
            TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4b),
            array_to_sequence( numpy.matmul(number_to_matrix( value(y) / self.x4), u4)),
            TOL
        )
        # Intermediate u_component test
        y = self.un6 / self.un5
        u6 = self.u6
        u5,r = to_std_uncertainty(self.u5)    
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y)/value(self.un5) ), u5))
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(y)/value(self.un6) ), numpy.asarray( [[u6,0],[0,0]])))
        ,   TOL
        )
        y = self.un5 / self.un6         
        equivalent_sequence(
            u_component(y,self.un5)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( value(y)/value(self.un5) ), u5))
        ,   TOL
        )
        equivalent_sequence(
            u_component(y,self.un6)
        ,   array_to_sequence( numpy.matmul(number_to_matrix( -value(y)/value(self.un6) ), numpy.asarray( [[u6,0],[0,0]])))
        ,   TOL
        )

        # Zero-valued numerator
        # Cases to consider: uc-uc, uc-numb, numb-uc, uc-ur, ur-uc
        u = 0.1
        n = ucomplex(0,u)
        d = ucomplex(1,1)
        q = n/d
        dq_dn = 1.0/value(d)
        dq_dd = 0.0
        self.assertTrue(
            equivalent_complex(q.x,0.0)
        )
        equivalent_sequence(
            u_component(q,n)
        ,   [dq_dn*u,-dq_dd*u,dq_dd*u,dq_dn*u]
        ,   TOL
        )

        d = 1
        q = n/d
        dq_dn = 1.0/value(d)
        dq_dd = 0.0
        self.assertTrue(
            equivalent_complex(q.x,0.0)
        )
        equivalent_sequence(
            u_component(q,n)
        ,   [dq_dn*u,-dq_dd*u,dq_dd*u,dq_dn*u]
        ,   TOL
        )

        u = 0.0
        n = 0
        d = ucomplex(1,1)
        q = n/d
        dq_dn = 1.0/value(d)
        dq_dd = 0.0
        self.assertTrue(
            equivalent_complex(q.x,0.0)
        )
        equivalent_sequence(
            u_component(q,n)
        ,   [dq_dn*u,-dq_dd*u,dq_dd*u,dq_dn*u]
        ,   TOL
        )

        z = ucomplex(0,1)
        x = ureal(1,1)
        q = z/x
        dq_dz = 1.0/value(x)
        dq_dx = -value(q)/value(x)
        self.assertTrue( equivalent_complex(value(q),0.0) )
        self.assertTrue(
            equivalent_sequence(u_component(q,z), [dq_dz.real,-dq_dz.imag,dq_dz.imag,dq_dz.real] ) 
        )
 
        z = ucomplex(1,1)
        x = ureal(0,1)
        q = x/z
        dq_dx = 1.0/value(z)
        self.assertTrue( equivalent_complex(value(q),0.0) )
        self.assertTrue(
            equivalent_sequence(u_component(q,x), [dq_dx.real,-dq_dx.imag,0,0] ) 
        )

        # c-uc
        y = self.x1 / self.un4
        equivalent_complex(value(y),self.x1/self.x4,TOL)
        equivalent_sequence(
            u_component(y,self.x1),
            (0.,0.,0.,0.)
        ,   TOL
        )
        u4 = numpy.asarray( [ [self.u4,0], [0,0] ])
        equivalent_sequence(
            u_component(y,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix(-value(y) / self.x4), u4)),
            TOL)

#-----------------------------------------------------
class GuideExampleH2Complex(unittest.TestCase):

    def setUp(self):

        self.v = ucomplex(4.999,(0.0032,0.0),independent=False)
        self.i = ucomplex(0.019661,(0.0000095,0.0),independent=False)
        self.phi = ucomplex(1.04446,(0.00075,0.0),independent=False)

        set_correlation(-0.36,self.v.real,self.i.real)
        set_correlation(0.86,self.v.real,self.phi.real)
        set_correlation(-0.65,self.i.real,self.phi.real)

    def test(self):
        _J_ = 1j
        TOL = 1E-10
        z = self.v * exp( _J_ * self.phi )/ self.i
        
        equivalent( uncertainty(z)[0],0.0699787279884,TOL)
        equivalent( uncertainty(z)[1],0.295716826846,TOL)
        equivalent( get_correlation(z),-0.591484610819,TOL)

        # The willink_hall function provides an alternative
        # calculation of the covariance, so just check that
        # it works.
        cv = willink_hall(z)[0]
        
        u_r = math.sqrt(cv[0])
        u_i = math.sqrt(cv[3])
        r = cv[1]/(u_r*u_i)

        equivalent( u_r,0.0699787279884,TOL)
        equivalent( u_i,0.295716826846,TOL)
        equivalent( r,-0.591484610819,TOL)

#-----------------------------------------------------
class ComplexFunctionTest(unittest.TestCase):

    def setUp(self):
        x1 = 1+2j
        u1 = (.5,0,0,.2)
        
        x2 = 2+2j
        u2 = (1.5,.7)
        
        # An intermediate un-complex for u_component testing
        self.un3 = result( ucomplex(x1,u1) * ucomplex(x2,u2) )

        # An intermediate un-real for u_component testing
        self.un4 = result( ureal(3.2,.4) / ureal(2.2,0.8) )
        
    def test(self):
        # self.python_atrigh_change()
        self.trig()
        self.trig_h()
        self.a_trig()
        self.a_trig_h()
        self.imag_stuff()
        self.misc()
        self.power()
       
    def trig(self):
        v =  -1 + 2j
        mat = numpy.asarray([[0.87,0],[0,0.95] ])
        x = ucomplex(v,( (mat[0,0], mat[1,1]) ) )
        
        s = sin(x)
        c = cos(x)
        t = tan(x)

        # sin -----------------------------------------        
        equivalent_complex(value(s),cmath.sin(v),TOL)
        equivalent_sequence(
            u_component(x,x),
            array_to_sequence( mat ),
            TOL)
        equivalent_sequence(
            u_component(s,x),
            array_to_sequence( numpy.matmul(number_to_matrix( cmath.cos(v) ), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(s) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = sin(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.cos(v3)), mat3)),
            TOL)

         # cos -----------------------------------------        
        equivalent_complex(value(c),cmath.cos(v),TOL)
        equivalent_sequence(
            u_component(c,x),
            array_to_sequence( numpy.matmul(number_to_matrix(-cmath.sin(v)), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(c) ) )
        
        # intermediate
        v3 = value(self.un3)
        s3 = cos(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(-cmath.sin(v3)), mat3)),
            TOL)

         # tan -----------------------------------------        
        equivalent_complex(value(t),cmath.tan(v),TOL)
        equivalent_sequence(
            u_component(t,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/( cmath.cos(v))**2), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(c) ) )
        
        # intermediate
        v3 = value(self.un3)
        s3 = tan(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/(cmath.cos(v3))**2), mat3)),
            TOL)
 
    def trig_h(self):
        v = -2 + 0.6j
        mat,r = to_std_uncertainty([1.87,0.,0.,0.55])
        x = ucomplex(v, (mat[0,0],mat[1,1]) )
        
        s = sinh(x)
        c = cosh(x)
        t = tanh(x)
        
        # sinh -----------------------------------------------
        equivalent_complex(value(s),cmath.sinh(v),TOL)
        equivalent_sequence(
            u_component(s,x),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.cosh(v)), mat)),
            TOL)
        self.assertTrue( math.isinf( dof(s) ) )
        
        # intermediate
        v3 = value(self.un3)
        s3 = sinh(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.cosh(v3)), mat3)),
            TOL)
        
        
        # cosh -----------------------------------------------      
        equivalent_complex(value(c),cmath.cosh(v),TOL)
        equivalent_sequence(
            u_component(c,x),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.sinh(v)), mat)),
            TOL)
        self.assertTrue(math.isinf( dof(c) ))

        # intermediate
        v3 = value(self.un3)
        s3 = cosh(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.sinh(v3)), mat3)),
            TOL)
        
        # tanh -----------------------------------------------      
        equivalent_complex(value(t),cmath.tanh(v),TOL)
        equivalent_sequence(
            u_component(t,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/(cmath.cosh(v))**2), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(t) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = tanh(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/(cmath.cosh(v3))**2), mat3)),
            TOL)

    def arcos(self,z):
        return -1j * cmath.log(z + 1j*cmath.sqrt(1-z*z))

    def arsin(self,z):
        return -1j * cmath.log(1j*z + cmath.sqrt(1 - z*z))
    
    def artan(self,z):
        return 1j * cmath.log((1j+z)/(1j-z)) / 2

    def a_trig(self):
        v = 0.32 + 1.1j
        mat = numpy.asarray([ [1.17,0],[0,0.35]])
        x = ucomplex(v,(mat[0,0], mat[1,1]) )
        
        s = asin(x)
        c = acos(x)
        t = atan(x)
        
        # asin -----------------------------------------------      
        equivalent_complex(value(s),self.arsin(v),TOL)
        den = 1 - v**2
        equivalent_sequence(
            u_component(s,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1/cmath.sqrt(den)), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(s) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = asin(self.un3)
        den = 1 - v3**2
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1/cmath.sqrt(den)), mat3)),
            TOL)

        # acos -----------------------------------------------      
        equivalent_complex(value(c),self.arcos(v),TOL)
        den = 1 - v**2
        equivalent_sequence(
            u_component(c,x),
            array_to_sequence( numpy.matmul(number_to_matrix(-1/cmath.sqrt(den)), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(c) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = acos(self.un3)
        den = 1 - v3**2
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(-1/cmath.sqrt(den)), mat3)),
            TOL)

        # atan -----------------------------------------------      
        equivalent_complex(value(t),self.artan(v),TOL)
        den = 1 + v**2
        equivalent_sequence(
            u_component(t,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/den), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(t) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = atan(self.un3)
        den = 1 + v3**2
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1/den), mat3)),
            TOL)

    def arcosh(self,z):
        return cmath.log(z + cmath.sqrt(z-1)*cmath.sqrt(z+1))

    def arsinh(self,z):
        return cmath.log(z + cmath.sqrt(z*z + 1.0))
    
    def artanh(self,z):
        return cmath.log((1+z)/(1-z)) / 2

    def python_atrigh_change(self):
        # The cmath implementation of inverse hyperbolic trig functions
        # may evolve in future Python releases, so keep an eye on it here.
        z = -2 + 3j
        assert (self.arsinh(z) != cmath.asinh(z)), "Library function has changed"
        for z in (2+3j,-2+3j,2-3j,-2-3j):
            equivalent_complex(self.arcosh(z),cmath.acosh(z),TOL)
            equivalent_complex(self.artanh(z),cmath.atanh(z),TOL)
                
    def a_trig_h(self):
        v = -2 + 3j
        mat = numpy.asarray( [[1.17,0],[0,0.35]] )
        x = ucomplex(v, ((mat[0,0], mat[1,1]) ) )
        
        s = asinh(x)
        c = acosh(x)
        t = atanh(x)
        
        # asinh -----------------------------------------------      
        equivalent_complex(value(s),self.arsinh(v),TOL)
        den = 1 + v**2
        equivalent_sequence(
            u_component(s,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/cmath.sqrt(den)), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(s) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = asinh(self.un3)
        den = 1 + v3**2
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1/cmath.sqrt(den)), mat3)),
            TOL)

        # acosh -----------------------------------------------      
        equivalent_complex(value(c),cmath.acosh(v),TOL)
        den = cmath.sqrt((v - 1) * (v + 1))
        equivalent_sequence(
            u_component(c,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/den), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(c) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = acosh(self.un3)
        den = cmath.sqrt((v3 - 1) * (v3 + 1))
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1/(den)), mat3)),
            TOL)

        # atanh -----------------------------------------------      
        equivalent_complex(value(t),cmath.atanh(v),TOL)
        den = 1 - v**2
        equivalent_sequence(
            u_component(t,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/den), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(t) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = atanh(self.un3)
        den = 1 - v3**2
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1/den), mat3)),
            TOL)

        
    def imag_stuff(self):
        v = -2+8j
        mat,r = to_std_uncertainty([10.17,0,0,11.35])
        x = ucomplex(v, (mat[0,0], mat[1,1]) )

        c = x.conjugate()
        re = x.real
        im = x.imag
        
        # conjugate -----------------------------------------------      
        equivalent_complex(value(c),v.conjugate(), TOL)
        equivalent_sequence(
            u_component(c,x),
            array_to_sequence( numpy.matmul(numpy.asarray([[1,0],[0,-1]]), mat) ),
            TOL)
        self.assertTrue( math.isinf( dof(c) ) )
        
        # intermediate
        s3 = self.un3.conjugate()
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(numpy.asarray([[1,0],[0,-1]]), mat3) ),
            TOL)

        # real -----------------------------------------------      
        equivalent_complex(value(re),complex(v.real),TOL)
        equivalent_sequence(
            u_component(re,x),
            array_to_sequence( numpy.matmul(numpy.asarray([[1,0],[0,0]]), mat) ),
            TOL)
        self.assertTrue( math.isinf( dof(re) ) )

        # intermediate
        s3 = self.un3.real
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(numpy.asarray([[1,0],[0,0]]), mat3) ),
            TOL)

        # imag -----------------------------------------------      
        equivalent_complex(value(im),complex(v.imag),TOL)
        equivalent_sequence(
            u_component(im,x),
            array_to_sequence( numpy.matmul(numpy.asarray([[0,1],[0,0]]), mat) ),
            TOL)
        self.assertTrue( math.isinf( dof(im) ) )

        # intermediate
        s3 = self.un3.imag
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(numpy.asarray([[0,1],[0,0]]), mat3) ),
            TOL)
        
    def misc(self):
        v = 1.1-0.8j
        mat,r = to_std_uncertainty([4.1,0,0,1.7])
        x = ucomplex(v, (mat[0,0], mat[1,1]) )

        l10 = log10(x)
        p = phase(x)
        
        e = exp(x)
        l = log(x)
        s = sqrt(x)
        n = mag_squared(x)
        a = magnitude(x)
        
        z = ucomplex(0, (mat[0,0], mat[1,1]) )
        self.assertRaises(ZeroDivisionError,magnitude,z)

        # log10 -----------------------------------------------------------        
        equivalent_complex(value(l10),cmath.log10(v),TOL)
        equivalent_sequence(
            u_component(l10,x)
        ,   array_to_sequence( numpy.matmul(number_to_matrix(LOG10_E/v), mat ))
        ,   TOL
        )
        self.assertTrue( math.isinf(dof(l10)) )
        
        # exp --------------------------------------------------------
        equivalent_complex(value(e),cmath.exp(v),TOL)
        equivalent_sequence(
            u_component(e,x),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.exp(v)), mat)),
            TOL)
        self.assertTrue( math.isinf(dof(e)) )
        
        # intermediate
        v3 = value(self.un3)
        s3 = exp(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(cmath.exp(v3)), mat3)),
            TOL)

        # log -----------------------------------------------------------        
        equivalent_complex(value(l),cmath.log(v),TOL)
        equivalent_sequence(
            u_component(l,x),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/v), mat )),
            TOL)
        self.assertTrue( math.isinf(dof(l)) )

        # intermediate
        v3 = value(self.un3)
        s3 = log(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(1.0/v3), mat3)),
            TOL)

        # sqrt ------------------------------------------------------------
        equivalent_complex(value(s),cmath.sqrt(v),TOL)
        equivalent_sequence(
            u_component(s,x),
            array_to_sequence( numpy.matmul(number_to_matrix(0.5/cmath.sqrt(v)), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(s) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = sqrt(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(0.5/cmath.sqrt(v3)), mat3)),
            TOL)
           
        # mag_squared ------------------------------------------------------------
        equivalent_complex(value(n),complex(abs(v)**2),TOL)
        equivalent_sequence(
            u_component(n,x),
            array_to_sequence( numpy.matmul(numpy.asarray([ [2*v.real,2*v.imag],[0,0] ]), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(n) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = mag_squared(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(numpy.asarray([ [2*v3.real,2*v3.imag],[0,0] ]), mat3)),
            TOL)

        # when argument is ureal, it returns self*self
        v4 = 2
        u4 = 0.5
        z4 = ureal(v4,u4)
        z4_mag = mag_squared(z4)
        self.assertTrue(
            equivalent(value(z4_mag),v4**2)
        )
        self.assertTrue(
            equivalent(uncertainty(z4_mag),2*v4*u4)
        )
        
        # magnitude ------------------------------------------------------------
        equivalent_complex(value(a),complex( abs(v) ),TOL)
        equivalent_sequence(
            u_component(a,x),
            array_to_sequence( numpy.matmul(numpy.asarray([ [v.real/abs(v),v.imag/abs(v)],[0,0] ]), mat )),
            TOL)
        self.assertTrue( math.isinf( dof(a) ) )

        # intermediate
        v3 = value(self.un3)
        s3 = magnitude(self.un3)
        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        equivalent_sequence(
            u_component(s3,self.un3),
            array_to_sequence( numpy.matmul(numpy.asarray([ [v3.real/abs(v3),v3.imag/abs(v3)],[0,0] ]), mat3)),
            TOL)

    def power(self):
        # Cases to consider: uc-uc, uc-numb, numb-uc, uc-ur, ur-uc
        # Need to check u_component in each case.
        
        TOL = 1E-14

        # uc-uc ---------------------------------------------------        
        vx = 2.1-1.8j
        vy = 1.1-0.5j
        matx,r = to_std_uncertainty([4.1,0,0,1.7])
        maty,r = to_std_uncertainty([0.1,0,0,1.1])
        x = ucomplex(vx, (matx[0,0], matx[1,1]) )
        y = ucomplex(vy, (maty[0,0], maty[1,1]) )

        p = x**y
        vp = vx**vy
        equivalent_complex(value(p),vp,TOL)
        
        dp_dx = vp*vy/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), matx )),
            TOL)
        dp_dy = cmath.log(vx)*vp
        equivalent_sequence(
            u_component(p,y),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dy), maty )),
            TOL)

        # intermediate
        p = x**self.un3
        v3 = value(self.un3)
        vp = vx**v3

        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        dp_dv3 = cmath.log(vx)* vp        
        equivalent_sequence(
            u_component(p,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv3), mat3 )),
            TOL)

        p = self.un3**y
        vp = v3**vy
        dp_dv3 = vp*vy/v3
        equivalent_sequence(
            u_component(p,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv3), mat3 )),
            TOL)

        # numb-uc ---------------------------------------------------        
        # x ** n, where x is a number
        vz = 2.5
        p = vz ** y
        vp = vz ** vy
        equivalent_complex(value(p),vp,TOL)
        dp_dy = cmath.log(vz)*vp
        equivalent_sequence(
            u_component(p,y),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dy), maty )),
            TOL)

        # intermediate
        p = vz**self.un3
        v3 = value(self.un3)
        vp = vz**v3

        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        dp_dv3 = cmath.log(vz) * vp
        
        equivalent_sequence(
            u_component(p,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv3), mat3 )),
            TOL)
        
        # uc-numb ---------------------------------------------------        
        # x ** n, where n is a number        
        vz = 2.5
        p = x ** vz 
        vp = vx ** vz
        equivalent_complex(value(p),vp,TOL)
        dp_dx = vp*vz/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), matx )),
            TOL)

        # intermediate
        p = self.un3**vz
        v3 = value(self.un3)
        vp = v3**vz

        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        dp_dv3 = vp*vz/v3
        
        equivalent_sequence(
            u_component(p,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv3), mat3 )),
            TOL)

        # uc-ur ------------------------------------------------
        vx = 2.1-1.8j
        vy = 1.1
        matx,r = to_std_uncertainty([4.1,0,0,1.7])
        uy = 0.12
        x = ucomplex(vx, (matx[0,0], matx[1,1]) )
        y = ureal(vy, uy )

        p = x**y
        vp = vx**vy
        equivalent_complex(value(p),vp,TOL)
        
        dp_dx = vp*vy/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), matx )),
            TOL)
        dp_dy = cmath.log(vx)*vp
        equivalent_sequence(
            u_component(p,y),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dy), numpy.asarray( [[uy,0],[0,0]] ) )),
            TOL)

        # intermediate
        p = self.un3**self.un4
        v3 = value(self.un3)
        v4 = value(self.un4)
        u4 = uncertainty(self.un4)
        vp = v3**v4

        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        mat4 = numpy.asarray( [[u4,0],[0,0]] )

        dp_dv4 = cmath.log(v3)* vp        
        equivalent_sequence(
            u_component(p,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv4), mat4 )),
            TOL)
        dp_dv3 = vp*v4/v3
        equivalent_sequence(
            u_component(p,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv3), mat3 )),
            TOL)
        
        # ur-uc ------------------------------------------------
        vx = -1.8
        vy = 1.1-0.5j
        ux = 4.1
        maty,r = to_std_uncertainty([0.1,0,0,1.1])
        x = ureal(vx,ux )
        y = ucomplex(vy, (maty[0,0], maty[1,1]) )

        p = x**y
        vp = vx**vy
        equivalent_complex(value(p),vp,TOL)
        
        dp_dx = vp*vy/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), numpy.asarray( [[ux,0],[0,0]] ) )),
            TOL)
        dp_dy = cmath.log(vx)*vp
        equivalent_sequence(
            u_component(p,y),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dy), maty )),
            TOL)

        # intermediate
        p = self.un4**self.un3
        v3 = value(self.un3)
        v4 = value(self.un4)
        u4 = uncertainty(self.un4)
        vp = v4**v3

        mat3,r = to_std_uncertainty( uncertainty(self.un3) )
        mat4 = numpy.asarray( [[u4,0],[0,0]] )

        dp_dv3 = cmath.log(v4)* vp        
        equivalent_sequence(
            u_component(p,self.un3),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv3), mat3 )),
            TOL)
        dp_dv4 = vp*v3/v4
        equivalent_sequence(
            u_component(p,self.un4),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dv4), mat4 )),
            TOL)
        
        # special cases ------------------------------------------------
        vx = 2.1-1.8j
        vy = 1.1-0.5j
        matx,r = to_std_uncertainty([4.1,0,0,1.7])
        maty,r = to_std_uncertainty([0.1,0,0,1.1])
        x = ucomplex(vx, (matx[0,0], matx[1,1]) )
        y = ucomplex(vy, (maty[0,0], maty[1,1]) )

        # x ** n, where n is 0        
        vz = 0.0
        p = x ** vz 
        vp = vx ** vz
        equivalent_complex(value(p),vp,TOL)
        dp_dx = vp*vz/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), matx )),
            TOL)

        # x ** n, where n is 1
        vz = 1.0
        p = x ** vz
        self.assertTrue( p is x )
        vp = vx ** vz
        equivalent_complex(value(p),vp,TOL)
        dp_dx = vp*vz/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), matx )),
            TOL)

        # negative real to fractional power ---------------------------------
        # ur-r 
        vx = -1.8
        vy = 0.5
        ux = 4.1
        
        x = ureal( vx,ux )
        y = vy

        p = x**y
        vp = complex(vx,0.)**vy
        equivalent_complex(value(p),vp,TOL)
        
        dp_dx = vp*vy/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), numpy.asarray( [[ux,0],[0,0]] ) )),
            TOL)
        dp_dy = cmath.log(vx)*vp
        equivalent_sequence(
            u_component(p,y),
            (0.,0.,0.,0.),
            TOL)

        # r-ur 
        vx = -1.8
        vy = 0.5
        uy = 4.1
        maty = numpy.array( (uy,0,0,0) )
        maty.shape = (2,2)
        
        x = vx
        y = ureal( vy,uy )

        p = x**y
        vp = complex(vx,0.)**vy
        equivalent_complex(value(p),vp,TOL)
        
        dp_dx = vp*vy/vx
        equivalent_sequence(
            u_component(p,x),
            (0.,0.,0.,0.),
            TOL)
            
        dp_dy = cmath.log(vx)*vp
        equivalent_sequence(
            u_component(p,y),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dy), maty )),
            TOL)

        # ur-ur 
        vx = -1.8
        vy = 0.5
        ux = 4.1
        uy = 0.4
        
        matx = numpy.array( (ux,0,0,0) )
        matx.shape = (2,2)
        maty = numpy.array( (uy,0,0,0) )
        maty.shape = (2,2)

        x = ureal(vx, ux )
        y = ureal(vy, uy )

        p = x**y
        vp = complex(vx,0)**vy
        equivalent_complex(value(p),vp,TOL)
        
        dp_dx = vp*vy/vx
        equivalent_sequence(
            u_component(p,x),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dx), matx )),
            TOL)
        dp_dy = cmath.log(vx)*vp
        equivalent_sequence(
            u_component(p,y),
            array_to_sequence( numpy.matmul(number_to_matrix(dp_dy), maty )),
            TOL)


#-----------------------------------------------------
class TestMultipleUNsWithComplexConstants(unittest.TestCase):
    """
    When defining ensembles of real or complex uncertain numbers,
    some elements might in fact have no uncertainty! In which
    case they are defined as constants.
    
    This seems a bit artificial: in effect the user is
    passing an uncertain number as part of an ensemble
    when it is not! However, we may have this situation
    in fitting scenarios (perhaps a parameter comes
    back with no uncertainty due to some odd data).
    So it is provided for. The creation of a constant
    should not harm subsequent calculations.
    
    """
        
    def test_ucomplex(self):
        values = [4.999+0j,0.019661+0j,1.04446j]
        uncert = [(0.0032,0.3),(0,0),(0.2,0.00075)]
        
        v,i,phi = multiple_ucomplex(values,uncert,5)

        self.assertTrue( equivalent_complex(v.x, values[0]) )
        self.assertTrue( equivalent_sequence(v.u, uncert[0]) )
        self.assertTrue(v.df == 5)

        self.assertTrue( equivalent_complex(i.x, values[1]) )
        self.assertTrue( equivalent_sequence(i.u, uncert[1]) )
        self.assertEqual(i.df, inf)

        self.assertTrue( equivalent_complex(phi.x, values[2]) )
        self.assertTrue( equivalent_sequence(phi.u, uncert[2]) )
        self.assertEqual(phi.df, 5)

        # We can set a zero correlation 
        set_correlation([0,0,0,0],v,i)
        set_correlation([0,0,0,0],i,phi)
        set_correlation([0,0,0,0],phi,v)
        
        # and set correlation between pairs in the ensemble
        set_correlation([0.5,0.1,0.2,0.3],v,phi)
        
#-----------------------------------------------------
class TestComplexUncertainRealMath(unittest.TestCase):

    def test_addition(self):
        x = ureal(1,1)
        a = 3
        b = 0.5
        z = complex(a,b)

        y = x + z
        self.assertTrue( isinstance(y,UncertainComplex) )
        
        self.assertTrue( equivalent_complex( complex(1+a,b), value(y), TOL) )
        self.assertTrue( equivalent( 1, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( 0, uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (1,0,0,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        xz = ucomplex(1,(1,0))
        yz = xz + z
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        # complex on right        
        y = z + x
        self.assertTrue( isinstance(y,UncertainComplex) )

        self.assertTrue( equivalent_complex( complex(1+a,b), value(y), TOL) )
        self.assertTrue( equivalent( 1, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( 0, uncertainty(y).imag, TOL) )
        
        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (1,0,0,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        yz = z + xz
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        # intermediate components
        # -- on the right
        a = 3
        b = 0.5
        z = complex(a,b)
        x1v = 3.1
        x2v = 4.1
        x1 = result(ureal(x1v,1) + ureal(x2v,1) )
        y = x1 + z

        ux1 = uncertainty(x1)
        self.assertTrue( equivalent_sequence( u_component(y,x1), (ux1,0,0,0) ) )

        # -- on the left
        y = z + x1
        self.assertTrue( equivalent_sequence( u_component(y,x1), (ux1,0,0,0) ) )
      

    def test_subtraction(self):
        x = ureal(1,1.0)
        a = 3
        b = 0.5
        z = complex(a,b)

        y = x - z
        self.assertTrue( isinstance(y,UncertainComplex) )

        self.assertTrue( equivalent_complex( complex(1-a,-b), value(y), TOL) )
        self.assertTrue( equivalent( 1, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( 0, uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (1,0,0,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        xz = ucomplex(1,(1,0))
        yz = xz - z
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )
        
        # complex on right        
        y = z - x
        self.assertTrue( isinstance(y,UncertainComplex) )

        self.assertTrue( equivalent_complex( complex(a-1,b), value(y), TOL) )
        self.assertTrue( equivalent( 1, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( 0, uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (-1,0,0,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        yz = z - xz
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        # intermediate components
        # -- on the right
        a = 3
        b = 0.5
        z = complex(a,b)
        x1v = 3.1
        x2v = 4.1
        x1 = result(ureal(x1v,1) + ureal(x2v,1) )
        y = x1 - z

        ux1 = uncertainty(x1)
        self.assertTrue( equivalent_sequence( u_component(y,x1), (ux1,0,0,0) ) )

        # -- on the left
        y = z - x1
        self.assertTrue( equivalent_sequence( u_component(y,x1), (-ux1,0,0,0) ) )
      
    def test_multiplication(self):
        x = ureal(1,1)
        a = 3
        b = 0.5
        z = complex(a,b)

        y = x * z
        self.assertTrue( isinstance(y,UncertainComplex) )

        self.assertTrue( equivalent_complex( complex(a,b), value(y), TOL) )
        self.assertTrue( equivalent( a, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( b, uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (z.real,0,z.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        xz = ucomplex(1,(1,0))
        yz = xz * z
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yz,xz), ) )
        
        # complex on right        
        y = z * x
        self.assertTrue( isinstance(y,UncertainComplex) )

        self.assertTrue( equivalent_complex( complex(a,b), value(y), TOL) )
        self.assertTrue( equivalent( a, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( b, uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (z.real,0,z.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        yz = z * xz
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yz,xz), ) )

        # intermediate components
        # -- on the right
        a = 3
        b = 0.5
        z = complex(a,b)
        x1v = 3.1
        x2v = 4.1
        x1 = result(ureal(x1v,1) + ureal(x2v,1) )
        y = x1 * z

        ux1 = uncertainty(x1)
        dy_dx = ux1 * z
        self.assertTrue( equivalent_sequence( u_component(y,x1), (dy_dx.real,0,dy_dx.imag,0) ) )

        # -- on the left
        y = z * x1
        self.assertTrue( equivalent_sequence( u_component(y,x1), (dy_dx.real,0,dy_dx.imag,0) ) )
      
    def test_division(self):
        xv = 2
        x = ureal(xv,1)
        a = 3.0
        b = 0.5
        z = complex(a,b)

        y = x / z
        self.assertTrue( isinstance(y,UncertainComplex) )

        inv_z = 1.0/z 
        self.assertTrue( equivalent_complex( xv/z, value(y), TOL) )
        self.assertTrue( equivalent( abs(inv_z.real), uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( abs(inv_z.imag), uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (inv_z.real,0,inv_z.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        xz = ucomplex(2,(1,0))
        yz = xz / z
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yz,xz), ) )

        # complex on right        
        y = z / x
        self.assertTrue( isinstance(y,UncertainComplex) )

        self.assertTrue( equivalent_complex( z/xv, value(y), TOL) )
        self.assertTrue( equivalent( abs(z.real)/xv**2, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( abs(z.imag)/xv**2, uncertainty(y).imag, TOL) )

        # components
        self.assertTrue( equivalent_sequence( u_component(y,x), (-z.real/xv**2,0,-z.imag/xv**2,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # same thing
        yz = z / xz
        self.assertTrue( equivalent_complex( value(yz), value(y), TOL) )
        self.assertTrue( equivalent( uncertainty(yz).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yz).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yz,xz), ) )

        # intermediate components
        # -- on the right
        a = 3
        b = 0.5
        z = complex(a,b)
        x1v = 3.1
        x2v = 4.1
        x1 = result(ureal(x1v,1) + ureal(x2v,1) )
        y = x1 / z

        ux1 = uncertainty(x1)
        dy_dx = ux1 * 1.0/z
        self.assertTrue( equivalent_sequence( u_component(y,x1), (dy_dx.real,0,dy_dx.imag,0) ) )

        # -- on the left
        y = z / x1
        dy_dx = -ux1 * z / value(x1)**2
        self.assertTrue( equivalent_sequence( u_component(y,x1), (dy_dx.real,0,dy_dx.imag,0) ) )

    def test_power(self):
        # complex on the right ----------------------------------
        xv = 3.1
        xu = 1
        x = ureal(xv,xu)
        xc = ucomplex(complex(xv,0),(xu,0))
        a = 1.2
        b = 0.3
        z = complex(a,b)

        y = x ** z
        yc = xc ** z
        
        self.assertTrue( isinstance(y,UncertainComplex) )
        self.assertTrue( equivalent_complex( xv ** z, value(y), TOL) )
        self.assertTrue( equivalent_complex( value(yc), value(y), TOL) )
        
        # components
        dy_dx =z * xv**(z-1) 
        self.assertTrue( equivalent_sequence( u_component(y,x), (dy_dx.real,0,dy_dx.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # make sure that results are equivalent to full complex calculation
        self.assertTrue( equivalent( uncertainty(yc).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yc).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yc,xc), ) )

        # Different numbers
        xv = -3.1
        xu = 1
        x = ureal(xv,xu)
        xc = ucomplex(complex(xv,0),(xu,0))
        a = -0.2
        b = -1.3
        z = complex(a,b)

        y = x ** z
        yc = xc ** z
        
        self.assertTrue( isinstance(y,UncertainComplex) )
        self.assertTrue( equivalent_complex( xv ** z, value(y), TOL) )
        self.assertTrue( equivalent_complex( value(yc), value(y), TOL) )

        # components
        dy_dx = z * xv**(z-1)
        self.assertTrue( equivalent_sequence( u_component(y,x), (dy_dx.real,0,dy_dx.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # make sure that results are equivalent to full complex calculation
        self.assertTrue( equivalent( uncertainty(yc).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yc).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yc,xc), ) )

        # complex on the left ----------------------------------
        xv = 3.1
        xu = 1
        x = ureal(xv,xu)
        xc = ucomplex(complex(xv,0),(xu,0))
        a = 1.2
        b = 0.3
        z = complex(a,b)

        y = z ** x
        yc = z ** xc
        yv = z ** xv
        
        self.assertTrue( isinstance(y,UncertainComplex) )
        self.assertTrue( equivalent_complex( yv, value(y), TOL) )
        self.assertTrue( equivalent_complex( value(yc), value(y), TOL) )

        # components
        dy_dx = yv * cmath.log(z) if z != 0 else 0
        self.assertTrue( equivalent_sequence( u_component(y,x), (dy_dx.real,0,dy_dx.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # make sure that results are equivalent to full complex calculation
        self.assertTrue( equivalent( uncertainty(yc).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yc).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yc,xc), ) )

        # different numbers
        xv = -3.1
        xu = 1
        x = ureal(xv,xu)
        xc = ucomplex(complex(xv,0),(xu,0))
        a = -0.2
        b = -1.3
        z = complex(a,b)

        y = z ** x
        yc = z ** xc
        yv = z ** xv
        
        self.assertTrue( isinstance(y,UncertainComplex) )
        self.assertTrue( equivalent_complex( yv, value(y), TOL) )
        self.assertTrue( equivalent_complex( value(yc), value(y), TOL) )

        # components
        dy_dx = yv * cmath.log(z) if z != 0 else 0
        self.assertTrue( equivalent_sequence( u_component(y,x), (dy_dx.real,0,dy_dx.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # make sure that results are equivalent to full complex calculation
        self.assertTrue( equivalent( uncertainty(yc).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yc).imag, uncertainty(y).imag, TOL) )
        
        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yc,xc), ) )

        # Special case z==0 here
        xv = 3.1
        xu = 1
        x = ureal(xv,xu)
        xc = ucomplex(complex(xv,0),(xu,0))
        a = 0
        b = 0
        z = complex(a,b)

        y = z ** x
        yc = z ** xc
        yv = z ** xv
        
        self.assertTrue( isinstance(y,UncertainComplex) )
        self.assertTrue( equivalent_complex( yv, value(y), TOL) )
        self.assertTrue( equivalent_complex( value(yc), value(y), TOL) )

        # components
        dy_dx = yv * cmath.log(z) if z != 0 else 0
        self.assertTrue( equivalent_sequence( u_component(y,x), (dy_dx.real,0,dy_dx.imag,0) ) )
        self.assertTrue( equivalent_sequence( u_component(y,z), (0,0,0,0) ) )

        # make sure that results are equivalent to full complex calculation
        self.assertTrue( equivalent( uncertainty(yc).real, uncertainty(y).real, TOL) )
        self.assertTrue( equivalent( uncertainty(yc).imag, uncertainty(y).imag, TOL) )

        self.assertTrue( equivalent_sequence( u_component(y,x), u_component(yc,xc), ) )

        # intermediate components
        # -- on the right
        a = 3
        b = 0.5
        z = complex(a,b)
        x1v = 3.1
        x2v = 4.1
        x1 = result(ureal(x1v,1) + ureal(x2v,1) )
        y = x1 ** z
        yv = value(y)

        ux1 = uncertainty(x1)
        dy_dx = z * value(x1)**(z-1) * ux1
        self.assertTrue( equivalent_sequence( u_component(y,x1), (dy_dx.real,0,dy_dx.imag,0) ) )

        # -- on the left
        y = z ** x1
        yv = value(y)
        dy_dx = yv * cmath.log(z) * ux1 if z != 0 else 0
        self.assertTrue( equivalent_sequence( u_component(y,x1), (dy_dx.real,0,dy_dx.imag,0) ) )

#-----------------------------------------------------
class TestGetSetCorrelation(unittest.TestCase):

    """
    tests the extended forms of get_correlation and
    set_correlation, where uncertain complex arguments
    are given.
    """
    
    def test(self):
        x1 = ureal(1,1)
        z1 = ucomplex(1,(1,1),independent=False)
        z2 = ucomplex(1,(1,1),independent=False)
        r = (.1,.2,.3,.4)
        set_correlation(r,z1,z2)
        check_r = get_correlation(z1,z2)
        self.assertTrue( equivalent_sequence(r,check_r) )

        r_t = (.1,.3,.2,.4)
        check_r = get_correlation(z2,z1)
        self.assertTrue( equivalent_sequence(r_t,check_r) )

        self.assertRaises(TypeError,set_correlation,r,z1,x1)
        self.assertRaises(TypeError,set_correlation,r,x1,z1)

    def test_with_mixed_unumbers(self):
        x1 = ureal(1,2,independent=False)
        z1 = ucomplex(1,.5,independent=False)
        
        r1 = .1
        set_correlation(r1,z1.real,x1)
        check_r = get_correlation(z1,x1)
        self.assertTrue( equivalent_sequence([r1,0,0,0],check_r) )
        
        r2 = -.3
        set_correlation(r2,z1.imag,x1)
        check_r = get_correlation(z1,x1)
        self.assertTrue( equivalent_sequence([r1,0,r2,0],check_r) )
        
        check_r = get_correlation(x1,z1)
        self.assertTrue( equivalent_sequence([r1,r2,0,0],check_r) )

        check_r = get_correlation(1.0,z1)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_correlation(1.0+7j)
        self.assertTrue( equivalent(0,check_r) )

        check_r = get_correlation(1.0+7j,z1)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_correlation(z1,1.0+7j)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_correlation(1.0+7j,x1)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_correlation(x1,1.0+7j)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )
      
        check_r = get_correlation(7,x1)
        self.assertTrue( equivalent(0,check_r) )

        check_r = get_correlation(x1,1.0)
        self.assertTrue( equivalent(0,check_r) )
        
#-----------------------------------------------------
class TestGetCovariance(unittest.TestCase):

    """
    tests get_covariance
    """
    
    def cv(self,u1,u2,r):
        return (
            u1[0]*r[0]*u2[0],
            u1[0]*r[1]*u2[1],
            u1[1]*r[2]*u2[0],
            u1[1]*r[3]*u2[1],
        )
    def test(self):
        ux1 = [0.25,0]
        uz1 = [0.33,1.5]
        uz2 = [.2,.4]
        x1 = ureal(1,ux1[0],independent=False)
        z1 = ucomplex(1,uz1,independent=False)
        z2 = ucomplex(1,uz2,independent=False)
        r = (.1,.2,.3,.4)
        
        set_correlation(r,z1,z2)
        check_r = get_covariance(z1,z2)
        self.assertTrue( equivalent_sequence(self.cv(uz1,uz2,r),check_r) )

        r_t = (.1,.3,.2,.4)
        check_r = get_covariance(z2,z1)
        self.assertTrue( equivalent_sequence(self.cv(uz2,uz1,r_t),check_r) )

        rx = [-.3,0,.7,0]
        set_correlation(rx[0],z1.real,x1)
        set_correlation(rx[2],z1.imag,x1)
        check_r = get_covariance(z1,x1)
        self.assertTrue( equivalent_sequence(self.cv(uz1,ux1,rx),check_r) )

        check_r = get_covariance(x1,z1)
        rx_t = [-.3,.7,0,0]
        self.assertTrue( equivalent_sequence(self.cv(ux1,uz1,rx_t),check_r) )

    def test_with_mixed_unumbers(self):
        ux1 = [2,0]
        x1 = ureal(1,ux1[0],independent=False)
        uz1 = [.5,.5]
        z1 = ucomplex(1,uz1,independent=False)
        
        r1 = .1
        set_correlation(r1,z1.real,x1)
        check_r = get_covariance(z1,x1)
        self.assertTrue( equivalent_sequence(self.cv(uz1,ux1,[r1,0,0,0]),check_r) )
        
        r2 = -.3
        set_correlation(r2,z1.imag,x1)
        check_r = get_covariance(z1,x1)
        self.assertTrue( equivalent_sequence(self.cv(uz1,ux1,[r1,0,r2,0]),check_r) )
        
        check_r = get_covariance(x1,z1)
        self.assertTrue( equivalent_sequence(self.cv(ux1,uz1,[r1,r2,0,0]),check_r) )

        check_r = get_covariance(1.0,z1)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_covariance(1.0+7j)
        self.assertTrue( equivalent(0,check_r) )

        check_r = get_covariance(1.0+7j,z1)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_covariance(z1,1.0+7j)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_covariance(1.0+7j,x1)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )

        check_r = get_covariance(x1,1.0+7j)
        self.assertTrue( equivalent_sequence([0]*4,check_r) )
      
        check_r = get_covariance(7,x1)
        self.assertTrue( equivalent(0,check_r) )

        check_r = get_covariance(x1,1.0)
        self.assertTrue( equivalent(0,check_r) )
 
#-----------------------------------------------------
class TestStringRepresentations(unittest.TestCase):
 
    def test(self):
        z = 1+0j
        u = 0.1 
        df = 'inf' 

        uc = ucomplex(z, u)
        s = re.search(r'ucomplex\(\((.*)\), u=\[(.*)\], r=(.*), df=(.*)\)', repr(uc))
        
        self.assertEqual(s.group(1),"{0.real:.16g}{0.imag:+.16g}j".format(z))
        self.assertEqual(s.group(2),"{0!r},{0!r}".format(u))
        self.assertEqual(s.group(3),repr(0.))
        self.assertEqual(s.group(4),"inf")
  
    def test_strange_cases(self):
        z = UncertainComplex._elementary(
            complex(inf,inf),
            inf,inf,
            None,
            inf,
            None,True
        )
        self.assertEqual( str(z), '(inf(inf)+inf(inf)j)')
        self.assertEqual( repr(z), 'ucomplex((inf+infj), u=[inf,inf], r=nan, df=inf)')
        
        z = UncertainComplex._elementary(
            complex(inf,inf),
            inf,inf,
            None,
            nan,
            None,True
        )
        self.assertEqual( str(z), '(inf(inf)+inf(inf)j)')
        self.assertEqual( repr(z), 'ucomplex((inf+infj), u=[inf,inf], r=nan, df=nan)')

        z = UncertainComplex._elementary(
            complex(inf,inf),
            inf,nan,
            None,
            nan,
            None,True
        )
        self.assertEqual( str(z), '(inf(inf)+inf(nan)j)')
        self.assertEqual( repr(z), 'ucomplex((inf+infj), u=[inf,nan], r=nan, df=nan)')

        z = UncertainComplex._elementary(
            complex(inf,inf),
            nan,nan,
            None,
            nan,
            None,True
        )
        self.assertEqual( str(z), '(inf(nan)+inf(nan)j)')
        self.assertEqual( repr(z), 'ucomplex((inf+infj), u=[nan,nan], r=nan, df=nan)')

        z = UncertainComplex._elementary(
            complex(inf,nan),
            nan,nan,
            None,
            nan,
            None,True
        )
        self.assertEqual( str(z), '(inf(nan)+nan(nan)j)')
        self.assertEqual( repr(z), 'ucomplex((inf+nanj), u=[nan,nan], r=nan, df=nan)')

        z = UncertainComplex._elementary(
            complex(nan,nan),
            nan,nan,
            None,
            nan,
            None,True
        )
        self.assertEqual( str(z), '(nan(nan)+nan(nan)j)')
        self.assertEqual( repr(z), 'ucomplex((nan+nanj), u=[nan,nan], r=nan, df=nan)')

        z = UncertainComplex._elementary(
            complex(1,1),
            0.1,0.1,
            None,
            nan,
            None,True
        )
        self.assertEqual( str(z), '(1.00(10)+1.00(10)j)')
        self.assertEqual( repr(z), 'ucomplex((1+1j), u=[0.1,0.1], r=0.0, df=nan)')

#-----------------------------------------------------
class TestMisc(unittest.TestCase):
 
    def test_constant(self):
        z = 1+0j
        uc = constant(z)
        self.assertTrue( _is_uncertain_complex_constant(uc) )
        self.assertRaises(TypeError,_is_uncertain_complex_constant,z)
        
        # Setting u=0 makes a constant (at the moment, should this be changed?)
        un = ucomplex(1j,0)
        self.assertTrue( _is_uncertain_complex_constant(un) )
        
    def test_nonzero(self):
        un = ucomplex(1+0j,1)
        self.assertTrue( bool(un) is True )
        un = ucomplex(1j,1)
        self.assertTrue( bool(un) is True )
        un = ucomplex(0+0j,1)
        self.assertTrue( bool(un) is not True )
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
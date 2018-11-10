import re
import unittest

from GTC import *
from GTC import context 
from GTC.vector import *
from GTC.nodes import *
from GTC.reporting import u_component
from GTC.lib import (
    UncertainReal,
    welch_satterthwaite,
    std_variance_real,
    set_correlation_real,
    get_correlation_real,
    real_ensemble,
    _is_uncertain_real_constant
)

from testing_tools import *

TOL = 1E-13 
LOG10_E = math.log10(math.e)

#----------------------------------------------------------------------------
class SimpleAttributesAndFunctions(unittest.TestCase):

    # An UncertainReal object has properties `x`, `u`, `v` and `df`.
    # There are corresponding functions: `value()`, `uncertainty()`, 
    # `variance()` and `dof()` 
    
    # An UncertainReal object also has `real`, `imag` and `conjugate`
    # properties.
    
    # An UncertainReal object may be implicitly converted to a 
    # float, a complex or a Boolean. 
    
    # The unary positive operation applied to an UncertainReal object 
    # creates a clone object.
    
    # The absolute value returns a float. 
    
    # The truth value of an uncertain number is equal to 

    #------------------------------------------------------------------------
    def setUp(self):
        self.x = x = 10.2
        self.u = u = 3.2
        self.df = df = 5
        self.label = label = 'me_too'
        
        self.un = ureal(x,u,df,label)
        self.const = constant(x,label)    
            
    def test_value(self):
        self.assertTrue( equivalent(self.x,value(self.un) ) )
        self.assertTrue( equivalent(self.x,value(self.const) ) )
        self.assertTrue( equivalent(self.x,self.un.x ) )
        self.assertTrue( equivalent(self.x,self.const.x ) )
      
    def test_uncertainty(self):
        self.assertTrue( equivalent(self.u,uncertainty(self.un) ) )
        self.assertTrue( equivalent(0,uncertainty(self.const) ) )
        self.assertTrue( equivalent(self.u,self.un.u ) )
        self.assertTrue( equivalent(0,self.const.u ) )
      
    def test_variance(self):
        self.assertTrue( equivalent(self.u**2,variance(self.un) ) )
        self.assertTrue( equivalent(0,variance(self.const) ) )
        self.assertTrue( equivalent(self.u**2,self.un.v ) )
        self.assertTrue( equivalent(0,self.const.v ) )
      
    def test_dof(self): 
        self.assertTrue( equivalent(self.df,dof(self.un) ) )
        self.assertTrue( equivalent(inf,dof(self.const) ) )
        self.assertTrue( equivalent(self.df,self.un.df ) )
        self.assertTrue( equivalent(inf,self.const.df ) )
        self.assertTrue( math.isinf( self.const.df ) )
 
    def test_conjugate(self):
        self.assertTrue( self.un.conjugate() is self.un )
        
    def test_real_imaginary(self):
        self.assertTrue( self.un.real == self.x )
        self.assertTrue( self.un.imag == 0 )
        
    def test_conversions(self):
        self.assertTrue( equivalent( value(self.un), self.x )  )
        z = value( self.un )
        self.assertTrue( equivalent( z.real, self.x ) )
        self.assertTrue( z.imag == 0 )
        self.assertTrue( bool(z.imag) == False )
        self.assertTrue( bool(z.real) == True )
        
    def test_positive_unary(self):
        un_plus = +self.un
        self.assertTrue(un_plus is not self.un)
        self.assertTrue( equivalent(self.x,value(un_plus) ) )
        self.assertTrue( equivalent(self.u,uncertainty(un_plus) ) )
        self.assertTrue( equivalent(self.df,dof(un_plus) ) )

    def test_absolute(self):
        un_minus = -self.un
        self.assertTrue( isinstance( abs(un_minus), float ) )
        self.assertTrue( equivalent(self.un.x,abs(un_minus) ) )

    def test_non_zero(self):
        x = ureal(1,1)
        self.assertTrue( bool(x) is True )
        x = ureal(0,1)
        self.assertTrue( bool(x) is not True )
        
    def test_is_constant(self): 
        x = constant(1.1)
        self.assertTrue( _is_uncertain_real_constant(x) )
        x = ureal(1,1)
        self.assertTrue( _is_uncertain_real_constant(x) is not True )
        x= 3
        self.assertRaises( TypeError, _is_uncertain_real_constant, x )
        
#----------------------------------------------------------------------------
class ArithmeticTestsReal(unittest.TestCase):

    # 
    
    #------------------------------------------------------------------------
    def setUp(self):
        self.x = ureal(1.,2.)
        self.w = ureal(2.,2.)
        self.z = ureal(3.,3.)
        
    #------------------------------------------------------------------------
    def testAddition(self):
        # Cases to consider: un-un, un-numb, numb-un
        # Need to check u_component in each case.
        # Adding zero is a special case
        
        #---------------------------------------------------------
        # += operator is not defined, but we check it
        x = self.x
        x += self.z
        equivalent(value(x),value(self.x)+value(self.z),TOL)
        
        equivalent(
            uncertainty(x)
        ,   math.sqrt( uncertainty(self.x)**2 + uncertainty(self.z)**2 )
        ,   TOL
        )
        self.assertTrue( math.isinf( dof(x) ) )

        #---------------------------------------------------------
        # Adding a constant on the right
        x = x + 10
        equivalent(value(x),value(self.x)+value(self.z)+10,TOL)
        
        equivalent(
            uncertainty(x)
        ,   math.sqrt(uncertainty(self.x)**2+uncertainty(self.z)**2)
        ,   TOL
        )

        # component of uncertainty
        x1 = result( self.x + self.w )
        x2 = x1 + 4
        equivalent(
            u_component(x2,x1)
        ,   uncertainty(x1)
        ,   TOL
        )
        
        #---------------------------------------------------------
        # Adding a constant on the left
        x = self.x + self.z
        x = 10 + x
        equivalent(value(x),value(self.x)+value(self.z)+10,TOL)
        
        equivalent(
            uncertainty(x)
        ,   math.sqrt(uncertainty(self.x)**2+uncertainty(self.z)**2)
        ,   TOL
        )
        
        # component of uncertainty
        x1 = result( self.x + self.w )
        x2 = 3 + x1
        equivalent(
            u_component(x2,x1)
        ,   uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # regular addition
        y = self.x + self.w
        equivalent(value(y),3.0,TOL)
        
        equivalent(uncertainty(y),2.0 * math.sqrt(2.0),TOL)
        self.assertTrue( math.isinf( dof(y) ) )

        # component of uncertainty
        x1 = result( y )
        x2 = x1 + self.z
        
        equivalent(value(x2),6.0,TOL)
        equivalent(
            u_component(x2,x1)
        ,   uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # trivial addition
        oldy = y
        y += 0
        self.assertTrue(oldy is y)
        y = y + 0
        self.assertTrue(oldy is y)
        y = 0 + y
        self.assertTrue(oldy is y)
        
        newy = y + 0j
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )

        newy = 0j + y 
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )

        #---------------------------------------------------------
        # Adding a complex constant on the left
        x = self.x 
        z = -10+6j
        y = z + x
        
        re = y.real 
        im = y.imag 
        
        equivalent(
            value(y.real),
            value(x)+z.real,
            TOL
        )
  
        equivalent(
            value(y.imag),
            z.imag,
            TOL
        )
  
        equivalent(
            uncertainty(y.real)
        ,   x.real.u
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   0.0
        ,   TOL
        )  

        #---------------------------------------------------------
        # Adding a complex constant on the right
        x = self.x 
        z = 10-3j
        y = x + z
        
        re = y.real 
        im = y.imag 
        
        equivalent(
            value(y.real),
            value(x)+z.real,
            TOL
        )
  
        equivalent(
            value(y.imag),
            z.imag,
            TOL
        )
  
        equivalent(
            uncertainty(y.real)
        ,   x.real.u
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   0.0
        ,   TOL
        )
                
    #------------------------------------------------------------------------
    def testSubtraction(self):
        # Cases to consider: un-un, un-numb, numb-un
        # Need to check u_component in each case.

        #---------------------------------------------------------
        # -= operator is not defined, but we check it
        x = self.x
        x -= self.z
        equivalent(value(x),value(self.x)-value(self.z),TOL)
        
        equivalent(
            uncertainty(x)
        ,   math.sqrt(uncertainty(self.x)**2+uncertainty(self.z)**2)
        ,   TOL
        )
        self.assertTrue( math.isinf( dof(x) ) )

        #---------------------------------------------------------
        # Subtracting a constant
        x = x - 10
        equivalent(value(x),value(self.x)-value(self.z)-10,TOL)

        equivalent(
            uncertainty(x)
        ,   math.sqrt(uncertainty(self.x)**2+uncertainty(self.z)**2)
        ,   TOL
        )
        
        # component of uncertainty
        x1 = result( self.x + self.w )
        x2 = x1 - 10
        equivalent(
            u_component(x2,x1)
        ,   uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # lhs is not an UN
        x = 10 - x
        equivalent(value(x),-value(self.x)+value(self.z)+20,TOL)

        equivalent(
            uncertainty(x)
        ,   math.sqrt(uncertainty(self.x)**2+uncertainty(self.z)**2)
        ,   TOL
        )

        # component of uncertainty
        x1 = result( self.x + self.w )
        x2 = 10 - x1
        equivalent(
            u_component(x2,x1)
        ,   -uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # regular subtraction        
        y = self.x - self.w
        equivalent(value(y),value(self.x)-value(self.w))
        equivalent(uncertainty(y),2.0 * math.sqrt(2.0))

        # component of uncertainty
        x1 = result( y )
        x2 = x1 - self.z
        equivalent(
            u_component(x2,x1)
        ,   uncertainty(x1)
        ,   TOL
        )
        
        #---------------------------------------------------------
        # trivial subtraction
        oldy = y
        y -= 0
        self.assertTrue(oldy is y)
        y = y - 0
        self.assertTrue(oldy is y)
        y = 0 - y
        self.assertTrue(oldy is not y)
        equivalent(value(y),-value(oldy))

        newy = y - 0j
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )

        newy = 0j - y 
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),-value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )
        
        #---------------------------------------------------------
        # Subtracting a complex constant on the left
        x = self.x 
        z = -10+6j
        y = z - x
        
        re = y.real 
        im = y.imag 
        
        equivalent(
            value(y.real),
            z.real-value(x),
            TOL
        )
  
        equivalent(
            value(y.imag),
            z.imag,
            TOL
        )
  
        equivalent(
            uncertainty(y.real)
        ,   x.real.u
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   0.0
        ,   TOL
        )  

        #---------------------------------------------------------
        # Subtracting a complex constant on the right
        x = self.x 
        z = 3.2+6.1j
        y = x - z
        
        re = y.real 
        im = y.imag 
        
        equivalent(
            value(y.real),
            value(x)-z.real,
            TOL
        )
  
        equivalent(
            value(y.imag),
            -z.imag,
            TOL
        )
  
        equivalent(
            uncertainty(y.real)
        ,   x.real.u
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   0.0
        ,   TOL
        )  
        
    #------------------------------------------------------------------------
    def testMultiplication(self):
        # Cases to consider: un-un, un-numb, numb-un
        # Need to check u_component in each case.

        #---------------------------------------------------------
        # *= operator is not defined, but we check it
        x = self.x
        x *= self.z
        equivalent(value(x),value(self.x)*value(self.z),TOL)

        equivalent(
            uncertainty(x)
        ,   math.sqrt(
                (value(self.z) * uncertainty(self.x))**2
            +   (value(self.x) * uncertainty(self.z))**2
            )
        ,   TOL
        )
        self.assertTrue( math.isinf( dof(x) ) )

        #---------------------------------------------------------
        # multiplying by a constant
        x = x * 10
        equivalent(value(x),(value(self.x)*value(self.z))*10,TOL)

        equivalent(
            uncertainty(x)
        ,   math.sqrt(
                (10*value(self.z) * uncertainty(self.x))**2
            +   (10*value(self.x) * uncertainty(self.z))**2
            )
        ,   TOL
        )
        
        # component of uncertainty
        x1 = result( self.x + self.w )
        x2 = x1 * 5
        equivalent(
            u_component(x2,x1)
        ,   5 * uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # lhs is not an UN
        x = 10 * x
        equivalent(value(x),(value(self.x)*value(self.z))*100,TOL)
        equivalent(
            uncertainty(x)
        ,   math.sqrt(
                (100*value(self.z) * uncertainty(self.x))**2
            +   (100*value(self.x) * uncertainty(self.z))**2
            )
        ,   TOL
        )

        # component of uncertainty
        x1 = result( self.x + self.w )
        x2 = 3 * x1
        equivalent(
            u_component(x2,x1)
        ,   3 * uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # regular multiplication
        y = self.z * self.w
        equivalent(value(y),6.0)
        equivalent(uncertainty(y),6.0 * math.sqrt(2.0))

        # component of uncertainty
        x1 = result( self.x * self.w )
        x2 = self.z * x1
        equivalent(
            u_component(x2,x1)
        ,   value(self.z) * uncertainty(x1)
        ,   TOL
        )

        #---------------------------------------------------------
        # trivial multiplication
        oldy = y
        y *= 1
        self.assertTrue(oldy is y)
        y = y * 1
        self.assertTrue(oldy is y)
        y = 1.0 * y
        self.assertTrue(oldy is y)

        newy = y*(1.0+0j)
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )

        newy = (1.0+0j)*y
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )
        
        #---------------------------------------------------------
        # Multiplying a complex constant on the left
        x = self.x 
        z = 10+4j
        y = z*x
        
        re = y.real 
        im = y.imag 
        
        equivalent(
            value(y.real),
            (z*value(x)).real,
            TOL
        )
  
        equivalent(
            value(y.imag),
            (z*value(x)).imag,
            TOL
        )
  
        equivalent(
            uncertainty(y.real)
        ,   z.real * x.u
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   z.imag * x.u
        ,   TOL
        )  

        #---------------------------------------------------------
        # Multiplying a complex constant on the right
        x = self.x 
        z = 3.2+6.1j
        y = x * z
        
        equivalent(
            value(y.real),
            (z*value(x)).real,
            TOL
        )
    
        equivalent(
            value(y.imag),
            (value(x)*z).imag,
            TOL
        )
  
        equivalent(
            uncertainty(y.real)
        ,   x.u * z.real
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   x.u * z.imag
        ,   TOL
        )  
    #------------------------------------------------------------------------
    def testDivision(self):
        # Cases to consider: un-un, un-numb, numb-un
        # Need to check u_component in each case.

        #---------------------------------------------------------
        # /= operator is not defined, but we check it
        x = self.x
        x /= self.z
        equivalent(value(x),value(self.x)/value(self.z),TOL)
        equivalent(
            uncertainty(x)/value(x)
        ,   math.sqrt(
                (uncertainty(self.x)/value(self.x))**2
            +   (uncertainty(self.z)/value(self.z))**2
            )
        ,   TOL
        )
        self.assertTrue( math.isinf(dof(x)) )

        #---------------------------------------------------------
        # regular division
        y = self.z / self.w
        equivalent(value(y),1.5)
        equivalent(uncertainty(y),1.5 * math.sqrt(2.0))

        # component of uncertainty
        x1 = result( self.x / self.w )
        x2 = x1 / self.x
        equivalent(
            u_component(x2,x1)
        ,   value(x1)/value(x2) * uncertainty(x1)
        ,   TOL
        )
        #---------------------------------------------------------
        # Divide by a constant
        y = self.w / 5.0
        equivalent(value(y),value(self.w)/5.0,TOL)
        equivalent(
            uncertainty(y)
        ,   uncertainty(self.w) / 5.0
        ,   TOL
        )
        equivalent(
            u_component(y,self.w)
        ,   uncertainty(self.w) / 5.0
        ,   TOL
        )
        
        # component of uncertainty
        x1 = result( self.x / self.w )
        x2 = x1 / 5.0
        equivalent(
            u_component(x2,x1)
        ,   uncertainty(x1) / 5
        ,   TOL
        )
        #---------------------------------------------------------
        # lhs is not UN
        y = 4.0 / self.w
        equivalent(value(y),4.0 / value(self.w),TOL)
        equivalent(
            u_component(y,self.w)
        ,   -4.0 / value(self.w)**2 * uncertainty(self.w) 
        ,   TOL
        )

        # component of uncertainty
        x1 = result( self.x / self.w )
        x2 = 4.0 / x1
        equivalent(
            u_component(x2,x1)
        ,   -4.0 * uncertainty(x1) / value(x1)**2
        ,   TOL
        )
        
        #---------------------------------------------------------
        # trivial division
        y = self.w / 1.0
        self.assertTrue( y is self.w )
        y = 1.0 / self.w
        self.assertTrue( y is not self.w )
        
        self.assertRaises(ZeroDivisionError,UncertainReal.__div__, self.z, 0)

        newy = y/(1.0+0j)
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[0],uncertainty(y),TOL) )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )

        newy = (1.0+0j)/y
        self.assertFalse( newy is y )
        self.assertTrue( equivalent(value(newy.real),1./value(y),TOL) )
        self.assertTrue( equivalent(value(newy.imag),0.0,TOL) )
        self.assertTrue( 
            equivalent(
                uncertainty(newy)[0]/value(newy.real),
                uncertainty(y)/value(y),TOL
            ) 
        )
        self.assertTrue( equivalent(uncertainty(newy)[1],0.0,TOL) )
        

        #---------------------------------------------------------
        # Dividing a complex constant on the left
        x = self.x 
        z = 10+4j
        y = z/x
        
        re = y.real 
        im = y.imag 
        
        equivalent(
            value(y.real),
            (z/value(x)).real,
            TOL
        )
  
        equivalent(
            value(y.imag),
            (z/value(x)).imag,
            TOL
        )
  
        norm = value(x)**2
        equivalent(
            uncertainty(y.real)
        ,   abs( z.real/norm*x.u )
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   abs( z.imag/norm*x.u )
        ,   TOL
        )  

        #---------------------------------------------------------
        # Dividing by a complex constant on the right
        x = self.x 
        z = 3.2+6.1j
        y = x / z
        
        equivalent(
            value(y.real),
            (value(x)/z).real,
            TOL
        )
    
        equivalent(
            value(y.imag),
            (value(x)/z).imag,
            TOL
        )
  
        norm = abs(z)**2
        equivalent(
            uncertainty(y.real)
        ,   x.u * abs(z.real)/norm
        ,   TOL
        )

        equivalent(
            uncertainty(y.imag)
        ,   x.u * abs(z.imag)/norm
        ,   TOL
        )  
    #------------------------------------------------------------------------
    def testNegation(self):
        x = self.x
        w = self.w
        y = -(x + w)
        equivalent(value(y),-value(value(self.x)+value(self.w)),TOL)
        u_x = u_component(y,self.x)
        equivalent(u_x,-u_component(self.x+self.w,self.x),TOL)

        # component of uncertainty
        x1 = result( self.x / self.w )
        x2 = -x1
        equivalent(
            u_component(x2,x1)
        ,   -1.0 * uncertainty(x1)
        ,   TOL
        )

#----------------------------------------------------------------------------
class TestUncertainReal(unittest.TestCase):
    """
    UncertainReal object special methods.
    Some of these handle mathematical operations and are tested
    elsewhere. Here testing is for repr() and creation 
    
    """
    def test_elementary(self):
        x = ureal(1,1)
        self.assertTrue( x.is_elementary )
        self.assertTrue( not x.is_intermediate )
        self.assertEqual( None, x.label )
        
        y = x + 10
        self.assertTrue( not y.is_elementary )
        self.assertTrue( not y.is_intermediate )
        self.assertEqual( None, y.label )
        
        y = result(x + 10,label='y')
        self.assertTrue( y.is_intermediate )
        self.assertEqual( 'y', y.label )
    
        z = constant(10,label='z')
        self.assertTrue( not z.is_elementary )
        self.assertTrue( not z.is_intermediate )
        self.assertEqual( 'z', z.label )
            
    def test(self):
        """
        The constructor takes arguments: context,x,u_comp,d_comp,i_comp [,node]
        
        A node object is provided when creating an elementary uncertain number 
        or an intermediate result.
        
        """
        x = 1.
        u = 0.1 
        df = 'inf'
        
        un = ureal(x,u)
        s = re.search(r'ureal\((.*),(.*),(.*)\)', repr(un))
        self.assertEqual(s.group(1), repr(x) )
        self.assertEqual(s.group(2), repr(u) )
        self.assertEqual(s.group(3), df )
        
        x = 1.2
        u = 0.2
        x_str= "1.20(20)" # takes account of uncertainty rounding

        un = ureal(x,u)

        rep = "ureal({!r},{!r},inf)".format(x,u)
        self.assertEqual( rep,repr(un) )
        self.assertEqual( x_str,str(un) )

        self.assertEqual( value(un), x)

        # These tests just look at the constructor and by-pass the context. 
        # Here is an uncertain number with no node
        u_comp = Vector()
        u_comp.extend( [(1,u)] )
        d_comp = Vector()
        d_comp.extend( [(2,2*u)] )
        i_comp = Vector()
        un = UncertainReal(x,u_comp,d_comp,i_comp)
        self.assertTrue( un._u_components is u_comp )
        self.assertTrue( un._i_components is i_comp )
        self.assertTrue( un._d_components is d_comp )

        # Here is an uncertain number with a node
        # args: uid,label,u,df [,independent=True]
        label = 'test'
        df = 10
        lf = Leaf(3,label,u,df)
        un = UncertainReal(x,u_comp,d_comp,i_comp,lf)
        self.assertTrue( un._u_components is u_comp )
        self.assertTrue( un._i_components is i_comp )
        self.assertTrue( un._d_components is d_comp )
        self.assertTrue( un._node.label is label )
        self.assertTrue( un._node.independent is True )
        self.assertTrue( un._node.df == df )
        
    def test_check_identity(self):
        """
        The `reporting.is_ureal` function should be able to pick
        an uncertain real number.
        
        """
        x = ureal(1,1)
        self.assertTrue( reporting.is_ureal(x) )

        y = ureal(1,1) * x
        self.assertTrue( reporting.is_ureal(y) )

        z = constant(6)        
        self.assertTrue( reporting.is_ureal(z) )

        z1 = ucomplex(1+3j,1)
        self.assertTrue( reporting.is_ureal( z1.real ) )
        
    def test_strange_rounding_cases(self):
        """
        The __str__ method 
        
        """
        x = UncertainReal._elementary(
            inf,inf,inf,
            None,True
        )
        self.assertEqual( str(x), 'inf(inf)')
        self.assertEqual( repr(x), 'ureal(inf,inf,inf)')

        x = UncertainReal._elementary(
            inf,inf,nan,
            None,True
        )
        self.assertEqual( str(x), 'inf(inf)')
        self.assertEqual( repr(x), 'ureal(inf,inf,nan)')
        
        x = UncertainReal._elementary(
            inf,nan,nan,
            None,True
        )
        self.assertEqual( str(x), 'inf(nan)')
        self.assertEqual( repr(x), 'ureal(inf,nan,nan)')
        
        x = UncertainReal._elementary(
            nan,nan,nan,
            None,True
        )
        self.assertEqual( str(x), 'nan(nan)')
        self.assertEqual( repr(x), 'ureal(nan,nan,nan)')

        x = UncertainReal._elementary(
            1,0.1,nan,
            None,True
        )
        self.assertEqual( str(x), '1.00(10)')
        self.assertEqual( repr(x), 'ureal(1.0,0.1,nan)')

        
#----------------------------------------------------------------------------
class TestComparisons(unittest.TestCase):

    # UncertainReal defines comparisons wrt the object value 
    
    #------------------------------------------------------------------------
    def test_ureal_comparison(self):
        # All float comparisons should apply
        x_value = -1.2
        x_greater = x_value + .1
        x_less = x_value - .1
        
        x = ureal(x_value,0.5)
        x_g = ureal(x_greater,0.25)
        x_l = ureal(x_less,1)
        x_false = ureal(0,1)

        self.assertTrue(x == x_value)
        self.assertTrue(x <= x_greater)
        self.assertTrue(x >= x_less)
        self.assertTrue(x < x_greater)
        self.assertTrue(x > x_less)
        self.assertTrue(x != x_greater)
        self.assertTrue(x != x_less)

        self.assertTrue(x <= x_g)
        self.assertTrue(x >= x_l)
        self.assertTrue(x < x_g)
        self.assertTrue(x > x_l)
        self.assertTrue(x != x_g)
        self.assertTrue(x != x_l)
        
        # Boolean tests depend on value 
        self.assertTrue( bool(x) )
        self.assertTrue( not bool(x_false) )
        self.assertTrue( x )
        self.assertTrue( not x_false )

        self.assertTrue( equivalent(abs(x_value),abs(x)) )

#----------------------------------------------------------------------------
class TestFunctionsReal(unittest.TestCase):
    
    def setUp(self):
        self.x1 = 1.7
        self.u1 = 0.1
        self.un1 = ureal(self.x1,self.u1) 
        
        self.x2 = 0.5
        self.u2 = 0.4
        self.un2 = ureal(self.x2,self.u2)
        
        self.x3 = -1.0
        self.u3 = 0.4
        self.un3 = ureal(self.x3,self.u3)

        self.x4 = 0.0
        self.u4 = 0.2
        self.un4 = ureal(self.x4,self.u4)
        
        self.x5 = 1.0
        self.u5 = 0.2
        self.un5 = ureal(self.x5,self.u5)

        # Create intermediate values for u_component checking
        self.x_12 = self.x1 * self.x2
        self.u_12 = math.sqrt((self.x1 * self.u2)**2 + (self.x2 * self.u1)**2)

        # By using `result` here the i_components buffer will be used
        # in the intermediate tests. 
        self.un_12 = result( self.un1 * self.un2 ) 
        
        self.un_22 = result( self.un2 / ureal(.3,.1) )

    # This test case is probably unnecessary
    # def test_conjugate(self):
        # """
        # conjugate() is implemented so as to simply return self
        
        # """
        # self.assertTrue( equivalent(self.un1.x, self.un1.conjugate().x) )
        # self.assertTrue( equivalent(self.un1.u, self.un1.conjugate().u) )
        # self.assertTrue( self.un1 is self.un1.conjugate() )
        
        # # 
        # x2 = result( +self.un5 )
        # xc = x2.conjugate() 
        # self.assertTrue(
            # equivalent_matt(
                # xc._i_components,
                # x2._i_components,
                # TOL
        # ))
 
    def test_Pwr(self):
        # ur-ur
        vv = self.x1 ** self.x2
        # The two components of uncertainty are:
        u1 = self.x2 * vv / self.x1 * self.u1
        u2 = vv * math.log( abs( self.x1) ) * self.u2
        y = self.un1 ** self.un2
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( u1, u_component(y,self.un1), TOL )
        equivalent( u2, u_component(y,self.un2), TOL )

        # 0 ** 1 is a tricky one
        y = self.un4 ** self.un5
        v = value(y)
        equivalent( v,  0., TOL  )
        equivalent( self.u4, u_component(y,self.un4), TOL )
        equivalent( 0., u_component(y,self.un5), TOL )

        # y = x ** 1                             
        y = self.un2 ** self.un5
        
        # The two components of uncertainty are:
        u1 = self.x5 * (self.x2 ** (self.x5 - 1)) * self.u2
        u2 = (self.x2 ** self.x5) * math.log( abs( self.x2) ) * self.u5

        v = value(y)
        equivalent( v,  self.x2 ** self.x5 , TOL  )
        equivalent( u1, u_component(y,self.un2), TOL )
        equivalent( u2, u_component(y,self.un5), TOL )

        # x ** number
        number = 2.5
        y = self.un1 ** number
        vv = self.x1 ** number
        v = value(y)
        equivalent( v,  vv, TOL  )
        # The components of uncertainty are:
        u1 = number * vv / self.x1 * self.u1        
        equivalent( u1, u_component(y,self.un1), TOL )

        # number ** x
        number = 2.5
        y = number ** self.un2
        vv = number ** self.x2 
        v = value(y)
        equivalent( v,  vv, TOL  )
        # The components of uncertainty are:
        u2 = vv * math.log( abs( number ) ) * self.u2
        equivalent( u2, u_component(y,self.un2), TOL )
 
        # x ** complex number
        number = .5+1.2j
        y = self.un1 ** number
        vv = self.x1 ** number
        v = value(y)
        equivalent_complex( v,  vv, TOL  )
        # The real/imag components of uncertainty are:
        dy_dx = number * vv / self.x1  
        u_re = abs(dy_dx.real * self.u1)
        u_im = abs(dy_dx.imag * self.u1)
        equivalent( u_im, y.imag.u, TOL )
        equivalent( u_re, y.real.u, TOL )

        # complex number ** x
        number = 2.5-.3j
        y = number ** self.un2
        vv = number ** self.x2 
        v = value(y)
        equivalent_complex( v,  vv, TOL  )
        # The real/imag components of uncertainty are:
        dy_dx = vv * cmath.log( number ) 
        u_re = abs(dy_dx.real * self.u2)
        u_im = abs(dy_dx.imag * self.u2)
        equivalent( u_im, y.imag.u, TOL )
        equivalent( u_re, y.real.u, TOL )
 
        # x ** 1
        y = self.un1 ** 1.0
        self.assertTrue( y is self.un1 )
        
        # x ** 0
        y = self.un1 ** 0.0
        equivalent( 1.0, value(y), TOL  )
        equivalent( 0.0, uncertainty(y), TOL  )

        # intermediate case
        y = self.un_12 ** self.un_22
        
        equivalent(
            value(self.un_22)*value(self.un_12)**(value(self.un_22)-1) * uncertainty(self.un_12),
            u_component(y,self.un_12),
            TOL )
        equivalent(
            math.log( abs(value(self.un_12)) ) * value(y) * uncertainty(self.un_22),
            u_component(y,self.un_22),
            TOL )
          

    def test_magnitude(self):
        """
        magnitude() is implemented as if it were sqrt(x**2)
        
        """
        self.assertRaises(ZeroDivisionError,magnitude,ureal(0,1))

        mag_x3 = magnitude(self.un3)
        equivalent( value(mag_x3), abs( self.x3 ), TOL )

        y = -1 * self.un2
        mag_y = magnitude(y)
        equivalent( value(mag_y), self.x2, TOL )
        equivalent( u_component(y,self.un2), -self.u2, TOL )
        equivalent( u_component(mag_y,self.un2), self.u2, TOL )

        # Intermediate tests (these use the i_component value)       
        y = +self.un_12
        mag_y = magnitude(y)
        equivalent( value(mag_y), self.x_12, TOL )
        equivalent( u_component(y,self.un_12), self.u_12, TOL )
        equivalent( u_component(mag_y,self.un_12), self.u_12, TOL )
        
        y = -1 * self.un_12
        mag_y = magnitude(y)
        equivalent( value(mag_y), self.x_12, TOL )
        equivalent( u_component(y,self.un_12), -self.u_12, TOL )
        equivalent( u_component(mag_y,self.un_12), self.u_12, TOL )


#-----------------------------------------------------
class TrigTestsReal(unittest.TestCase):

    def setUp(self):
        self.x = 0.7
        self.u = 0.3
        self.un = ureal(self.x,self.u)

        self.x1 = -0.2
        self.u1 = 0.1
        self.un1 = ureal(self.x1,self.u1)

        # Using `result` means that the i_components
        # buffer is used for the intermediate components.
        self.x2 = self.x * self.x1
        self.un2 = result( self.un * self.un1 )
        
    def testSine(self):
        # There are no special cases of inputs to test
        y = sin(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, math.sin( self.x ), TOL )
        equivalent( u, math.cos( self.x ) * self.u, TOL )
        self.assertTrue( math.isinf(df) )

        # This will test an intermediate component of uncertainty
        vv = math.sin(self.x2)
        uu = math.cos(self.x2) * uncertainty( self.un2 )
        y = sin( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )
        
    def testCosine(self):
        # There are no special cases of inputs to test
        y = cos(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, math.cos( self.x ),TOL )
        equivalent( u, math.sin( self.x) * self.u, TOL )
        self.assertTrue( math.isinf(df) )
        
        # This will test an intermediate component of uncertainty
        vv = math.cos(self.x2)
        uu = -math.sin(self.x2) * uncertainty( self.un2 )
        y = cos( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )

    def testTangent(self):
        # There are no special cases of inputs to test
        y = tan(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, math.tan( self.x ), TOL )
        uu = self.u/(math.cos( self.x )**2)
        equivalent( u, uu, TOL )
        self.assertTrue( math.isinf(df) )

        # This will test an intermediate component of uncertainty
        vv = math.tan(self.x2)
        uu = uncertainty( self.un2 ) /(math.cos( self.x2 )**2)
        y = tan( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )
        
class FunctionTestsReal(unittest.TestCase):
    def setUp(self):
        self.x1 = 1.7
        self.u1 = 0.1
        self.un1 = ureal(self.x1,self.u1) 
        
        self.x2 = 0.5
        self.u2 = 0.4
        self.un2 = ureal(self.x2,self.u2)
        
        self.x3 = -1.0
        self.u3 = 0.4
        self.un3 = ureal(self.x3,self.u3)

        self.x4 = 0.0
        self.u4 = 0.2
        self.un4 = ureal(self.x4,self.u4)
        
        self.x5 = 1.0
        self.u5 = 0.2
        self.un5 = ureal(self.x5,self.u5)

        # Create intermediate values for u_component checking
        self.x_12 = self.x1 * self.x2
        self.u_12 = math.sqrt((self.x1 * self.u2)**2 + (self.x2 * self.u1)**2)
        
        # By using `result` here the i_components buffer will be used
        # in the intermediate tests. 
        self.un_12 = result( self.un1 * self.un2 ) 
        
        self.un_22 = result( self.un2 / ureal(.3,.1) )

    def testLog(self):
        y = log(self.un1)
        v = value(y)
        u = uncertainty(y)
        equivalent( v, math.log( self.x1 ), TOL )
        equivalent( u, self.u1 / self.x1, TOL )

        y = log(self.un2)
        v = value(y)
        u = uncertainty(y)
        equivalent( v, math.log( self.x2 ), TOL )
        equivalent( u, self.u2 / self.x2, TOL )

        self.assertRaises(ValueError,log,self.un3)

        # This will test an intermediate component of uncertainty
        vv = math.log(self.x_12)
        uu = 1/value(self.x_12) * uncertainty( self.un_12 )
        y = log( self.un_12 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un_12), TOL )

    def testLog10(self):
        y = log10(self.un1)
        v = value(y)
        u = uncertainty(y)
        equivalent( v, LOG10_E * math.log( self.x1 ), TOL )
        equivalent( u, LOG10_E * self.u1 / self.x1, TOL )

        y = log10(self.un2)
        v = value(y)
        u = uncertainty(y)
        equivalent( v, LOG10_E * math.log( self.x2 ), TOL )
        equivalent( u, LOG10_E * self.u2 / self.x2, TOL )

        self.assertRaises(ValueError,log10,self.un3)

        # This will test an intermediate component of uncertainty
        vv = math.log(self.x_12)
        uu = 1/value(self.x_12) * uncertainty( self.un_12 )
        y = log10( self.un_12 )
        v = value(y)
        equivalent( v,  LOG10_E * vv, TOL  )
        equivalent( LOG10_E * uu, uncertainty(y), TOL )
        equivalent( LOG10_E * uu, u_component(y,self.un_12), TOL )
        
    def testSqrt(self):
        y = sqrt(self.un1)
        v = value(y)
        vv = math.sqrt(self.x1)
        uu = self.u1 / ( 2.0 * vv )
        
        equivalent( v,  vv, TOL  )
        equivalent( uu, u_component(y,self.un1), TOL )

        y = sqrt(self.un2) 
        v = value(y)
        vv = math.sqrt(self.x2)
        uu = self.u2 / ( 2.0 * vv )
        
        equivalent( v,  vv, TOL  )
        equivalent( uu, u_component(y,self.un2), TOL )

        self.assertRaises(ValueError,sqrt,self.un3)

        # This will test an intermediate component of uncertainty
        vv = math.sqrt(self.x_12)
        uu = 0.5/vv * uncertainty( self.un_12 )
        y = sqrt( self.un_12 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un_12), TOL )

    def testExp(self):
        # There are no special cases of inputs to test
        vv = math.exp(self.x1)
        uu = self.u1 * vv
        y = exp( self.un1 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, u_component(y,self.un1), TOL )
        equivalent( uu, uncertainty(y), TOL )

        vv = math.exp(self.x2)
        uu = self.u2 * vv
        y = exp( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, u_component(y,self.un2), TOL )

        vv = math.exp(self.x3)
        uu = self.u3 * vv
        y = exp( self.un3 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, u_component(y,self.un3), TOL )

        # This will test an intermediate component of uncertainty
        vv = math.exp(self.x_12)
        uu = vv * uncertainty( self.un_12 )
        y = exp( self.un_12 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un_12), TOL )
        
    def testMagSquared(self):
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

    def testMagnitude(self):
        v1 = 1.5
        u1 = 1
        v2 = -.5
        u2 = 1
        v3 = 0.0
        u3 = 1
        z1 = ureal(v1,u1)
        z1_mag = magnitude(z1)
        self.assertTrue(
            equivalent(value(z1_mag),v1)
        )
        self.assertTrue(
            equivalent(uncertainty(z1),u1)
        )
        
        z = ureal(v2,u2)
        z_mag = magnitude(z)
        self.assertTrue(
            equivalent(value(z_mag),abs(v2) )
        )
        self.assertTrue(
            equivalent(uncertainty(z),u2)
        )
        z = ureal(v3,u3)
        self.assertRaises( ZeroDivisionError, magnitude, z )

    def testPhase(self):
        v1 = 1.5
        u1 = 1
        z1 = ureal(v1,u1)
        self.assertEqual( phase(z1), 0.0 )
        
#-----------------------------------------------------
class TestGetCovariance(unittest.TestCase):

    def test_with_mixed_unumbers(self):
    
        ux1 = 2
        x1 = ureal(1,ux1,independent=False)
        uz1 = .5
        z1 = ureal(1,uz1,independent=False)
        
        r1 = .1
        set_correlation(r1,z1,x1)
        check_r = get_covariance(z1,x1)
        self.assertTrue( equivalent(r1,check_r) )
        
        r2 = -.3
        set_correlation(r2,z1,x1)
        check_r = get_covariance(z1,x1)
        self.assertTrue( equivalent(r2,check_r) )

    def test_simple_correlation(self):
        x1 = ureal(0,1,5,None,independent=True)
        x2 = ureal(0,1,5,None,independent=True)

        self.assertRaises(
            RuntimeError,
            set_correlation_real,x1,x2,.1
        )

        # self correlation must be unity
        self.assertRaises(
            RuntimeError,
            set_correlation_real,x1,x1,.1
        )

        x1 = ureal(0,1,5,None,independent=False)
        x2 = ureal(0,1,5,None,independent=False)
        
        # Self correlation
        self.assertEqual( 1, get_correlation_real(x1,x1) )
        
        r = 0.65
        set_correlation_real(x1,x2,r)
        
        self.assertTrue( equivalent( r, get_correlation_real(x1,x2) ) )

        # When we delete one node the other remains
        x1_uid = x1._node.uid
        x2_uid = x2._node.uid
        self.assertTrue( x1_uid in context._context._registered_leaf_nodes )
        self.assertTrue( x2_uid in context._context._registered_leaf_nodes )
        del x1
        self.assertTrue( x1_uid not in context._context._registered_leaf_nodes )
        self.assertTrue( x2_uid in context._context._registered_leaf_nodes )

        # self.assertEqual(1, len(c._registered_leaf_nodes) )
        # self.assertTrue( x2._node.uid in c._registered_leaf_nodes )
        
        # Correlation matrix is not trimmed when `x1` is destroyed
        self.assertEqual(2, len(x2._node.correlation) )
        self.assertTrue( x2_uid in x2._node.correlation )
        self.assertTrue( x1_uid in x2._node.correlation )
        
        self.assertEqual(2, len(x2._node.correlation) )
        self.assertEqual(1.0, x2._node.correlation[x2_uid] )
        
#-----------------------------------------------------
class ArcTrigTests(unittest.TestCase):
    def setUp(self):
        self.x = 0.7
        self.u = 0.3
        self.un = ureal(self.x,self.u)
        
    def test(self):
        PI = math.pi
        arg1 = 0.75             # 1st quadrant
        arg2 = arg1 + PI / 2    # 2nd quadrant
        arg3 = arg1-PI          # 3rd quadrant
        arg4 = arg1 - PI/2      # 4th quadrant 
        u = 0.05

        x1 = ureal(arg1,u)        
        x2 = ureal(arg2,u)        
        x3 = ureal(arg3,u)        
        x4 = ureal(arg4,u)

        # atan2 test ---------------------
        # u = y/x
        df_dy = lambda y,x: x/(x**2 + y**2)
        df_dx = lambda y,x: -y/(x**2 + y**2)
        
        x_cpt = result( cos(x1) )
        y_cpt = result( sin(x1) )
        x = atan2(y_cpt,x_cpt)
        
        cos_x1 = math.cos(arg1)
        sin_x1 = math.sin(arg1)
        
        equivalent(value(x),arg1,TOL)
        equivalent(u_component(x,x1),u,TOL)

        # mixed argument types
        x = atan2(y_cpt,cos_x1)
        equivalent(value(x),arg1,TOL)
        u_y = df_dy(sin_x1,cos_x1) * uncertainty(y_cpt)
        equivalent(u_component(x,y_cpt),u_y,TOL)

        x = atan2(sin_x1,x_cpt)
        equivalent(value(x),arg1,TOL)
        u_x = df_dx(sin_x1,cos_x1) * uncertainty(x_cpt)
        equivalent(u_component(x,x_cpt),u_x,TOL)
        
        # intermediate -----
        x = atan2(y_cpt,x_cpt)
        xv = value(x_cpt)
        yv = value(y_cpt)
        den = xv**2 + yv**2
        equivalent(
            u_component(x,x_cpt),
            uncertainty(x_cpt)* -yv/den,
            TOL)
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)* xv/den,
            TOL)

        # illegal 
        self.assertRaises(TypeError,atan2,y_cpt,1+5j) 
        self.assertRaises(TypeError,atan2,1+5j,x_cpt,) 
        
        # trivial 
        y = atan2(ureal(0,1),ureal(0,1))
        self.assertEqual( value(y), 0 )
        self.assertEqual( uncertainty(y), 0 )
        self.assertEqual( dof(y), inf )   
 
        y = atan2(0,ureal(0,1))
        self.assertEqual( value(y), math.atan2(0,0) )
        self.assertEqual( uncertainty(y), 0 )
        self.assertEqual( dof(y), inf )   
 
        y = atan2(ureal(0,1),0)
        self.assertEqual( value(y), math.atan2(0,0) )
        self.assertEqual( uncertainty(y), 0 )
        self.assertEqual( dof(y), inf ) 
        
        # ----------------------        
        x_cpt = result( cos(x2) )
        y_cpt = result( sin(x2) )
        x = atan2(y_cpt,x_cpt)
        
        cos_x = math.cos(arg2)
        sin_x = math.sin(arg2)
        equivalent(value(x),arg2,TOL)
        equivalent(u_component(x,x2),u,TOL)
        
        # intermediate -----
        xv = value(x_cpt)
        yv = value(y_cpt)
        den = xv**2 + yv**2
        equivalent(
            u_component(x,x_cpt),
            uncertainty(x_cpt)* -yv/den,
            TOL)
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)* xv/den,
            TOL)

        # mixed argument types
        x = atan2(y_cpt,cos_x)
        equivalent(value(x),arg2,TOL)
        u_y = df_dy(sin_x,cos_x) * uncertainty(y_cpt)
        equivalent(u_component(x,y_cpt),u_y,TOL)

        x = atan2(sin_x,x_cpt)
        equivalent(value(x),arg2,TOL)
        u_x = df_dx(sin_x,cos_x) * uncertainty(x_cpt)
        equivalent(u_component(x,x_cpt),u_x,TOL)
        
        # ----------------------        
        x_cpt = result( cos(x3) )
        y_cpt = result( sin(x3) )
        x = atan2(y_cpt,x_cpt)
        
        cos_x = math.cos(arg3)
        sin_x = math.sin(arg3)
        equivalent(value(x),arg3,TOL)
        equivalent(u_component(x,x3),u,TOL)

        # intermediate -----
        xv = value(x_cpt)
        yv = value(y_cpt)
        den = xv**2 + yv**2
        equivalent(
            u_component(x,x_cpt),
            uncertainty(x_cpt)* -yv/den,
            TOL)
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)* xv/den,
            TOL)

        # mixed argument types
        x = atan2(y_cpt,cos_x)
        equivalent(value(x),arg3,TOL)
        u_y = df_dy(sin_x,cos_x) * uncertainty(y_cpt)
        equivalent(u_component(x,y_cpt),u_y,TOL)

        x = atan2(sin_x,x_cpt)
        equivalent(value(x),arg3,TOL)
        u_x = df_dx(sin_x,cos_x) * uncertainty(x_cpt)
        equivalent(u_component(x,x_cpt),u_x,TOL)
        
        # ----------------------        
        x_cpt = result( cos(x4) )
        y_cpt = result( sin(x4) )
        x = atan2(y_cpt,x_cpt)
        
        cos_x = math.cos(arg4)
        sin_x = math.sin(arg4)
        equivalent(value(x),arg4,TOL)
        equivalent(u_component(x,x4),u,TOL)

        # intermediate -----
        xv = value(x_cpt)
        yv = value(y_cpt)
        den = xv**2 + yv**2
        equivalent(
            u_component(x,x_cpt),
            uncertainty(x_cpt)* -yv/den,
            TOL)
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)* xv/den,
            TOL)

        # mixed argument types
        x = atan2(y_cpt,cos_x)
        equivalent(value(x),arg4,TOL)
        u_y = df_dy(sin_x,cos_x) * uncertainty(y_cpt)
        equivalent(u_component(x,y_cpt),u_y,TOL)

        x = atan2(sin_x,x_cpt)
        equivalent(value(x),arg4,TOL)
        u_x = df_dx(sin_x,cos_x) * uncertainty(x_cpt)
        equivalent(u_component(x,x_cpt),u_x,TOL)
        
        # asin test ---------------------------------
        y_cpt = result( sin(x1) )
        x = asin(y_cpt)
        equivalent(value(x),arg1,TOL)
        equivalent(uncertainty(x),u,TOL)
        equivalent(u_component(x,x1),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)
        

        # Maps to PI/2 - arg1        
        y_cpt = result( sin(x2) )
        x = asin(y_cpt)
        equivalent(value(x),PI/2-arg1,TOL)
        equivalent(u_component(x,x2),-u,TOL)
        equivalent(uncertainty(x),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        # Maps to  - arg1        
        y_cpt = result( sin(x3) )
        x = asin(y_cpt)
        equivalent(value(x),-arg1,TOL)
        equivalent(u_component(x,x3),-u,TOL)
        equivalent(uncertainty(x),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        y_cpt = result( sin(x4) )
        x = asin(y_cpt)
        equivalent(value(x),arg4,TOL)
        equivalent(u_component(x,x4),u,TOL)
        equivalent(uncertainty(x),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        # acos test ---------------------------------
        y_cpt = result( cos(x1) )
        x = acos(y_cpt)
        equivalent(value(x),arg1,TOL)
        equivalent(u_component(x,x1),u,TOL)
        equivalent(uncertainty(x),u,TOL)

        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            -uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        y_cpt = result( cos(x2) )
        x = acos(y_cpt)
        equivalent(value(x),arg2,TOL)
        equivalent(u_component(x,x2),u,TOL)
        equivalent(uncertainty(x),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            -uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        # Maps to  PI - arg1        
        y_cpt = result( cos(x3) )
        x = acos(y_cpt)
        equivalent(value(x),PI - arg1,TOL)
        equivalent(u_component(x,x3),-u,TOL)
        equivalent(uncertainty(x),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            -uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        # Maps to  PI/2 - arg1        
        y_cpt = result( cos(x4) )
        x = acos(y_cpt)
        equivalent(value(x),PI/2 - arg1,TOL)
        equivalent(u_component(x,x4),-u,TOL)
        equivalent(uncertainty(x),u,TOL)
        
        # intermediate -----
        equivalent(
            u_component(x,y_cpt),
            -uncertainty(y_cpt)/math.sqrt(1-value(y_cpt)**2),
            TOL)

        # atan test ---------------------------------
        t = result( tan(x1) )
        x = atan(t)
        equivalent(value(x),arg1,TOL)
        equivalent(u_component(x,x1),u,TOL)
        equivalent(uncertainty(x),u,TOL)

        # intermediate -----
        equivalent(
            u_component(x,t),
            uncertainty(t)/(1+value(t)**2),
            TOL)

        # mapped to arg1 - PI/2
        t = result( tan(x2) )
        x = atan(t)
        equivalent(value(x),arg1 - PI/2,TOL)
        equivalent(u_component(x,x2),u,TOL)
        equivalent(uncertainty(x),u,TOL)

        # intermediate -----
        equivalent(
            u_component(x,t),
            uncertainty(t)/(1+value(t)**2),
            TOL)

        # mapped to arg1
        t = result( tan(x3) )
        x = atan(t)
        equivalent(value(x),arg1,TOL)
        equivalent(u_component(x,x3),u,TOL)
        equivalent(uncertainty(x),u,TOL)

        # intermediate -----
        equivalent(
            u_component(x,t),
            uncertainty(t)/(1+value(t)**2),
            TOL)

        t = result( tan(x4) )
        x = atan(t)
        equivalent(value(x),arg4,TOL)
        equivalent(u_component(x,x4),u,TOL)
        equivalent(uncertainty(x),u,TOL)

        # intermediate -----
        equivalent(
            u_component(x,t),
            uncertainty(t)/(1+value(t)**2),
            TOL)

        # illegal input values -----------------------
        illegal1 = ureal(1.01,1)
        illegal2 = ureal(-1.01,1)

        self.assertRaises(ValueError,asin,illegal1)        
        self.assertRaises(ValueError,asin,illegal2)        
        self.assertRaises(ValueError,acos,illegal1)        
        self.assertRaises(ValueError,acos,illegal2)        

#-----------------------------------------------------
class HyperbolicTrigTestsReal(unittest.TestCase):
    def setUp(self):
        self.x = 0.44
        self.u = 0.13
        self.un = ureal(self.x,self.u)

        self.x1 = -0.43
        self.u1 = 0.2
        self.un1 = ureal(self.x1,self.u1)

        self.x2 = self.x * self.x1
        self.un2 = result( self.un * self.un1 )
        
    def testhSine(self):
        y = sinh(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, math.sinh( self.x ), TOL )
        equivalent( u, math.cosh( self.x ) * self.u, TOL )
        self.assertTrue( math.isinf(df) )

        # This will test an intermediate component of uncertainty
        vv = math.sinh(self.x2)
        uu = math.cosh(self.x2) * uncertainty( self.un2 )
        y = sinh( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )
        
    def testhCosine(self):
        y = cosh(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, math.cosh( self.x ),TOL )
        equivalent( u, math.sinh( self.x) * self.u, TOL )
        self.assertTrue( math.isinf(df) )
        
        # This will test an intermediate component of uncertainty
        vv = math.cosh(self.x2)
        uu = math.sinh(self.x2) * uncertainty( self.un2 )
        y = cosh( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( abs( uu ), uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )

    def testhTangent(self):
        y = tanh(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, math.tanh( self.x ), TOL )
        uu = self.u/(math.cosh( self.x )**2)
        equivalent( u, uu, TOL )
        self.assertTrue( math.isinf(df) )

        # This will test an intermediate component of uncertainty
        vv = math.tanh(self.x2)
        uu = uncertainty( self.un2 ) /(math.cosh( self.x2 )**2)
        y = tanh( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )

#-----------------------------------------------------
class InverseHyperbolicTrigTestsReal(unittest.TestCase):
    def setUp(self):
        self.x = 1.44
        self.u = 0.13
        self.un = ureal(self.x,self.u)

        self.x1 = 2.03
        self.u1 = 0.2
        self.un1 = ureal(self.x1,self.u1)

        self.x2 = self.x * self.x1
        self.un2 = result( self.un * self.un1 )

        self.x3 = 0.44
        self.u3 = 0.23
        self.un3 = ureal(self.x3,self.u3)

        self.x4 = .13
        self.u4 = 0.2
        self.un4 = ureal(self.x4,self.u4)

        self.x5 = self.x3 * self.x4
        self.un5 = result( self.un3 * self.un4 )
        
    def testArcSineh(self):
        log_form = lambda x: math.log(x + math.sqrt(1 + x**2))
        derivative = lambda x: 1./math.sqrt(1 + x**2)
        
        y = asinh(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, log_form( self.x ), TOL )
        equivalent( u, derivative( self.x ) * self.u, TOL )
        self.assertTrue( math.isinf(df) )

        # This will test an intermediate component of uncertainty
        vv = log_form(self.x2)
        uu = derivative(self.x2) * uncertainty( self.un2 )
        y = asinh( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )
        
    def testArcCosineh(self):
        log_form = lambda x: math.log(x + math.sqrt(x**2 - 1))
        derivative = lambda x: 1./(math.sqrt(x-1) * math.sqrt(x+1))

        y = acosh(self.un)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, log_form( self.x ),TOL )
        equivalent( u, derivative( self.x) * self.u, TOL )
        self.assertTrue( math.isinf(df) )
        
        # This will test an intermediate component of uncertainty
        vv = log_form(self.x2)
        uu = derivative(self.x2) * uncertainty( self.un2 )
        y = acosh( self.un2 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( abs( uu ), uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un2), TOL )

    def testArcTangenth(self):
        log_form = lambda x: math.log((1+x)/(1-x))/2.
        derivative = lambda x: 1./(1 - x**2)

        y = atanh(self.un3)
        v = value(y)
        u = uncertainty(y)
        df = dof(y)
        equivalent( v, log_form( self.x3 ), TOL )
        uu = derivative(self.x3) * self.u3
        equivalent( u, uu, TOL )
        self.assertTrue( math.isinf(df) )

        # This will test an intermediate component of uncertainty
        vv = log_form(self.x5)
        uu = uncertainty( self.un5 ) * derivative( self.x5 )
        y = atanh( self.un5 )
        v = value(y)
        equivalent( v,  vv, TOL  )
        equivalent( uu, uncertainty(y), TOL )
        equivalent( uu, u_component(y,self.un5), TOL )
        
#-----------------------------------------------------
class GuideExampleH2(unittest.TestCase):

    def test_no_dof(self):
        TOL = 1E-10

        v = ureal(4.999,0.0032,independent=False)
        i = ureal(0.019661,0.0000095,independent=False)
        phi = ureal(1.04446,0.00075,independent=False)

        set_correlation(-0.36,v,i)
        set_correlation(0.86,v,phi)
        set_correlation(-0.65,i,phi)
        
        r = v * cos(phi)/ i
        x = v * sin(phi)/ i
        z = v / i

        equivalent( uncertainty(r),0.0699787279884,TOL)
        equivalent( uncertainty(x),0.295716826846,TOL)
        equivalent( uncertainty(z),0.236602971835,TOL)

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),uncertainty(r),TOL)
        equivalent( math.sqrt(welch_satterthwaite(x)[0]),uncertainty(x),TOL)
        equivalent( math.sqrt(welch_satterthwaite(z)[0]),uncertainty(z),TOL)

        equivalent( get_correlation(r,x),-0.591484610819,TOL)
        equivalent( get_correlation(x,z),0.992797472722,TOL)
        equivalent( get_correlation(r,z),-0.490623905441,TOL)

    def test_with_dof(self):
        TOL = 1E-10

        # It is illegal to do this now
        # v = ureal(4.999,0.0032,5,independent=False)
        # i = ureal(0.019661,0.0000095,5,independent=False)
        # phi = ureal(1.04446,0.00075,5,independent=False)
        # Instead, we must use multiple_ureal
        
        # Once the inputs are in an ensemble we can calculate dof
        x = [4.999,0.019661,1.04446]
        u = [0.0032,0.0000095,0.00075]
        v,i,phi = multiple_ureal(x,u,5)
        
        set_correlation(-0.36,v,i)
        set_correlation(0.86,v,phi)
        set_correlation(-0.65,i,phi)

        r = v * cos(phi)/ i
        x = v * sin(phi)/ i
        z = v / i

        # The presence of finite DoF should not alter previous results
        equivalent( uncertainty(r),0.0699787279884,TOL)
        equivalent( uncertainty(x),0.295716826846,TOL)
        equivalent( uncertainty(z),0.236602971835,TOL)

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),math.sqrt( std_variance_real(r) ),TOL)
        equivalent( math.sqrt(welch_satterthwaite(x)[0]),uncertainty(x),TOL)
        equivalent( math.sqrt(welch_satterthwaite(z)[0]),uncertainty(z),TOL)

        equivalent( get_correlation(r,x),-0.591484610819,TOL)
        equivalent( get_correlation(x,z),0.992797472722,TOL)
        equivalent( get_correlation(r,z),-0.490623905441,TOL)

        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL)
        
        # Add another influence and it breaks again, but earlier results OK
        f = ureal(1,0.0032,5,independent=False)
        self.assertRaises(RuntimeError,set_correlation,0.3,f,i)
        
        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL)

    def test_with_dof_2(self):
        """
        Same test as above, but now ensure that the 
        ensemble calculation can deal with influences 
        that are not in sequence
        """
        TOL = 1E-10

        # The dummy variables mean that those included 
        # in the ensemble will not be a consecutive 
        # series of IDs, provided we include the 
        # dummies in the equations, as is done below
        # with zero weight.
        v = ureal(4.999,0.0032,5,independent=False)
        dummy_1 = ureal(1,1)
        i = ureal(0.019661,0.0000095,5,independent=False)
        dummy_2 = ureal(1,1)
        phi = ureal(1.04446,0.00075,5,independent=False)
        dummy_3 = ureal(1,1)

        real_ensemble( [v,i,phi], 5 )

        set_correlation(-0.36,v,i)
        set_correlation(0.86,v,phi)
        set_correlation(-0.65,i,phi)

        r = v * cos(phi)/ i + (0*dummy_1)
        x = v * sin(phi)/ i + (0*dummy_2)
        z = v / i + (0*dummy_3)

        # The presence of finite DoF should not alter previous results
        equivalent( uncertainty(r),0.0699787279884,TOL)
        equivalent( uncertainty(x),0.295716826846,TOL)
        equivalent( uncertainty(z),0.236602971835,TOL)

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),math.sqrt( std_variance_real(r) ),TOL)
        equivalent( math.sqrt(welch_satterthwaite(x)[0]),uncertainty(x),TOL)
        equivalent( math.sqrt(welch_satterthwaite(z)[0]),uncertainty(z),TOL)

        equivalent( get_correlation(r,x),-0.591484610819,TOL)
        equivalent( get_correlation(x,z),0.992797472722,TOL)
        equivalent( get_correlation(r,z),-0.490623905441,TOL)

        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL) 
        
    def test_with_dof_multiple_ureal(self):
        TOL = 1E-10

        x = [4.999,0.019661,1.04446]
        u = [0.0032,0.0000095,0.00075]
        labels = ['v','i','phi']

        v,i,phi = multiple_ureal(x,u,5,labels)
        
        set_correlation(-0.36,v,i)
        set_correlation(0.86,v,phi)
        set_correlation(-0.65,i,phi)

        r = v * cos(phi)/ i
        x = v * sin(phi)/ i
        z = v / i

        # The presence of finite DoF should not alter previous results
        equivalent( uncertainty(r),0.0699787279884,TOL)
        equivalent( uncertainty(x),0.295716826846,TOL)
        equivalent( uncertainty(z),0.236602971835,TOL)

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),math.sqrt( std_variance_real(r) ),TOL)
        equivalent( math.sqrt(welch_satterthwaite(x)[0]),uncertainty(x),TOL)
        equivalent( math.sqrt(welch_satterthwaite(z)[0]),uncertainty(z),TOL)

        equivalent( get_correlation(r,x),-0.591484610819,TOL)
        equivalent( get_correlation(x,z),0.992797472722,TOL)
        equivalent( get_correlation(r,z),-0.490623905441,TOL)

        # Dof calculation should be legal
        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL)
        
        # Add another influence and it breaks, but earlier results OK
        f = ureal(1,0.0032,5,independent=False)
        self.assertRaises(RuntimeError,set_correlation,0.3,f,i)
                
        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL)
        

#-----------------------------------------------------
class TestMultipleUNsWithRealConstants(unittest.TestCase):
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
    def test_ureal(self):
    
        x = [4.999,19.661E-3,1.04446]
        u = [3.2E-3,0,7.5E-4]
        labels = ['V','I','phi']
        
        v,i,phi = multiple_ureal(x,u,4,labels)
        
        self.assertTrue(v.x == x[0])
        self.assertTrue(v.u == u[0])
        self.assertTrue(v.df == 4)

        self.assertTrue(i.x == x[1])
        self.assertTrue(i.u == u[1])
        self.assertTrue(i.df == inf)
        
        self.assertTrue(phi.x == x[2])
        self.assertTrue(phi.u == u[2])
        self.assertTrue(phi.df == 4)

        # We can set a zero correlation ...
        set_correlation(0.0,v,i)
        set_correlation(0.0,i,phi)
        set_correlation(0.0,phi,v)
        
        # but not a finite one to a constant
        self.assertRaises(RuntimeError,set_correlation,0.1,v,i)
        self.assertRaises(RuntimeError,set_correlation,0.1,phi,i)
        set_correlation(0.1,phi,v)

#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
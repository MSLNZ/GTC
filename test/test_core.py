import unittest
import warnings 

from GTC import *

from testing_tools import *

TOL = 1E-13 

#----------------------------------------------------------------------------
class TestAPIFunctions(unittest.TestCase):
    """
    Test some general functions:
        - component
        - constant
        - variance
        - ureal + ucomplex check argument types
        - ureal properties: real, imag 
    """
    def test_ureal_float_interface(self):
        x = ureal(1.0,1.0)
        self.assertTrue( equivalent(x.x,x.real.x) )
        self.assertTrue( equivalent(x.u,x.real.u) )
        self.assertTrue( x is x.real )
        
        self.assertTrue( equivalent(0,x.imag.x) )
        self.assertTrue( equivalent(0,x.imag.u) )
       
        y = x + ureal(2,2)
        self.assertTrue( equivalent(y.x,y.real.x) )
        self.assertTrue( equivalent(y.u,y.real.u) )
        self.assertTrue( y is y.real )

        self.assertTrue( equivalent(0,y.imag.x) )
        self.assertTrue( equivalent(0,y.imag.u) )
        
    def test_ureal_arg_typs(self):
        """
        `ureal` should not accept anything but number types
        in its numerical arguments
        
        """
        x = "s"
        u = 0.1
        nu = 6.5
        self.assertRaises(TypeError,ureal,x,u,nu)
        x = ureal(1,1)
        self.assertRaises(TypeError,ureal,x,u,nu)
        x = 1
        u = 'd'
        self.assertRaises(TypeError,ureal,x,u,nu)
        u = ureal(1,1)
        self.assertRaises(TypeError,ureal,x,u,nu)
        u = 1
        nu = 'd'
        self.assertRaises(TypeError,ureal,x,u,nu)
        nu = ureal(1,1)
        self.assertRaises(TypeError,ureal,x,u,nu)
        
    def test_ucomplex_arg_typs(self):
        """
        `ureal` should not accept anything but number types
        in its numerical arguments
        
        """
        x = "s"
        u = (0.1,0.2)
        nu = 6.5
        self.assertRaises(TypeError,ucomplex,x,u,nu)
        x = ucomplex(1,1)
        self.assertRaises(TypeError,ucomplex,x,u,nu)
        x = 1
        u = 'd'
        self.assertRaises(TypeError,ucomplex,x,u,nu)
        u = ucomplex(1,1)
        self.assertRaises(TypeError,ucomplex,x,u,nu)
        u = 1
        nu = 'd'
        self.assertRaises(TypeError,ucomplex,x,u,nu)
        nu = ucomplex(1,1)
        self.assertRaises(TypeError,ucomplex,x,u,nu)
        
    def test_component(self):
        """
        component() should return the absolute value of a real component of uncertainty
        it should return the u_bar value of a complex component of uncertainty
        and it should return 0 if either argument is just a number.
        
        """
        x1 = ureal(1,1)
        x2 = -x1
        c = component(x2,x1)
        self.assertNotEqual(c,reporting.u_component(x2,x1))
        self.assertEqual(c,abs(reporting.u_component(x2,x1)))

        z1 = ucomplex(1+1j,(2,3))
        z2 = -z1
        c = component(z2,z1)
        uc = reporting.u_component(z2,z1)
        self.assertEqual(c,reporting.u_bar(uc))

        self.assertEqual(component(z2,3),0)
        self.assertEqual(component(x2,3),0)

    def test_constant(self):
        """
        A constant has no uncertainty
        it is of the uncertain type associated with its value type
        it may be labelled
        
        """
        x = constant(3)
        self.assertEqual( label(x), None )
        self.assertEqual( value(x),3) 
        self.assertEqual( uncertainty(x),0)
        
        x = constant(3,label='x')
        self.assertEqual( label(x), 'x' )
        self.assertEqual( value(x),3) 
        self.assertEqual( uncertainty(x),0)

        z = constant(3+1j)
        self.assertEqual( value(z),3+1j) 
        self.assertEqual( uncertainty(z),(0,0) )
        self.assertEqual( label(z), None )

        z = constant(3+1j,label='z')
        self.assertEqual( value(z),3+1j) 
        self.assertEqual( uncertainty(z),(0,0) )
        self.assertEqual( label(z), 'z' )

    def test_variance(self):
        """
        variance returns the std variance of a real number,
        or the std cv of a complex number,
        or 0
        
        """
        x = ureal(1,2)
        self.assertEqual(variance(x),4)

        cv = (5,3,3,4)        
        z = ucomplex(1+1j,cv)
        cv_z = variance(z)

        for i in range(4):
            self.assertTrue( equivalent(cv[i],cv_z[i],TOL) )
            
        self.assertEqual(variance(4),0)

    def test_get_correlation(self):
        # If the product of variances is zero but the covariance
        # is also zero, then return a zero correlation coefficient.
        #
        l = constant(1)
        r = ureal(1,1) * ureal(1,1)
        cc = get_correlation(l,r)
        self.assertTrue( equivalent(cc,0.0) )
        
        # Asking for correlation wrt a number gives zero
        #
        x = ureal(1,1)
        cc = get_correlation(x,1.5)
        self.assertTrue( equivalent(cc,0.0) )
 
        z = ucomplex(1,1)
        cc = get_correlation(z,1.5+6j)
        self.assertTrue( equivalent_sequence(cc,[0.0,0.0,0.0,0.0]) )

        cc = get_correlation(1.5+6j)
        self.assertTrue( equivalent(cc,0.0) )

        # You can have complex number (implied) with no imaginary
        # component. If that is the only argument to get_correlation
        # then return zero
        cc = get_correlation(1.5)
        self.assertTrue( equivalent(cc,0.0) )
     
        # uc = ucomplex(1 + 1j, 0.1)
        # ur = ureal(2, 0.2)
        # r = 0.3
        # self.assertRaises(TypeError,set_correlation,r,uc,ur)
        
        # uc = ucomplex(1 + 1j, 0.1,independent=False)
        # ur = ureal(2, 0.2,independent=False)
        # set_correlation(r,uc,ur)
        # rr = get_correlation(uc,ur)
        
        
    def test_result(self):
            # `result()` should create a clone of the uncertain 
            # number object passed as an argument
            x = ureal(1.5,1)
            y = ureal(2,2)
            z = x**y 
            z_ = result(z)
            
            self.assertTrue( z is not z_ )
            self.assertTrue( equivalent(z.x,z_.x) )
            self.assertTrue( equivalent(z.u,z_.u) )
            self.assertTrue( equivalent(z.df,z_.df) )
            self.assertEqual( len(z._u_components), len(z_._u_components)) 
            self.assertEqual( len(z._i_components)+1, len(z_._i_components))    # a reference to self is added
            self.assertTrue( equivalent_sequence(z._u_components.values(),z_._u_components.values()) )
            
            x = ucomplex(1.2+.4j,1)
            y = ucomplex(-.5+1.1j,1)
            z = x * y 
            z_ = result(z)
            
            self.assertTrue( z is not z_ )
            self.assertTrue( equivalent_complex(z.x,z_.x) )
            self.assertTrue( equivalent_sequence(z.u,z_.u) )
            self.assertTrue( equivalent(z.df,z_.df) )
            self.assertEqual( len(z.real._u_components), len(z_.real._u_components)) 
            self.assertEqual( len(z.imag._u_components), len(z_.imag._u_components)) 
            self.assertEqual( len(z.real._i_components)+1, len(z_.real._i_components)) 
            self.assertEqual( len(z.imag._i_components)+1, len(z_.imag._i_components)) 
            self.assertTrue( equivalent_sequence(z.real._u_components.values(),z_.real._u_components.values()) )
            self.assertTrue( equivalent_sequence(z.imag._u_components.values(),z_.imag._u_components.values()) )

            # When applied to an elementary uncertain number nothing will happen 
            # unless the `label` argument is used. In that case, if the elementary 
            # uncertain number label is already set a warning is issued, 
            # otherwise label is assigned to the uncertain number.
            x = ureal(1.5,1)
            xx = result(x)
            self.assertTrue( x is xx )
            
            xx = result(x,label = 'x')
            self.assertTrue( x is xx )
            self.assertEqual( x.label, 'x' )
            
            x = ureal(1.5,1,label='x')
            xx = result(x,label='x')
            self.assertTrue( x is xx )
            self.assertEqual( x.label, 'x' )
 
            # From warnings documentation 2.7.16
            # Note that with Python 3 there is `assertWarns`
            with warnings.catch_warnings(record=True) as w:            
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                
                # Trigger a warning.
                xx = result(x,'xx')

                self.assertTrue( len(w) == 1 )
                self.assertTrue( issubclass(w[-1].category, RuntimeWarning) )
                self.assertTrue( x is xx )
                self.assertEqual( x.label, 'x' )
                
#----------------------------------------------------------------------------
class TestMultipleUN(unittest.TestCase):

    """
    """
    
    def test_multiple_ureal_creation(self):
        values = [4.999, 0.019661]
        uncert = [0.0032, 0.0000095]
        # this raises a RuntimeError: invalid `label_seq`: 'abc'
        self.assertRaises(RuntimeError,multiple_ureal,values, uncert, 5, label_seq='abc')
    
    def test_multiple_ucomplex_creation(self):
        values = [4.999 + 0j, 0.019661 + 0j]
        uncert = [(0.0032, 0.0), (0.0000095, 0.0)]
        # this raises a RuntimeError: invalid `label_seq`: 'abc'
        self.assertRaises(RuntimeError,multiple_ureal,values, uncert, 5, label_seq='abc')
    
#----------------------------------------------------------------------------
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

    def test_illegal_set_correlation(self):
        x1 = ureal(1,2,3,independent=False)
        z1 = ucomplex(1,.5,3,independent=False)
        x2 = ureal(1,2,3,independent=False)
        z2 = ucomplex(1,.5,3,independent=False)
        
        r = 0.5
        
        self.assertRaises(RuntimeError,set_correlation,r,x1,x2)
        self.assertTrue( set_correlation(r,z1) is None )
        
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
        
#----------------------------------------------------------------------------
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
        
#----------------------------------------------------------------------------
class TestPolymorphicMaths(unittest.TestCase):
    """
    math functions will assume that an uncertain number argument is provided,
    but should then try to find a suitable math or cmath library function 
    instead, finally raising TypeError if nothing can be done
    """
    def test_unary(self):
        unary_fns = [
            ('cos', 1.0, 1.0+.5j)
        ,   ('sin', 1.0, 1.0+.5j)
        ,   ('tan', 1.0, 1.0+.5j)
        ,   ('acos', 0.5, 0.5+0.5j)
        ,   ('asin', 0.5, 0.5+0.5j)
        ,   ('atan', 0.5, 0.5+0.5j)
        ,   ('exp', -0.5, -0.5+0.5j)
        ,   ('log', 0.5, 0.5+0.5j)
        ,   ('log10', 0.5, 0.5+0.5j)
        ,   ('sqrt',  0.5, 0.5+0.5j)
        ,   ('sinh', 1.0, 1.0+.5j)
        ,   ('cosh', 1.0, 1.0+.5j)
        ,   ('tanh', 1.0, 1.0+.5j)
        ,   ('acosh', 1.5, 1.5+0.5j)
        ,   ('asinh', 1.5, 1.5+0.5j)
        ,   ('atanh', 0.5, 0.5+0.5j)
        ]
        # This is intended to test the selection of 
        # an appropriate library or exception, not numerical accuracy.
        for d in unary_fns:
            self.assertTrue( equivalent(
                eval("{}({})".format(d[0],d[1])),
                eval("math.{}({})".format(d[0],d[1]))
            ) )
            self.assertTrue( equivalent_complex(
                eval("{}({})".format(d[0],d[2])),
                eval("cmath.{}({})".format(d[0],d[2]))
            ) )
            eval("self.assertRaises(TypeError,{},'s')".format(d[0]))
            
    def test_special_cases(self):
        """
        Note magnitude, mag_squared and pow do not use libraries
        
        """
        # phase:
        x = 1.0+1j
        self.assertTrue(
            equivalent(cmath.phase(x), phase(x) )
        )
        self.assertRaises(TypeError,phase,'z')
        
        # atan2:
        y, x = 1.0, 1.0
        self.assertTrue(
            equivalent(math.atan2(y,x), atan2(y,x))
        )
        self.assertRaises(TypeError,atan2,y,'x')
        self.assertRaises(TypeError,atan2,'y',x)
        self.assertRaises(TypeError,atan2,.5+.5j,.2+.2j)
       
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
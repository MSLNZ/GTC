import unittest

from GTC import *
from GTC.lib import (
    UncertainReal,
    complex_ensemble
)

from testing_tools import *

TOL = 1E-13 

#-----------------------------------------------------
class TestWillinkHall(unittest.TestCase):

    def test_pathological_cases(self):
        x1 = ucomplex( 0, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 0, (0.51,0.33,0.33,0.31), 3 )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 6 )

        # The measurement equation
        x = x1*x2*x3

        equivalent( dof(x1), 5, 1E-15)
        equivalent( dof(x2), 3, 1E-15)
        equivalent( dof(x3), 6, 1E-15)
        # The components of uncertainty will all be zero
        self.assertTrue( dof(x) is nan )
        
        # Illegal correlation
        v,i = multiple_ucomplex(
            [4.999+0j,0.019661+0j],
            [(0.0032,0.0),(0.0000095,0.0)],
            5
        )
        phi = ucomplex(1.04446j,(0.0,0.00075),5,independent=False)

        self.assertRaises(RuntimeError,set_correlation,0.86,v.real,phi.imag)
        
    def testDoF1(self):
        # Test case from Metrologia 2002,39,365
        # Here define variables with covariance matrices
        TOL = 1E-2
        
        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31), 3 )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 6 )

        # The measurement equation
        x = x1 + x2 + x3
        
        equivalent( dof(x1), 5, 1E-15)
        equivalent( dof(x2), 3, 1E-15)
        equivalent( dof(x3), 6, 1E-15)

        self.assertTrue( equivalent( dof(x), 11.34 , TOL ) )

    def testDoF2(self):
        
        # Here define variables with standard uncertainties
        # and correlation coefficients

        TOL = 1E-2
        q1 = ucomplex( 1, (0.9789,0.51962), 5,independent=False )
        q2 = ucomplex( 1, (0.71414,0.5567), 3,independent=False )
        q3 = ucomplex( 1, (0.67082,1.28452), 6,independent=False )

        set_correlation( -0.6678, q1 )
        set_correlation( 0.82994, q2 )
        set_correlation( 0.32494, q3 )

        x = q1 + q2 + q3;
        
        self.assertTrue( equivalent( dof(x), 11.34 , TOL ) )

    def testDoF3(self):
        # Here we use a mixture of float and complex
        # uncertain numbers and expect to get the
        # usual GUM result for Example H1.
        
        TOL = 1E-10

        d_1 = ucomplex(0.0,(5.8,0),24)
        d_2 = ureal(0.0,3.9,5)
        d_3 = ucomplex(0.0,(6.7,0.0),8)

        # Note values from the guide are only approximate
        d = d_1 + d_2 + d_3
        df = dof(d)
        self.assertTrue( equivalent(df,25.4472507774,TOL) )

        alpha_s = ureal(0.0000115,tb.uniform(0.000002))
        delta_alpha = ureal(0.0,tb.uniform(0.000001),50)
        
        theta_1 = ucomplex(-0.1,(0.2,0.0))
        theta_2 = ucomplex(0.0,(tb.arcsine(0.5),0.0))
        delta_theta = ucomplex(0.0,(tb.uniform(0.05),0.0),2)

        Ls = ucomplex(5e7,(25.0,0.0),18)        

        theta = theta_1 + theta_2 
        df = dof(theta)
        self.assertTrue( math.isinf(df) )

        x1 = Ls * delta_alpha * theta       
        x2 = Ls * alpha_s * delta_theta        
        y = Ls + d - (x1 + x2)
        df = dof(y)
        self.assertTrue( equivalent(df,16.7521475092,TOL) )
        
    # Do a few tests of the invalid calculation to try to
    # cover various limit cases within the algorithm
    def testDoF4(self):
        
        # Correlations should cause the calculation to abort

        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31), 5 )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 5 )
        x = x1 + x2 + x3
        
        self.assertRaises(RuntimeError,set_correlation,0.5,x1.real,x2.real)

        # If we declare that they are in an ensemble then its OK
        complex_ensemble( (x1,x2,x3), 5 )
        self.assertTrue( not math.isnan(dof(x)) )

    def testDoF5(self):
        # Correlations should cause the calculation to abort

        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31), 5 )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 5 )
        x = x1 + x2 + x3

        self.assertRaises(RuntimeError,set_correlation,0.5,x1.real,x3.imag)

        # If we declare that they are in an ensemble then its OK
        complex_ensemble( (x1,x2,x3), 5 )
        self.assertTrue( not math.isnan(dof(x)) )

    def testDoF6(self):
        # Correlations should cause the calculation to abort

        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31), 5 )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 5 )
        x = x1 + x2 + x3

        self.assertRaises(RuntimeError,set_correlation,0.5,x2.imag,x3.real)

        # If we declare that they are in an ensemble then its OK
        complex_ensemble( (x1,x2,x3), 5 )
        self.assertTrue( not math.isnan(dof(x)) )

    def testDoF7(self):
        # Correlations should cause the calculation to abort

        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31), 5 )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 5 )
        x = x1 + x2 + x3

        self.assertRaises(RuntimeError,set_correlation,0.5,x2.imag,x3.imag)

        # If we declare that they are in an ensemble then its OK
        complex_ensemble( (x1,x2,x3), 5 )
        self.assertTrue( not math.isnan(dof(x)) )

    def testDoF8(self):
        # Look at the numerical stability of the dof calculation
        
        TOL = 1E-14
        
        cv1 = (0.96,-0.34,-0.34,0.27)
        cv2 = (0.51,0.33,0.33,0.31)
        cv3 = (0.45,0.28,0.28,1.65)
        
        scale = 1E-14

        scale_cv = lambda cv: [scale*x for x in cv ]
        
        x1 = ucomplex( 1, scale_cv(cv1), 5 )
        x2 = ucomplex( 1, scale_cv(cv2), 3 )
        x3 = ucomplex( 1, scale_cv(cv3), 6 )
        x = x1 + x2 + x3

        self.assertTrue( equivalent( dof(x), 11.3409777904914 , TOL ) )
        
    def testDoF9(self):
        TOL = 1E-5
        
        # v = ucomplex(4.999+0j,(0.0032,0.0),5,independent=False)
        # i = ucomplex(0.019661+0j,(0.0000095,0.0),5,independent=False)
        # phi = ucomplex(1.04446j,(0.0,0.00075),5,independent=False)
        
        v,i,phi = multiple_ucomplex(
            [4.999+0j,0.019661+0j,1.04446j],
            [(0.0032,0.0),(0.0000095,0.0),(0.0,0.00075)],
            5
        )

        set_correlation(-0.36,v.real,i.real)
        set_correlation(0.86,v.real,phi.imag)
        set_correlation(-0.65,i.real,phi.imag)

        z = v * exp(phi)/ i

        equivalent( uncertainty(z.real),0.0699787279884,TOL)
        equivalent( uncertainty(z.imag),0.295716826846,TOL)

        equivalent( get_correlation(z),-0.591484610819,TOL)

        equivalent( dof(z),5,TOL)

    def testDoF10(self):
        TOL = 1E-5
        
        values = [4.999+0j,0.019661+0j,1.04446j]
        uncert = [(0.0032,0.0),(0.0000095,0.0),(0.0,0.00075)]
        v,i,phi = multiple_ucomplex(values,uncert,5,('v','i','phi'))

        set_correlation(-0.36,v.real,i.real)
        set_correlation(0.86,v.real,phi.imag)
        set_correlation(-0.65,i.real,phi.imag)

        z = v * exp(phi)/ i

        equivalent( uncertainty(z.real),0.0699787279884,TOL)
        equivalent( uncertainty(z.imag),0.295716826846,TOL)

        equivalent( get_correlation(z),-0.591484610819,TOL)

        equivalent( dof(z),5,TOL)
        
        self.assertEqual(v.label,'v')
        self.assertEqual(i.label,'i')
        self.assertEqual(phi.label,'phi')

    def test_correlated_infinite_dof(self):
        # the routine should not have a problem when
        # influences with infinite dof are correlated

        # This is the same calculation as `testDoF7`
        # but with infinite dof for x2 and x3
        
        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31) )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65) )
        x = x1 + x2 + x3
        set_correlation(0.5,x2.imag,x3.imag)
        self.assertTrue( dof(x) is not nan )
 
    def test_correlated_ensemble(self):
        # test when non-consecutive influences 
        # are present in an ensemble
        
        TOL = 1E-5
        
        # Dummy UNs are created between the usual values
        v = ucomplex(4.999+0j,(0.0032,0.0),5,independent=False)
        dum1 = ucomplex( 1,1 )
        i = ucomplex(0.019661+0j,(0.0000095,0.0),5,independent=False)
        dum2 = ucomplex( 1,1 )
        phi = ucomplex(1.04446j,(0.0,0.00075),5,independent=False)

        complex_ensemble( (v,i,phi), 5 )

        set_correlation(-0.36,v.real,i.real)
        set_correlation(0.86,v.real,phi.imag)
        set_correlation(-0.65,i.real,phi.imag)

        # The dummy values are included with zero weight,
        # so the DoF calculation should not change.
        z = (v + 0*dum1) * exp(phi)/ (i + 0.0*dum2)

        equivalent( uncertainty(z.real),0.0699787279884,TOL)
        equivalent( uncertainty(z.imag),0.295716826846,TOL)

        equivalent( get_correlation(z),-0.591484610819,TOL)

        self.assertTrue( nan is not dof(z) )
        equivalent( dof(z),5,TOL)

    def test_illegal_complex_ensemble(self):
    
        # Check for inconsistent numbers of DoF 
        x1 = ucomplex( 1, (0.96,-0.34,-0.34,0.27), 5 )
        x2 = ucomplex( 1, (0.51,0.33,0.33,0.31) )
        x3 = ucomplex( 1, (0.45,0.28,0.28,1.65), 3 )
                 
        # Check for non-elementary UNs 
        # NB, this assertion may be removed in production
        y2 = x1 + x2 
        self.assertRaises(
            AssertionError,complex_ensemble,[x1,y2],5
        )        
              
    def testDoFMixedTypes(self):
        TOL = 1E-5
        
        # i = ucomplex(0.019661+0j,(0.0000095,0.0),5,independent=False)
        # phi = ucomplex(1.04446j,(0.0,0.00075),5,independent=False)
        
        i,phi = multiple_ucomplex(
            [0.019661+0j,1.04446j],
            [(0.0000095,0.0),(0.0,0.00075)],
            5
        )
        v = ureal(4.999,0.0032,5,independent=False)

        z = v * exp(phi)/ i
        
        # These are the values when correlation is ignored
        equivalent( uncertainty(z.real),0.194117890168,TOL)
        equivalent( uncertainty(z.imag),0.200665630894,TOL)

        equivalent( get_correlation(z),0.05820381031584,TOL)

        equivalent( dof(z),8.23564845495,TOL)              
#-----------------------------------------------------
class WelchSatterthwaiteExtensions(unittest.TestCase):
 
    def test_with_complex(self):
        """
        The real and imaginary components of a complex
        influence may be correlated
        
        """
        df_x1, df_x2, df_z1 = 5,6,7
        r = 0.1
        z1 = ucomplex(1,[1,r,r,1],df_z1)
        
        x1 = ureal(1,1,df_x1,independent=False)
        x2 = ureal(1,1,df_x2,independent=False)

        m = magnitude(z1)
        
        y = m + x1 + x2
        
        v_z1 = component(y,z1.real)**2\
               + component(y,z1.imag)**2\
               + 2*component(y,z1.real)*r*component(y,z1.imag)
        
        den = sum([
                component(y,x1)**4/df_x1,
                component(y,x2)**4/df_x2,
                v_z1**2/df_z1,
            ])
        nu_eff = variance(y)**2 / den
        
        df = dof(y)
        
        self.assertTrue( equivalent(nu_eff,df) )

        # but if others are correlated it fails
        self.assertRaises(RuntimeError,set_correlation,r,x1,x2)
        
    def test_with_complex_2(self):
        """
        The real and imaginary components of a complex
        quantity should behave as real UNs
        
        """
        x_re, u_x_re = -0.0152, 0.0040
        x_im, u_x_im = -0.0071, 0.0040 
        df_x = inf
        
        y_re, u_y_re = 0.0, 0.00057
        y_im, u_y_im = 0.0, 0.00072
        df_y = 3.0
        
        xre = ureal(x_re,u_x_re,df_x)
        xim = ureal(x_im,u_x_im,df_x)
        
        yre = ureal(y_re,u_y_re,df_y)
        yim = ureal(y_im,u_y_im,df_y)
        
        x = ucomplex(complex(x_re,x_im), [u_x_re,u_x_im], df_x)
        y = ucomplex(complex(y_re,y_im), [u_y_re,u_y_im], df_y)

        z = x + y
        zre = xre + yre
        zim = xim + yim
        
        self.assertTrue( dof(zre) != 0 )
        self.assertTrue( dof(zim) != 0 )
        
        self.assertTrue( equivalent(dof(z.real),dof(zre) ) )
        self.assertTrue( equivalent(dof(z.imag),dof(zim) ) )

    def test_with_infinity_1(self):
        """
        Influences may be correlated when they have
        infinite degrees of freedom
        
        """
        df_x3 = 5        
        x1 = ureal(1,1,independent=False)
        x2 = ureal(1,1,independent=False)
        x3 = ureal(1,1,df_x3)
        
        set_correlation(0.1,x1,x2)
        
        y = x1 + x2 + x3

        den = component(y,x3)**4/df_x3
        
        df = dof(y)
        nu_eff = variance(y)**2 / den
        
        self.assertTrue( equivalent(nu_eff,df))

    def test_with_infinity_2(self):
        """
        Complex components may be correlated when they have
        infinite degrees of freedom
        
        """
        df_x1, df_x2 = 5,6,
        r = 0.1
        x1 = ureal(1,1,df_x1)
        x2 = ureal(1,1,df_x2)
        z1 = ucomplex(1,[1,r,r,1])

        m = magnitude(z1)
        
        y = x1 + x2 + m
        
        den = sum([
                component(y,x1)**4/df_x1,
                component(y,x2)**4/df_x2,
            ])
        nu_eff = variance(y)**2 / den
        df = dof(y)
        
        self.assertTrue( equivalent(nu_eff,df))
        
#-----------------------------------------------------
class TestDoF(unittest.TestCase):

    """
    Additional tests related to DoF and ensemble calculations
    """

    def test_single_complex(self):
        """
        The DoF of a single complex quantity is a 
        special case in the WS algorithm 
        """
        z = ucomplex( (1+1j), [1.0,0.5,0.5,1.0], 6.0 )
        z_mag_sq = mag_squared(z)
        
        self.assertTrue( isinstance(z_mag_sq,UncertainReal) )
        self.assertTrue( equivalent(z_mag_sq.df,6) )
        

#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
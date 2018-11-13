import unittest

from GTC import *
from GTC.lib import (
    real_ensemble,
    welch_satterthwaite,
    UncertainReal
)

from testing_tools import *

TOL = 1E-13 

#----------------------------------------------------------------------------
class SimpleWSCases(unittest.TestCase):
    def test(self):
        x = ureal(1,1,4)
        self.assertEqual(4,x.df)
        self.assertEqual(4,welch_satterthwaite(x)[1] )
        
        # product with zero values 
        x1 = ureal(0,1,4)
        x2 = ureal(0,1,3)
        y = x1 * x2
        self.assertEqual(0,y.u)
        self.assertTrue(nan is y.df)
        
        # Pathological case - not sure it can be created in practice
        x1 = ureal(1,1)
        x2 = UncertainReal._elementary(1,0,4,label=None,independent=True)
        x3 = UncertainReal._elementary(1,0,3,label=None,independent=True)
        y = x1 + x2 + x3
        self.assertEqual( len(y._u_components), 3 )
        self.assertTrue(inf is y.df)

#----------------------------------------------------------------------------
class SimpleComplexMagnitude(unittest.TestCase):
    def test(self):
        mag = magnitude( ucomplex(1+1j,[.5,.1,.1,.5],4) )
        self.assertEqual(4,mag.df)
        
#----------------------------------------------------------------------------
class GuideExampleH1(unittest.TestCase):

    def setUp(self):
        self.d_1 = ureal(0.0,5.8,24)
        self.d_2 = ureal(0.0,3.9,5)
        self.d_3 = ureal(0.0,6.7,8)

        self.alpha_s = ureal(0.0000115,tb.uniform(0.000002))
        self.delta_alpha = ureal(0.0,tb.uniform(0.000001),50)
        
        self.theta_1 = ureal(-0.1,0.2)
        self.theta_2 = ureal(0.0,tb.arcsine(0.5))
        self.delta_theta = ureal(0.0,tb.uniform(0.05),2)

        self.Ls = ureal(5e7,25,18)        

    def test(self):
        TOL = 1E-10
        
        # Note values from the guide are only approximate
        d = self.d_1 + self.d_2 + self.d_3
        u = uncertainty(d)
        df = dof(d)
        equivalent(u,9.68194195397,TOL)
        equivalent(df,25.4472507774,TOL)

        theta = self.theta_1 + self.theta_2 
        v = value(theta)
        u = uncertainty(theta)
        df = dof(theta)
        equivalent(v,-0.1,TOL)
        equivalent(u,0.406201920232,TOL)
        self.assertTrue( math.isinf(df) )

        x1 = self.Ls * self.delta_alpha * theta       
        u = uncertainty(x1)
        equivalent(u,2.88675134595,TOL)

        x2 = self.Ls * self.alpha_s * self.delta_theta        
        u = uncertainty(x2)
        equivalent(u,16.5988202392,TOL)

        y = self.Ls + d - (x1 + x2)
        u = uncertainty(y)
        df = dof(y)
        equivalent(u,31.6637674111,TOL)
        equivalent(df,16.7521475092,TOL)

#----------------------------------------------------------------------------
class GuideExampleH1SIUnits(unittest.TestCase):
    # Check that the use of very small uncertainty values
    # does not pose a problem.
    
    def setUp(self):
        self.d_1 = ureal(0.0,5.8E-9,24)
        self.d_2 = ureal(0.0,3.9E-9,5)
        self.d_3 = ureal(0.0,6.7E-9,8)

        self.alpha_s = ureal(11.5E-6,tb.uniform(2E-6))
        self.delta_alpha = ureal(0.0,tb.uniform(1E-6),50)
        
        self.theta_1 = ureal(-0.1,0.2)
        self.theta_2 = ureal(0.0,tb.arcsine(0.5))
        self.delta_theta = ureal(0.0,tb.uniform(0.05),2)

        self.Ls = ureal(5e-2,25E-9,18)        

    def test(self):
        TOL = 1E-10
        
        # Note values from the guide are only approximate
        d = self.d_1 + self.d_2 + self.d_3
        u = uncertainty(d)
        df = dof(d)
        equivalent(u,9.68194195397E-9,TOL)
        equivalent(df,25.4472507774,TOL)

        theta = self.theta_1 + self.theta_2 
        v = value(theta)
        u = uncertainty(theta)
        df = dof(theta)
        equivalent(v,-0.1,TOL)
        equivalent(u,0.406201920232,TOL)
        self.assertTrue( math.isinf(df) )

        x1 = self.Ls * self.delta_alpha * theta       
        u = uncertainty(x1)
        equivalent(u,2.88675134595E-9,TOL)

        x2 = self.Ls * self.alpha_s * self.delta_theta        
        u = uncertainty(x2)
        equivalent(u,16.5988202392E-9,TOL)

        y = self.Ls + d - (x1 + x2)
        u = uncertainty(y)
        df = dof(y)
        equivalent(u,31.6637674111E-9,TOL)
        equivalent(df,16.7521475092,TOL)

#-----------------------------------------------------
class GuideExampleH2(unittest.TestCase):

    def setUp(self):
        self.v = ureal(4.999,0.0032,independent=False)
        self.i = ureal(0.019661,0.0000095,independent=False)
        self.phi = ureal(1.04446,0.00075,independent=False)

        set_correlation(-0.36,self.v,self.i)
        set_correlation(0.86,self.v,self.phi)
        set_correlation(-0.65,self.i,self.phi)

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

        # The old way! 
        # NB, I think there are now multiple instances of this test 
        # v = ureal(4.999,0.0032,5,independent=False)
        # i = ureal(0.019661,0.0000095,5,independent=False)
        # phi = ureal(1.04446,0.00075,5,independent=False)
        
        v,i,phi = multiple_ureal([4.999,0.019661,1.04446],[0.0032,0.0000095,0.00075],5)

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

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),uncertainty(r),TOL)
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

        real_ensemble( [v,i,phi],5 )

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

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),uncertainty(r),TOL)
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

        equivalent( math.sqrt(welch_satterthwaite(r)[0]),uncertainty(r),TOL)
        equivalent( math.sqrt(welch_satterthwaite(x)[0]),uncertainty(x),TOL)
        equivalent( math.sqrt(welch_satterthwaite(z)[0]),uncertainty(z),TOL)

        equivalent( get_correlation(r,x),-0.591484610819,TOL)
        equivalent( get_correlation(x,z),0.992797472722,TOL)
        equivalent( get_correlation(r,z),-0.490623905441,TOL)

        # Dof calculation should be legal
        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL)
        
        # Add another influence and it breaks again, but earlier results OK
        f = ureal(1,0.0032,5,independent=False)
        self.assertRaises(RuntimeError,set_correlation,0.3,f,i)
        
        equivalent( dof(r),5,TOL)
        equivalent( dof(x),5,TOL)
        equivalent( dof(z),5,TOL)

        
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
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
import unittest

from GTC import *

from testing_tools import *

TOL = 1E-13 
 
#-----------------------------------------------------
class TestMisc(unittest.TestCase):

    def test_illegal_u_component_cases(self):
        # Cannot ask for ...
        
        # non-intermediate real component
        x1 = ureal(1,1) 
        x2 = x1 + ureal(2,1)
        y = x1**x2
        self.assertRaises(RuntimeError,rp.u_component,y,x2)
        
        # non-intermediate complex component
        x1 = ucomplex(10,1)
        x2 = x1**2 
        y = magnitude(x1*3j + x2) 
        self.assertRaises(RuntimeError,rp.u_component,y,x2)
        
        # non-intermediate complex component
        y = x1*3j + x2
        self.assertRaises(RuntimeError,rp.u_component,y,x2)
 
        # non-intermediate real component 
        x1 = ureal(10,1)
        x2 = x1**2 
        y = x1*3j + x2 
        self.assertRaises(RuntimeError,rp.u_component,y,x2)
        
        # y must be an uncertain type
        self.assertRaises(RuntimeError,rp.u_component,10.0,x2)
 
    def test_simple_cases(self):
        # Return zero when ...
        
        # real y, uncertain complex constant 
        x1 = ureal(10,1)
        x2 = constant(1+5j) 
        self.assertTrue( equivalent_sequence(
            rp.u_component(x1,x2),
            (0.,0.,0.,0.),
            TOL
        ))
        
        # real y, complex-valued x 
        y = ureal(1,1) 
        self.assertTrue( equivalent_sequence(
            rp.u_component(y,1+7j),
            (0.,0.,0.,0.),
            TOL
        ))
        
        # complex y, uncertain complex constant 
        x1 = ucomplex(10,1)
        x2 = constant(1+5j) 
        self.assertTrue( equivalent_sequence(
            rp.u_component(x1,x2),
            (0.,0.,0.,0.),
            TOL
        ))
        
        
#-----------------------------------------------------
class TestInCoverage(unittest.TestCase):
    """
    Functions that take a coverage factor and refurn dof
    """
    
    def test_k_to_dof(self):
        """
        1-D cases
        """
        for p in (90,95,99):
            for df in (1,5,10,25,55,1001):
                k = rp.k_factor(df,p)
                dof = rp.k_to_dof(k,p)
                self.assertTrue( equivalent(df,dof,1E10) )
            

        # Illegal cases
        k = -16.58005817 
        self.assertRaises(RuntimeError,rp.k_to_dof,k)
        
        k = rp.k_factor(1.9)
        self.assertRaises(RuntimeError,rp.k_to_dof,k,p=100)
        self.assertRaises(RuntimeError,rp.k_to_dof,-k)
   
    def test_k2_to_dof(self):
        """
        2-D cases
        """
        for p in (90,95,99):
            for df in (2,5,10,25,55,1001):
                k = math.sqrt( rp.k2_factor_sq(df,p) )
                dof = rp.k2_to_dof(k,p)
                self.assertTrue( equivalent(df,dof,1E10) )
 
        # Illegal cases
        k = math.sqrt( rp.k2_factor_sq(2) ) + 0.1
        self.assertRaises(RuntimeError,rp.k2_to_dof,k)
        
        k = math.sqrt( rp.k2_factor_sq(2) )
        self.assertRaises(RuntimeError,rp.k2_to_dof,k,p=100)
        self.assertRaises(RuntimeError,rp.k2_to_dof,-k)
  
#-----------------------------------------------------
class TestCoverage(unittest.TestCase):
    
    """
    The tests are made for 95% and 99% coverage using
    values calculated to 10 digits by R-2.9.2
    
    """
           
    def test_k_factor(self):
        TOL = 1E-7
        data_95 = (
            (2,4.302652730),(3,3.182446305),(5,2.570581836), (10,2.228138852), (50,2.008559112), (inf,1.959963985)
        )
        self.assertTrue(
            equivalent(data_95[0][1],rp.k_factor(data_95[0][0],95),1E-2 ) 
        )
        self.assertTrue(
            equivalent(data_95[1][1],rp.k_factor(data_95[1][0],95),1E-3 ) 
        )
            
        for df,k in data_95:
            self.assertTrue( equivalent(k,rp.k_factor(df),TOL) )

        data_99 = (
            (2,9.924843201),(3,5.84090931),(5,4.032142984), (10,3.169272673), (50,2.677793271), (inf,2.575829304)
        )
        self.assertTrue(
            equivalent(data_99[0][1],rp.k_factor(data_99[0][0],99),0.2 ) 
        )
        self.assertTrue(
            equivalent(data_99[1][1],rp.k_factor(data_99[1][0],99),0.02 ) 
        )
        for df,k in data_99:
            self.assertTrue( equivalent(k,rp.k_factor(df,99),TOL ) )
            
        # illegal `df`
        self.assertRaises(RuntimeError,rp.k_factor,0.5)
        
    def test_k2_factor(self):
        TOL = 1E-7
        data_95 = (
           (2,28.248893784), (5,4.166614906), 
            (10,3.075528764), (50,2.550142994), 
            (inf,2.447746831),
        )
        # illegal `df`
        self.assertRaises(RuntimeError,rp.k2_factor_sq,1)
        
        self.assertTrue(
            equivalent(data_95[1][1]**2,rp.k2_factor_sq(data_95[1][0],95),0.2 ) 
        )
        self.assertTrue(
            equivalent(data_95[2][1]**2,rp.k2_factor_sq(data_95[2][0],95),0.1 ) 
        )
        self.assertTrue(
            equivalent(data_95[3][1]**2,rp.k2_factor_sq(data_95[3][0],95),0.1 ) 
        )
        self.assertTrue(
            equivalent(data_95[4][1]**2,rp.k2_factor_sq(data_95[4][0],95),0.01 ) 
        )

        for df,k in data_95:
            self.assertTrue(
                equivalent(
                    k,
                    math.sqrt( 
                        reporting.k2_factor_sq(df)
                    ),
                    TOL) 
        )

        data_99 = (
           (2,141.414284993), (5,6.708203932), (10,4.222036715), (50,3.215529821), (inf,3.034854259)               
        )
        
        self.assertTrue(
            equivalent(data_99[1][1]**2,rp.k2_factor_sq(data_99[1][0],99),1.2 ) 
        )
        self.assertTrue(
            equivalent(data_99[2][1]**2,rp.k2_factor_sq(data_99[2][0],99),0.6 ) 
        )
        self.assertTrue(
            equivalent(data_99[3][1]**2,rp.k2_factor_sq(data_99[3][0],99),0.4 ) 
        )

        for df,k in data_99:
            self.assertTrue(
                equivalent(
                    k,
                    math.sqrt( 
                        reporting.k2_factor_sq(df,99)
                    ),
                    TOL
                ) 
            )          
 
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

    def test_limited_budget(self):
        x1 = ureal(0,1)
        x2 = ureal(0,1)
        x3 = ureal(0,1)
        y = x1 + x2 + x3 
        
        seq = rp.budget(y)
        self.assertEqual( len(seq), 3 )
        seq = rp.budget(y,max_number=2)
        self.assertEqual( len(seq), 2 )
        
    def test_real(self):
        """
        The budget of a real quantity will consist of a
        list of named tuples with elements for the
        labels and values (the magnitude)
        of the components of uncertainty.

        When a complex quantity has been involved, expect to
        see the components of uncertainty in terms of the
        real and imaginary components.

        A sequence of influences may be given, which may include
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

        self.assertTrue( equivalent(b[0].u,2.0,TOL) )
        self.assertTrue( equivalent(b[1].u,3*.2,TOL) )
        self.assertTrue( equivalent(b[2].u,0.1,TOL) )
        self.assertTrue( equivalent(b[3].u,0,TOL) )

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

        self.assertTrue( equivalent(b[0].u,1.0,TOL) )
        self.assertTrue( equivalent(b[1].u,0.5,TOL) )
        self.assertTrue( equivalent(b[2].u,math.sqrt((.1**2 + .2**2)/2),TOL) )

        b = reporting.budget(y,[ux1,uz1])
        self.assertEqual( len(b), 2)
        
        self.assertTrue( equivalent(b[0].u,1.0,TOL) )
        self.assertTrue( equivalent(b[1].u,math.sqrt((.1**2 + .2**2)/2),TOL) )


#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
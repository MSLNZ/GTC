import unittest
import sys
import math
import cmath
import numpy
import operator
import collections
try:
    xrange  # Python 2
except NameError:
    xrange = range

TOL = 1E-13
DIGITS = 13

from GTC import *
from GTC.LU import *

from testing_tools import *

#-----------------------------------------------------------------------------------
def one_port_xratio(measured,nominal):
    """
    'nominal' and 'measured' are 3-element sequences for the three artefacts used
    to calibrate the one-port.

    returns the three elements: E_D, E_S, E_R

    Equations are from R. W. Beatty and A. C. Macpherson, proc. I.R.E. vol 41, (9) pp 1112-1119 (1953),
    also in R. W. Beatty, "Invariance of the cross ratio applied to microwave network analysis"
    (NBS TN-623, Sept. 1972)

    NB
    * The NBS Note has two transcription errors in the formulae.
    * This algorithm appears to be about 20% SLOWER than using linear algebra (with mst.GUM)
    * The results of this method and the LA method differ when problems are not well-conditioned
    
    """
    m_1, m_2, m_3 = measured
    n_1, n_2, n_3 = nominal

    n_12 = n_1 * n_2     
    n_23 = n_2 * n_3    
    n_31 = n_1 * n_3

    dm_12 = m_1 - m_2
    dm_23 = m_2 - m_3
    dm_31 = m_3 - m_1

    # Common demoninator
    den = n_12 * dm_12 + n_23 * dm_23 + n_31 * dm_31

    # Directivity
    num = n_12 * m_3 * dm_12 + n_23 * m_1 * dm_23 + n_31 * m_2 * dm_31
    E_D = num / den

    # Source match
    num = n_1 * dm_23 + n_2 * dm_31 + n_3 * dm_12
    E_S = -num / den

    # Reflection tracking
    num = dm_12 * dm_23 * dm_31 * (n_1 - n_2) * (n_2 - n_3) * (n_3 - n_1)
    E_R = num / den**2

    return dict(E_D=E_D,E_S=E_S,E_R=E_R)

#-----------------------------------------------------------------------------------
def one_port(measured,nominal):
    """
    -> a sequence containing E_D, E_S, E_R for a 1-port 

    'measured','nominal' are 3-element sequences
        
    """    
    # H =   [ (nominal[0], unity, -nominal[0] * measured[0]),
    #           etc
    #       ]
    H = numpy.array( [ (n,1.0,-n * m) for m,n in zip(measured,nominal) ] )
    b = numpy.array( measured )
    
    ABC = solve( H,b )

    E_D=ABC[1]
    E_S=-ABC[2]
    E_R=ABC[0] - ABC[1] * ABC[2]
    
    return dict(E_D=E_D,E_S=E_S,E_R=E_R)


#-----------------------------------------------------
class TestLUScalar(unittest.TestCase):
    """
    The LU module provides functions to:
        * solve a system of linear equations,
        * evaluate the determinant of a matrix,
    
    """
    def test(self):
        
        a = numpy.array(
            [ (17.99753493515198, 18.08102451776513, 6),
              (627.1639334786039, 27.20225830408064, 8.998767467575989),
              (27.20225830408064, 628.4433741012242, 9.040512258882565)
            ])

        b = numpy.array( (627.8036537899141, 1845.420819171415, 1851.64204122553) )

        # determinant        
        numpy_det = numpy.linalg.det(a)
        a_lu,i,p = ludcmp( a.copy() )
        lu_det = ludet(a_lu,p)
        self.assert_( equivalent(numpy_det,lu_det) )

        # Solve system of equations
        numpy_soln = numpy.linalg.solve( a, b )
        lu_soln = solve( a,b )
        for b1,b2 in zip(numpy_soln,lu_soln):
            self.assert_( equivalent(b1,b2) )    

#-----------------------------------------------------
class TestLUInvProduct(unittest.TestCase):
    """
    The LU module provides a function to evaluate inv(a).b
    where `b` is a 2D array
    
    """
    def test(self):
        a = numpy.array(
            [ (17.99753493515198, 18.08102451776513, 6),
              (627.1639334786039, 27.20225830408064, 8.998767467575989),
              (27.20225830408064, 628.4433741012242, 9.040512258882565)
            ])

        # Calculate the inverse matrix
        a_inv = invab(a,numpy.identity(3))
        a_b = numpy.dot(a,a_inv)
        for i in xrange(3):
            for j in xrange(3):
                if i == j:
                    self.assertTrue( equivalent(1.0,a_b[i,i]) )   
                else:
                    self.assertTrue( equivalent(0.0,a_b[i,j]) )    

        # Solve `a.x=b` for `b` a matrix 
        x = numpy.array([ (-2,1), (-3,2), (4,4) ])
        b = numpy.dot(a,x)
        
        a_b = invab(a,b)
        for i in xrange(a_b.shape[0]):
            for j in xrange(a_b.shape[1]):
                self.assertTrue( equivalent(x[i,j],a_b[i,j]) )

#-----------------------------------------------------
class TestLUComplex(unittest.TestCase):
    """
    The LU module can be used to solve systems of
    linear equations involving complex quantities.
    
    """
    def test_one_port_x_ratio(self):
        """
        The Beatty cross-ratio method should give the same results
        
        """
        measured = (-0.188 - 0.902j, 0.239 + 0.936j, 0.006 + 0.007j)
        nominal = (-1 + 0j,1 + 0j,0 + 0j)

        result1 = one_port_xratio(measured,nominal)
        result2 = one_port(measured,nominal)

        for k in result1:
            equivalent_complex(result1[k],result2[k],TOL)

    def test_known_errors(self):
        """
        Create some 'measured' data by passing known 
        standard values through a known error network
        and check that the errors can be recovered.
        
        """
        error = dict(E_D = 0.04+0.01j,E_S=0.15-0.02j,E_R=0.97+0.01j)
        standards = dict(G_1=0.03,G_2=0.95+0.1j,G_3=-0.97-0.01j)
        
        # transforms gamma to one seen behind the error box 
        measure = lambda G: error['E_D'] + G*error['E_R']/(1-error['E_S']*G)
        
        nominal = standards.values()
        measured = [ measure(G) for G in nominal ]

        result2 = one_port(measured,nominal)
        for k in error:
            equivalent_complex(error[k],result2[k],TOL)

#-----------------------------------------------------
class TestLUUncertainComplex(unittest.TestCase):
    """
    This test uses analytical results by Rodriguez, for the
    one-port calibration case, to compare the sensitivity
    coefficients obtained after a one-port calibration.
    
    """
    def test(self):
        seq_to_complex = lambda seq: complex(seq[0],seq[2])

        #-------------------------------------------------------
        # The values of the standard artefacts 
        # Use the unity uncertainty for each standard
        # and construct a vector of uncertain number inputs.
        #
        Open = 0.98 + 0.01j
        Short = -0.95 -0.03j
        Load = 0.0 + 0.0j

        Gx = [Open, Short, Load]
        G = [ ucomplex(x,[1,1]) for x in Gx ]

        # The measured values 
        # These are just dreamed up
        Gm = [-0.188 - 0.902j, 0.239 + 0.936j, 0.006 + 0.007j ]
        T = [ ucomplex(x,[1,1]) for x in Gm ]

        p1 = one_port(T,G)
        E_S = p1['E_S']
        E_R = p1['E_R']
        E_D = p1['E_D']

        # Values to put in Rodriguez equations
        e_s = value(E_S)
        e_r = value(E_R)
        e_d = value(E_D)
        g_sc = value(G[0])
        g_oc = value(G[1])
        g_ld = value(G[2])
        a = value(T[2])
        b = value(T[1])
        c = value(T[0])

        ## =================== Rodriguez calculated equivalents ====================
        # Note that there is a systematic sign error in Rodriguez equation,
        # negative signs here correct Rodriguez sign error
        de_s_d_g_sc = -(1-e_s*g_oc) / (g_sc * (g_sc -g_oc)) # short
        de_s_d_g_oc = -(1-e_s*g_sc) / (g_oc * (g_oc - g_sc)) # open
        de_s_d_g_ld = -( (1 - e_s*g_oc)*(1-e_s*g_sc)/(g_oc*g_sc) ) #load

        de_s_d_C =(1-e_s*g_sc)**2 * (1-e_s*g_oc) / (g_sc * e_r *(g_sc -g_oc)) # short
        de_s_d_B = (1-e_s*g_oc)**2 * (1-e_s*g_sc) / (g_oc * e_r * (g_oc - g_sc)) # open
        de_s_d_A= ( (1 - e_s*g_oc)*(1-e_s*g_sc)/(g_oc*g_sc*e_r) ) #load

        ## ============ One port error terms and covariances for full solution =====================
        dE_S_d_g_sc = seq_to_complex( rp.u_component(E_S,G[0]) ) # short
        dE_S_d_g_oc = seq_to_complex( rp.u_component(E_S,G[1]) ) # open
        dE_S_d_g_ld = seq_to_complex( rp.u_component(E_S,G[2]) ) # load

        dE_S_d_C = seq_to_complex( rp.u_component(E_S,T[0]) )
        dE_S_d_B = seq_to_complex( rp.u_component(E_S,T[1]) )
        dE_S_d_A = seq_to_complex( rp.u_component(E_S,T[2]) )

        equivalent_complex(de_s_d_g_sc,dE_S_d_g_sc,TOL)
        equivalent_complex(de_s_d_g_oc,dE_S_d_g_oc,TOL)
        equivalent_complex(de_s_d_g_ld,dE_S_d_g_ld,TOL)

        equivalent_complex(de_s_d_C,dE_S_d_C,TOL)
        equivalent_complex(de_s_d_B,dE_S_d_B,TOL)
        equivalent_complex(de_s_d_A,dE_S_d_A,TOL)

        ## Compare error terms from full method with
        ## Rodriguez calculated equivalents 
        equivalent_complex( e_s, ((a-b)/g_oc-(a-c)/g_sc)/(c-b), TOL)
        equivalent_complex( e_r ,(a-b)*(a-c)*(1/g_oc-1/g_sc)/(c-b), TOL)
        equivalent_complex( e_d , a, TOL)
        
#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
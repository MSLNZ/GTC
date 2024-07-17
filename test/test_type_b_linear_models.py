import unittest
import sys
import math
import cmath
import numpy
import operator
import collections

TOL = 1E-13
DIGITS = 13

from GTC import *
from GTC import type_b_linear_models as LM

from GTC import cholesky

from testing_tools import *

#----------------------------------------------------------------------------
class TestSVDWLS(unittest.TestCase):

    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test3(self):
        # weighted least squares
        # from http://www.stat.ufl.edu/~winner/sta6208/reg_ex/cholest.r 
        
        # dLDL ~ lnDOSE, weights=r
        y = (-35.8,-45.0,-52.7,-49.7,-58.2,-66.0)   # dLDL
        DOSE = (1,2.5,5,10,20,40)    
        x = [ math.log(x_i) for x_i in DOSE ]
        
        w = (15,17,12,14,18,13) 
        sig = [ 1.0/math.sqrt(w_i) for w_i in w ]
        
        N = len(y) 
        M = 2

        def fn(x_i):
            return [x_i,1]
        
        fit = LM.wls(x,y,sig,fn)  
        
        a = fit.beta
        self.assertTrue( equivalent(a[0],-7.3753,tol=1E-4) )
        self.assertTrue( equivalent(a[1],-36.9588,tol=1E-4) )
        
        # s2 = fit.ssr/(N-M)
        # cv = s2*svdvar(v,w)
        
        # self.assertTrue( equivalent(math.sqrt(cv[1,1]),2.2441,tol=1E-4) )
        # self.assertTrue( equivalent(math.sqrt(cv[0,0]),0.9892,tol=1E-4) )

#----------------------------------------------------------------------------
class TestSVDOLS(unittest.TestCase):

    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test1(self):
        # Simple example 
        
        def fn(x_i):
            return [x_i,1]
            
        M = 2 
        N = 10
        
        x = [ float(x_i) for x_i in range(N) ]
        y = [ 2*x_i + 1.5 for x_i in x ]
    
        a = LM.ols(x,y,fn=fn).beta
        
        self.assertTrue( equivalent(a[1],1.5) )
        self.assertTrue( equivalent(a[0],2.0) )
   
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test5(self):
        # From halweb.uc3m.es/esp/Personal/personas/durban/esp/web/notes/gls.pdf
        #
        # year, GNP.deflator, GNP, Unemployed, Armed.Forces, Population, Year, Employed
        s =  """
            1947         83.0 234.289      235.6        159.0    107.608 1947   60.323
            1948         88.5 259.426      232.5        145.6    108.632 1948   61.122
            1949         88.2 258.054      368.2        161.6    109.773 1949   60.171
            1950         89.5 284.599      335.1        165.0    110.929 1950   61.187
            1951         96.2 328.975      209.9        309.9    112.075 1951   63.221
            1952         98.1 346.999      193.2        359.4    113.270 1952   63.639
            1953         99.0 365.385      187.0        354.7    115.094 1953   64.989
            1954        100.0 363.112      357.8        335.0    116.219 1954   63.761
            1955        101.2 397.469      290.4        304.8    117.388 1955   66.019
            1956        104.6 419.180      282.2        285.7    118.734 1956   67.857
            1957        108.4 442.769      293.6        279.8    120.445 1957   68.169
            1958        110.8 444.546      468.1        263.7    121.950 1958   66.513
            1959        112.6 482.704      381.3        255.2    123.366 1959   68.655
            1960        114.2 502.601      393.1        251.4    125.368 1960   69.564
            1961        115.7 518.173      480.6        257.2    127.852 1961   69.331
            1962        116.9 554.894      400.7        282.7    130.081 1962   70.551

        """.strip().split()

        step = 8
        M = int( len(s)/step )
        P = 3
        
        x = []
        y = []
        for i in range(M):
            x.append([
                1.0,
                float(s[i*step+2]),
                float(s[i*step+5])
            ])
            y.append( float(s[i*step+7]) )        
        
        a = LM.ols(x,y).beta
        
        self.assertTrue( equivalent(a[0],88.93880,tol=1E-5) )
        self.assertTrue( equivalent(a[1],0.06317,tol=1E-5) )
        self.assertTrue( equivalent(a[2],-0.40974,tol=1E-5) )
                
        # s2 = chisq/(M-P)
        # cv = s2*svdvar(v,w)
        
        # se = [
            # math.sqrt(cv[0,0]),
            # math.sqrt(cv[1,1]),
            # math.sqrt(cv[2,2])
        # ]
                  
        # self.assertTrue( equivalent(se[0],13.78503,tol=1E-5) )
        # self.assertTrue( equivalent(se[1],0.01065,tol=1E-5) )
        # self.assertTrue( equivalent(se[2],0.15214,tol=1E-5) )
        
        # Output from R calculation for covariance matrix (see reference)      
        strings="""
                1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05 8.305119e-06 2.577985e-06 8.002303e-07 2.483989e-07 7.710529e-08 2.393419e-08
                3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05 8.305119e-06 2.577985e-06 8.002303e-07 2.483989e-07 7.710529e-08
                9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05 8.305119e-06 2.577985e-06 8.002303e-07 2.483989e-07
                2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05 8.305119e-06 2.577985e-06 8.002303e-07
                9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05 8.305119e-06 2.577985e-06
                2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05 8.305119e-06
                8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05 2.675539e-05
                2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04 8.619393e-05
                8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04 2.776784e-04
                2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03 8.945559e-04
                8.305119e-06 2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03 2.881860e-03
                2.577985e-06 8.305119e-06 2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02 9.284069e-03
                8.002303e-07 2.577985e-06 8.305119e-06 2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02 2.990913e-02
                2.483989e-07 8.002303e-07 2.577985e-06 8.305119e-06 2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01 9.635387e-02
                7.710529e-08 2.483989e-07 8.002303e-07 2.577985e-06 8.305119e-06 2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00 3.104092e-01
                2.393419e-08 7.710529e-08 2.483989e-07 8.002303e-07 2.577985e-06 8.305119e-06 2.675539e-05 8.619393e-05 2.776784e-04 8.945559e-04 2.881860e-03 9.284069e-03 2.990913e-02 9.635387e-02 3.104092e-01 1.000000e+00""".strip().split()

        cv = numpy.array(strings, dtype=float)
        cv.shape = 16,16 
                
        coef = LM.gls(x,y,cv).beta
        
        # Values agree well with reference
        self.assertTrue( equivalent(coef[0],94.89887752,tol=1E-7) )
        self.assertTrue( equivalent(coef[1],0.06738948,tol=1E-7) )
        self.assertTrue( equivalent(coef[2],-0.47427391,tol=1E-7) )
        
        # chisq = 0.0 
        # for i in range(M):
            # afunc_i = fn(x[i])
            # s = math.fsum(
                    # coef[j]*afunc_i[j]
                        # for j in range(P)
                # )
            # tmp = value( y[i] - s )
            # chisq += tmp*tmp 
            
        # s2 = chisq/(M-P)
        # cv = s2*svdvar(v,w)
        
        # se = [
            # math.sqrt(cv[0,0]),
            # math.sqrt(cv[1,1]),
            # math.sqrt(cv[2,2])
        # ]
 
        # # The std errors agree with the reference values
        # self.assertTrue( equivalent(se[0],14.15760467,tol=1E-5) )
        # self.assertTrue( equivalent(se[1],0.01086675,tol=1E-5) )
        # self.assertTrue( equivalent(se[2],0.15572652,tol=1E-5) )
        
# #----------------------------------------------------------------------------
# class TestSVDLinearSystems(unittest.TestCase):
    
    # """
    # Use SVD to solve linear systems of equations 
    # """

    # #------------------------------------------------------------------------
    # # This test does not use uncertain numbers for the data
    # def test1(self):
        # data = ([
            # [2, -3],
            # [4, 1]
        # ])
        # b = [-2,24]
        # x_expect =[ 5, 4 ]

        # a = numpy.array( data, dtype=float )

        # x = LM.solve(a,b)
        
        # for i,j in zip(x,x_expect):
            # self.assertTrue( equivalent(i,j) )
 
    # #------------------------------------------------------------------------
    # # This test does not use uncertain numbers for the data
    # def test2(self):
        # data = ([
            # [2, 1, 3],
            # [2, 6, 8],
            # [6, 8, 18]
        # ])
        # b = [1,3,5]
        # x_expect = [ 3./10., 4./10., 0.] 


        # a = numpy.array( data, dtype=float )
        
        # x = LM.solve(a,b)

        # for i,j in zip(x,x_expect):
            # self.assertTrue( equivalent(i,j) )
 

#----------------------------------------------------------------------------
class TestUncertainNumberSVDOLS(unittest.TestCase):

    #------------------------------------------------------------------------
    def test_simple(self):
        """Trivial straight line
        
        uncertain numbers in y only with uncertainty of unity

        """
        x = numpy.array([ float(x_i) for x_i in range(10) ])
        y = numpy.array([ ureal(y_i,1) for y_i in x ])

        def fn(x_i):
            return [x_i,1]

        fit = LM.ols(x,y,fn)
        b,a = fit.beta
        
        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)

        from test_fitting import simple_sigma_abr   
        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            [1.0 for x_i in x ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)

        equivalent(r,a.get_correlation(b),TOL)        
        
        # Horizontal line
        x = numpy.array([ float(x_i) for x_i in range(5) ])
        y = numpy.array([ ureal(2,1) for y_i in x ])
        
        def fn_inv(y_i,beta):
            if abs(beta[0]) > 1E-13:
                return (y_i - beta[1])/beta[0]
            else:
                return beta[1]
                
        fit = LM.ols(x,y,fn)
        b,a = fit.beta
        
        # x_0 = fit.x_from_y( [1.5] )
        # equivalent(value(x_0),value(a),TOL)
        # equivalent(uncertainty(x_0),uncertainty(a),TOL)

        # Incorrect input sequences
        self.assertRaises(
            RuntimeError,
            LM.ols,
            numpy.array([1, 2, 3]), numpy.array([]), fn
         )

        self.assertRaises(
            RuntimeError,
            LM.ols,
            numpy.array([1, 2, 3, 4]), numpy.array([1, 2, 3]), fn
        )

    #------------------------------------------------------------------------    
    def test_willink(self):
        """
        This case is taken from
        R Willink, Metrologia 45 (2008) 63-67

        The test ensures that the DoF calculation, when using
        the LS to interpolate to another data point, is correct.
        
        """
        def fn(x_i):
            return [x_i,1]

        nu = 2
        u = 0.1
        x = numpy.array([ value(x_i) for x_i in [-1,0,1] ])
        y = numpy.array([ ureal(y_i,u,df=nu) for y_i in x ])

        fit = LM.ols(x,y,fn)
        b,a = fit.beta

        equivalent( value(a),0.0,TOL)
        equivalent( value(b),1.0,TOL)
        
        equivalent(0.0,a.get_correlation(b),TOL)        

        # See below eqns (12) and (13), in section 2.1
        x_13 = 2.0/3.0
        y_13 = a + b*x_13
        equivalent(y_13.v,5.0*u**2/9.0,TOL)        
        equivalent(y_13.df,25.0*nu/17.0,TOL)         

#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
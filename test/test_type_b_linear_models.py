import unittest
import sys
import math
import cmath
import numpy
import operator
import collections

import numpy as np

TOL = 1E-13
DIGITS = 13

from GTC import *

from GTC import cholesky

from testing_tools import *

#----------------------------------------------------------------------------
class TestSVDWLS(unittest.TestCase):

    def test1wls(self):
        """ISO/TS 28037:2010, p 13"""
        
        # Better numbers using R:
    
        # fit <- lm(y~x,weights=w)
        # fit.sum <- summary(fit)
        # coef(fit)
        # sqrt(diag(fit.sum$cov))

        x = [1,2,3,4,5,6]
        y_data = (3.3,5.6,7.1,9.3,10.7,12.1)
        y = [ ureal(y_i,0.5) for y_i in y_data ]
                
        def fn(x_i):
            return [x_i,1]
        
        fit = lmb.wls(x,y,fn=fn)  
        b, a = fit.beta 
        
        TOL = 1E-6
        self.assertTrue( equivalent(value(a),1.866667,TOL) )
        self.assertTrue( equivalent(value(b),1.757143,TOL) )
        self.assertTrue( equivalent(a.u,0.4654747,TOL) )
        self.assertTrue( equivalent(b.u,0.1195229,TOL) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.050,TOL) )
        
        self.assertEqual( a.df,inf )

        self.assertTrue( equivalent(fit.ssr,1.665,1E-3) )

        TOL = 0.001
        y0 = ureal(10.5,0.5)
        x0 = (y0 - a)/b
        self.assertTrue( equivalent(x0.x,4.913,TOL) )
        self.assertTrue( equivalent(x0.u,0.322,TOL) )
        self.assertEqual( x0.df,inf )
        
        # The type-A evaluation should be equivalent
        u_y = [0.5]*len(x)
        fit = lma.wls(x,y,u_y,fn=fn)
        TOL = 1E-6
        self.assertTrue( equivalent(value(a),1.866667,TOL) )
        self.assertTrue( equivalent(value(b),1.757143,TOL) )
        self.assertTrue( equivalent(a.u,0.4654747,TOL) )
        self.assertTrue( equivalent(b.u,0.1195229,TOL) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.050,TOL) )
        
        
    def test2wls(self):
        """ISO/TS 28037:2010, p 15"""
        
        # Better numbers using R:
    
        # fit <- lm(y~x,weights=w)
        # fit.sum <- summary(fit)
        # coef(fit)
        # sqrt(diag(fit.sum$cov))

        x = [1,2,3,4,5,6]
        u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        y_data = (3.2, 4.3, 7.6, 8.6, 11.7, 12.8)
        y = [ ureal(y_i, u_i) for y_i, u_i in zip(y_data,u_y)]

        def fn(x_i):
            return [x_i,1]
        
        fit = lmb.wls(x,y,fn=fn)  
        b, a = fit.beta 
        
        TOL = 1E-4
        self.assertTrue( equivalent(value(a),0.8852,TOL) )
        self.assertTrue( equivalent(value(b),2.0570,TOL) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.08227848,1E-6) )
        self.assertEqual( a.df,inf )

        TOL = 0.001
        y0 = ureal(10.5,1.0)
        x0 = (y0 - a)/b
        self.assertTrue( equivalent(x0.x,4.674,TOL) )
        self.assertTrue( equivalent(x0.u,0.533,TOL) )
        self.assertEqual( x0.df,inf )

        # Cross check with straight-line fits
        fit = type_a.line_fit_wls(x,y,u_y)
        a, b = fit.a_b 
        self.assertTrue( equivalent(value(a),0.8852,TOL) )
        self.assertTrue( equivalent(value(b),2.0570,TOL) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.08227848,1E-6) )
        self.assertEqual( a.df,inf )       

        # The type-A evaluation should be equivalent
        fit = lmb.wls(x,y,u_y,fn=fn)  
        b, a = fit.beta 
        self.assertTrue( equivalent(value(a),0.8852,TOL) )
        self.assertTrue( equivalent(value(b),2.0570,TOL) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.08227848,1E-6) )

        # The gls evaluation should be equivalent
        cv = np.diag([ u_i**2 for u_i in u_y ])
        fit = lma.gls(x,y,cv,fn=fn)

        b, a = fit.beta   
        self.assertTrue( equivalent(value(a),0.8852,TOL) )
        self.assertTrue( equivalent(value(b),2.0570,TOL) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.08227848,1E-6) )

        fit = lmb.gls(x,y,cv,fn=fn)

        b, a = fit.beta   
        self.assertTrue( equivalent(value(a),0.8852,TOL) )
        self.assertTrue( equivalent(value(b),2.0570,TOL) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(get_covariance(a,b),-0.08227848,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )

    def test3wls(self):
        """        
            Results compared with R, using
        
            fit <- lm(y~x,weights=1/u^2)
            summary(fit)
            vcov(fit)
        """
        x = [1,2,3,4,5,6]
        u_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        y_data = [3.014,5.225,7.004,9.061,11.201,12.762]
        y = [ ureal(y_i, u_i) for y_i, u_i in zip(y_data, u_y) ]
                
        def fn(x_i):
            return [x_i,1]
        
        # This is a RWLS problem
        # fit = lmb.wls(x,y,fn=fn)  
        fit = lmb.wls(x,y,u_y,fn=fn)  
        b, a = fit.beta 
        
        s2 = fit.ssr
        sigma = math.sqrt(s2/(len(x)-2))
        
        self.assertTrue( equivalent(a.x,1.13754,1E-5))
        self.assertTrue( equivalent(b.x,1.97264,1E-5))
        self.assertTrue( equivalent(sigma*a.u,0.12261,1E-5))
        self.assertTrue( equivalent(sigma*b.u,0.04118,1E-5))
        self.assertTrue( equivalent(sigma*sigma*get_covariance(a,b),-0.004408553,1E-5))

    #------------------------------------------------------------------------
    def test3(self):
        # weighted least squares
        # from http://www.stat.ufl.edu/~winner/sta6208/reg_ex/cholest.r 
        
        # dLDL ~ lnDOSE, weights=r
        DOSE = (1,2.5,5,10,20,40)    
        x = [ math.log(x_i) for x_i in DOSE ]
        
        # This is really a RWLS problem, so we adjust the weights
        # knowing the sample estimate of sigma. This simulates a
        # WLS problem for testing
        w = (15,17,12,14,18,13) 
        sigma = 11.58269171402602
        u_y = [ sigma/(math.sqrt(w_i)) for w_i in w ]
        y_data = (-35.8,-45.0,-52.7,-49.7,-58.2,-66.0)   # dLDL
        y = [ ureal(y_i,u_i) for y_i, u_i in zip(y_data,u_y) ]
        
        N = len(y) 
        M = 2

        def fn(x_i):
            return [x_i,1]
        
        fit = lmb.wls(x,y,u_y,fn)  
        
        a = fit.beta
        self.assertTrue( equivalent(value(a[0]),-7.3753,tol=1E-4) )
        self.assertTrue( equivalent(value(a[1]),-36.9588,tol=1E-4) )
        
        self.assertTrue( equivalent(uncertainty(a[0]),0.9892,tol=1E-4) )
        self.assertTrue( equivalent(uncertainty(a[1]),2.2441,tol=1E-4) )

        # The type-A evaluation should be equivalent
        fit = lma.wls(x,y,u_y,fn)  
        a = fit.beta
        self.assertTrue( equivalent(value(a[0]),-7.3753,tol=1E-4) )
        self.assertTrue( equivalent(value(a[1]),-36.9588,tol=1E-4) )

        self.assertTrue( equivalent(uncertainty(a[0]),0.9892,tol=1E-4) )
        self.assertTrue( equivalent(uncertainty(a[1]),2.2441,tol=1E-4) )

#----------------------------------------------------------------------------
class TestSVDOLS(unittest.TestCase):

    #------------------------------------------------------------------------
    def test1(self):
        # This test does not use uncertain numbers for the data
        
        def fn(x_i):
            return [x_i,1]
            
        M = 2 
        N = 10
        
        x = [ float(x_i) for x_i in range(N) ]
        y = [ 2*x_i + 1.5 for x_i in x ]
    
        # The type-B evaluation takes generic arguments
        # Here we provide floating point data so that is 
        # what it returns for coefficients
        fit = lmb.ols(x,y,fn=fn)
        
        # The fit should be exact
        a = fit.beta        
        self.assertTrue( equivalent(a[1],1.5) )
        self.assertTrue( equivalent(a[0],2.0) )
        self.assertTrue( fit.ssr < 1E-13 )

        # The type-A evaluation should be equivalent
        fit = lma.ols(x,y,fn=fn)

        a = fit.beta        
        self.assertTrue( equivalent(value(a[1]),1.5) )
        self.assertTrue( equivalent(value(a[0]),2.0) )
        self.assertTrue( fit.ssr < 1E-13 )
        self.assertTrue( equivalent(uncertainty(a[1]),0) )
        self.assertTrue( equivalent(uncertainty(a[0]),0) )

    def test1UN(self):
        
        def fn(x_i):
            return [x_i,1]
            
        M = 2 
        N = 10
        
        x = [ float(x_i) for x_i in range(N) ]
        y_data = [ 2*x_i + 1.5 for x_i in x ]
        u_y = [ 1.0 ] * len(x)
        y = [ ureal(y_i,u_i) for y_i, u_i in zip(y_data,u_y) ]
    
        a = lmb.ols(x,y,fn=fn).beta
        
        self.assertTrue( equivalent(value(a[1]),1.5) )
        self.assertTrue( equivalent(value(a[0]),2.0) )

        # self.assertTrue( abs(uncertainty(a[1])) < 1E-13 )
        
    #------------------------------------------------------------------------
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
        
        # This is a problem where sigma is estimated from the sample. 
        # To simulate the type-B case we take the known sample estimate. 
        u_y = 0.5459192295102885
        
        for i in range(M):
            x.append([
                1.0,
                float(s[i*step+2]),
                float(s[i*step+5])
            ])
            y.append( ureal( float(s[i*step+7]), u_y) )        
        
        fit = lmb.ols(x,y)
        a = fit.beta
        
        self.assertTrue( equivalent(value(a[0]),88.93880,tol=1E-5) )
        self.assertTrue( equivalent(value(a[1]),0.06317,tol=1E-5) )
        self.assertTrue( equivalent(value(a[2]),-0.40974,tol=1E-5) )
                
        self.assertTrue( equivalent(uncertainty(a[0]),13.78503,tol=1E-5) )
        self.assertTrue( equivalent(uncertainty(a[1]),0.01065,tol=1E-5) )
        self.assertTrue( equivalent(uncertainty(a[2]),0.15214,tol=1E-5) )
        
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
        
        # # The cv matrix can be rescaled without affecting the outcome
        # # But the uncertain numbers need appropriate uncertainties    
        # sigma = .5150725 # 0.5507220949272323 
        # cv = sigma**2 * cv
        # # sigma = 0.5424430452827115 # residuals with cv 
        # # y = [ ureal(y_i.x,math.sqrt(cv[i,i])) for i, y_i in enumerate(y)  ]      
        # y = [ ureal(y_i.x,sigma) for i, y_i in enumerate(y)  ]      
        fit = lmb.gls(x,y,cv)
        
        coef = fit.beta
        # Values agree well with reference
        self.assertTrue( equivalent(value(coef[0]),94.89887752,tol=1E-7) )
        self.assertTrue( equivalent(value(coef[1]),0.06738948,tol=1E-7) )
        self.assertTrue( equivalent(value(coef[2]),-0.47427391,tol=1E-7) )

        type_a_ssr = 3.825178 # Known from type-A testing
        self.assertTrue( equivalent(fit.ssr,type_a_ssr,tol=1E-5) ) 
        
        # # math.sqrt( fit.ssr/(M-P) )
        # print(uncertainty(coef[0]))
        # print(uncertainty(coef[1]))
        # print(uncertainty(coef[2]))
        # # The reference only assumed a covariance matrix with terms
        # # proportional to the actual values and also took the residuals without accounting
        # # for the covariance structure.
        # # This sigma must be applied to the uncertainties to match the published values.
        # self.assertTrue( equivalent(uncertainty(coef[0]),13.94477,tol=1E-5) )
        # self.assertTrue( equivalent(uncertainty(coef[1]),0.010703,tol=1E-5) )
        # self.assertTrue( equivalent(uncertainty(coef[2]),0.153385,tol=1E-5) )
 
    #------------------------------------------------------------------------
    def test7gls(self):
        """
        Example from ISO/TS 28037:2010 "Determination and use of straight-line 
        calibration functions", Section 9: "Model for uncertainties and covariances
        associated with the y_i"
        
        """
        M = 10
        P = 2
        x = range(1,11)
        y_data = (1.3, 4.1, 6.9, 7.5, 10.2, 12.0, 14.5, 17.1, 19.5, 21.0)
        
        cv = np.diag([2] * 10)
        for i in range(5):
            for j in range(i+1,5):
                cv[i][j] = cv[j][i] = 1
        for i in range(5,10):
            cv[i][i] = 5
        for i in range(5,10):
            for j in range(i+1,10):
                cv[i][j] = cv[j][i] = 4
        
        x = []
        for x_i in range(1,11):
            x.append( [x_i,1] )
        y = [ ureal(y_i,math.sqrt(cv[i,i])) for i, y_i in enumerate(y_data) ]
         
        fit = lmb.gls(x,y,cv)
        fit = lmb.ols(x,y)
        
        a = fit.beta

        # Values agree well with reference
        # self.assertTrue( equivalent(value(a[0]),2.2014,tol=1E-4) )
        # self.assertTrue( equivalent(value(a[1]),-0.6456,tol=1E-4) )
        # self.assertTrue( equivalent(fit.ssr,2.07395,tol=1E-5) ) 

        # self.assertTrue( equivalent(uncertainty(a[0]),0.2015,tol=1E-4) )
        # self.assertTrue( equivalent(uncertainty(a[1]),1.2726,tol=1E-4) )
        # self.assertTrue( equivalent(get_covariance(a[0],a[1]),-0.1669,tol=1E-4) )
        
        
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

        fit = lmb.ols(x,y,fn)
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
                
        fit = lmb.ols(x,y,fn)
        b,a = fit.beta
        
        # Incorrect input sequences
        self.assertRaises(
            RuntimeError,
            lmb.ols,
            numpy.array([1, 2, 3]), numpy.array([]), fn
         )

        self.assertRaises(
            RuntimeError,
            lmb.ols,
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

        fit = lmb.ols(x,y,fn)
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
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
from GTC import type_a_SVD as SVD
from GTC.lib import _is_constant

from testing_tools import * 

#----------------------------------------------------------------------------
class TestSVDWLS(unittest.TestCase):

    """
    WLS problems
    """    
    
    #------------------------------------------------------------------------
    def test3svdfit(self):
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
            # for linear fits 
            return [x_i,1]
        
        # a, ssr, w, v = SVD.svdfit(x,y,sig,fn)  
        a, cv, ssr = SVD.svdfit(x,y,sig,fn)  
 
        self.assertTrue( equivalent(a[0],-7.3753,tol=1E-4) )
        self.assertTrue( equivalent(a[1],-36.9588,tol=1E-4) )
        
        s2 = ssr/(N-M)
        # cv = s2*SVD.svdvar(v,w)
        cv = s2*cv
        
        self.assertTrue( equivalent(math.sqrt(cv[1,1]),2.2441,tol=1E-4) )
        self.assertTrue( equivalent(math.sqrt(cv[0,0]),0.9892,tol=1E-4) )

    #------------------------------------------------------------------------
    def test3wls(self):
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
            # for linear fits 
            return [x_i,1]
        
        fit = SVD.rwls(x,y,sig,fn)  
 
        a = fit.beta

        self.assertTrue( equivalent( value(a[0]),-7.3753,tol=1E-4) )
        self.assertTrue( equivalent( value(a[1]),-36.9588,tol=1E-4) )
        
        self.assertTrue( 
            equivalent(uncertainty(a[1]),2.2441,tol=1E-4) 
        )
        self.assertTrue( 
            equivalent(uncertainty(a[0]),0.9892,tol=1E-4) 
        )

#----------------------------------------------------------------------------
class TestSVDOLS(unittest.TestCase):

    """
    OLS problems
    """
    
    #------------------------------------------------------------------------
    def test1(self):
        # Simple example 
        
        def fn(x_i):
            # for linear fits 
            return [x_i,1]
            
        P = 2 
        N = 10
        # sig = np.ones( (N,) )
        
        x = [ float(x_i) for x_i in range(N) ]
        y = [ 2*x_i + 1.5 for x_i in x ]
    
        # a, ssr, w, v = SVD.svdfit(x,y,sig,fn)
        ls = SVD.ols(x,y,fn)
        
        self.assertTrue( equivalent(ls.beta[1].x,1.5) )
        self.assertTrue( equivalent(ls.beta[0].x,2.0) )
        
        # This is a perfect fit so we expect the coefficients
        # to have no uncertainty
        self.assertTrue( _is_constant(ls.beta[1]) )
        self.assertTrue( _is_constant(ls.beta[0]) )
 
    #------------------------------------------------------------------------
    def test2svdfit(self):
        # From http://www.stat.ufl.edu/~winner/Regression_Examples.html
        # "Matrix Form of Multiple Regression - British Calorie Burning Experiment"
        #
        # Description: Measurements of Heat Production (calories) at various
        # Body Masses (kgs) and Work levels (Calories/hour) on a stationary bike.
        #
        # First column: Body Masses (kgs)
        # Second column: Work levels (Calories/hour)
        # Third column: Heat production (calories)

        s =  """
            43.7      19     177
            43.7      43     279
            43.7      56     346
            54.6      13     160
            54.6      19     193
            54.6      43     280
            54.6      56     335
            55.7      13     169
            55.7      26     212
            55.7    34.5     244
            55.7      43     285
            58.8      13     181
            58.8      43     298
            60.5      19     212
            60.5      43     317
            60.5      56     347
            61.9      13     186
            61.9      19     216
            61.9    34.5     265
            61.9      43     306
            61.9      56     348
            66.7      13     209
            66.7      43     324
            66.7      56     352
        """.strip().split()

        step = 3
        N = int( len(s)/step )
        M = 3 

        x = []
        y = []
        for i in range(0,N):
            x.append([
                1.0,
                float(s[i*step+0]),
                float(s[i*step+1])
            ])
            y.append( float(s[i*step+2]) )        
         
        a, cv, ssr = SVD.svdfit(x,y)  
        
        self.assertTrue( equivalent(a[2],3.939524,tol=1E-6) )
        self.assertTrue( equivalent(a[1],1.696528,tol=1E-6) )
        self.assertTrue( equivalent(a[0],28.312636,tol=1E-6) )

        s2 = ssr/(N-M)
        self.assertTrue( equivalent(s2,112.3413,tol=1E-4) )

        # cv = s2*SVD.svdvar(v,w)
        cv = s2*cv
        r_cv = numpy.array([
            [403.2310528, -6.517061844, -0.691727735],
            [-6.5170618,  0.112537458,  0.001218182],
            [-0.6917277,  0.001218182,  0.018260902]
        ])
        
        for i,j in zip( cv.flat, r_cv.flat):
            self.assertTrue( equivalent(i,j,tol=1E-7) )

    #------------------------------------------------------------------------
    def test2wls(self):
        # From http://www.stat.ufl.edu/~winner/Regression_Examples.html
        # "Matrix Form of Multiple Regression - British Calorie Burning Experiment"
        #
        # Description: Measurements of Heat Production (calories) at various
        # Body Masses (kgs) and Work levels (Calories/hour) on a stationary bike.
        #
        # First column: Body Masses (kgs)
        # Second column: Work levels (Calories/hour)
        # Third column: Heat production (calories)

        s =  """
            43.7      19     177
            43.7      43     279
            43.7      56     346
            54.6      13     160
            54.6      19     193
            54.6      43     280
            54.6      56     335
            55.7      13     169
            55.7      26     212
            55.7    34.5     244
            55.7      43     285
            58.8      13     181
            58.8      43     298
            60.5      19     212
            60.5      43     317
            60.5      56     347
            61.9      13     186
            61.9      19     216
            61.9    34.5     265
            61.9      43     306
            61.9      56     348
            66.7      13     209
            66.7      43     324
            66.7      56     352
        """.strip().split()

        step = 3
        N = int( len(s)/step )
        M = 3 

        sig = [1]*N
        x = []
        y = []
        for i in range(0,N):
            x.append([
                1.0,
                float(s[i*step+0]),
                float(s[i*step+1])
            ])
            y.append( float(s[i*step+2]) )        
        
        def fn(x_i):
            return x_i 
 
        # The reference determines sigma from the residuals
        fit = SVD.rwls(x,y,sig,fn)  
 
        a = fit.beta
        
        self.assertTrue( equivalent(value(a[2]),3.939524,tol=1E-6) )
        self.assertTrue( equivalent(value(a[1]),1.696528,tol=1E-6) )
        self.assertTrue( equivalent(value(a[0]),28.312636,tol=1E-6) )

        r_cv = numpy.array([
            [403.2310528, -6.517061844, -0.691727735],
            [-6.5170618,  0.112537458,  0.001218182],
            [-0.6917277,  0.001218182,  0.018260902]
        ])
 
        self.assertTrue( equivalent(variance(a[2]),0.018260902,tol=1E-6) )
        self.assertTrue( equivalent(variance(a[1]),0.112537458,tol=1E-6) )
        self.assertTrue( equivalent(variance(a[0]),403.2310528,tol=1E-6) )
 
        for i in range(3):
            for j in range(i):
                cv1 = get_covariance(a[i],a[j])
                cv2 = r_cv[i,j]
                self.assertTrue( 
                    equivalent(cv1,cv2,tol=1E-7) 
                )
                
    #------------------------------------------------------------------------
    def test3ols(self):
        # Example from Walpole + Myers, but the numerical results
        # were done using R, because Walpole made an error with
        # their t-distribution 'k' factor.
        
        # In R:
            # fit <- lm(y~x)
            # summary(fit)
            # vcov(fit)

        # Also used in test_type_a.py

        x = numpy.array([3.,7.,11.,15.,18.,27.,29.,30.,30.,31.,31.,32.,33.,33.,34.,36.,36.,
             36.,37.,38.,39.,39.,39.,40.,41.,42.,42.,43.,44.,45.,46.,47.,50.])
        y = numpy.array([5.,11.,21.,16.,16.,28.,27.,25.,35.,30.,40.,32.,34.,32.,34.,37.,38.,
             34.,36.,38.,37.,36.,45.,39.,41.,40.,44.,37.,44.,46.,46.,49.,51.])

        N = int( len(x) )
        M = 2 

        def fn(x_i):
            return [x_i,1]
 
        fit = SVD.ols(x,y,fn)
        
        TOL = 1E-5
        self.assertTrue( equivalent( value(fit.beta[1]), 3.82963, TOL) )
        self.assertTrue( equivalent( uncertainty(fit.beta[1]), 1.768447, TOL) )
        self.assertTrue( equivalent( value(fit.beta[0]), 0.90364, TOL) )
        self.assertTrue( equivalent( uncertainty(fit.beta[0]), 0.05011898, TOL) )

    #------------------------------------------------------------------------
    def test4ols(self):
        # From http://www.stat.ufl.edu/~winner/sta6208/reg_ex/spiritsgg.rout
        #

        # input year, consume,   income,    price,     price_inc
        s =  """
            1870      1.9565    1.7669    1.9176    1.085290622
            1871      1.9794    1.7766    1.9059    1.0727794664
            1872      2.012     1.7764    1.8798    1.0582076109
            1873      2.0449    1.7942    1.8727    1.0437520901
            1874      2.0561    1.8156    1.8984    1.0456047588
            1875      2.0678    1.8083    1.9137    1.0582867887
            1876      2.0561    1.8083    1.9176    1.0604435105
            1877      2.0428    1.8067    1.9176    1.0613826313
            1878      2.029     1.8166    1.942     1.0690300561
            1879      1.998     1.8041    1.9547    1.0834765257
            1880      1.9884    1.8053    1.9379    1.0734503961
            1881      1.9835    1.8242    1.9462    1.0668786317
            1882      1.9773    1.8395    1.9504    1.0602881218
            1883      1.9748    1.8464    1.9504    1.0563258232
            1884      1.9629    1.8492    1.9723    1.0665693273
            1885      1.9396    1.8668    2         1.0713520463
            1886      1.9309    1.8783    2.0097    1.0699568759
            1887      1.9271    1.8914    2.0146    1.0651369356
            1888      1.9239    1.9166    2.0146    1.0511322133
            1889      1.9414    1.9363    2.0097    1.0379073491
            1890      1.9685    1.9548    2.0097    1.0280847145
            1891      1.9727    1.9453    2.0097    1.0331054336
            1892      1.9736    1.9292    2.0048    1.0391872279
            1893      1.9499    1.9209    2.0097    1.0462283305
            1894      1.9432    1.951     2.0296    1.0402870323
            1895      1.9569    1.9776    2.0399    1.0315028317
            1896      1.9647    1.9814    2.0399    1.0295245786
            1897      1.971     1.9819    2.0296    1.0240678137
            1898      1.9719    1.9828    2.0146    1.0160379262
            1899      1.9956    2.0076    2.0245    1.0084180116
            1900      2         2         2         1
            1901      1.9904    1.9939    2.0048    1.0054666734
            1902      1.9752    1.9933    2.0048    1.0057693272
            1903      1.9494    1.9797    2         1.0102540789
            1904      1.9332    1.9772    1.9952    1.0091037831
            1905      1.9139    1.9924    1.9952    1.0014053403
            1906      1.9091    2.0117    1.9905    0.9894616494
            1907      1.9139    2.0204    1.9813    0.9806473966
            1908      1.8886    2.0018    1.9905    0.9943550804
            1909      1.7945    2.0038    1.9859    0.9910669728
            1910      1.7644    2.0099    2.0518    1.0208468083
            1911      1.7817    2.0174    2.0474    1.0148706256
            1912      1.7784    2.0279    2.0341    1.00305735
            1913      1.7945    2.0359    2.0255    0.9948916941
            1914      1.7888    2.0216    2.0341    1.0061832212
            1915      1.8751    1.9896    1.9445    0.9773321271
            1916      1.7853    1.9843    1.9939    1.0048379781
            1917      1.6075    1.9764    2.2082    1.1172839506
            1918      1.5185    1.9965    2.27      1.136989732
            1919      1.6513    2.0652    2.243     1.0860933566
            1920      1.6247    2.0369    2.2567    1.1079090775
            1921      1.5391    1.9723    2.2988    1.1655427673
            1922      1.4922    1.9797    2.3723    1.1983128757
            1923      1.4606    2.0136    2.4105    1.1971096544
            1924      1.4551    2.0165    2.4081    1.1941978676
            1925      1.4425    2.0213    2.4081    1.1913619948
            1926      1.4023    2.0206    2.4367    1.205928932
            1927      1.3991    2.0563    2.4284    1.1809560862
            1928      1.3798    2.0579    2.431     1.1813013266
            1929      1.3782    2.0649    2.4363    1.1798634316
            1930      1.3366    2.0582    2.4552    1.1928869886
            1931      1.3026    2.0517    2.4838    1.2106058391
            1932      1.2592    2.0491    2.4958    1.2179981455
            1933      1.2365    2.0766    2.5048    1.2062024463
            1934      1.2549    2.089     2.5017    1.1975586405
            1935      1.2527    2.1059    2.4958    1.1851464932
            1936      1.2763    2.1205    2.4838    1.1713275171
            1937      1.2906    2.1205    2.4636    1.1618014619
            1938      1.2721    2.1182    2.458     1.1604192239
        """.strip().split()

        step = 5
        N = int( len(s)/step )
        M = 3

        x = []
        y = []
        for i in range(0,N):
            x.append([
                1.0,
                float(s[i*step+2]),
                float(s[i*step+3])
            ])
            y.append( float(s[i*step+1]) )        
        
        # def fn(x_i):
            # return x_i 
 
        fit = SVD.ols(x,y)

        a = fit.beta
        self.assertTrue( equivalent(value(a[0]),4.6117077,tol=1E-7) )
        self.assertTrue( equivalent(value(a[1]),-0.1184552,tol=1E-7) )
        self.assertTrue( equivalent(value(a[2]),-1.2317419,tol=1E-7) )

        self.assertTrue( equivalent(uncertainty(a[0]),0.15261611,tol=1E-7) )
        self.assertTrue( equivalent(uncertainty(a[1]),0.10885040,tol=1E-7) )
        self.assertTrue( equivalent(uncertainty(a[2]),0.05024342,tol=1E-7) )

    #------------------------------------------------------------------------
    def test5gls(self):
        # From halweb.uc3m.es/esp/Personal/personas/durban/esp/web/notes/gls.pdf
        # which in turn came from Julian J. Faraway, 
        # "Practical Regression and Anova using R", 2002
        # https://cran.r-project.org/doc/contrib/Faraway-PRA.pdf
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
        N = int( len(s)/step )
        M = 3
        
        x = []
        y = []
        for i in range(0,N):
            x.append([
                1.0,
                float(s[i*step+2]),
                float(s[i*step+5])
            ])
            y.append( float(s[i*step+7]) )        
        
        # def fn(x_i):
            # return x_i 
 
        fit = SVD.ols(x,y)

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

        V = numpy.array(strings, dtype=float)
        V.shape = 16,16 
        
        fit = SVD.gls(x,y,V)
 
        a = fit.beta
        sigma = math.sqrt( fit.ssr/(N-M) )
        
        # Values agree well with reference
        self.assertTrue( equivalent(value(a[0]),94.89887752,tol=1E-7) )
        self.assertTrue( equivalent(value(a[1]),0.06738948,tol=1E-7) )
        self.assertTrue( equivalent(value(a[2]),-0.47427391,tol=1E-7) )
        
        # Values agree well with reference
        # Note that the reference only assumed a covariance matrix with terms
        # proportional to the actual values. The residuals are used to estimate
        # sigma squared and sigma must be applied to the uncertainties to match 
        # the published values.
        self.assertTrue( equivalent(sigma*uncertainty(a[0]),14.15760467,tol=1E-5) )
        self.assertTrue( equivalent(sigma*uncertainty(a[1]),0.01086675,tol=1E-5) )
        self.assertTrue( equivalent(sigma*uncertainty(a[2]),0.15572652,tol=1E-5) )

    #------------------------------------------------------------------------
    def test6gls(self):
        """
        Example from ISO/TS 28037:2010 "Determination and use of straight-line 
        calibration functions", Section 9: "Model for uncertainties and covariances
        associated with the y_i"
        
        """
        M = 10
        P = 2
        x = range(1,11)
        y = (1.3, 4.1, 6.9, 7.5, 10.2, 12.0, 14.5, 17.1, 19.5, 21.0)
        
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
         
        fit = SVD.gls(x,y,cv,lambda x: x)
        a = fit.beta

        # Values agree well with reference
        self.assertTrue( equivalent(value(a[0]),2.2014,tol=1E-4) )
        self.assertTrue( equivalent(value(a[1]),-0.6456,tol=1E-4) )

        # Values agree well with reference
        self.assertTrue( equivalent(uncertainty(a[0]),0.2015,tol=1E-4) )
        self.assertTrue( equivalent(uncertainty(a[1]),1.2726,tol=1E-4) )

    #------------------------------------------------------------------------
    def test7wls(self):
        # From halweb.uc3m.es/esp/Personal/personas/durban/esp/web/notes/gls.pdf
        # which in turn came from Julian J. Faraway, 
        # "Practical Regression and Anova using R", 2002
        # https://cran.r-project.org/doc/contrib/Faraway-PRA.pdf

        momentum = (4, 6, 8, 10, 12, 15, 20, 30, 75, 150)
        energy = (0.345, 0.287, 0.251, 0.225, 0.207, 0.186, 0.161, 0.132, 0.084, 0.060)
        crossx = (367, 311, 295, 268, 253, 239, 220, 213, 193, 192)
        sd = (17, 9, 9, 7, 7, 6, 6, 6, 5, 5)

        x = np.array( energy, dtype=float)
        y = np.array( crossx, dtype=float )
        
        def fn(x_i):
            return [x_i, 1] 
 
        fit = SVD.ols(x,y,fn)

        a = fit.beta
        self.assertTrue( equivalent(value(a[0]),619.71,tol=1E-2) )
        self.assertTrue( equivalent(value(a[1]),135.00,tol=1E-2) )

        self.assertTrue( equivalent(uncertainty(a[0]),47.68,tol=1E-2) )
        self.assertTrue( equivalent(uncertainty(a[1]),10.08,tol=1E-2) )

        u_y = np.array( sd, dtype=float)

        fit = SVD.rwls(x,y,u_y,fn)
 
        a = fit.beta
        
        # Values agree well with reference
        self.assertTrue( equivalent(value(a[0]),530.835,tol=1E-3) )
        self.assertTrue( equivalent(value(a[1]),148.473,tol=1E-3) )
        
        # Values agree well with reference
        self.assertTrue( equivalent(uncertainty(a[0]),47.550,tol=1E-3) )
        self.assertTrue( equivalent(uncertainty(a[1]),8.079,tol=1E-3) )
        
        # Alternative formulation with x data in MxP array 
        X = np.array( [ fn(x_i) for x_i in x ], dtype=float)
        
        fit = SVD.rwls(X,y,u_y,lambda x: x)

        a = fit.beta
        # Values agree well with reference
        self.assertTrue( equivalent(value(a[0]),530.835,tol=1E-3) )
        self.assertTrue( equivalent(value(a[1]),148.473,tol=1E-3) )
        
        # Values agree well with reference
        self.assertTrue( equivalent(uncertainty(a[0]),47.550,tol=1E-3) )
        self.assertTrue( equivalent(uncertainty(a[1]),8.079,tol=1E-3) )
    #------------------------------------------------------------------------
    def test_bevington(self):
        """
        Example from Bevington Table 6.1
        Some calculations done in R

        In R:
            fit <- lm(y~x)
            summary(fit)
            vcov(fit)
        
        """
        # Also used in test_type_a.py
        
        x = numpy.array([4.,8.,12.5,16.,20.,25.,31.,36.,40.,40.])
        y = numpy.array([3.7,7.8,12.1,15.6,19.8,24.5,30.7,35.5,39.4,39.5])

        N = int( len(x) )
        M = 2 

        def fn(x_i):
            return [x_i,1]
 
        fit = SVD.ols(x,y,fn)
        
        TOL = 1E-5
        a = fit.beta[1]
        b = fit.beta[0]
        self.assertTrue( equivalent( value(a), -0.222142, TOL) )
        self.assertTrue( equivalent( uncertainty(a), 0.06962967, TOL) )
        self.assertTrue( equivalent( value(b), 0.992780, TOL) )
        self.assertTrue( equivalent( uncertainty(b), 0.002636608, TOL) )
        self.assertTrue( equivalent( a.u*b.u*get_correlation(a,b), -0.0001616271, TOL) )


    #------------------------------------------------------------------------
    def test_H3(self):
        """H3 from the GUM
        """
        t_k = numpy.array([21.521,22.012,22.512,23.003,23.507,23.999,24.513,25.002,25.503,26.010,26.511])
        b_k = numpy.array([-0.171,-0.169,-0.166,-0.159,-0.164,-0.165,-0.156,-0.157,-0.159,-0.161,-0.160])
        theta = numpy.array([ t_k_i - 20 for t_k_i in t_k ])

        N = len(theta)
        M = 2 

        def fn(x_i):
            return [x_i,1]
 
        fit = SVD.ols(theta,b_k,fn,M)
        
        TOL = 1E-5
        a = fit.beta[1]
        b = fit.beta[0]

        # Compare with GUM values

        self.assertTrue( equivalent(value(a),-0.1712,1E-4) )
        self.assertTrue( equivalent(value(b),0.00218,1E-5) )
        self.assertTrue( equivalent(get_correlation(a,b),-0.930,1E-3) )
        
        b_30 = a + b*(30 - 20)
        self.assertTrue( equivalent(b_30.x,-0.1494,1E-4) )
        self.assertTrue( equivalent(b_30.u,0.0041,1E-4) )
        self.assertTrue( equivalent(b_30.df,9,1E-13) )

        # # `y_from_x` is the predicted single `y` response
        # # which has greater variability        
        # b_30 = fit.y_from_x(30 - 20)
        # self.assertTrue( not b_30.is_intermediate )
        # self.assertTrue( equivalent(b_30.x,-0.1494,1E-4) )
        # b_30 = fit.y_from_x(30 - 20,y_label='b_30')
        # self.assertTrue( equivalent(b_30.x,-0.1494,1E-4) )
        # self.assertTrue( b_30.is_intermediate )

    #------------------------------------------------------------------------
    def test_A5(self):
        """CITAC 3rd edition

        Test the calibration curve aspect
        """
        x = numpy.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9])
        y = numpy.array([0.028, 0.029, 0.029, 0.084, 0.083, 0.081, 0.135, 0.131, 0.133, 0.180,
                  0.181, 0.183, 0.215, 0.230, 0.216])

        N = int( len(x) )
        M = 2 

        def fn(x_i):
            return [x_i,1]
 
        def fn_inv(y_i,beta):
            if abs(beta[0]) > 1E-13:
                return (y_i - beta[1])/beta[0]
            else:
                return beta[1]

        fit = SVD.ols(x,y,fn)

        TOL = 1E-5
        b,a = fit.beta

        self.assertTrue(equivalent(value(a),0.0087,1E-4))
        self.assertTrue(equivalent(value(b),0.2410,1E-4))
        self.assertTrue(equivalent(uncertainty(a),0.0029,1E-4))
        self.assertTrue(equivalent(uncertainty(b),0.0050,1E-4))

        # # The classical uncertainty
        # xmean = type_a.mean(x)
        # sxx = sum( (x_i-xmean)**2 for x_i in x )
        # S = math.sqrt(fit.ssr/(N-2))

        # c_0 = fit.x_from_y( [0.0712, 0.0716] )
        # _x = c_0.x
        # u_c_0 = S*math.sqrt(1.0/2 + 1.0/N + (_x-xmean)**2 / sxx)/b.x

        # self.assertTrue(equivalent(u_c_0,c_0.u,TOL))
        # self.assertEqual(c_0.df,N-2)

        # # Now in the opposite sense
        # y_0 = fit.y_from_x(_x)
        # u_y_0 = S*math.sqrt(1.0 + 1.0/N + (_x-xmean)**2/sxx)
        
        # self.assertTrue(equivalent(value(y_0),0.0714,TOL))
        # self.assertTrue(equivalent(u_y_0,y_0.u,TOL))
        # self.assertEqual(y_0.df,N-2)
        
# #----------------------------------------------------------------------------
# class TestSVDLinearSystems(unittest.TestCase):
    
    # """
    # Using SVD to solve linear systems of equations 
    # """

    # #------------------------------------------------------------------------
    # def test1(self):
        # # Simple example
        # data = ([
            # [2, -3],
            # [4, 1]
        # ])
        # b = [-2,24]
        # x_expect =[ 5, 4 ]

        # a = numpy.array( data, dtype=float )
        
        # x = SVD.solve(a,b)

        # for i,j in zip(x,x_expect):
            # self.assertTrue( equivalent(i,j) )
 
    # #------------------------------------------------------------------------
    # def test2(self):
        # # Simple example
        # data = ([
            # [2, 1, 3],
            # [2, 6, 8],
            # [6, 8, 18]
        # ])
        # b = [1,3,5]
        # x_expect = [ 3./10., 4./10., 0.] 


        # a = numpy.array( data, dtype=float )

        # x = SVD.solve(a,b)

        # for i,j in zip(x,x_expect):
            # self.assertTrue( equivalent(i,j) )     

#=====================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
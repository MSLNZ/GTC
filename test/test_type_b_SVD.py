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
from GTC import type_b_SVD as SVD
from GTC.type_a_SVD import svdvar

from GTC import cholesky

from testing_tools import *

#----------------------------------------------------------------------------
# Note, this test suite exercises the SVD implementation using floats.
# It is in this type-B testing module because these routines are only 
# used from type_b_SVD.py (type_a_SVD.py uses Numpy routines)
#
class TestSVD(unittest.TestCase):

    """
    Check the decomposition of a real matrix 

    In general, we expect that is u,w,v is the SVD then:
    
        u * u.T is an identity matrix 
        v * v.T is an identity matrix 
        u * w * v.T is the original matrix 
        
    The routine does not sort the values in ``w`` 
    
    It seems that different methods of evaluating SVD lead to 
    different factorisations. So you cannot expect to get 
    agreement with the individual matrices unless you know 
    it is the same algorithm.   
    """
    
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test1(self):
        # From https://en.wikipedia.org/wiki/Singular_value_decomposition 
        # Note that NR does implement the full SVD assumed in the worked example
        data = (
            [1,0,0,0,2],
            [0,0,3,0,0],
            [0,0,0,0,0],
            [0,4,0,0,0],
        )
        
        U,S,V = numpy.linalg.svd(
            numpy.array( data )
        )
        
        a = numpy.array( data, dtype=float )
        M,N = a.shape 
        
        u,w,v = SVD.svd_decomp(a)
        
        ww = numpy.zeros( (N,N) )
        w_sorted = []
        for i in range(N): 
            ww[i,i] = w[i]
            w_sorted.append( w[i] )
        w_sorted.sort(reverse=True)
        
        # Check the diagonal values
        for i in range(min(M,N)):
            self.assertTrue( equivalent(w_sorted[i],S[i]) )

        # Check that u * u.T is an identity matrix
        check = numpy.matmul(u,u.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
      
        # Check that v * v.T is an identity matrix
        check = numpy.matmul(v,v.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )

        # Check that u * w * v.T is the original matrix 
        check = numpy.matmul(numpy.matmul(u,ww),v.T) 
        original = numpy.array( data )
        for i,j in zip( original.flat, check.flat):
            self.assertTrue( equivalent(i,j) )                

    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test2(self):
        
        data = (
            [2,5],
            [2,5],
        )
        
        na = numpy.array( data, dtype=float )
        U,S,V = numpy.linalg.svd(na)

        a = numpy.array( data, dtype=float )
        M,N = a.shape 
        
        u,w,v = SVD.svd_decomp(a)
        
        ww = numpy.zeros( [N,N] )
        w_sorted = []
        for i in range(N): 
            ww[i,i] = w[i]
            w_sorted.append( w[i] )
        w_sorted.sort(reverse=True)
        
        # Check the diagonal values
        for i in range(min(M,N)):
            self.assertTrue( equivalent(w_sorted[i],S[i]) )

        # Check that u * u.T is an identity matrix
        check = numpy.matmul(u.T,u) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )

        # Check that v * v.T is an identity matrix
        check = numpy.matmul(v,v.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
            
        # Check that u * w * v.T is the original matrix 
        check = numpy.matmul(u,numpy.matmul(ww,v.T)) 
        original = numpy.array( data, dtype=float )
        for i,j in zip( original.flat, check.flat):
            self.assertTrue( equivalent(i,j) )                
 
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test3(self):
        data = (
            [1,2,3],
            [4,5,6],
            [7,8,9],
        )
        
        na = numpy.array( data )
        U,S,V = numpy.linalg.svd(na)
        
        a = numpy.array( data, dtype=float )
        M,N = a.shape 
        
        u,w,v = SVD.svd_decomp(a)
        
        ww = numpy.zeros( [N,N] )
        w_sorted = []
        for i in range(N): 
            ww[i,i] = w[i]
            w_sorted.append( w[i] )
        w_sorted.sort(reverse=True)
        
        # Check the diagonal values
        for i in range(min(M,N)):
            self.assertTrue( equivalent(w_sorted[i],S[i]) )

        # Check that u * u.T is an identity matrix
        check = numpy.matmul(u.T,u) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
      
        # Check that u * w * v.T is the original matrix 
        check = numpy.matmul(u,numpy.matmul(ww,v.T)) 
        original = numpy.array( data, dtype=float )
        for i,j in zip( original.flat, check.flat):
            self.assertTrue( equivalent(i,j) )                

        # Check that v.T agrees with values reported on NR forum 
        check = numpy.array([
           [-0.479671,   -0.572368,   -0.665064],
           [ 0.776691,    0.075686,   -0.625318],
           [ 0.408248,   -0.816497,    0.408248],
        ])
        for i,j in zip( v.T.flat, check.flat):
            self.assertTrue( equivalent(i,j,1E-6) )
 
        # Check that v * v.T is an identity matrix
        check = numpy.matmul(v,v.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
 
    #------------------------------------------------------------------------
    def test4(self):
        # This test does not use uncertain numbers for the data
        # See http://numerical.recipes/forum/showthread.php?t=2236
        data = (
            [2,5,2],
            [5,2,5],
        )
        
        na = numpy.array( data )
        U,S,V = numpy.linalg.svd(na)   
        
        a = numpy.array( data, dtype=float )
        M,N = a.shape 
        
        u,w,v = SVD.svd_decomp(a)
        
        ww = numpy.zeros( [N,N] )
        w_sorted = []
        for i in range(N): 
            ww[i,i] = w[i]
            w_sorted.append( w[i] )
        w_sorted.sort(reverse=True)
        
        # Check the diagonal values
        for i in range(min(M,N)):
            self.assertTrue( equivalent(w_sorted[i],S[i]) )

        # Check that u * u.T is an identity matrix
        check = numpy.matmul(u,u.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
 
        # Check that v * v.T is an identity matrix
        check = numpy.matmul(v,v.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
 
        # Check that u * w * v.T is the original matrix 
        check = numpy.matmul(u,numpy.matmul(ww,v.T)) 
        original = numpy.array( data, dtype=float )
        for i,j in zip( original.flat, check.flat):
            self.assertTrue( equivalent(i,j) )                
      
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test5(self):
        # This came from http://numerical.recipes/forum/showthread.php?t=1437
        # The discussion there uses the later C++ version of SVD in NR(3) 
        data = (
            [0.299000,    0.587000,    0.114000],
            [-0.168636,   -0.331068,    0.499704],
            [0.499813,   -0.418531,   -0.081282],
        )
        
        na = numpy.array( data )
        U,S,V = numpy.linalg.svd(na)
        
        a = numpy.array( data, dtype=float )
        M,N = a.shape 
                
        u,w,v = SVD.svd_decomp(a)
        
        ww = numpy.zeros( [N,N] )
        w_sorted = []
        for i in range(N): 
            ww[i,i] = w[i]
            w_sorted.append( w[i] )
        w_sorted.sort(reverse=True)
        
        # Check the diagonal values
        for i in range(min(M,N)):
            self.assertTrue( equivalent(w_sorted[i],S[i]) )

        # Check that u * u.T is an identity matrix
        check = numpy.matmul(u.T,u)
        idn = la.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
            
        # Check that v * v.T is an identity matrix
        check = numpy.matmul(v,v.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
  
        # Check that u * w * v.T is the original matrix 
        check = numpy.matmul(u,numpy.matmul(ww,v.T)) 
        original = numpy.array( data, dtype=float )
        for i,j in zip( original.flat, check.flat):
            self.assertTrue( equivalent(i,j) )       
            
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test6(self):
        # This came from http://numerical.recipes/forum/showthread.php?t=1437
        # The discussion there uses the later C++ version of SVD in NR(3) 
        data = ([
            [2.000000,    2.000000,    5.000000],
            [4.000000,    5.000000,    1.000000],
            [7.000000,    8.000000,    9.000000],
           [13.000000,   11.000000,   12.000000],  
        ])
        na = numpy.array( data )
        U,S,V = numpy.linalg.svd(na)
        
        a = numpy.array( data )
        M,N = a.shape 
                
        u,w,v = SVD.svd_decomp(a)
        
        ww = numpy.zeros( [N,N] )
        w_sorted = []
        for i in range(N): 
            ww[i,i] = w[i]
            w_sorted.append( w[i] )
        w_sorted.sort(reverse=True)
        
        # Check the diagonal values
        for i in range(min(M,N)):
            self.assertTrue( equivalent(w_sorted[i],S[i]) )

        # Check that u * u.T is an identity matrix
        check = numpy.matmul(u.T,u) 
        idn = la.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
            
        # Check that v * v.T is an identity matrix
        check = numpy.matmul(v,v.T) 
        idn = numpy.identity( check.shape[0] )
        for i,j in zip( idn.flat, check.flat):
            self.assertTrue( equivalent(i,j) )
  
        # Check that u * w * v.T is the original matrix 
        check = numpy.matmul(u,numpy.matmul(ww,v.T)) 
        original = numpy.array( data, dtype=float )
        for i,j in zip( original.flat, check.flat):
            self.assertTrue( equivalent(i,j) )      

#----------------------------------------------------------------------------
class TestSVDWLS(unittest.TestCase):

    """
    WLS problems
    """
    
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
            # for linear fits 
            return [x_i,1]
        
        a, chisq, w, v = SVD.svdfit(x,y,sig,fn)  
 
        self.assertTrue( equivalent(a[0],-7.3753,tol=1E-4) )
        self.assertTrue( equivalent(a[1],-36.9588,tol=1E-4) )
        
        s2 = chisq/(N-M)
        cv = s2*svdvar(v,w)
        
        self.assertTrue( equivalent(math.sqrt(cv[1,1]),2.2441,tol=1E-4) )
        self.assertTrue( equivalent(math.sqrt(cv[0,0]),0.9892,tol=1E-4) )

#----------------------------------------------------------------------------
class TestSVDOLS(unittest.TestCase):

    """
    OLS problems
    """
    
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test1(self):
        # Simple example 
        
        def fn(x_i):
            # for linear fits 
            return [x_i,1]
            
        M = 2 
        N = 10
        sig = [1]*N 
        
        x = [ float(x_i) for x_i in range(N) ]
        y = [ 2*x_i + 1.5 for x_i in x ]
    
        a, chisq, w, v = SVD.svdfit(x,y,sig,fn)
        
        self.assertTrue( equivalent(a[1],1.5) )
        self.assertTrue( equivalent(a[0],2.0) )
 
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test2(self):
        # From http://www.stat.ufl.edu/~winner/Regression_Examples.html
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
        
        def fn(x_i):
            return x_i 
 
        sig = numpy.ones( (N,) )
        a, chisq, w, v = SVD.svdfit(x,y,sig,fn)
        
        self.assertTrue( equivalent(a[2],3.939524,tol=1E-6) )
        self.assertTrue( equivalent(a[1],1.696528,tol=1E-6) )
        self.assertTrue( equivalent(a[0],28.312636,tol=1E-6) )

        s2 = chisq/(N-M)
        self.assertTrue( equivalent(s2,112.3413,tol=1E-4) )

        cv = s2*svdvar(v,w)
        r_cv = numpy.array([
            [403.2310528, -6.517061844, -0.691727735],
            [-6.5170618,  0.112537458,  0.001218182],
            [-0.6917277,  0.001218182,  0.018260902]
        ])
        
        for i,j in zip( cv.flat, r_cv.flat):
            self.assertTrue( equivalent(i,j,tol=1E-7) )
 
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test4(self):
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
        
        def fn(x_i):
            return x_i 
 
        sig = numpy.ones( (N,) )
        a, chisq, w, v = SVD.svdfit(x,y,sig,fn)

        self.assertTrue( equivalent(a[0],4.6117077,tol=1E-7) )
        self.assertTrue( equivalent(a[1],-0.1184552,tol=1E-7) )
        self.assertTrue( equivalent(a[2],-1.2317419,tol=1E-7) )
                
        s2 = chisq/(N-M)
        cv = s2*svdvar(v,w)

        se = [
            math.sqrt(cv[0,0]),
            math.sqrt(cv[1,1]),
            math.sqrt(cv[2,2])
        ]
        
        self.assertTrue( equivalent(se[0],0.15261611,tol=1E-7) )
        self.assertTrue( equivalent(se[1],0.10885040,tol=1E-7) )
        self.assertTrue( equivalent(se[2],0.05024342,tol=1E-7) )
   
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
        
        def fn(x_i):
            return x_i 
 
        sig = numpy.ones( (N,) )
        a, chisq, w, v = SVD.svdfit(x,y,sig,fn)
        
        self.assertTrue( equivalent(a[0],88.93880,tol=1E-5) )
        self.assertTrue( equivalent(a[1],0.06317,tol=1E-5) )
        self.assertTrue( equivalent(a[2],-0.40974,tol=1E-5) )
                
        s2 = chisq/(N-M)
        cv = s2*svdvar(v,w)
        
        se = [
            math.sqrt(cv[0,0]),
            math.sqrt(cv[1,1]),
            math.sqrt(cv[2,2])
        ]

        r = numpy.identity(M) 
        for i in range(M):
            for j in range(i+1):
                den = se[i]*se[j]
                r[i,j] = r[j,i] = cv[i,j]/den 
                  
        self.assertTrue( equivalent(se[0],13.78503,tol=1E-5) )
        self.assertTrue( equivalent(se[1],0.01065,tol=1E-5) )
        self.assertTrue( equivalent(se[2],0.15214,tol=1E-5) )
        
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
        
        K = cholesky.cholesky_decomp(V)
        Kinv = cholesky.cholesky_inv(K)
        
        X = numpy.array( x )
        Y = numpy.array( y ).T
        
        Z = numpy.matmul(Kinv,Y) 
        Q = numpy.matmul(Kinv,X) 

        # X is N by M 
        x = []
        y = []
        for i in range(N):
            x.append( [ Q[i,j] for j in range(M) ] )
            y.append( Z[i] )
         
        a, chisq, w, v = SVD.svdfit(x,y,sig,fn)

        s2 = chisq/(N-M)
        cv = s2*svdvar(v,w)
        
        se = [
            math.sqrt(cv[0,0]),
            math.sqrt(cv[1,1]),
            math.sqrt(cv[2,2])
        ]
        
        # Values agree well with reference
        self.assertTrue( equivalent(a[0],94.89887752,tol=1E-7) )
        self.assertTrue( equivalent(a[1],0.06738948,tol=1E-7) )
        self.assertTrue( equivalent(a[2],-0.47427391,tol=1E-7) )
        
        # The std errors do not agree with the reference values, which are incorrect!
        # If a straightforward R OLS is carried out, 
        # i.e. add this to these steps in the reference
        # > gl <- lm(z~B-1)
        # > summary(gl,cor=T)
        # then we get the following standard errors
        self.assertTrue( equivalent(se[0],13.94477,tol=1E-5) )
        self.assertTrue( equivalent(se[1],0.01070,tol=1E-5) )
        self.assertTrue( equivalent(se[2],0.15339,tol=1E-5) )
        # Note, that I am sure the reference is wrong because using 
        # the formula in Draper & Smith p79 
        # (or wikipedia: https://en.wikipedia.org/wiki/Confidence_region )
        # the sum of squares for the residuals is 
        #   Z'Z - b'Q'Z 
        # I evaluated this in R and it is not the same as the reference 
        # value, but it is the same value that I obtain here.
        
#----------------------------------------------------------------------------
class TestSVDLinearSystems(unittest.TestCase):
    
    """
    Use SVD to solve linear systems of equations 
    """

    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test1(self):
        data = ([
            [2, -3],
            [4, 1]
        ])
        b = [-2,24]
        x_expect =[ 5, 4 ]

        a = numpy.array( data, dtype=float )

        x = SVD.solve(a,b)
        
        for i,j in zip(x,x_expect):
            self.assertTrue( equivalent(i,j) )
 
    #------------------------------------------------------------------------
    # This test does not use uncertain numbers for the data
    def test2(self):
        data = ([
            [2, 1, 3],
            [2, 6, 8],
            [6, 8, 18]
        ])
        b = [1,3,5]
        x_expect = [ 3./10., 4./10., 0.] 


        a = numpy.array( data, dtype=float )
        
        x = SVD.solve(a,b)

        for i,j in zip(x,x_expect):
            self.assertTrue( equivalent(i,j) )
 

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

        fit = SVD.ols(x,y,fn)
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
                
        fit = SVD.ols(x,y,fn,fn_inv)
        b,a = fit.beta
        
        x_0 = fit.x_from_y( [1.5] )
        equivalent(value(x_0),value(a),TOL)
        equivalent(uncertainty(x_0),uncertainty(a),TOL)

        # Incorrect input sequences
        self.assertRaises(
            RuntimeError,
            SVD.ols,
            numpy.array([1, 2, 3]), numpy.array([]), fn
         )

        self.assertRaises(
            RuntimeError,
            SVD.ols,
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

        fit = SVD.ols(x,y,fn)
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
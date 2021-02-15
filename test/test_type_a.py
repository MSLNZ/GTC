import unittest
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

import numpy

from GTC import *

from GTC.lib import (
    UncertainReal, 
    UncertainComplex,
    set_correlation_real,
    real_ensemble,
    complex_ensemble,
    append_real_ensemble
)

from testing_tools import *

TOL = 1E-13 
            
#---------------------------------------------------------
class StdDataSets(object):
    """
    See section 5 in:
    'Design and us of reference data sets for testing scientific software'
    M. G. Cox and P. M. Harris, Analytica Chemica Acta 380 (1999) 339-351.
    
    """
    
    def __init__(self,mu,h,q,n):
        self._mu = mu
        self._h = h
        self._q = q
        self._n = n

    def seq(self,k=1):
        self._k = k
        
        N = self._n
        a = numpy.array( xrange(-N,N+1) ) * self._h
        q = self._q ** self._k
        
        return (self._mu + q) - a

    def mean(self):
        return self._mu + (self._q ** self._k)

    def std(self):
        N = self._n
        return self._h * math.sqrt((N + 0.5)*(N+1)/3.0)
        
#-----------------------------------------------------
class TestTypeA(unittest.TestCase):

    def testMean(self):
        TOL = 1E-12
        
        data_ = StdDataSets(mu=3.172,h=0.1,q=1.5,n=1000)

        for k in range(5):
            seq = data_.seq(k)
            self.assertTrue( equivalent( numpy.mean(seq) , data_.mean(), TOL) )
            self.assertTrue( equivalent( type_a.mean(seq) , data_.mean(), TOL) )
            self.assertTrue( equivalent( type_a.mean(seq) , numpy.mean(seq), TOL) )

        # takes an iterable object 
        rng = range(100)
        itr = iter(rng)
        self.assertTrue( equivalent( type_a.mean(itr) , type_a.mean(rng), TOL) )
        
    def testUNMean(self):
        """A sequence of uncertain numbers"""
        
        TOL = 1E-12
        data_ = StdDataSets(mu=-3.172,h=0.2,q=1.5,n=10)
        
        seq = [ ureal(x_i,1) for x_i in data_.seq() ]
        self.assertTrue( equivalent( type_a.mean(seq) , data_.mean(), TOL) )
        
        useq = la.uarray(seq)
        self.assertTrue( equivalent( type_a.mean(useq) , data_.mean(), TOL) )

    def testStd(self):
        TOL = 1E-13
        
        data_ = StdDataSets(mu=3.172,h=0.1,q=1.5,n=1000)
        
        for k in range(5):
            seq = data_.seq(k)
            
            N = float(len(seq))
            root_N = math.sqrt(N)
            
            # numpy.std divides the variance by len(seq), not len(seq)-1, unless ddof=1
            self.assertTrue( equivalent( numpy.std(seq,ddof=1) , data_.std(), TOL) )
            
            self.assertTrue( equivalent(
                root_N * type_a.standard_uncertainty(seq) ,
                data_.std(), TOL)
            )
            self.assertTrue( equivalent(
                root_N * type_a.standard_uncertainty(seq) ,
                numpy.std(seq,ddof=1), TOL )
            )            

            useq = la.uarray(seq)
            self.assertTrue( equivalent(
                root_N * type_a.standard_uncertainty(useq) ,
                data_.std(), TOL)
            )
    
    def testUNStd(self):
        TOL = 1E-13
        
        data_ = StdDataSets(mu=-3.172,h=0.1,q=1.5,n=10)
        
        seq = [ ureal(x_i,1) for x_i in data_.seq() ]
        
        N = float(len(seq))
        root_N = math.sqrt(N)
        
        self.assertTrue( equivalent(
            root_N * type_a.standard_uncertainty(seq) ,
            data_.std(), TOL)
        )

        useq = la.uarray(seq)
        self.assertTrue( equivalent(
            root_N * type_a.standard_uncertainty(useq) ,
            data_.std(), TOL)
        )

    def testComplexMean(self):
        TOL = 1E-12

        mu = complex(3.172,-0.123)
        
        re_data_ = StdDataSets(mu=mu.real,h=0.1,q=1.5,n=1000)
        im_data_ = StdDataSets(mu=mu.imag,h=0.1,q=1.5,n=1000)

        for k in range(5):
            re_seq = re_data_.seq(k)
            im_seq = im_data_.seq(k)
            zseq = [ complex(i,j) for i,j in izip(re_seq,im_seq) ]
        
            self.assertTrue(
                equivalent_complex(
                    type_a.mean(zseq),
                    complex( re_data_.mean(),im_data_.mean() ),
                    TOL
                )
            )

    def testUNComplexMean(self):
        TOL = 1E-12

        mu = complex(-3.172,0.123)
        
        re_data_ = StdDataSets(mu=mu.real,h=0.1,q=1.5,n=10)
        im_data_ = StdDataSets(mu=mu.imag,h=0.1,q=1.5,n=10)

        re_seq = re_data_.seq()
        im_seq = im_data_.seq()
        zseq = [ ucomplex(complex(i,j),1.0) for i,j in izip(re_seq,im_seq) ]
        
        equivalent_complex(
            type_a.mean(zseq),
            complex( re_data_.mean(),im_data_.mean() ),
            TOL
        )

        uzseq = la.uarray(zseq)
        equivalent_complex(
            type_a.mean(uzseq),
            complex( re_data_.mean(),im_data_.mean() ),
            TOL
        )
        
    def testComplexUncertainties(self):
        TOL = 1E-12

        mu = complex(3.172,-0.123)
        
        re_data_ = StdDataSets(mu=mu.real,h=0.1,q=1.5,n=1000)
        im_data_ = StdDataSets(mu=mu.imag,h=0.1,q=1.5,n=1000)
        
        for k in range(5):
            re_seq = re_data_.seq(k)
            im_seq = im_data_.seq(k)
            zseq = [ complex(i,j) for i,j in izip(re_seq,im_seq) ]

            root_N = math.sqrt( len(zseq) )            

            (u_re,u_im), r = type_a.standard_uncertainty(zseq)

            self.assertTrue( equivalent(r,1.0,TOL) )
            self.assertTrue( equivalent( root_N * u_re , re_data_.std(), TOL) )
            self.assertTrue( equivalent( root_N * u_im , im_data_.std(), TOL) )

    def testUNComplexUncertainties(self):
        TOL = 1E-12

        mu = complex(-3.172,-0.123)
        
        re_data_ = StdDataSets(mu=mu.real,h=0.1,q=1.5,n=10)
        im_data_ = StdDataSets(mu=mu.imag,h=0.1,q=1.5,n=10)
        
        re_seq = re_data_.seq()
        im_seq = im_data_.seq()
        zseq = [ ucomplex( complex(i,j), 1) for i,j in izip(re_seq,im_seq) ]

        root_N = math.sqrt( len(zseq) )            

        (u_re,u_im), r = type_a.standard_uncertainty(zseq)

        self.assertTrue( equivalent(r,1.0,TOL) )
        self.assertTrue( equivalent( root_N * u_re , re_data_.std(), TOL) )
        self.assertTrue( equivalent( root_N * u_im , im_data_.std(), TOL) )

        uzseq = la.uarray(zseq)
        (u_re,u_im), r = type_a.standard_uncertainty(uzseq)

        self.assertTrue( equivalent(r,1.0,TOL) )
        self.assertTrue( equivalent( root_N * u_re , re_data_.std(), TOL) )
        self.assertTrue( equivalent( root_N * u_im , im_data_.std(), TOL) )

    def testTypeAComplex(self):
        TOL = 1E-12

        mu = complex(3.172,-0.123)
        
        re_data_ = StdDataSets(mu=mu.real,h=0.1,q=1.5,n=1000)
        im_data_ = StdDataSets(mu=mu.imag,h=0.1,q=1.5,n=1000)
        
        for k in range(5):
            re_seq = re_data_.seq(k)
            im_seq = im_data_.seq(k)
            zseq = [ complex(i,j) for i,j in izip(re_seq,im_seq) ]

            root_N = math.sqrt( len(zseq) )            

            z = type_a.estimate(zseq)
            
            self.assertTrue( equivalent(get_correlation(z),1.0,TOL) )
            self.assertEqual( dof(z), len(zseq)-1 )
            self.assertTrue( equivalent_complex(value(z),complex( re_data_.mean(),im_data_.mean() ),TOL) )
            
            u = uncertainty(z)
            self.assertTrue( equivalent( root_N * u[0] , re_data_.std(), TOL) )
            self.assertTrue( equivalent( root_N * u[1] , im_data_.std(), TOL) )

    def testUNTypeAComplex(self):
        TOL = 1E-12

        mu = complex(3.172,0.123)
        
        re_data_ = StdDataSets(mu=mu.real,h=0.1,q=1.5,n=10)
        im_data_ = StdDataSets(mu=mu.imag,h=0.1,q=1.5,n=10)
        
        re_seq = re_data_.seq()
        im_seq = im_data_.seq()
        zseq = [ complex(i,j) for i,j in izip(re_seq,im_seq) ]

        root_N = math.sqrt( len(zseq) )            

        z = type_a.estimate(zseq)
        
        self.assertTrue( equivalent(get_correlation(z),1.0,TOL) )
        self.assertEqual( dof(z), len(zseq)-1 )
        self.assertTrue( equivalent_complex(value(z),complex( re_data_.mean(),im_data_.mean() ),TOL) )
        
        u = uncertainty(z)
        self.assertTrue( equivalent( root_N * u[0] , re_data_.std(), TOL) )
        self.assertTrue( equivalent( root_N * u[1] , im_data_.std(), TOL) )

    def testTypeAReal(self):
        TOL = 1E-12

        data_ = StdDataSets(mu=-11.342,h=0.1,q=1.5,n=1000)
        
        for k in range(5):
            seq = data_.seq(k)
            
            root_N = math.sqrt(len(seq))
            
            x = type_a.estimate(seq)
            
            self.assertEqual( dof(x), len(seq)-1 )
            self.assertTrue( equivalent(value(x),data_.mean(),TOL) )
            self.assertTrue( equivalent( root_N * uncertainty(x) , data_.std(), TOL) )

    def testUNTypeAReal(self):
        TOL = 1E-12

        data_ = StdDataSets(mu=11.342,h=0.1,q=1.5,n=10)
        
        seq = data_.seq()
        
        root_N = math.sqrt(len(seq))
        
        x = type_a.estimate(seq)
        
        self.assertEqual( dof(x), len(seq)-1 )
        self.assertTrue( equivalent(value(x),data_.mean(),TOL) )
        self.assertTrue( equivalent( root_N * uncertainty(x) , data_.std(), TOL) )


#-----------------------------------------------------
class TestEnsembleWS(unittest.TestCase):
    
    """
    Test case for the type_a function that defines an ensemble
    of estimates, which are based on a multivariate sample.

    Also checks that the WS routine can handle the ensemble
    calculation.
    """
    
    def test_GUM_H2_wo_labels(self):
        TOL = 1E-5
        
        V = [5.007,4.994,5.005,4.990,4.999]
        I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        phi = [1.0456,1.0438,1.0468,1.0428,1.0433]

        data = (V,I,phi)

        seq = type_a.multi_estimate_real(data)
        v,i,p = seq

        # Check that the calculation of covariance and the
        # definition of ureals is correct.
        self.assertTrue( equivalent(value(v),4.9990,TOL) )
        self.assertTrue( equivalent(uncertainty(v),0.0032,TOL) )
        self.assertTrue( equivalent(dof(v),4,TOL) )

        self.assertTrue( equivalent(value(i),0.019661,TOL) )
        self.assertTrue( equivalent(uncertainty(i),0.0000095,TOL) )
        self.assertTrue( equivalent(dof(i),4,TOL) )

        self.assertTrue( equivalent(value(p),1.04446,TOL) )
        self.assertTrue( equivalent(uncertainty(p),0.00075,TOL) )
        self.assertTrue( equivalent(dof(p),4,TOL) )

        self.assertTrue( equivalent(get_correlation(v,i),-0.36,1E-2) )
        self.assertTrue( equivalent(get_correlation(i,p),-0.65,1E-2) )
        self.assertTrue( equivalent(get_correlation(v,p),0.86,1E-2) )

        # Perform the data analysis and check that calculations
        # are correct.
        r = v/i*cos(p)
        x = v/i*sin(p)
        z = v/i

        # Comparing with the numbers in the GUM
        TOL = 1E-3
        self.assertTrue( equivalent(value(r),127.732,TOL) )
        self.assertTrue( equivalent(uncertainty(r), 0.071,TOL) )
        self.assertTrue( equivalent(dof(r),4,TOL) )

        self.assertTrue( equivalent(value(x), 219.847,TOL) )
        self.assertTrue( equivalent(uncertainty(x), 0.295,TOL) )
        self.assertTrue( equivalent(dof(x),4,TOL) )

        self.assertTrue( equivalent(value(z), 254.260,TOL) )
        self.assertTrue( equivalent(uncertainty(z), 0.236,TOL) )
        self.assertTrue( equivalent(dof(z),4,TOL) )

        equivalent( get_correlation(r,x),-0.588,TOL)
        equivalent( get_correlation(x,z),0.993,TOL)
        equivalent( get_correlation(r,z),-0.485,TOL)

    def test_GUM_H2(self):
        """Test that labels are correctly assigned
        
        No need to repeat all of the above tests again
        
        """
        TOL = 1E-5
        
        V = [5.007,4.994,5.005,4.990,4.999]
        I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        phi = [1.0456,1.0438,1.0468,1.0428,1.0433]

        data = (V,I,phi)
        labels = ('V','I','phi')
        seq = type_a.multi_estimate_real(data,labels=labels)
        v,i,p = seq
 
        self.assertEqual(v.label,labels[0])
        self.assertEqual(i.label,labels[1])
        self.assertEqual(p.label,labels[2])
        
        # Check that the calculation of covariance and the
        # definition of ureals is correct.
        self.assertTrue( equivalent(value(v),4.9990,TOL) )
        self.assertTrue( equivalent(uncertainty(v),0.0032,TOL) )
        self.assertTrue( equivalent(dof(v),4,TOL) )

        self.assertTrue( equivalent(value(i),0.019661,TOL) )
        self.assertTrue( equivalent(uncertainty(i),0.0000095,TOL) )
        self.assertTrue( equivalent(dof(i),4,TOL) )

        self.assertTrue( equivalent(value(p),1.04446,TOL) )
        self.assertTrue( equivalent(uncertainty(p),0.00075,TOL) )
        self.assertTrue( equivalent(dof(p),4,TOL) )
        
    def test_GUM_H2_illegal(self):
        """Test illegal cases:
            - different length sequences
            - incompatible length labels
            
        """
        V = [5.007,4.994,5.005,4.990,4.999]
        I = [19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3]
        phi = [1.0456,1.0438,1.0468,1.0428,1.0433]

        data = (V,I,phi)
        labels = ('V','I','phi')
        
        self.assertRaises(RuntimeError,type_a.multi_estimate_real,data,labels[:-1])

        data = (V[:-1],I,phi)
        self.assertRaises(RuntimeError,type_a.multi_estimate_real,data,labels)
        self.assertRaises(RuntimeError,type_a.multi_estimate_real,data)
        data = (V,I,phi[:-1])
        self.assertRaises(RuntimeError,type_a.multi_estimate_real,data)

    def test_GUM_H2_wo_labels_complex(self):
        TOL = 1E-5
        
        V = [ complex(x,0) for x in (5.007,4.994,5.005,4.990,4.999) ]
        I = [ complex(x,0) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3)]
        phi = [ complex(0,y) for y in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        data = (V,I,phi)

        seq = type_a.multi_estimate_complex(data)
        v,i,p = seq

        # Check that the calculation of covariance and the
        # definition of ureals is correct.
        self.assertTrue( equivalent_complex(value(v),4.9990,TOL) )
        self.assertTrue( equivalent(uncertainty(v.real),0.0032,TOL) )
        self.assertTrue( equivalent(dof(v),4,TOL) )

        self.assertTrue( equivalent_complex(value(i),0.019661,TOL) )
        self.assertTrue( equivalent(uncertainty(i.real),0.0000095,TOL) )
        self.assertTrue( equivalent(dof(i),4,TOL) )

        self.assertTrue( equivalent_complex(value(p),0+1.04446j,TOL) )
        self.assertTrue( equivalent(uncertainty(p.imag),0.00075,TOL) )
        self.assertTrue( equivalent(dof(p),4,TOL) )

        self.assertTrue( equivalent(get_correlation(v.real,i.real),-0.36,1E-2) )
        self.assertTrue( equivalent(get_correlation(i.real,p.imag),-0.65,1E-2) )
        self.assertTrue( equivalent(get_correlation(v.real,p.imag),0.86,1E-2) )

        # Perform the data analysis and check that calculations
        # are correct.
        z = v/i*exp(p)

        # Comparing with the numbers in the GUM
        TOL = 1E-3
        self.assertTrue( equivalent(value(z.real),127.732,TOL) )
        self.assertTrue( equivalent(uncertainty(z.real), 0.071,TOL) )
        self.assertTrue( equivalent(dof(z),4,TOL) )

        self.assertTrue( equivalent(value(z.imag), 219.847,TOL) )
        self.assertTrue( equivalent(uncertainty(z.imag), 0.295,TOL) )

        equivalent( get_correlation(z.real,z.imag),-0.588,TOL)

    def test_GUM_H2_complex(self):
        TOL = 1E-5
        
        V = [ complex(x,0) for x in (5.007,4.994,5.005,4.990,4.999) ]
        I = [ complex(x,0) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3)]
        phi = [ complex(0,y) for y in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        data = (V,I,phi)
        labels = ('V','I','phi')

        seq = type_a.multi_estimate_complex(data,labels=labels)
        v,i,p = seq

        self.assertEqual(v.label,labels[0])
        self.assertEqual(i.label,labels[1])
        self.assertEqual(p.label,labels[2])
        
        # Check that the calculation of covariance and the
        # definition of ureals is correct.
        self.assertTrue( equivalent_complex(value(v),4.9990,TOL) )
        self.assertTrue( equivalent(uncertainty(v.real),0.0032,TOL) )
        self.assertTrue( equivalent(dof(v),4,TOL) )

        self.assertTrue( equivalent_complex(value(i),0.019661,TOL) )
        self.assertTrue( equivalent(uncertainty(i.real),0.0000095,TOL) )
        self.assertTrue( equivalent(dof(i),4,TOL) )

        self.assertTrue( equivalent_complex(value(p),0+1.04446j,TOL) )
        self.assertTrue( equivalent(uncertainty(p.imag),0.00075,TOL) )
        self.assertTrue( equivalent(dof(p),4,TOL) )

        self.assertTrue( equivalent(get_correlation(v.real,i.real),-0.36,1E-2) )
        self.assertTrue( equivalent(get_correlation(i.real,p.imag),-0.65,1E-2) )
        self.assertTrue( equivalent(get_correlation(v.real,p.imag),0.86,1E-2) )

    def test_GUM_H2_complex_illegal(self):
        """Test illegal cases:
            - different length sequences
            - incompatible length labels
            
        """
        V = [ complex(x,0) for x in (5.007,4.994,5.005,4.990,4.999) ]
        I = [ complex(x,0) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3)]
        phi = [ complex(0,y) for y in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        data = (V,I,phi)
        labels = ('V','I','phi')

        
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data,labels[:-1])

        data = (V[:-1],I,phi)
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data,labels)
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data)
        data = (V,I,phi[:-1])
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data) 
        
#-----------------------------------------------------
class TestEnsembleWSComplex(unittest.TestCase):
    
    """
    Test case for the type_a function that defines an ensemble
    of estimates, which are based on a multivariate sample.

    Also checks that the WS routine can handle the ensemble
    calculation.
    """
    

    def test_GUM_H2_wo_labels_complex(self):
        TOL = 1E-5
        
        V = [ complex(x,0) for x in (5.007,4.994,5.005,4.990,4.999) ]
        I = [ complex(x,0) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3)]
        phi = [ complex(0,y) for y in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        data = (V,I,phi)

        seq = type_a.multi_estimate_complex(data)
        v,i,p = seq

        # Check that the calculation of covariance and the
        # definition of ureals is correct.
        self.assertTrue( equivalent_complex(value(v),4.9990,TOL) )
        self.assertTrue( equivalent(uncertainty(v.real),0.0032,TOL) )
        self.assertTrue( equivalent(dof(v),4,TOL) )

        self.assertTrue( equivalent_complex(value(i),0.019661,TOL) )
        self.assertTrue( equivalent(uncertainty(i.real),0.0000095,TOL) )
        self.assertTrue( equivalent(dof(i),4,TOL) )

        self.assertTrue( equivalent_complex(value(p),0+1.04446j,TOL) )
        self.assertTrue( equivalent(uncertainty(p.imag),0.00075,TOL) )
        self.assertTrue( equivalent(dof(p),4,TOL) )

        self.assertTrue( equivalent(get_correlation(v.real,i.real),-0.36,1E-2) )
        self.assertTrue( equivalent(get_correlation(i.real,p.imag),-0.65,1E-2) )
        self.assertTrue( equivalent(get_correlation(v.real,p.imag),0.86,1E-2) )

        # Perform the data analysis and check that calculations
        # are correct.
        z = v/i*exp(p)

        # Comparing with the numbers in the GUM
        TOL = 1E-3
        self.assertTrue( equivalent(value(z.real),127.732,TOL) )
        self.assertTrue( equivalent(uncertainty(z.real), 0.071,TOL) )
        self.assertTrue( equivalent(dof(z),4,TOL) )

        self.assertTrue( equivalent(value(z.imag), 219.847,TOL) )
        self.assertTrue( equivalent(uncertainty(z.imag), 0.295,TOL) )

        equivalent( get_correlation(z.real,z.imag),-0.588,TOL)

    def test_GUM_H2_complex(self):
        TOL = 1E-5
        
        V = [ complex(x,0) for x in (5.007,4.994,5.005,4.990,4.999) ]
        I = [ complex(x,0) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3)]
        phi = [ complex(0,y) for y in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        data = (V,I,phi)
        labels = ('V','I','phi')

        seq = type_a.multi_estimate_complex(data,labels=labels)
        v,i,p = seq

        self.assertEqual(v.label,labels[0])
        self.assertEqual(i.label,labels[1])
        self.assertEqual(p.label,labels[2])
        
        # Check that the calculation of covariance and the
        # definition of ureals is correct.
        self.assertTrue( equivalent_complex(value(v),4.9990,TOL) )
        self.assertTrue( equivalent(uncertainty(v.real),0.0032,TOL) )
        self.assertTrue( equivalent(dof(v),4,TOL) )

        self.assertTrue( equivalent_complex(value(i),0.019661,TOL) )
        self.assertTrue( equivalent(uncertainty(i.real),0.0000095,TOL) )
        self.assertTrue( equivalent(dof(i),4,TOL) )

        self.assertTrue( equivalent_complex(value(p),0+1.04446j,TOL) )
        self.assertTrue( equivalent(uncertainty(p.imag),0.00075,TOL) )
        self.assertTrue( equivalent(dof(p),4,TOL) )

        self.assertTrue( equivalent(get_correlation(v.real,i.real),-0.36,1E-2) )
        self.assertTrue( equivalent(get_correlation(i.real,p.imag),-0.65,1E-2) )
        self.assertTrue( equivalent(get_correlation(v.real,p.imag),0.86,1E-2) )

    def test_GUM_H2_complex_illegal(self):
        """Test illegal cases:
            - different length sequences
            - incompatible length labels
            
        """
        V = [ complex(x,0) for x in (5.007,4.994,5.005,4.990,4.999) ]
        I = [ complex(x,0) for x in (19.663E-3,19.639E-3,19.640E-3,19.685E-3,19.678E-3)]
        phi = [ complex(0,y) for y in (1.0456,1.0438,1.0468,1.0428,1.0433) ]

        data = (V,I,phi)
        labels = ('V','I','phi')
        
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data,labels[:-1])

        data = (V[:-1],I,phi)
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data,labels)
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data)
        data = (V,I,phi[:-1])
        self.assertRaises(RuntimeError,type_a.multi_estimate_complex,data)
    
    
    def test_WS_complex_WH_ex1(self):
        """
        This is Example 1 in 
            "An extension to GUM methodology:
            degrees-of-freedom calculations for correlated
            multidimensional estimates", by 
            Willink and Hall
            
        """
        s11 = [
            0.0242-0.0101j, -0.0023+0.2229j,
            0.0599+0.0601j, 0.0433+0.2100j,
            -0.0026+0.0627j
        ]
        S11 = type_a.estimate( s11, label='S11' )
        self.assertTrue( equivalent_complex(S11.x,(0.02450+0.10912j),tol=1E-8) )
        self.assertTrue( equivalent_sequence(S11.v,(0.1530E-3,-0.0797E-3,-0.0797E-3,2.0947E-3),tol=1E-7) )

        g_prime = [
            0.1648-0.0250j, 0.1568-0.0179j,
            0.1598-0.1367j, 0.1198-0.0045j,
            0.3162-0.1310j
        ]
        G_prime = type_a.estimate( g_prime, label='G_prime' )        
        self.assertTrue( equivalent_complex(G_prime.x,(0.18348-0.06302j),tol=1E-8) )
        self.assertTrue( equivalent_sequence(G_prime.v,(1.1646E-3,-0.6459E-3,-0.6459E-3,0.8478E-3),tol=1E-7) )
   
        G = result(G_prime - S11,label = 'G')
        self.assertTrue( equivalent_complex(G.x,(0.15898-0.17214j),tol=1E-8) )
        self.assertTrue( equivalent_sequence(G.v,(1.3175E-3,-0.7256E-3,-0.7256E-3,2.9425E-3),tol=1E-7) )
        self.assertTrue( equivalent(G.df,6.85,tol=5E-3) )
  
    def test_WS_complex_WH_ex2(self):
        """
        This is Example 2 in 
            "An extension to GUM methodology:
            degrees-of-freedom calculations for correlated
            multidimensional estimates", by 
            Willink and Hall
            
        """
        g_prime = [
            -0.220+0.008j, -0.217-0.010j,
            -0.222-0.003j, -0.207-0.018j,
            -0.219-0.009j
        ]
        G_prime = type_a.estimate( g_prime, label='G_prime' )        
        self.assertTrue( equivalent_complex(G_prime.x,(-0.2170-0.0064j),tol=1E-10) )
        self.assertTrue( 
            equivalent_sequence(
                G_prime.v,
                (0.6900E-5,-0.8550E-5,-0.8550E-5,1.8660E-5),
                tol=1E-10
            ) 
        )

        s11 = [
            -0.072+0.066j, -0.075+0.073j,
            -0.087+0.069j, -0.087+0.092j,
            -0.064+0.071j, -0.079+0.067j,
            -0.069+0.072j
        ]
        s12 = [
            0.112+0.903j, 0.115+0.891j,
            0.094+0.888j, 0.098+0.898j,
            0.092+0.899j, 0.072+0.892j,
            0.102+0.906j
        ]
        s21 = [
            0.106+0.900j, 0.098+0.900j,
            0.085+0.893j, 0.106+0.886j,
            0.114+0.907j, 0.089+0.882j,
            0.108+0.893j
        ]
        s22 = [
            0.044+0.106j, 0.029+0.091j,
            0.019+0.096j, 0.036+0.099j,
            0.026+0.102j, 0.026+0.088j,
            0.022+0.096j
        ]
        # Correlated estimates
        S11, S12, S21, S22 = type_a.multi_estimate_complex(
            (s11,s12,s21,s22),
            labels =('S11','S12','S21','S22')
        ) 
   
        # The variance-covariance matrix
        self.assertTrue( 
            equivalent_sequence(
                S11.v,
                (1.0973E-5,-0.4908E-5,-0.4908E-5,1.1116E-5),
                tol=1E-8
            ) 
        )
        self.assertTrue( 
            equivalent_sequence(
                S12.v,
                (2.9259E-5,0.4088E-5,0.4088E-5,0.6272E-5),
                tol=1E-8
            ) 
        )
        self.assertTrue( 
            equivalent_sequence(
                S21.v,
                (1.6116E-5,0.7010E-5,0.7010E-5,1.0707E-5),
                tol=1E-8
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                S22.v,
                (1.0497E-5,0.4235E-5,0.4235E-5,0.5449E-5),
                tol=1E-8
            ) 
        )
        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S11,S12),
                (0.3592E-5,0.4946E-5,0.1949E-5,0.0707E-5),
                tol=1E-9
            ) 
        )
        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S11,S21),
                (0.9020E-5,0.7486E-5,0.3878E-5,-0.3395E-5),
                tol=1E-9
            ) 
        )
        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S11,S22),
                (0.0401E-5,0.2354E-5,0.2354E-5,0.0568E-5),
                tol=1E-9
            ) 
        )
        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S12,S11),
                (0.3592E-5,0.1949E-5,0.4946E-5,0.0707E-5),
                tol=1E-9
            ) 
        )
 
        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S12,S21),
                (0.8211E-5,1.0010E-5,0.8231E-5,0.1878E-5),
                tol=1E-9
            ) 
        )
 
        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S12,S22),
                (0.7568E-5,0.5425E-5,0.3160E-5,0.3493E-5),
                tol=1E-9
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S21,S11),
                (0.9020E-5,0.3878E-5,0.7486E-5,-0.3395E-5),
                tol=1E-9
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S21,S12),
                (0.8211E-5,0.8231E-5,1.0010E-5,0.1878E-5),
                tol=1E-9
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S21,S22),
                (0.5187E-5,0.6068E-5,0.1153E-5,0.4224E-5),
                tol=1E-9
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S22,S11),
                (0.0401E-5,0.2354E-5,0.2354E-5,0.0568E-5),
                tol=1E-9
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S22,S12),
                (0.7568E-5,0.3160E-5,0.5425E-5,0.3493E-5),
                tol=1E-9
            ) 
        )

        self.assertTrue( 
            equivalent_sequence(
                get_covariance(S22,S21),
                (0.5187E-5,0.1153E-5,0.6068E-5,0.4224E-5),
                tol=1E-9
            ) 
        )

        # Calculation of measurement result
        num = G_prime - S11
        den = S12*S21 + S22*num
        G = result( num/den, label = 'G')
        self.assertTrue( equivalent_complex(G.x,(0.1516+0.1317j),tol=5E-5) )
        self.assertTrue( equivalent_sequence(G.v,(3.035E-5,-2.583E-5,-2.583E-5,4.199E-5),tol=1E-8) )
        self.assertTrue( equivalent(G.df,9.01,tol=5E-3) )
    
#-----------------------------------------------------
#
from test_fitting import simple_sigma_abr

class TestLineFit(unittest.TestCase):

    """
    Tests of the type_a.line_fit function

    NB, these tests call the routine twice (once internally)
    and go though the chi-square calculation in the first call.
    """

    TOL = 1E-5
    
    def test_integer_x_values(self):
        """
        The integer arithmetic of Python can be a problem
        when the `x` data are integers.
        """
        # Data - independent variable
        # NB these are integers and will cause
        # type_b.line_fit to fail if the extended division is not implemented
        x_short = [2,4,8,16,32]

        # x will simply repeat each values 3 times, i.e.: [2,2,2,4,4,4,...]
        # This way we don't need to use weighted LS
        x = []
        for x_i in x_short:
            x.extend( 3*[x_i] )

        # Data - dependent variable
        y = (1032.,	1021.,	1016., 3012.,	3001.,	3022., 7089.,	7111.,	
        7080., 15102.,	15087.,	15105., 30469.,	30461.,	30481.)

        result_1 = type_a.line_fit(x,y)
   
        # Alternatively, group the data into repeated observations
        # and create uncertain numbers for each. Propagate the
        # uncertainty to `a` and `b`.
        # This is not equivalent to the first method in terms of uncertainty!
        y = [(1032.,1021.,1016.),
             (3012.,3001.,3022.),
             (7089.,7111.,7080.),
             (15102.,15087.,15105.),
             (30469.,30461.,30481.)
        ]

        y_est = [ ta.estimate(y_i) for y_i in y]
        result_2 = type_b.line_fit(x_short,y_est)

        self.assertTrue( equivalent( value(result_1.a_b.a), value(result_2.a_b.a), tol=1E-10 ) )
        self.assertTrue( equivalent( value(result_1.a_b.b), value(result_2.a_b.b), tol=1E-10 ) )
        
    def test_walpole(self):
        """
        Example from Walpole + Myers, but the numerical results
        were done using R, because Walpole made an error with
        their t-distribution 'k' factor.
        
        In R:
            fit <- lm(y~x)
            summary(fit)
            vcov(fit)
            
        """
        x = [3.,7.,11.,15.,18.,27.,29.,30.,30.,31.,31.,32.,33.,33.,34.,36.,36.,
             36.,37.,38.,39.,39.,39.,40.,41.,42.,42.,43.,44.,45.,46.,47.,50.]
        y = [5.,11.,21.,16.,16.,28.,27.,25.,35.,30.,40.,32.,34.,32.,34.,37.,38.,
             34.,36.,38.,37.,36.,45.,39.,41.,40.,44.,37.,44.,46.,46.,49.,51.]

        TOL = 1E-5
        fit = type_a.line_fit(x,y)
        a,b = fit.a_b
        self.assertTrue( equivalent( value(a), 3.82963, self.TOL) )
        self.assertTrue( equivalent( uncertainty(a), 1.768447, self.TOL) )
        self.assertTrue( equivalent( value(b), 0.90364, self.TOL) )
        self.assertTrue( equivalent( uncertainty(b), 0.05011898, self.TOL) )

        # prediction would be based on 31 dof, but the std uncertainty is key
        x_0 = 20.0
        y_0 = a + b*x_0
        self.assertTrue( equivalent( value(y_0), 21.9025, self.TOL) )
        self.assertTrue( equivalent( uncertainty(y_0), 0.877939, self.TOL) )

    def test_bevington(self):
        """
        Example from Bevington Table 6.1
        Some calculations done in R

        In R:
            fit <- lm(y~x)
            summary(fit)
            vcov(fit)
        
        """
        x = [4.,8.,12.5,16.,20.,25.,31.,36.,40.,40.]
        y = [3.7,7.8,12.1,15.6,19.8,24.5,30.7,35.5,39.4,39.5]
        a,b = type_a.line_fit(x,y).a_b
        
        self.assertTrue( equivalent( value(a), -0.222142, self.TOL) )
        self.assertTrue( equivalent( uncertainty(a), 0.06962967, self.TOL) )
        self.assertTrue( equivalent( value(b), 0.992780, self.TOL) )
        self.assertTrue( equivalent( uncertainty(b), 0.002636608, self.TOL) )
        self.assertTrue( equivalent( a.u*b.u*get_correlation(a,b), -0.0001616271, self.TOL) )

        a_u,b_u,r_ = simple_sigma_abr(x,y)
        self.assertTrue( equivalent( uncertainty(a), a_u, self.TOL) )
        self.assertTrue( equivalent( uncertainty(b), b_u, self.TOL) )
        self.assertTrue( equivalent( get_correlation(a,b), r_, self.TOL) )

    def test_H3(self):
        """H3 from the GUM
        """
        t_k = (21.521,22.012,22.512,23.003,23.507,23.999,24.513,25.002,25.503,26.010,26.511)
        b_k = (-0.171,-0.169,-0.166,-0.159,-0.164,-0.165,-0.156,-0.157,-0.159,-0.161,-0.160)
        theta = [ t_k_i - 20 for t_k_i in t_k ]

        fit = type_a.line_fit(theta,b_k)
        a,b = fit.a_b

        # Compare with GUM values

        self.assertTrue( equivalent(value(a),-0.1712,1E-4) )
        self.assertTrue( equivalent(value(b),0.00218,1E-5) )
        self.assertTrue( equivalent(get_correlation(a,b),-0.930,1E-3) )
        
        b_30 = a + b*(30 - 20)
        self.assertTrue( equivalent(b_30.x,-0.1494,1E-4) )
        self.assertTrue( equivalent(b_30.u,0.0041,1E-4) )
        self.assertTrue( equivalent(b_30.df,9,1E-13) )

        # `y_from_x` is the predicted single `y` response
        # which has greater variability        
        b_30 = fit.y_from_x(30 - 20)
        self.assertTrue( not b_30.is_intermediate )
        self.assertTrue( equivalent(b_30.x,-0.1494,1E-4) )
        b_30 = fit.y_from_x(30 - 20,y_label='b_30')
        self.assertTrue( equivalent(b_30.x,-0.1494,1E-4) )
        self.assertTrue( b_30.is_intermediate )

    def test_A5(self):
        """CITAC 3rd edition

        Test the calibration curve aspect
        """
        x_data = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9]
        y_data = [0.028, 0.029, 0.029, 0.084, 0.083, 0.081, 0.135, 0.131, 0.133, 0.180,
                  0.181, 0.183, 0.215, 0.230, 0.216]

        fit = ta.line_fit(x_data,y_data)
        a, b = fit.a_b
        c_0 = fit.x_from_y( [0.0712, 0.0716] )

        # The classical uncertainty
        N = len(x_data)
        xmean = type_a.mean(x_data)
        sxx = sum( (x_i-xmean)**2 for x_i in x_data )
        S = math.sqrt(fit.ssr/(N-2))

        _x = c_0.x
        u_c_0 = S*math.sqrt(1.0/2 + 1.0/N + (_x-xmean)**2 / sxx)/b.x

        self.assertTrue(equivalent(u_c_0,c_0.u,TOL))
        self.assertEqual(c_0.df,N-2)

        # Now in the opposite sense
        y_0 = fit.y_from_x(_x)
        u_y_0 = S*math.sqrt(1.0 + 1.0/N + (_x-xmean)**2/sxx)
        
        self.assertTrue(equivalent(value(y_0),0.0714,TOL))
        self.assertTrue(equivalent(u_y_0,y_0.u,TOL))
        self.assertEqual(y_0.df,N-2)
   
#-----------------------------------------------------
class TestCombineComponents(unittest.TestCase):

    def test_mean(self):
        """
        Using the type_a functions should always
        return number types.

        Generate a sequence of uncertain reals
        for V and then do a type_a and type_b
        analysis.

        Combine the results to merge both
        types of uncertainty.
        
        """
        TOL = 1E-13
        
        V = [5.007,4.994,5.005,4.990,4.999]
        mu_V = math.fsum(V)/len(V)
        sd = type_a.standard_uncertainty(V)

        u = 0.01
        x = [ ureal(v_i,u) for v_i in V ]

        mean = type_a.mean(x)

        self.assertTrue( isinstance(mean,float) )
        self.assertTrue( equivalent(mean,mu_V, TOL) )
        
        mu_a = type_a.estimate(x)
        self.assertTrue( equivalent(mu_a.x,mu_V, TOL) )
        self.assertTrue( equivalent(mu_a.u,sd, TOL) )

        mu_b = result( sum(x)/len(x) )
        self.assertTrue( rp.is_ureal(mu_b) )
        self.assertTrue( equivalent(mu_b.u,u/math.sqrt(len(x)), TOL) )

        mu = result( type_a.merge(mu_a,mu_b) )
        
        u_c = math.sqrt(mu_a.v + mu_b.v)
        self.assertTrue( equivalent(mu.u,u_c, TOL) )
        self.assertTrue( equivalent(component(mu,mu_a),mu_a.u, TOL) )
        self.assertTrue( equivalent(component(mu,mu_b),mu_b.u, TOL) )

        # Now try again with a systematic error
        e_sys = ureal(0,u,label="e_sys")
        x = [ v_i + e_sys for v_i in V ]

        mean = type_a.mean(x)

        self.assertTrue( isinstance(mean,float) )
        self.assertTrue( equivalent(mean,mu_V, TOL) )

        mu_a = type_a.estimate(x)
        self.assertTrue( equivalent(mu_a.x,mu_V, TOL) )
        self.assertTrue( equivalent(mu_a.u,sd, TOL) )

        mu_b = result( sum(x)/len(x) )
        self.assertTrue( rp.is_ureal(mu_b) )
        self.assertTrue( equivalent(mu_b.u,u, TOL) )

        mu = result( type_a.merge(mu_a,mu_b) )
        
        u_c = math.sqrt(mu_a.v + mu_b.v)
        self.assertTrue( equivalent(mu.u,u_c, TOL) )
        self.assertTrue( equivalent(component(mu,mu_a),mu_a.u, TOL) )
        self.assertTrue( equivalent(component(mu,mu_b),mu_b.u, TOL) )
        self.assertTrue( equivalent(component(mu,e_sys),u, TOL) )
        
        self.assertRaises(RuntimeError,type_a.merge,mu_a+1E-13,mu_b)

    def test_line_fit(self):
        """
        This is based on H3 in the GUM
        
        """
        TOL = 1E-13

        # Thermometer readings (degrees C)
        t = (21.521,22.012,22.512,23.003,23.507,23.999,24.513,25.002,25.503,26.010,26.511)

        # Observed differences with calibration standard (degrees C)
        b = (-0.171,-0.169,-0.166,-0.159,-0.164,-0.165,-0.156,-0.157,-0.159,-0.161,-0.160)

        # Arbitrary offset temperature (degrees C)
        t_0 = 20.0
        
        # Calculate the temperature relative to t_0
        t_rel = [ t_k - t_0 for t_k in t ]

        #--------------------------------------------------------
        # Case 1: each b_k is independent, but with known uncertainty
        u_b = 0.01
        b_b1 = [ ureal(b_k,u_b) for b_k in b ]

        # Type-A least-squares regression
        
        slope_intercept = type_a.line_fit(t_rel,b_b1).a_b
        
        y_1_a = result( slope_intercept[0] )
        y_2_a = result( slope_intercept[1])

        # Type-B least-squares regression
        slope_intercept = type_b.line_fit(t_rel,b_b1).a_b
        
        y_1_b = result( slope_intercept[0] )
        y_2_b = result( slope_intercept[1])

        self.assertTrue( equivalent(y_1_a.x,y_1_b.x,TOL) )
        self.assertTrue( equivalent(y_2_a.x,y_2_b.x,TOL) )

        y_1 = result( type_a.merge(y_1_a,y_1_b) )
        y_2 = result( type_a.merge(y_2_a,y_2_b) )

        uc_y1 = math.sqrt(y_1_a.v + y_1_b.v)
        uc_y2 = math.sqrt(y_2_a.v + y_2_b.v)

        self.assertTrue( equivalent(y_1.u,uc_y1,TOL) )
        self.assertTrue( equivalent(y_2.u,uc_y2,TOL) )

        self.assertTrue( equivalent(component(y_1,y_1_a),y_1_a.u, TOL) )
        self.assertTrue( equivalent(component(y_1,y_1_b),y_1_b.u, TOL) )
        
        self.assertTrue( equivalent(component(y_2,y_2_a),y_2_a.u, TOL) )
        self.assertTrue( equivalent(component(y_2,y_2_b),y_2_b.u, TOL) )

        #--------------------------------------------------------
        # Case 2: a common systematic error
        e_sys = ureal(0,u_b)
        b_b2 = [ b_k + e_sys for b_k in b ]

        # Type-A least-squares regression
        slope_intercept = type_a.line_fit(t_rel,b_b2).a_b
        
        y_1_a = result( slope_intercept[0] )
        y_2_a = result( slope_intercept[1])

        # Type-B least-squares regression
        slope_intercept = type_b.line_fit(t_rel,b_b2).a_b
        
        y_1_b = result( slope_intercept[0] )
        y_2_b = result( slope_intercept[1])

        self.assertTrue( equivalent(y_1_a.x,y_1_b.x,TOL) )
        self.assertTrue( equivalent(y_2_a.x,y_2_b.x,TOL) )

        y_1 = result( type_a.merge(y_1_a,y_1_b) )
        y_2 = result( type_a.merge(y_2_a,y_2_b) )

        uc_y1 = math.sqrt(y_1_a.v + y_1_b.v)
        uc_y2 = math.sqrt(y_2_a.v + y_2_b.v)

        self.assertTrue( equivalent(y_1.u,uc_y1,TOL) )
        self.assertTrue( equivalent(y_2.u,uc_y2,TOL) )

        self.assertTrue( equivalent(component(y_1,y_1_a),y_1_a.u, TOL) )
        self.assertTrue( equivalent(component(y_1,y_1_b),y_1_b.u, TOL) )
        self.assertTrue( equivalent(component(y_1,e_sys),e_sys.u, TOL) )
        
        self.assertTrue( equivalent(component(y_2,y_2_a),y_2_a.u, TOL) )
        self.assertTrue( equivalent(component(y_2,e_sys),0.0, TOL) )

    def test_complex(self):
        """
        It should be possible to merge complex results too
        """
        TOL = 1E-13
        
        e_r = ureal(0,.5)
        e_i = ureal(0,0.25)
        
        data = (
            (1.1 + e_r) + (3.5j + 1j* e_i),
            (2.1 + e_r) + (2.9j + 1j* e_i),
            (1.9 + e_r) + (3.0j + 1j* e_i),
            (1.5 + e_r) + (2.5j + 1j* e_i)
        )
        
        mu = type_a.estimate(data)
        
        mean = sum(data)/len(data)

        m = type_a.merge(mu,mean)

        self.assertTrue( equivalent(m.real.v,mu.real.v+mean.real.v,TOL) )
        self.assertTrue( equivalent(m.imag.v,mu.imag.v+mean.imag.v,TOL) )

    def test_illegal(self):
        """
        Can't merge uncrtain numbers with different values 
        """
        val = 0
        x1 = ureal(val,1)
        x2 = ucomplex(val-1j,1)

        self.assertRaises(RuntimeError,type_a.merge,x1,x2)
        self.assertRaises(RuntimeError,type_a.merge,x2,x1)   
  
        val = 0.1
        x1 = ureal(val,1)
        x2 = ucomplex(val,1)
        # This should be OK even though they are different types
        x = type_a.merge(x1,x2)
        self.assertTrue( equivalent(val,value(x)) )
  
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
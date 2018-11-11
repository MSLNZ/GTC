import unittest
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

import numpy

from GTC import *

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

    def testUNMean(self):
        """A sequence of uncertain numbers"""
        
        TOL = 1E-12
        data_ = StdDataSets(mu=-3.172,h=0.2,q=1.5,n=10)
        
        seq = [ ureal(x_i,1) for x_i in data_.seq() ]
        self.assertTrue( equivalent( type_a.mean(seq) , data_.mean(), TOL) )
        

    def testStd(self):
        TOL = 1E-13
        
        data_ = StdDataSets(mu=3.172,h=0.1,q=1.5,n=1000)
        
        for k in range(5):
            seq = data_.seq(k)
            
            # numpy.std divides the variance by len(seq), not len(seq)-1,
            # unless ddof=1
            N = float(len(seq))
            root_N = math.sqrt(N)
            
            self.assertTrue( equivalent( numpy.std(seq,ddof=1) , data_.std(), TOL) )
            self.assertTrue( equivalent(
                root_N * type_a.standard_uncertainty(seq) ,
                data_.std(), TOL)
            )
            self.assertTrue( equivalent(
                root_N * type_a.standard_uncertainty(seq) ,
                numpy.std(seq,ddof=1), TOL )
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
        self.assertTrue( equivalent(
            root_N * type_a.standard_uncertainty(seq) ,
            data_.std(), TOL )
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
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
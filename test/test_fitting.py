import unittest
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    xrange = range

import sys
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
EPSILON = sys.float_info.epsilon 
            
#-----------------------------------------------------
def simple_sigma_abr(x,y,u_y=None):
    """
    The uncertainty in `a` and `b` as well as `r`
    
    eqn 15.2.9 from NR, p 663
    
    """
    if u_y is None:
        weights = False
        u_y = [1.0] * len(x)
    else:
        weights = True
        
    N = len(x)
    v = [ u_i*u_i for u_i in u_y ]
        
    S = sum( 1.0/v_i for v_i in v)
    S_x = sum( x_i/v_i for x_i,v_i in izip(x,v) )
    S_xx = sum( x_i**2/v_i for x_i,v_i in izip(x,v) )

    S_y = sum( y_i/v_i for y_i,v_i in izip(y,v) )

    k = S_x / S
    t = [ (x_i - k)/u_y_i for x_i,u_y_i in izip(x,u_y) ]

    S_tt = sum( t_i*t_i for t_i in t )

    b = sum( t_i*y_i/u_y_i/S_tt for t_i,y_i,u_y_i in izip(t,y,u_y) )
    a = (S_y - b*S_x)/S

    delta = S*S_xx - S_x**2
    sigma_a = math.sqrt( S_xx/delta ) 
    sigma_b = math.sqrt( S/delta )
    r = -S_x/(delta*sigma_a*sigma_b)

    if not weights:
        # Need chi-square to adjust the values
        f = lambda x_i,y_i,u_y_i: ((y_i - a - b*x_i))**2 
        chisq = sum( f(x_i,y_i,u_y_i) for x_i,y_i,u_y_i in izip(x,y,u_y) )
        data_u = math.sqrt( chisq/(N-2) )
        sigma_a *= data_u
        sigma_b *= data_u

    return sigma_a, sigma_b, r
    
#-----------------------------------------------------
class TestLineFitScaledWeighted(unittest.TestCase):

    """
    Tests of line_fit_rwls 
    """

    def test_type_a(self):
        """
        Results compared with R, using
        
            fit <- lm(y~x,weights=1/u^2)
            summary(fit)
            vcov(fit)
            
        """
        x = [1,2,3,4,5,6]
        y = [3.014,5.225,7.004,9.061,11.201,12.762]
        u_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        fit = type_a.line_fit_rwls(x,y,u_y)
        a, b = fit.a_b

        self.assertTrue( equivalent(a.x,1.13754,1E-5))
        self.assertTrue( equivalent(b.x,1.97264,1E-5))
        self.assertTrue( equivalent(a.u,0.12261,1E-5))
        self.assertTrue( equivalent(b.u,0.04118,1E-5))
        self.assertTrue( equivalent(a.u*b.u*a.get_correlation(b),-0.004408553,1E-5))
        self.assertEqual(a.df,len(x)-2)
        self.assertEqual(b.df,len(x)-2)
 
    def test_type_b(self):
        """
        We can treat the problem above as type-B. In that
        case, the y uncertainties are the weights. We do 
        not expect the same uncertainties. 
        
        The R commands are:
            fit <- lm(y~x,weights=1/u^2)
            fit.sum <- summary(fit)
            sqrt(diag(fit.sum$cov))     # the std uncertainties
            fit.sum$cov[1,2]    # the covariance
            
        """
        x = [1,2,3,4,5,6]
        y = [ 
            ureal(y_i,u_i) for y_i,u_i in zip(
                [3.014,5.225,7.004,9.061,11.201,12.762],
                [0.2,0.2,0.2,0.4,0.4,0.4]
            ) 
        ]
        fit = tb.line_fit_wls(x,y)
        a, b = fit.a_b

        self.assertTrue( equivalent(a.x,1.13754,1E-5))
        self.assertTrue( equivalent(b.x,1.97264,1E-5))
        self.assertTrue( equivalent(a.u,0.21188326 ,1E-8))
        self.assertTrue( equivalent(b.u,0.07115681,1E-8))
        self.assertTrue( equivalent(a.u*b.u*a.get_correlation(b),-0.01316456,1E-7))
 
        # The same result should be obtained if we give the uncertainties explicitly
        #(but we must still use them in the uncertain number definition of y_i)
        u_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        y = [ 
            ureal(y_i,u_i) for y_i,u_i in zip(
                [3.014,5.225,7.004,9.061,11.201,12.762],
                u_y
            ) 
        ]
        fit = tb.line_fit_wls(x,y,u_y)
        a, b = fit.a_b

        self.assertTrue( equivalent(a.x,1.13754,1E-5))
        self.assertTrue( equivalent(b.x,1.97264,1E-5))
        self.assertTrue( equivalent(a.u,0.21188326 ,1E-8))
        self.assertTrue( equivalent(b.u,0.07115681,1E-8))
        self.assertTrue( equivalent(a.u*b.u*a.get_correlation(b),-0.01316456,1E-7))

        # If we change the y_i uncertainties, we expect the estimates to
        # remain the same, but the uncertainties to change.
        y = [ 
            ureal(y_i,1.0) for y_i in [3.014,5.225,7.004,9.061,11.201,12.762] 
            ]
        fit = tb.line_fit_wls(x,y,u_y)
        a, b = fit.a_b

        self.assertTrue( equivalent(a.x,1.13754,1E-5))
        self.assertTrue( equivalent(b.x,1.97264,1E-5))
        self.assertTrue( abs(a.u-0.21188326) > 1E-8)
        self.assertTrue( abs(b.u-0.07115681) > 1E-8)
        self.assertTrue( abs(a.u*b.u*a.get_correlation(b)+0.01316456) > 1E-7)

#-----------------------------------------------------
class TestLineFitWeighted(unittest.TestCase):

    """
    Tests of the line_fit function when weights are used
    """
    
    def test_iso28037_wls1(self):
        """ ISO/TS 28037:2010, p 13
        
        Better numbers using R:
        
            fit <- lm(y~x,weights=w)
            fit.sum <- summary(fit)
            coef(fit)
            sqrt(diag(fit.sum$cov))
        """
        u_y = 0.5
        x = [1,2,3,4,5,6]
        y = [ ureal(y_i,u_y) for y_i in (3.3,5.6,7.1,9.3,10.7,12.1) ]

        fit = type_b.line_fit_wls(x,y)
        a, b = fit.a_b

        TOL = 1E-6
        self.assertTrue( equivalent(a.x,1.866667,TOL) )
        self.assertTrue( equivalent(b.x,1.757143,TOL) )
        self.assertTrue( equivalent(a.u,0.4654747,TOL) )
        self.assertTrue( equivalent(b.u,0.1195229,TOL) )
        self.assertTrue( equivalent(a.get_correlation(b)*a.u*b.u,-0.050,TOL) )
        self.assertEqual( a.df,inf )

        self.assertTrue( equivalent(fit.ssr,1.665,1E-3) )

        TOL = 0.001
        y0 = ureal(10.5,0.5)
        x0 = (y0 - a)/b
        self.assertTrue( equivalent(x0.x,4.913,TOL) )
        self.assertTrue( equivalent(x0.u,0.322,TOL) )
        self.assertEqual( x0.df,inf )
        
    def test_iso28037_wls1_ta(self):
        """ ISO/TS 28037:2010, p 13
        
        Now use the type-A routine

        Better numbers using R:
        
            fit <- lm(y~x,wieghts=w)
            fit.sum <- summary(fit)
            coef(fit)
            sqrt(diag(fit.sum$cov))

            """
        x = [1,2,3,4,5,6]
        y = (3.3,5.6,7.1,9.3,10.7,12.1) 
        u_y = [0.5] * len(x)

        fit = type_a.line_fit_wls(x,y,u_y)
        a, b = fit.a_b

        TOL = 1E-6
        self.assertTrue( equivalent(a.x,1.866667,TOL) )
        self.assertTrue( equivalent(b.x,1.757143,TOL) )
        self.assertTrue( equivalent(a.u,0.4654747,TOL) )
        self.assertTrue( equivalent(b.u,0.1195229,TOL) )
        self.assertTrue( equivalent(a.get_correlation(b)*a.u*b.u,-0.050,TOL) )
        self.assertEqual( a.df,inf )

        self.assertTrue( equivalent(fit.ssr,1.665,1E-3) )

        TOL = 0.001
        y0 = ureal(10.5,0.5)
        x0 = (y0 - a)/b
        self.assertTrue( equivalent(x0.x,4.913,TOL) )
        self.assertTrue( equivalent(x0.u,0.322,TOL) )
        self.assertEqual( x0.df,inf )

        # Same thing but using the function in LineFitWLS
        x0 = fit.x_from_y([10.5],0.5,'x_label','y_label')
        self.assertTrue( x0.is_intermediate )
        self.assertTrue( equivalent(x0.x,4.913,TOL) )
        self.assertTrue( equivalent(x0.u,0.322,TOL) )
        self.assertEqual( x0.df,inf )
        self.assertEqual( x0.label,'x_label' )

        x0 = fit.x_from_y([10.5],0.5)
        self.assertTrue( not x0.is_intermediate )
        self.assertTrue( equivalent(x0.x,4.913,TOL) )
        self.assertTrue( equivalent(x0.u,0.322,TOL) )
        self.assertEqual( x0.df,inf )

    def test_iso28037_wls2(self):
        """ ISO/TS 28037:2010, p 14

        Better numbers using R:
        
            fit <- lm(y~x,wieghts=w)
            fit.sum <- summary(fit)
            coef(fit)
            sqrt(diag(fit.sum$cov))
        """
        x = [1,2,3,4,5,6]
        y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
        u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        
        y = [ ureal(y_i,u_y_i) for y_i, u_y_i in zip(y,u_y) ]

        fit = type_b.line_fit_wls(x,y)
        a, b = fit.a_b

        self.assertTrue( equivalent(a.x,0.8852,1E-4) )
        self.assertTrue( equivalent(b.x,2.0570,1E-4) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )
        self.assertTrue( equivalent(a.get_correlation(b)*a.u*b.u,-0.08227848,1E-6) )
        self.assertEqual( a.df,inf )

        TOL = 0.001
        y0 = ureal(10.5,1.0)
        x0 = (y0 - a)/b
        self.assertTrue( equivalent(x0.x,4.674,TOL) )
        self.assertTrue( equivalent(x0.u,0.533,TOL) )
        self.assertEqual( x0.df,inf )

    def test_iso28037_wls2_option(self):
        """ ISO/TS 28037:2010, p 14
        """
        x = [1,2,3,4,5,6]
        y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
        u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        u_y_dummy = [11,12,3,6,10,2,9]
        
        y = [ ureal(y_i,u_y_i) for y_i, u_y_i in zip(y,u_y_dummy) ]

        fit = type_b.line_fit_wls(x,y,u_y)
        a, b = fit.a_b

        TOL = 0.001
        self.assertTrue( equivalent(a.x,0.885,TOL) )
        self.assertTrue( equivalent(b.x,2.057,TOL) )
        # Don't expect the same uncertainty, because that
        # will depend on the uncertain numbers
        self.assertTrue( not abs(a.u - 0.530) < TOL )
        self.assertTrue( not abs(b.u - 0.178) < TOL )
        self.assertTrue( equivalent(fit.ssr,4.131,TOL) )
        self.assertEqual( a.df,inf )

    def test_iso28037_wls2_ta(self):
        """ ISO/TS 28037:2010, p 14
        
        Values here calculated using R
            fit <- lm(y~x,wieghts=w)
            fit.sum <- summary(fit)
            coef(fit)
            sqrt(diag(fit.sum$cov))
        
        """
        x = [1,2,3,4,5,6]
        y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
        u_y = [0.5,0.5,0.5,1.0,1.0,1.0]
        
        fit = ta.line_fit_wls(x,y,u_y)
        a, b = fit.a_b

        self.assertTrue( equivalent(a.x,0.8852,1E-4) )
        self.assertTrue( equivalent(b.x,2.0570,1E-4) )
        self.assertTrue( equivalent(a.u,0.5297081,1E-6) )
        self.assertTrue( equivalent(b.u,0.1778920,1E-6) )
        self.assertTrue( equivalent(fit.ssr,4.131,1E-3) )
        self.assertTrue( equivalent(a.get_correlation(b)*a.u*b.u,-0.08227848,1E-6) )
        self.assertEqual( a.df,inf )

        TOL = 0.001
        y0 = ureal(10.5,1.0)
        x0 = (y0 - a)/b
        self.assertTrue( equivalent(x0.x,4.674,TOL) )
        self.assertTrue( equivalent(x0.u,0.533,TOL) )
        self.assertEqual( x0.df,inf )

        # Same thing but using the function in LineFitWLS
        x0 = fit.x_from_y([10.5],1.0,'x_label','y_label')
        self.assertTrue( equivalent(x0.x,4.674,TOL) )
        self.assertTrue( equivalent(x0.u,0.533,TOL) )
        self.assertEqual( x0.df,inf )
        self.assertEqual( x0.label,'x_label' )


    def test_simple_scaled(self):
        """
        Straight line with non-trivial `a` and `b` and non-unity weight

        """
        N = 10
        a0 =10
        b0 = -3
        u0 = .2
        u = [u0] * N
        
        x = [ float(x_i) for x_i in xrange(10) ]
        y = [ ureal(b0*x_i + a0,u0) for x_i in x ]

        a,b = type_b.line_fit_wls(x,y,u).a_b

        equivalent(a.x,a0,TOL)
        equivalent(b.x,b0,TOL)

        # The uncertainties in `a` and `b` should match
        sig_a, sig_b, r = simple_sigma_abr(x,y,u)
        
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)
        
        # So should the correlation between `a` and `b`
        equivalent(r,a.get_correlation(b),TOL)             
        
    def test_simple_scaled_ta(self):
        """
        Straight line with non-trivial `a` and `b` and non-unity weight

        """
        N = 10
        a0 =10
        b0 = -3
        u0 = .2
        u = [u0] * N
        
        x = [ float(x_i) for x_i in xrange(10) ]
        y = [ ureal(b0*x_i + a0,u0) for x_i in x ]

        a,b = ta.line_fit_wls(x,y,u).a_b

        equivalent(a.x,a0,TOL)
        equivalent(b.x,b0,TOL)

        # The uncertainties in `a` and `b` should match
        sig_a, sig_b, r = simple_sigma_abr(x,y,u)
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)
        
        # So should the correlation between `a` and `b`
        equivalent(r,a.get_correlation(b),TOL)             
        
    def test_pearson_york(self):
        """
        Use Pearson-York data but just weight the y-data
        Results (for the parameters) are compared to similar
        calculation in R.
        
        In R:
        
            fit <- lm(y~x,wieghts=w)
            fit.sum <- summary(fit)
            coef(fit)
            sqrt(diag(fit.sum$cov))
        """
        
        y=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wy=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        x=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]

        uy=[1./math.sqrt(wy_i) for wy_i in wy ]
        y = [ ureal(y_i,u_i) for y_i, u_i in zip(y,uy) ]

        initial = type_b.line_fit_wls(x,y)
        a,b = initial.a_b
        
        self.assertTrue( equivalent(a.x,9.4302,1E-4) )
        self.assertTrue( equivalent(b.x,-1.5862,1E-4) )
        
        TOL = 1E-7
        self.assertTrue( equivalent( variance(a), 0.011384624, TOL) )
        self.assertTrue( equivalent( variance(b), 0.0004400409, TOL) )
        self.assertTrue( equivalent( a.u*b.u*a.get_correlation(b), -0.002211235, TOL) )

        TOL = 1E-7
        a_u,b_u,r_ = simple_sigma_abr(x,y,uy)
        self.assertTrue( equivalent( uncertainty(a), a_u, TOL) )
        self.assertTrue( equivalent( uncertainty(b), b_u, TOL) )
        self.assertTrue( equivalent( a.get_correlation(b), r_, TOL) )
        
    def test_pearson_york_ta(self):
        """
        Use Pearson-York data but just weight the y-data
        Results (for the parameters) are compared to similar
        calculation in R (but it doesn't give the uncertainties
        in the parameters).
        
        Better numbers using R:
            fit <- lm(y~x,wieghts=w)
            fit.sum <- summary(fit)
            coef(fit)
            sqrt(diag(fit.sum$cov))

            """
        TOL = 1E-4
        
        y=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wy=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        x=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]

        uy=[1./math.sqrt(wy_i) for wy_i in wy ]

        initial = type_a.line_fit_wls(x,y,uy)
        a,b = initial.a_b
        
        self.assertTrue( equivalent(a.x,9.4302,TOL) )
        self.assertTrue( equivalent(b.x,-1.5862,TOL) )
        
        TOL = 1E-7
        self.assertTrue( equivalent( variance(a), 0.011384624, TOL) )
        self.assertTrue( equivalent( variance(b), 0.0004400409, TOL) )
        self.assertTrue( equivalent( a.u*b.u*a.get_correlation(b), -0.002211235, TOL) )
        
        a_u,b_u,r_ = simple_sigma_abr(x,y,uy)
        self.assertTrue( equivalent( uncertainty(a), a_u, TOL) )
        self.assertTrue( equivalent( uncertainty(b), b_u, TOL) )
        self.assertTrue( equivalent( a.get_correlation(b), r_, TOL) )

    def test_fn_line_fit_wls_not_uncertain_real(self):
        # type_b.line_fit_wls should raise ValueError if y is not a
        # sequence of uncertain real numbers
        x = list(range(10))
        y = list(range(10))
        u_y = [1.0] * 10
        self.assertRaises(ValueError, tb.line_fit_wls, x, y, u_y)

#---------------------------------------------------------------
class UncertainLineTests(unittest.TestCase):

    def test_simple(self):
        """Trivial straight line
        
        uncertain number in y only with unity weight

        """
        x = [ float(x_i) for x_i in xrange(10) ]
        y = [ ureal(y_i,1) for y_i in x ]

        fit = type_b.line_fit(x,y)
        a,b = fit.a_b # default weight is unity

        self.assertTrue(a is fit.intercept)
        self.assertTrue(b is fit.slope)

        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)

        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            [1.0 for x_i in x ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)

        equivalent(r,a.get_correlation(b),TOL)        

    def test_simple_errors_in_x_and_y(self):
        """Trivial straight line
        
        uncertain number in x and y with unity weights.
        We expect the uncertainty to increase by sqrt(2).

        """
        N = 10
        y = [ ureal(y_i,1) for y_i in xrange(N) ]
        x = [ ureal(x_i,1) for x_i in xrange(N) ]

        a,b = type_b.line_fit( x,y ).a_b
        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)

        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            [ math.sqrt(2) for x_i,y_i in zip(x,y) ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)

        equivalent(r,a.get_correlation(b),TOL)        

        a,b = type_b.line_fit_wls( x,y ).a_b
        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)

        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            [ math.sqrt(2) for x_i,y_i in zip(x,y) ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)

        equivalent(r,a.get_correlation(b),TOL) 

    def test_simple_scaled(self):
        """Straight line with non-trivial `a` and `b`

        uncertain number in y only, but non-unity weight

        """
        N = 10
        a0 =10
        b0 = -3
        u0 = .2
        u = [u0] * N
        
        x = [ value(x_i) for x_i in xrange(10) ]
        y = [ ureal(b0*x_i + a0,u_i) for x_i,u_i in izip(x,u) ]

        a,b = type_b.line_fit_wls(x,y).a_b

        equivalent( value(a),a0,TOL)
        equivalent( value(b),b0,TOL)

        # The uncertainties in `a` and `b` should match
        sig_a, sig_b, r = simple_sigma_abr(
            x,
            [ value(y_i) for y_i in y],
            [ y_i.u for y_i in y ]
        )
        equivalent(uncertainty(a),sig_a,TOL)
        equivalent(uncertainty(b),sig_b,TOL)
        
        # So should the correlation between `a` and `b`
        equivalent(r,a.get_correlation(b),TOL)             

    def test_simple_scaled_errors_in_x_and_y(self):
        """Straight line with non-trivial `a` and `b`

        uncertain number in x and y, with non-unity weight.

        """
        a0 =10
        b0 = -3
        u0 = .2

        N = 10        
        x = [ ureal(x_i,u0) for x_i in xrange(N) ]
        y = [ ureal( b0 * value(x_i) + a0, 1) for x_i in x ]    # unity uncertainty

        u = [uncertainty(y_i - b0*x_i) for x_i,y_i in izip(x,y)]

        a,b = type_b.line_fit_wls(x,y).a_b

        equivalent( value(a),a0,TOL)
        equivalent( value(b),b0,TOL)

        # The uncertainties in `a` and `b` should match
        # when we feed in the `d` array
        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            u
        )
        equivalent(uncertainty(b),sig_b,TOL)
        equivalent(uncertainty(a),sig_a,TOL)

        # So should the correlation between `a` and `b`
        equivalent(r,a.get_correlation(b),TOL)        

        # We can calculate the uncertainty sequence directly
        d = [ math.sqrt(1 + (b0 * u0)**2) for i in xrange(N) ]
        sig_a, sig_b, r = simple_sigma_abr(
            [value(x_i) for x_i in x],
            [value(y_i) for y_i in y],
            d
        )
        equivalent(uncertainty(b),sig_b,TOL)
        equivalent(uncertainty(a),sig_a,TOL)

    def test_fn_line_fit_not_uncertain_real(self):
        # type_b.line_fit should raise ValueError if y is not a
        # sequence of uncertain real numbers
        x = list(range(10))
        y = list(range(10))
        self.assertRaises(ValueError, tb.line_fit, x, y)


class DoFLineTest(unittest.TestCase):
    """
    Test that the dof calculation is working correctly.
    """
    def test_willink(self):
        """
        This case is taken from
        R Willink, Metrologia 45 (2008) 63-67

        The test ensures that the DoF calculation, when using
        the LS to interpolate to another data point, is correct.
        
        """
        nu = 2
        u = 0.1
        x = [ value(x_i) for x_i in [-1,0,1] ]
        y = [ ureal(y_i,u,df=nu) for y_i in x ]

        a,b = type_b.line_fit(x,y).a_b # default weight is unity

        equivalent( value(a) ,0.0,TOL)
        equivalent( value(b) ,1.0,TOL)
        
        equivalent(0.0,a.get_correlation(b),TOL)        

        # See below eqns (12) and (13), in section 2.1
        x_13 = 2.0/3.0
        y_13 = a + b*x_13
        equivalent(y_13.v,5.0*u**2/9.0,TOL)        
        equivalent(y_13.df,25.0*nu/17.0,TOL)         
        
#--------------------------------------------------------------------
# Helper functions for testing Weighted Total Least Squares fit
#
#--------------------------------------------------------------------
ITMAX = 100
CGOLD = 0.3819660
ZEPS = 1E-10
def brent(a,b,c,fn,tol=math.sqrt(EPSILON)):
    """Return xmin and fn(xmin) bracketed by a, b, and c

    `b` must be between `a` and `c` and fn(b) must be less than
    both fn(a) and fn(c).

    `tol` - the fractional precision

    See also Numerical Recipes in C, 2nd ed, Section 10.3
    
    """
    e = 0.0

    a = a if a < c else c
    b = a if a > c else c

    x = w = v = b
    fx = fw = fv = fn(x)

    for i in xrange(ITMAX):

        xm = 0.5*(a + b)
        tol1 = tol*abs(x) + ZEPS
        tol2 = 2.0*tol1

        if abs(x - xm) <= tol2 - 0.5*(b - a):
            # Exit here
            return x, fx

        if abs(e) > tol1:
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            p = (x - v)*q - (x - w)*r
            q = 2.0*(q - r)
            if q > 0.0: p = -p
            q = abs(q)
            etemp, e = e, d
            if abs(p) >= abs(0.5*q*etemp) or p <= q*(a - x) or p >= q*(b - x):
                e = a - x if x >= xm else b - x
                d = CGOLD*e
            else:
                d = p/q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = math.copysign(tol1,xm - x)
        else:
            e = a - x if x >= xm else b - x
            d = CGOLD*e

        u = x + d if abs(d) >= tol1 else x + math.copysign(tol1,d)
        fu = fn(u)

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
                
            v, w, x, = w, x, u
            fv, fw, fx, = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b= u

            if fu <= fw or w == x:
                v, w, = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    raise RuntimeError("Exceeded iteration limit in 'brent'")

from GTC.type_b import _arrays
# #------------------------------------------------------------------
# def _arrays(sin_a,cos_a,sin_2a,cos_2a,x,y,u2_x,u2_y,cov):
    # """Returns the set of arrays needed for the Chi-sq calculation

    # Equations reference: M Krystek and M Anton,
    # Meas. Sci. Technol. 22 (2011) 035101 (9pp)

    # Note, this utility function can work equally with real numbers
    # or uncertain real numbers. This allows us to us it in the
    # checking routines too.
    
    # """
    # sin_a_2 = sin_a**2
    # cos_a_2 = cos_a**2
    # two_sin_cos_a = 2.0*sin_a*cos_a

    # # Note an alternative (perhaps more stable numerically is the following
    # # it will need cos_2a and sin_2a to be passed in as arguments too.
    
    # # eqn(53)
    # g_k = [
        # (u2_x_i + u2_y_i)/2.0 - (u2_x_i - u2_y_i)*cos_2a/2.0 - 2.0*cov_i*sin_2a
            # for u2_x_i, u2_y_i, cov_i in izip(u2_x,u2_y,cov)
    # ]
# ##    # eqn(32)
# ##    g_k = [
# ##        u2_x_i*sin_a_2 + u2_y_i*cos_a_2 - two_sin_cos_a*cov_i
# ##            for u2_x_i, u2_y_i, cov_i in izip(u2_x,u2_y,cov)
# ##    ]

    # N = len(g_k)

    # # eqn(33), but without sqrt
    # u2 = 1.0/(
        # sum( 1.0/g_k_i for g_k_i in g_k ) / N
    # )
    # # eqn(34)
    # w_k = [ u2/g_k_i for g_k_i in g_k ]

    # # Eqns (35,36,43)
    # x_bar = sum(
        # w_k_i*x_i for w_k_i,x_i in izip(w_k,x)
    # ) / N
    
    # y_bar = sum(
        # w_k_i*y_i for w_k_i,y_i in izip(w_k,y)
    # ) / N

    # p_hat = y_bar*cos_a - x_bar*sin_a    

    # # eqn(31)
    # v_k = [ y_i*cos_a - x_i*sin_a - p_hat for x_i,y_i in izip(x,y) ]  

    # return v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat

#--------------------------------------------------------------------
class _Chi(object):

    """
    Defines a callable object that can be used in the Brent minimum-search.
    The `x` and `y` data are used to initialise the object. The __call__
    method can then be expressed as a function of `alpha` only. 
    """
    
    def __init__(self,x,y):
        self.x = [ value(x_i) for x_i in x ]
        self.y = [ value(y_i) for y_i in y ]

        self.u2_x = [ variance(x_i) for x_i in x ]
        self.u2_y = [ variance(y_i) for y_i in y ]
        
        self.cov = [ uncertainty(x_i)*x_i.get_correlation(y_i)*uncertainty(y_i)
                         for x_i,y_i in izip(x,y)]
        
    #--------------------------------------------------------------------
    def arrays(self,alpha):
        """Return v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat
        
        """
        sin_a = math.sin(alpha)
        sin_2a = math.sin(2.0*alpha)
        cos_a = math.cos(alpha)
        cos_2a = math.cos(2.0*alpha)

        return _arrays(sin_a,cos_a,sin_2a,cos_2a,self.x,self.y,self.u2_x,self.u2_y,self.cov)
    
    def __call__(self,alpha):
        """Function to be minimised wrt to `alpha`
        """
        v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat = self.arrays(alpha)
        
        # Eqn (30)
        chi_2 = sum(
            v_k_i**2/g_k_i for v_k_i,g_k_i in izip(v_k,g_k)
        )
        
        return chi_2
    
#--------------------------------------------------------------------
HALF_PI = math.pi/2.0
def _WTLS(x,y,a0,b0):
    """Return a,b,u_a,u_b,u_ab

    Best-fit to
        y = b*x + a    

    `x` - independent variable data sequence of uncertain real numbers
    `y` - dependent variable data sequence of uncertain real numbers
    `a0` - estimated line intercept
    `b0` - estimated line slope

    Based on method of M Krystek and M Anton,
    Meas. Sci. Technol. 22 (2011) 035101 (9pp)
    
    """
    for x_i,y_i in izip(x,y):
        assert reporting.is_ureal(x_i)
        assert reporting.is_ureal(y_i)
        
    # initial value for `alpha`
    alpha0 = math.atan(b0)
    data = _Chi(x,y)

    # `brent` requires three points that bracket the minimum.
    # Seach for the minimum chi-squared wrt alpha
    x1 = alpha0 - HALF_PI
    x2 = alpha0 + HALF_PI

    alpha,fn_alpha = brent(x1,alpha0,x2,data)

    # Get the arrays at the root (not needed except for p_hat!?)     
    v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat = data.arrays(alpha)

    N = len(g_k)
    
    # Note we have reversed the definitions of `a` and `b` here
    b = math.tan(alpha)
    a = p_hat/math.cos(alpha)

    # Uncertainty calculation only------------------------------------
    sin_a = math.sin(alpha)
    sin_2a = math.sin(2.0*alpha)
    cos_a = math.cos(alpha)
    cos_2a = math.cos(2.0*alpha)

    # Just use real numbers
    g_k = [ value(g_k_i) for g_k_i in g_k ]
    x = [ value(x_i) for x_i in x ]
    y = [ value(y_i) for y_i in y ]
    u2_x = [ value(u2_x_i) for u2_x_i in u2_x ]
    u2_y = [ value(u2_y_i) for u2_y_i in u2_y ]
    
    v_k,u2_x,u2_y,g_k,u2,x_bar,y_bar,p_hat = _arrays(
        sin_a,cos_a,sin_2a,cos_2a,x,y,u2_x,u2_y,data.cov
    )
    
    v_ka = [ -y_i*sin_a - x_i*cos_a for x_i,y_i in izip(x,y)]
    v_kaa = [ -v_k_i - p_hat for v_k_i in v_k ]
    f_k = [ v_k_i**2 for v_k_i in v_k ]
    f_ka = [ 2.0*v_k_i*v_ka_i for v_k_i,v_ka_i in izip(v_k,v_ka)]
    f_kaa = [ 2.0*(v_ka_i**2 + v_k_i*v_kaa_i) for v_k_i,v_ka_i,v_kaa_i in izip(v_k,v_ka,v_kaa)]

    g_ka = [ 
        (u2_x_i-u2_y_i)*sin_2a - 2.0*cov_i*cos_2a 
            for u2_x_i,u2_y_i,cov_i in izip(u2_x,u2_y,data.cov)
    ]
    k = 2.0*(cos_a**2 - sin_a**2)
    g_kaa = [ k*(u2_x_i - u2_y_i) for u2_x_i,u2_y_i in izip(u2_x,u2_y) ]
    Hpp = 2*N/u2
    H_alpha_p = -2.0*sum(
        (v_ka_i*g_k_i - g_ka_i*v_k_i) / g_k_i**2
            for g_k_i,v_k_i,g_ka_i,v_ka_i in izip(g_k,v_k,g_ka,v_ka)
    )
    H_alpha_alpha = sum(
            f_kaa_i/g_k_i
        -   2.0*f_ka_i*g_ka_i/g_k_i**2
        +   2.0*g_ka_i**2*f_k_i/g_k_i**3
        -   g_kaa_i*f_k_i/g_k_i**2
       for g_k_i,g_ka_i,g_kaa_i,f_k_i,f_ka_i,f_kaa_i in izip(g_k,g_ka,g_kaa,f_k,f_ka,f_kaa)
    )
    NN = 2.0 / (Hpp*H_alpha_alpha - H_alpha_p**2)
    var_p = NN*H_alpha_alpha
    var_alpha = NN*Hpp
    cov_alphap = -NN*H_alpha_p

    # Convert to a,b variances
    # NB our `a` and `b` are the opposite of Krystek and Anton
    var_b = var_alpha/cos_a**4
    var_a = (
        var_alpha*p_hat*p_hat*sin_a**2 +
        var_p*cos_a**2 +
        2.0*cov_alphap*p_hat*sin_a*cos_a
    ) /cos_a**4
    cov_ab = (
        var_alpha*p_hat*sin_a +
        cov_alphap*cos_a
    ) /cos_a**4

    u_a,u_b = math.sqrt(var_a), math.sqrt(var_b)
    u_ab = cov_ab / (u_a*u_b)
    
    return a,b,u_a,u_b,u_ab

#-------------------------------------------------------------------------
class TestLineFitTLS(unittest.TestCase):
    """
    The WTSL is described in M Krystek and M Anton,
    Meas. Sci. Technol. 22 (2011) 035101 (9pp)
    """
    def test_iso_ts28037_2010_fn(self):
        """p21 of the standard
        """
        x = [1.2,1.9,2.9,4.0,4.7,5.9]
        u_x = 0.2
        x = [ ureal(x_i,u_x) for x_i in x ]
        
        y = [3.4,4.4,7.2,8.5,10.8,13.5]
        u_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        y = [ ureal(y_i,u_y_i) for y_i, u_y_i in zip(y,u_y) ]

        a_b = type_b.line_fit(x,y).a_b

        a0, b0 = a_b
        fit = type_b.line_fit_wtls(x,y,a_b=a_b)
        a, b = fit.a_b
        
        TOL = 0.004
        self.assertTrue( equivalent(a.x,0.5788,TOL) )
        self.assertTrue( equivalent(a.u,0.4764,TOL) )  # This is the critical case for TOL
        self.assertTrue( equivalent(b.x,2.1597,TOL) )
        self.assertTrue( equivalent(b.u,0.1355,TOL) )
        self.assertTrue( equivalent(get_correlation(a,b)*b.u*a.u,-0.0577,TOL) )
        self.assertTrue( equivalent(fit.ssr,2.743,TOL) )

        # Test default initial estimate
        fit = type_b.line_fit_wtls(x,y)
        a, b = fit.a_b
        
        self.assertTrue( equivalent(a.x,0.5788,TOL) )
        self.assertTrue( equivalent(a.u,0.4764,TOL) )  # This is the critical case for TOL
        self.assertTrue( equivalent(b.x,2.1597,TOL) )
        self.assertTrue( equivalent(b.u,0.1355,TOL) )
        self.assertTrue( equivalent(get_correlation(a,b)*b.u*a.u,-0.0577,TOL) )
        self.assertTrue( equivalent(fit.ssr,2.743,TOL) )

    def test_iso_ts28037_2010_ta(self):
        """p21 of the standard
        """
        x = [1.2,1.9,2.9,4.0,4.7,5.9]
        u_x = 0.2
        x = [ ureal(x_i,u_x) for x_i in x ]
        
        y = [3.4,4.4,7.2,8.5,10.8,13.5]
        u_y = [0.2,0.2,0.2,0.4,0.4,0.4]
        y = [ ureal(y_i,u_y_i) for y_i, u_y_i in zip(y,u_y) ]

        a_b = type_b.line_fit(x,y).a_b

        a0, b0 = a_b
        fit = type_a.line_fit_wtls(x,y, a0_b0=a_b, u_x=[u_x]*6, u_y=u_y)
        a, b = fit.a_b
        
        TOL = 0.004
        self.assertTrue( equivalent(a.x,0.5788,TOL) )
        self.assertTrue( equivalent(a.u,0.4764,TOL) )  # This is the critical case for TOL
        self.assertTrue( equivalent(b.x,2.1597,TOL) )
        self.assertTrue( equivalent(b.u,0.1355,TOL) )
        self.assertTrue( equivalent(a.get_correlation(b)*b.u*a.u,-0.0577,TOL) )
        self.assertTrue( equivalent(fit.ssr,2.743,TOL) )
  
        # Test default initial estimate
        fit = type_a.line_fit_wtls(x,y, u_x=[u_x]*6, u_y=u_y)
        a, b = fit.a_b
        
        self.assertTrue( equivalent(a.x,0.5788,TOL) )
        self.assertTrue( equivalent(a.u,0.4764,TOL) )  # This is the critical case for TOL
        self.assertTrue( equivalent(b.x,2.1597,TOL) )
        self.assertTrue( equivalent(b.u,0.1355,TOL) )
        self.assertTrue( equivalent(a.get_correlation(b)*b.u*a.u,-0.0577,TOL) )
        self.assertTrue( equivalent(fit.ssr,2.743,TOL) )
  
    def test_Lybanon_fn(self):
        """
        Test the WTLS method on the Pearson-York data.
        This does not test the correlations between x-y pairs.
        The the a and b uncertainty values come from
        A H Kalantar, Meas Sci Technol. 3 (1992) 1113
        
        """        
        # pearson_york_testdata
        # see, e.g., Lybanon, M. in Am. J. Phys 52 (1), January 1984 
        xin=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wx=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        yin=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        wy=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        uxin=[1./math.sqrt(wx_i) for wx_i in wx ]
        uyin=[1./math.sqrt(wy_i) for wy_i in wy ]

        x = [ ureal(xin_i,uxin_i) for xin_i,uxin_i in izip(xin,uxin) ]
        y = [ ureal(yin_i,uyin_i) for yin_i,uyin_i in izip(yin,uyin) ]

        # a0, b0 = tb.line_fit(x,y).a_b
        # result = tb.line_fit_wtls(x,y,a_b=(a0,b0))
        
        # Use default estimate to start
        result = tb.line_fit_wtls(x,y)
        a,b = result.a_b
        
        # 'exact' values of a and b from 
        # Neri et al J Phys E 22(1989) 215-217
        a0 = 5.47991022
        b0 = -0.480533407
        equivalent(a.x,a0,1E-7)
        equivalent(b.x,b0,1E-7)

        # Compare with published values for the error:
        # A H Kalantar, Meas Sci Tech (1992) 1113
        equivalent(a.u,0.29193,1E-5)
        equivalent(b.u,0.057617,1E-6)

        equivalent(-0.0162,a.u*b.u*get_correlation(a,b),1E-3)

        # From refs in R J Murray, 
        #   "Weighted Least-Squares Curve Fiting with Errors in all Variables", 
        # ASPRS Annual Convention & Exposition. Baltimore: ACSM/ASPRS, 1994
        equivalent(1.4833,result.ssr/(result.N-2),1E-4)

        # Compare between different implementations
        # This highlights a difference between the 
        # Krystek calculation (_WTLS) of uncertainty 
        # and the more direct uncertain numbers approach.
        # NB Krystek reports u(a)=0.2924 and claims
        # agreement with Kalantar. However, we see here
        # that he could have done better.
        _a,_b,u_a,u_b,u_ab = _WTLS(x,y,a0,b0)
        equivalent(a.x,_a,1E-7)
        equivalent(b.x,_b,1E-7)
        equivalent(b.u,u_b,1E-4)
        equivalent(a.u,u_a,1E-3)

    def test_Lybanon_fn_ext_weights(self):
        """
        Test the WTLS method on the Pearson-York data.
        Use explicit weights
        
        """        
        # pearson_york_testdata
        # see, e.g., Lybanon, M. in Am. J. Phys 52 (1), January 1984 
        xin=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wx=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        yin=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        wy=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        uxin=[1./math.sqrt(wx_i) for wx_i in wx ]
        uyin=[1./math.sqrt(wy_i) for wy_i in wy ]
        
        N = len(xin)

        x = [ ureal(xin_i,1) for xin_i in xin ]
        y = [ ureal(yin_i,1) for yin_i in yin ]

        # a0, b0 = tb.line_fit(x,y).a_b
        # result = tb.line_fit_wtls(x,y,a_b=(a0,b0),u_x=uxin,u_y=uyin)

        result = tb.line_fit_wtls(x,y,u_x=uxin,u_y=uyin)
        a,b = result.a_b
        
        # 'exact' values of a and b from 
        # Neri et al J Phys E 22(1989) 215-217
        # We still expect these values from the fit.
        a0 = 5.47991022
        b0 = -0.480533407
        equivalent(a.x,a0,1E-7)
        equivalent(b.x,b0,1E-7)

        # From refs in R J Murray, 
        #   "Weighted Least-Squares Curve Fiting with Errors in all Variables", 
        # ASPRS Annual Convention & Exposition. Baltimore: ACSM/ASPRS, 1994
        equivalent(1.4833,result.ssr/(result.N-2),1E-4)

        # Compare with published values for the error:
        # A H Kalantar, Meas Sci Tech (1992) 1113
        # NB Do not expect agreement now!
        self.assertTrue( abs(a.u-0.29193) > 1E-5)
        self.assertTrue( abs(b.u-0.057617) > 1E-6)
        self.assertTrue( abs(a.u*b.u*get_correlation(a,b)+0.0162) > 1E-3)

    def test_Lybanon_ta(self):
        """
        Test the WTLS method on the Pearson-York data.
        This does not test the correlations between x-y pairs.
        The the a and b uncertainty values come from
        A H Kalantar, Meas Sci Technol. 3 (1992) 1113
        
        """        
        # pearson_york_testdata
        # see, e.g., Lybanon, M. in Am. J. Phys 52 (1), January 1984 
        xin=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wx=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        yin=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        wy=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        uxin=[1./math.sqrt(wx_i) for wx_i in wx ]
        uyin=[1./math.sqrt(wy_i) for wy_i in wy ]

        x = [ ureal(xin_i,uxin_i) for xin_i,uxin_i in izip(xin,uxin) ]
        y = [ ureal(yin_i,uyin_i) for yin_i,uyin_i in izip(yin,uyin) ]

        # a0, b0 = tb.line_fit(x,y).a_b
        # result = ta.line_fit_wtls(x,y,u_x=uxin,u_y=uyin,a0_b0=(a0,b0))

        result = ta.line_fit_wtls(x,y,u_x=uxin,u_y=uyin)
        a,b = result.a_b
        # 'exact' values of a and b from 
        # Neri et al J Phys E 22(1989) 215-217
        a0 = 5.47991022
        b0 = -0.480533407
        equivalent(a.x,a0,1E-7)
        equivalent(b.x,b0,1E-7)

        # Compare with published values for the error:
        # A H Kalantar, Meas Sci Tech (1992) 1113
        equivalent(a.u,0.29193,1E-5)
        equivalent(b.u,0.057617,1E-6)

        equivalent(-0.0162,a.u*b.u*get_correlation(a,b),1E-3)

        # From refs in R J Murray, 
        #   "Weighted Least-Squares Curve Fiting with Errors in all Variables", 
        # ASPRS Annual Convention & Exposition. Baltimore: ACSM/ASPRS, 1994
        equivalent(1.4833,result.ssr/(result.N-2),1E-4)

        # Expect good agreement here because its 
        # the same algorithm, but the published formulae!
        # appear to be numerically poor.
        _a,_b,u_a,u_b,u_ab = _WTLS(x,y,a0,b0)
        equivalent(a.x,_a,1E-7)
        equivalent(b.x,_b,1E-7)
        equivalent(u_a,0.29193,5E-4)
        equivalent(u_b,0.057617,5E-5)

    def test_Lybanon_reversed_fn(self):
        """
        The method should work the same with x and y reversed
        
        """
        yin=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wy=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        xin=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        wx=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        uxin=[1./math.sqrt(wx_i) for wx_i in wx ]
        uyin=[1./math.sqrt(wy_i) for wy_i in wy ]

        x = [ ureal(xin_i,uxin_i) for xin_i,uxin_i in izip(xin,uxin) ]
        y = [ ureal(yin_i,uyin_i) for yin_i,uyin_i in izip(yin,uyin) ]

        # a0, b0 = tb.line_fit(x,y).a_b
        # result = tb.line_fit_wtls(x,y,a_b=(a0,b0))

        result = tb.line_fit_wtls(x,y)
        a,b = result.a_b

        # Now test what we get...
        a0 = 5.47991022
        b0 = -0.480533407

        b1 = 1.0/b0
        a1 = -a0*b1

        equivalent(b.x,b1,1E-6)
        equivalent(a.x,a1,1E-6)

        # Know in the direct case that
        # var(slope) = 0.057617
        # var(intercept) = 0.29193
        # Use LPU to propagate to the inverse case:
        #
        # var(b) = (0.057617)**2/b0**4
        # var(a) = (0.29193)**2/b0**2
        #   + a0**2/b0**4 * (0.057617)**2
        #   + 2 * a0/b0**3 * cov(a0,b0)
        # use cov(a0,b0) = 0.0161861960708
        # from calculation above. 
        equivalent(b.u,(0.057617)/b0**2,1E-5)
        var_a = (0.29193)**2/b0**2 + + a0**2/b0**4 * (0.057617)**2 + 2 * a0/b0**3 * 0.0161861960708
        equivalent(a.u,math.sqrt(var_a),1E-5)

    def test_Lybanon_reversed_ta(self):
        """
        The method should work the same with x and y reversed
        
        """
        yin=[0.0,0.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4]
        wy=[1000.0,1000.0,500.0,800.0,200.0,80.0,60.0,20.0,1.8,1.0]
        
        xin=[5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5]
        wx=[1.0,1.8,4.0,8.0,20.0,20.0,70.0,70.0,100.0,500.0]

        uxin=[1./math.sqrt(wx_i) for wx_i in wx ]
        uyin=[1./math.sqrt(wy_i) for wy_i in wy ]

        x = [ ureal(xin_i,uxin_i) for xin_i,uxin_i in izip(xin,uxin) ]
        y = [ ureal(yin_i,uyin_i) for yin_i,uyin_i in izip(yin,uyin) ]

        # a0, b0 = ta.line_fit(x,y).a_b
        # a,b = ta.line_fit_wtls(x,y,u_x=uxin,u_y=uyin,a0_b0=(a0,b0)).a_b

        a,b = ta.line_fit_wtls(x,y,u_x=uxin,u_y=uyin).a_b

        # Now test what we get...
        a0 = 5.47991022
        b0 = -0.480533407

        b1 = 1.0/b0
        a1 = -a0*b1

        equivalent(b.x,b1,1E-6)
        equivalent(a.x,a1,1E-6)

        # Know in the direct case that
        # var(slope) = 0.057617
        # var(intercept) = 0.29193
        # Use LPU to propagate to the inverse case:
        #
        # var(b) = (0.057617)**2/b0**4
        # var(a) = (0.29193)**2/b0**2
        #   + a0**2/b0**4 * (0.057617)**2
        #   + 2 * a0/b0**3 * cov(a0,b0)
        # use cov(a0,b0) = 0.0161861960708
        # from calculation above. 
        # Note, again the agreeement using 
        # Krystek is much worse than using
        # uncertain numbers.
        equivalent(b.u,(0.057617)/b0**2,1E-5)
        var_a = (0.29193)**2/b0**2 + + a0**2/b0**4 * (0.057617)**2 + 2 * a0/b0**3 * 0.0161861960708
        equivalent(a.u,math.sqrt(var_a),1E-5)

    def test_mathioulakis(self):
        """
        """
        t_r = (
            (0.1335,0.0106),
            (9.9303,0.0107),
            (19.8774,0.0108),
            (29.8626,0.0110),
            (39.8619,0.0113),
            (49.8241,0.0115),
            (59.7264,0.0098),
            (69.6918,0.0101),
            (79.6740,0.0104),
            (89.7215,0.0107),
        )

        t = (
            (0.0010,0.0175),
            (10.0867,0.0186),
            (20.2531,0.0183),
            (30.3477,0.0186),
            (40.6713,0.0188),
            (50.8418,0.0191),
            (60.9435,0.0202),
            (71.1146,0.0204),
            (81.3378,0.0210),
            (91.5948,0.0217),
        )

        y_data = [ ureal(x,u) for x,u in t_r ]
        x_data = [ ureal(x,u) for x,u in t ]

        # a0, b0 = type_b.line_fit(x_data,y_data).a_b
        # a,b = type_b.line_fit_wtls(x_data,y_data,a_b=(a0,b0)).a_b

        a,b = type_b.line_fit_wtls(x_data,y_data).a_b
        self.assertTrue( equivalent(a.x,0.0991,1E-4) )
        self.assertTrue( equivalent(a.u,0.0123,1E-4) )
        self.assertTrue( equivalent(b.x,0.97839,1E-5))
        self.assertTrue( equivalent(b.u,0.00024,1E-5))

        # This value differs in sign and by one order
        # of magnitude from that reported in the paper.
        # However, their value is clearly wrong because
        # it gives a correlation coefficient > 1.
        # A check with R on the unweighted fit leads
        # to a negative sign and we think they just reported
        # 1E-5 instead of 1E-6 in the paper.
        r = a.get_correlation(b)*a.u*b.u
        self.assertTrue( equivalent(r,-2.41E-6,1E-8) )
     
    def test_tromans(self):
        """
        This data came from an example that caused a failure in
        the brent routine while searching for a root.
        
        The results are not important, just that the 
        routine returns
        
        """
        x = [0.1,0.2,0.5,1.0,2 ]
        ux = [0.007421502, 0.014840599, 0.037099814, 0.074199146, 0.148398052 ]
        y = [2319.00,4620.00,12784.00,21705.00,45952.00 ]
        uy = [ 150.00,340.71,553.54,956.76,1649.68 ]
    
        un_x = [
            ureal(x_i,u_i,label="x_{}".format(i) ) 
                for i,x_i,u_i in zip(xrange(len(x)),x,ux)
        ]
        un_y = [
            ureal(y_i,u_i,label="y_{}".format(i)) 
                for i,y_i,u_i in zip(xrange(len(y)),y,uy)
        ]
        
        # OLS initial estimate
        # NB the default method of obtaining an estimate returns 
        # different results!
        initial = type_a.line_fit(x,y)         
        result = type_b.line_fit_wtls(un_x,un_y,a_b=initial.a_b)

        a,b = result.a_b
        
        self.assertTrue( equivalent(a.x,9.23517,1E-4) )
        self.assertTrue( equivalent(a.u,279.577,1E-3) )
        self.assertTrue( equivalent(b.x,23304.6,1E-1))
        self.assertTrue( equivalent(b.u,1311.37,1E-2))
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'
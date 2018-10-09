"""

"""
from __future__ import division

import decimal
import numbers
import math
import cmath

from itertools import izip

from GTC.lib_real import *
from GTC.vector import *
from GTC.nodes import *
from GTC.named_tuples import VarianceCovariance, StandardUncertainty, GroomedUncertainComplex

inf = float('inf')
nan = float('nan') 

__all__ = (
    'UncertainComplex',
    'std_variance_covariance_complex',
    'VarianceCovariance',
    'StandardUncertainty',
    'willink_hall',
)

#---------------------------------------------------------------------------
def z_to_seq( z ):
    """Return a 4-element sequence (re, -im, im, re)

    Parameter
    ---------
    z : complex
    
    """
    z = complex(z)
    re,im = z.real,z.imag
    return (re, -im, im, re)

#---------------------------------------------------------------------------
class UncertainComplex(object):
    
    """
    A class representing uncertain complex numbers

    """
    
    __slots__ = (
        'real'
    ,   'imag'
    ,   '_value'
    ,   '_u'                    
    ,   '_r'                    
    ,   '_v'                    
    ,   '_label'
    ,   '_context'
    ,   'is_elementary'         
    ,   'is_intermediate'       
    )

    #------------------------------------------------------------------------
    def __init__(self,r,i):
        """
        UncertainComplex(r,i)
        
        Encapsulate a pair of uncertain real number objects

        Parameters
        ----------
        r, i : UncertainReal
        
        """
        # TODO: real and imaginary components do not always 
        # satisfy the following, is this a problem?
        # assert i.is_elementary == r.is_elementary
        # assert i.is_intermediate == r.is_intermediate
        
        assert i._context is r._context 
        
        self.real = r
        self.imag = i
        self._value = complex(r.x,i.x)
        
        self._context = r._context
        self.is_elementary = r.is_elementary
        self.is_intermediate = r.is_intermediate

    #------------------------------------------------------------------------
    def _round(self,digits,df_decimals):
        """
        Return a `RoundedUncertainComplex` 
        
        `digits` specifies the number of significant digits of 
        in the least component uncertainty that will be retained. 
        
        The components of the value will use the same precision. 
        
        The degrees-of-freedom will be represented using 
        `df_decimals` decimal places.

        `df_decimals` specifies the number of decimal places 
        reported for the degrees-of-freedom.
        
        Degrees-of-freedom are greater than 1E6 are set to `inf`.
        
        """
        v11, v12, v21, v22 = self.v 
        re_u = math.sqrt( v11 )
        im_u = math.sqrt( v22 )
        
        den = (re_u*im_u)
        r = v12/den if v12 != 0.0 else 0.0
        
        if v11 != 0 or v22 != 0:
            re = self.real 
            im = self.imag 
            
            # Real and imaginary component uncertainties are different
            # find the lesser uncertainty and round to two digits,
            # then express the results in this precision.
            u = min(re.u, im.u)
            # However, if one component is constant use the other 
            if u == 0.0:
                u = max(re.u, im.u)
        
            log10_u = math.log10( u )
            if log10_u.is_integer(): log10_u += 1 
            
            # The least power of 10 above the value of `u`
            exponent = math.ceil( log10_u ) 
            
            # In fixed-point, precision is the number of decimal places. 
            decimal_places = 0 if exponent-digits >= 0 else int(digits-exponent)
        
            factor = 10**(exponent-digits)
            
            re_x = factor*round(re.x/factor)
            re_u = factor*round(re.u/factor)
            
            im_x = factor*round(im.x/factor)
            im_u = factor*round(im.u/factor)

            # Get the numerals representing uncertainty 
            # When the uncertainty is to the left of the 
            # decimal point there will be `digits` numerals 
            # but to the right of the decimal point there will
            # be sufficient to reach the units column.
            
            # TODO: generalise so that we can use the format 
            # specifier to control this. Let the precision parameter
            # be the number of significant digits in the uncertainty 
            # and format the result accordingly.
            
            # Also need to generalise so that it works with 
            # E and G presentations
            if decimal_places <= 1:
                re_u_digits = "{1:.{0}f}".format(decimal_places,re_u)
                im_u_digits = "{1:.{0}f}".format(decimal_places,im_u)
            else:
                re_u_digits = "{:.0f}".format(re_u/factor)
                im_u_digits = "{:.0f}".format(im_u/factor)

            r_factor = 10**(-3)
            r = r_factor*round(r/r_factor) 
            
            df_factor = 10**(-df_decimals)       
            df = df_factor*math.floor(self.df/df_factor)
            if df > 1E6: df = float('inf')
            
            return GroomedUncertainComplex(
                x = complex(re_x,im_x),
                u = [re_u,im_u],
                r = r,
                df = df,
                label = self.label,
                precision = decimal_places,
                df_decimals = df_decimals,
                re_u_digits = re_u_digits,
                im_u_digits = im_u_digits
            )
        else:
            # A constant 
            # Just use Python's default fixed-point precision
            return GroomedUncertainComplex(
                x = self.x,
                u = [0.0, 0.0],
                r = r,
                df = inf,
                label = self.label,
                precision = 6,
                df_decimals = 0,
                re_u_digits = 0,
                im_u_digits = 0
            )

    #------------------------------------------------------------------------
    def __repr__(self):
        gself = self._round(16,3)
        if self.label is None:
            s = ("ucomplex(({1.real:.{0}g}{1.imag:+.{0}g}j), "
                "u=[{2[0]:.{0}g},{2[1]:.{0}g}], "
                "r={3:.{0}g}, df={5:.{4}g}"
                ")").format( 
                gself.precision,
                gself.x,
                gself.u,
                gself.r,
                gself.df_decimals,
                gself.df
            )
        else:
            s = ("ucomplex(({1.real:.{0}g}{1.imag:+.{0}g}j), "
                "u=[{2[0]:.{0}g},{2[1]:.{0}g}], "
                "r={3:.{0}g}, df={5:.{4}g}, "
                "label='{6!s}')").format( 
                gself.precision,
                gself.x,
                gself.u,
                gself.r,
                gself.df_decimals,
                gself.df,
                gself.label
            )
                      
        return s

    #------------------------------------------------------------------------
    def __str__(self):  
        gself = self._round(2,0)
        return "({1.real:.{0}f}({2}){1.imag:+.{0}f}({3})j)".format(
            gself.precision,
            gself.x,
            gself.re_u_digits,
            gself.im_u_digits
        )
        
    #------------------------------------------------------------------------
    def __neg__(self):
        return UncertainComplex(-self.real,-self.imag)

    #------------------------------------------------------------------------
    def __pos__(self):
        return UncertainComplex(+self.real,+self.imag)

    #------------------------------------------------------------------------
    def __eq__(self,other):
        return complex(self.x) == other

    #------------------------------------------------------------------------
    def __ne__(self,other):
        return complex(self.x) != other
    
    #-----------------------------------------------------------------
    # Boolean depends on value
    def __nonzero__(self):
        return self._value == 0
        
    #------------------------------------------------------------------------
    def __abs__(self):
        return abs( self._value )
    
    #------------------------------------------------------------------------
    def conjugate(self):
        """Return the complex conjugate

        An UncertainComplex object is created by negating the imaginary
        component.

        Returns
        -------
        UncertainComplex
        
        """
        # NB unary '+' clones the object
        return UncertainComplex(+self.real,-self.imag)  
        
    #------------------------------------------------------------------------
    @property
    def x(self):
        """Return the value 

        :returns: complex
        
        **Example**::
            >>> uc = ucomplex(1+2j,(.3,.2))
            >>> uc.x
            (1+2j)

        .. note:: ``uc.x`` is equivalent to ``complex(uc)`` and ``value(uc)``
        
        """
        return self._value

    @property
    def u(self):
        """Return standard uncertainties for the real and imaginary components

        :returns: 2-element sequence of float
        
        **Example**::
            >>> uc = ucomplex(1+2j,(.5,.5))
            >>> uc.u
            standard_uncertainty(real=0.5, imag=0.5)

        .. note:: ``uc.u`` is equivalent to ``uncertainty(uc)``
        
        """        
        try:
            return self._u 
        except AttributeError: 
            self.real.u
            self.imag.u
            self._u = StandardUncertainty(self.real.u,self.imag.u)
            
            return self._u 

    @property
    def v(self):
        """Return the variance-covariance matrix

        The uncertainty of an uncertain complex number can be associated with
        a 4-element variance-covariance matrix.

        :returns: 4-element sequence of float
        
        **Example**::
            >>> uc = ucomplex(1+2j,(.5,.5))
            >>> uc.v
            variance_covariance(rr=0.25, ri=0.0, ir=0.0, ii=0.25)

        .. note:: ``uc.v`` is equivalent to ``variance(uc)``
        
        """
        try:
            return self._v 
        except AttributeError: 
            cv = std_variance_covariance_complex(self)
            self._v = cv
        
            return self._v

    @property
    def r(self):
        """Return the correlation coefficient

        :returns: float
        
        """
        try:
            return self._r 
        except AttributeError: 
            try:
                cv = self._v
            except AttributeError:
                cv = std_variance_covariance_complex(self)
                self._v = cv
                
            self._r = cv[1]/(cv[0]*cv[3]) if cv[1] != 0.0 else 0.0
        
            return self._r
            
    @property
    def df(self):
        """Return the degrees-of-freedom 

        When the object is not an elementary uncertain number, the 
        effective degrees-of-freedom is calculated by the function
        :func:`~library_complex.willink_hall`.

        :returns: float
        
        **Example**::
            >>> uc = ucomplex(1+2j,(.3,.2),3)
            >>> uc.df
            3

        .. note:: 
        
            ``uc.df`` is equivalent to ``dof(uc)``
        
        """
        return willink_hall(self)[1]

    
    # @property
    # def s(self):
        # """Return a summary string

        # The summary string is composed of the label (when defined), the value,
        # the standard uncertainties, the correlation coefficient
        # and the degrees-of-freedom. 

        # :returns: string
        
        # **Example** ::
            # >>> uc = ucomplex(1+2j,(.3,.2),3,label='x')
            # >>> uc.s
            # 'x: (1.00+2.00j), u=[0.30,0.20], r=0.00, df=3.0'
        
            # >>> uc = ucomplex(1+2j,(.3,.2),3)
            # >>> uc.s
            # '(1.00+2.00j), u=[0.30,0.20], r=0.00, df=3.0'

        # .. note::

            # ``uc.s`` is equivalent to ``summary(uc)``

            # The calculation of degrees-of-freedom is invalid
            # if ``df=nan``.

            # When ``df=inf`` is indicated, the of degrees-of-freedom 
            # is greater than 1E6.          
        
        # """
        # label = self._label
        # if label is None:
            # return "%s, u=%s, r=%s, df=%s" % _summary(self)
        # else:
            # return "%s: %s, u=%s, r=%s, df=%s" % ( (label,) + _summary(self) )

    #--------------------------------------------
    @property
    def label(self):
        """The `label` attribute

        .. note:: ``un.label`` is equivalent to ``label(un)``
        
        **Example**::
            >>> z = ucomplex(2.5+.3j,(1,1),label='z')
            >>> z.label
            'z'
            
        """
        try:
            return self._label
        except AttributeError:
            return None

    #------------------------------------------------------------------------
    @classmethod
    def univariate_uc(
        cls,arg,z,dz_dx
    ):
        """
        Create an uncertain complex number as a function of one argument.

        This is a utility method for implementing mathematical
        functions of uncertain complex numbers.

        The parameter 'arg' is the UncertainComplex argument to the
        function, 'z' is the complex value of the function and 'dz_dx'
        is the Jacobian matrix of function value z with respect
        to the real and imaginary components of the function argument.
        
        Parameters
        ----------
        cls : the UncertainComplex class object
        arg : UncertainComplex
        z : complex
        dz_dx : 4-element sequence of float
        
        Returns
        -------
        UncertainComplex
        
        """
        return cls(
            UncertainReal(
                arg._context,
                z.real,
                merge_weighted_vectors(
                    arg.real._u_components,dz_dx[0],
                    arg.imag._u_components,dz_dx[1],
                ),
                merge_weighted_vectors(
                    arg.real._d_components,dz_dx[0],
                    arg.imag._d_components,dz_dx[1],
                ),
                merge_weighted_vectors(
                    arg.real._i_components,dz_dx[0],
                    arg.imag._i_components,dz_dx[1],
                ),
            ),
            UncertainReal(
                arg._context,
                z.imag,
                merge_weighted_vectors(
                    arg.real._u_components,dz_dx[2],
                    arg.imag._u_components,dz_dx[3],
                ),
                merge_weighted_vectors(
                    arg.real._d_components,dz_dx[2],
                    arg.imag._d_components,dz_dx[3],
                ),
                merge_weighted_vectors(
                    arg.real._i_components,dz_dx[2],
                    arg.imag._i_components,dz_dx[3],
                )
            )
        )
    #------------------------------------------------------------------------
    @classmethod
    def bivariate_uc_uc(
        cls,
        lhs,rhs,
        z,
        dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
        dz_dr, # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
        context
    ):
        """
        Create an uncertain complex number as a bivariate function

        This is a utility method for implementing mathematical
        functions of uncertain complex numbers.

        The parameters 'lhs' and 'rhs' are the UncertainComplex
        arguments to the function, 'z' is the complex value of the
        function and 'dz_dl' and 'dz_dr' are the Jacobian matrices
        of the function value z with respect to the real and imaginary
        components of the function's left and right arguments.
        
        Parameters
        ----------
        cls : the UncertainComplex class object
        lhs, rhs : UncertainComplex
        z : complex
        dz_dl, dz_dr : 4-element sequence of float
        
        Returns
        -------
        UncertainComplex
        
        """
        lhs_r = lhs.real
        lhs_i = lhs.imag
        rhs_r = rhs.real
        rhs_i = rhs.imag

        u_lhs_real, u_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._u_components,(dz_dl[0],dz_dl[2]),
            lhs_i._u_components,(dz_dl[1],dz_dl[3])
        )
        u_rhs_real, u_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._u_components,(dz_dr[0],dz_dr[2]),
            rhs_i._u_components,(dz_dr[1],dz_dr[3])
        )
        d_lhs_real, d_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._d_components,(dz_dl[0],dz_dl[2]),
            lhs_i._d_components,(dz_dl[1],dz_dl[3])
        )
        d_rhs_real, d_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._d_components,(dz_dr[0],dz_dr[2]),
            rhs_i._d_components,(dz_dr[1],dz_dr[3])
        )
        i_lhs_real, i_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._i_components,(dz_dl[0],dz_dl[2]),
            lhs_i._i_components,(dz_dl[1],dz_dl[3])
        )
        i_rhs_real, i_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._i_components,(dz_dr[0],dz_dr[2]),
            rhs_i._i_components,(dz_dr[1],dz_dr[3])
        )
        return cls(
            UncertainReal(
                context,
                z.real,
                merge_vectors(
                    u_lhs_real, u_rhs_real
                ),
                merge_vectors(
                    d_lhs_real, d_rhs_real
                ),
                merge_vectors(
                    i_lhs_real, i_rhs_real
                )
            ),
            UncertainReal(
                context,
                z.imag,
                merge_vectors(
                    u_lhs_imag,u_rhs_imag
                ),
                merge_vectors(
                    d_lhs_imag,d_rhs_imag
                ),
                merge_vectors(
                    i_lhs_imag, i_rhs_imag
                )
            )
        )
    #------------------------------------------------------------------------
    @classmethod
    def bivariate_uc_ur(
        cls,
        lhs,rhs,
        z,
        dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
        dz_dr, # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
        context
    ):
        """
        Create an uncertain complex number as a bivariate function

        This is a utility method for implementing mathematical
        functions of uncertain complex numbers.

        The parameter 'lhs' is an UncertainComplex argument to the
        function, 'rhs' is an uncertain real number argument.
        'z' is the complex value of the function and 'dz_dl' and
        'dz_dr' are the Jacobian matrices of the function value z
        with respect to the real and imaginary components of the
        function's left and right arguments.
        
        Parameters
        ----------
        cls : the UncertainComplex class object
        lhs : UncertainComplex
        rhs : UncertainReal
        z : complex
        dz_dl, dz_dr : 4-element sequence of float
        
        Returns
        -------
        UncertainComplex
                
        """
        lhs_r = lhs.real
        lhs_i = lhs.imag

        u_lhs_real, u_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._u_components,(dz_dl[0],dz_dl[2]),
            lhs_i._u_components,(dz_dl[1],dz_dl[3])
        )

        u_rhs_real, u_rhs_imag = scale_vector_twice(rhs._u_components,(dz_dr[0],dz_dr[2]))
        
        d_lhs_real, d_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._d_components,(dz_dl[0],dz_dl[2]),
            lhs_i._d_components,(dz_dl[1],dz_dl[3])
        )

        d_rhs_real, d_rhs_imag = scale_vector_twice(rhs._d_components,(dz_dr[0],dz_dr[2]))

        i_lhs_real, i_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._i_components,(dz_dl[0],dz_dl[2]),
            lhs_i._i_components,(dz_dl[1],dz_dl[3])
        )

        i_rhs_real, i_rhs_imag = scale_vector_twice(rhs._i_components,(dz_dr[0],dz_dr[2]))
        
        return cls(
            UncertainReal(
                context,
                z.real,
                merge_vectors(
                    u_lhs_real,u_rhs_real
                ),
                merge_vectors(
                    d_lhs_real,d_rhs_real
                ),
                merge_vectors(
                    i_lhs_real, i_rhs_real
                )
            ),
            UncertainReal(
                context,
                z.imag,
                merge_vectors(
                    u_lhs_imag,u_rhs_imag
                ),
                merge_vectors(
                    d_lhs_imag,d_rhs_imag
                ),
                merge_vectors(
                    i_lhs_imag, i_rhs_imag
                )
            )
        )
    #------------------------------------------------------------------------
    @classmethod
    def bivariate_uc_n(
        cls,
        lhs,rhs,
        z,
        dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
        dz_dr, # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
        context
    ):
        """
        Create an uncertain complex number as a bivariate function

        This is a utility method for implementing mathematical
        functions of uncertain complex numbers.

        The parameter 'lhs' is an UncertainComplex argument to the
        function, 'rhs' is a real number. 'z' is the complex value
        of the function and 'dz_dl' and 'dz_dr' are the Jacobian
        matrices of the function value z with respect to the real
        and imaginary components of the function's left and right
        arguments.
        
        Parameters
        ----------
        cls : the UncertainComplex class object
        lhs : UncertainComplex
        rhs : float
        z : complex
        dz_dl, dz_dr : 4-element sequence of float
        
        Returns
        -------
        UncertainComplex
                
        """
        lhs_r = lhs.real
        lhs_i = lhs.imag

        u_lhs_real, u_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._u_components,(dz_dl[0],dz_dl[2]),
            lhs_i._u_components,(dz_dl[1],dz_dl[3])
        )

        d_lhs_real, d_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._d_components,(dz_dl[0],dz_dl[2]),
            lhs_i._d_components,(dz_dl[1],dz_dl[3])
        )

        i_lhs_real, i_lhs_imag = merge_weighted_vectors_twice(
            lhs_r._i_components,(dz_dl[0],dz_dl[2]),
            lhs_i._i_components,(dz_dl[1],dz_dl[3])
        )

        return cls(
            UncertainReal(
                context,
                z.real,
                u_lhs_real,
                d_lhs_real,
                i_lhs_real
            ),
            UncertainReal(
                context,
                z.imag,
                u_lhs_imag,
                d_lhs_imag,
                i_lhs_imag
            )
        )
    #------------------------------------------------------------------------
    @classmethod
    def bivariate_ur_uc(
        cls,
        lhs,rhs,
        z,
        dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
        dz_dr, # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
        context
    ):
        """
        Create an uncertain complex number as a bivariate function

        This is a utility method for implementing mathematical
        functions of uncertain complex numbers.

        The parameter 'lhs' is an uncertain real number argument and
        'rhs' is an uncertain complex number argument.
        'z' is the complex value of the function and 'dz_dl' and
        'dz_dr' are the Jacobian matrices of the function value z with
        respect to the real and imaginary components of the function's
        left and right arguments.
        
        Parameters
        ----------
        cls : the UncertainComplex class object
        lhs : UncertainReal
        rhs : UncertainComplex 
        z : complex
        dz_dl, dz_dr : 4-element sequence of float
        
        Returns
        -------
        UncertainComplex
                
        """
        rhs_r = rhs.real
        rhs_i = rhs.imag

        u_lhs_real, u_lhs_imag = scale_vector_twice(lhs._u_components,(dz_dl[0],dz_dl[2]))
        
        u_rhs_real, u_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._u_components,(dz_dr[0],dz_dr[2]),
            rhs_i._u_components,(dz_dr[1],dz_dr[3])
        )

        d_lhs_real, d_lhs_imag = scale_vector_twice(lhs._d_components,(dz_dl[0],dz_dl[2]))
        
        d_rhs_real, d_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._d_components,(dz_dr[0],dz_dr[2]),
            rhs_i._d_components,(dz_dr[1],dz_dr[3])
        )

        i_lhs_real, i_lhs_imag = scale_vector_twice(lhs._i_components,(dz_dl[0],dz_dl[2]))
        
        i_rhs_real, i_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._i_components,(dz_dr[0],dz_dr[2]),
            rhs_i._i_components,(dz_dr[1],dz_dr[3])
        )

        return cls(
            UncertainReal(
                context,
                z.real,
                merge_vectors(
                    u_lhs_real,u_rhs_real
                ),
                merge_vectors(
                    d_lhs_real,d_rhs_real
                ),
                merge_vectors(
                    i_lhs_real, i_rhs_real
                )
            ),
            UncertainReal(
                context,
                z.imag,
                merge_vectors(
                    u_lhs_imag,u_rhs_imag
                ),
                merge_vectors(
                    d_lhs_imag,d_rhs_imag
                ),
                merge_vectors(
                    i_lhs_imag, i_rhs_imag
                )
            )
        )
    #------------------------------------------------------------------------
    @classmethod
    def bivariate_n_uc(
        cls,
        lhs,rhs,
        z,
        dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
        dz_dr, # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
        context
    ):
        """
        Create an uncertain complex number as a bivariate function 

        This is a utility method for implementing mathematical
        functions of uncertain complex numbers.

        The parameter 'lhs' is a real number and 'rhs' is an uncertain
        complex number.
        'z' is the complex value of the function and 'dz_dl' and
        'dz_dr' are the Jacobian matrices of the function value z with
        respect to the real and imaginary components of the function's
        left and right arguments.
        
        Parameters
        ----------
        cls : the UncertainComplex class object
        lhs : float 
        rhs : UncertainComplex
        z : complex
        dz_dl, dz_dr : 4-element sequence of float
        
        Returns
        -------
        UncertainComplex
                
        """
        rhs_r = rhs.real
        rhs_i = rhs.imag

        u_rhs_real, u_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._u_components,(dz_dr[0],dz_dr[2]),
            rhs_i._u_components,(dz_dr[1],dz_dr[3])
        )

        d_rhs_real, d_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._d_components,(dz_dr[0],dz_dr[2]),
            rhs_i._d_components,(dz_dr[1],dz_dr[3])
        )

        i_rhs_real, i_rhs_imag = merge_weighted_vectors_twice(
            rhs_r._i_components,(dz_dr[0],dz_dr[2]),
            rhs_i._i_components,(dz_dr[1],dz_dr[3])
        )
        
        return cls(
            UncertainReal(
                context,
                z.real,
                u_rhs_real,
                d_rhs_real,
                i_rhs_real
            ),
            UncertainReal(
                context,
                z.imag,
                u_rhs_imag,
                d_rhs_imag,
                i_rhs_imag
            )
        )
    #------------------------------------------------------------------------
    def __add__(self,rhs):
        """
        Return the uncertain complex number sum.
        
        Parameter
        ---------
        rhs : UncertainComplex, or UncertainReal, or complex
        
        Returns ``NotImplemented`` otherwise

        Returns
        -------
        UncertainComplex
        
        """
        # TODO: in the shortcuts like x+0 -> x, we ensure
        # that results are never elementary, because 
        # `elementary` status is tested for when archiving. 
        # If this is not really a requirement of archiving 
        # (ie: think about it) then some of the additional 
        # steps below can be removed.
        
        lhs = self
        if isinstance(rhs,UncertainComplex):
            r = self.real + rhs.real
            i = self.imag + rhs.imag
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,UncertainReal):
            r = self.real + rhs
            # Force `i` to be an intermediate uncertain number
            i = +self.imag 
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,numbers.Real):
            if rhs == 0.0:
                return self
            else:            
                r = self.real + rhs
                # Force `i` to be an intermediate uncertain number
                i = +self.imag
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,numbers.Complex):
            if rhs == 0.0:
                return self
            else:
                # Force addition between uncertain numbers
                r = self.real + self.real._context.constant_real( rhs.real, label=None )
                i = self.imag + self.real._context.constant_real( rhs.imag, label=None )
                return UncertainComplex(r,i)
                
        else:
            return NotImplemented
        
    def __radd__(self,lhs):
        if isinstance(lhs,UncertainReal):
            r = lhs + self.real
            # Force `i` to be an intermediate uncertain number
            i = +self.imag
            return UncertainComplex(r,i)
            
        elif isinstance(lhs,complex):
            if lhs == 0.0:
                return self
            else:
                # Force addition between uncertain numbers
                r = self.real._context.constant_real( lhs.real, label=None ) + self.real
                i = self.real._context.constant_real( lhs.imag, label=None ) + self.imag
                return UncertainComplex(r,i)
                
        elif isinstance(lhs,numbers.Real):
            if lhs == 0.0:
                return self
            else:            
                r = lhs + self.real
                # Force `i` to be an intermediate uncertain number
                i = +self.imag
                return UncertainComplex(r,i)
                
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __sub__(self,rhs):
        if isinstance(rhs,UncertainComplex):
            r = self.real - rhs.real
            i = self.imag - rhs.imag
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,UncertainReal):
            r = self.real - rhs
            i = self.imag - 0.0
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,numbers.Real):
            if rhs == 0.0:
                return self
            else:
                r = self.real - rhs
                # Force `i` to be an intermediate uncertain number
                i = +self.imag
                return UncertainComplex(r,i)
                
        elif isinstance(rhs,complex):
            if rhs == 0.0:
                return self
            else:
                # Force subtraction between uncertain numbers
                r = self.real - self.real._context.constant_real( rhs.real, label=None )
                i = self.imag - self.real._context.constant_real( rhs.imag, label=None )
                return UncertainComplex(r,i)
                
        else:
            return NotImplemented
        
    def __rsub__(self,lhs):
        if isinstance(lhs,UncertainReal):
            r = lhs - self.real
            return UncertainComplex(r,-self.imag)
            
        elif isinstance(lhs,numbers.Real):
            if lhs == 0.0:
                return -self
            else:
                r = lhs - self.real
                return UncertainComplex(r,-self.imag)
                
        elif isinstance(lhs,complex):
            if lhs == 0.0:
                return -self
            else:
                r = lhs.real - self.real
                i = lhs.imag - self.imag
                return UncertainComplex(r,i)
                
        else:
            return NotImplemented
    
    #------------------------------------------------------------------------
    def __mul__(self,rhs):
        lhs = self
        if isinstance(rhs,UncertainComplex):
            l = lhs._value
            r = rhs._value
            z = l * r

            dz_dl = z_to_seq( r )                
            dz_dr = z_to_seq( l )            
        
            return UncertainComplex.bivariate_uc_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
            
        elif isinstance(rhs,UncertainReal):
            l = lhs._value
            r = rhs.x
            z = l * r
            
            dz_dl = z_to_seq( r )                
            dz_dr = z_to_seq( l )            

            return UncertainComplex.bivariate_uc_ur(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
            
        elif isinstance(rhs,numbers.Complex):
            if rhs == 1.0:
                return self
            else:            
                l = lhs._value
                r = rhs
                z = l * r

            dz_dl = z_to_seq( r )                
            dz_dr = z_to_seq( 0.0 )            
            
            return UncertainComplex.bivariate_uc_n(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
            
        else:
            return NotImplemented
        
    def __rmul__(self,lhs):
        rhs = self
        if isinstance(lhs,UncertainReal):
            l = lhs.x
            r = rhs._value
            z = l * r
            
            dz_dr = z_to_seq( l )                
            dz_dl = z_to_seq( r )            

            return UncertainComplex.bivariate_ur_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                rhs.real._context
            )
            
        elif isinstance(lhs,numbers.Complex):
            if lhs == 1.0:
                return self
            else:            
                l = lhs
                r = rhs._value
                z = l * r 

            dz_dr = z_to_seq( l )                
            dz_dl = z_to_seq( 0.0 )            
            
            return UncertainComplex.bivariate_n_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                rhs.real._context
            )
            
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __truediv__(self,rhs):
        return self.__div__(rhs)
        
    def __div__(self,rhs):
        lhs = self
        if isinstance(rhs,UncertainComplex):
            l = lhs._value
            r = rhs._value
            z = l / r

            dz_dl = z_to_seq( 1.0 / r ) #z / l if l !=0 else 0 )                
            dz_dr = z_to_seq( -z / r )            
        
            return UncertainComplex.bivariate_uc_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
            
        elif isinstance(rhs,UncertainReal):
            l = lhs._value
            r = rhs.x
            
            z = l / r
            
            dz_dl = z_to_seq( 1.0 / r ) #z / l if l !=0 else 0 )                
            dz_dr = z_to_seq( -z / r )            

            return UncertainComplex.bivariate_uc_ur(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
            
        elif isinstance(rhs,numbers.Complex):
            if rhs == 1.0:
                return self
            else:            
                l = lhs._value
                r = 1.0 * rhs  # ensures we do not get integer division problems
            
            z = l / r

            dz_dl = z_to_seq( 1.0 / r ) #z / l if l !=0 else 0 )                
            dz_dr = z_to_seq( 0.0 )            
            
            return UncertainComplex.bivariate_uc_n(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
            
        else:
            return NotImplemented
 
    def __rtruediv__(self,lhs):
        return self.__rdiv__(lhs)
        
    def __rdiv__(self,lhs):
        rhs = self
        if isinstance(lhs,UncertainReal):
            r = rhs._value
            l = lhs.x
            z = l / r
            
            dz_dr = z_to_seq( -z / r )                
            dz_dl = z_to_seq( 1.0 / r ) #z / l if l !=0 else 0 )            

            return UncertainComplex.bivariate_ur_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                rhs.real._context
            )
            
        elif isinstance(lhs,numbers.Complex):
            r = rhs._value
            l = 1.0 * lhs # ensures we do not get integer division problems
            
            z = l / r

            dz_dr = z_to_seq( -z / r )                
            dz_dl = z_to_seq( 0.0 )            
            
            return UncertainComplex.bivariate_n_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                rhs.real._context
            )
            
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __pow__(self,rhs):
        lhs = self
        if isinstance(rhs,UncertainComplex):
            zl = lhs._value
            zr = rhs._value
            z = zl ** zr
            dz_dl = z_to_seq( zr * z / zl )
            dz_dr = z_to_seq( cmath.log(zl) * z if zl != 0 else 0  )
        
            return UncertainComplex.bivariate_uc_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
        elif isinstance(rhs,UncertainReal):
            zl = lhs._value
            zr = rhs.x
            z = zl ** zr
            dz_dl = z_to_seq( zr * z / zl )
            dz_dr = z_to_seq( cmath.log(zl) * z if zl != 0 else 0  )

            return UncertainComplex.bivariate_uc_ur(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                lhs.real._context
            )
        elif isinstance(rhs,(complex,float,int,long)):
            if rhs == 1.0:
                return self
            else:
                zl = lhs._value
                zr = rhs
                z = zl ** zr
                dz_dl = z_to_seq( zr * z / zl )
                dz_dr = z_to_seq( 0.0 )
                   
                return UncertainComplex.bivariate_uc_n(
                    lhs,rhs,
                    z,
                    dz_dl,
                    dz_dr,
                    lhs.real._context
                )
        else:
            return NotImplemented
        
    def __rpow__(self,lhs):        
        rhs = self
        if isinstance(lhs,UncertainReal):
            zl = lhs.x
            zr = rhs._value
            z = zl ** zr
            dz_dl = z_to_seq( zr * z / zl )
            dz_dr = z_to_seq( cmath.log(zl) * z if zl != 0 else 0 )

            return UncertainComplex.bivariate_ur_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                rhs.real._context
            )
        elif isinstance(lhs,(complex,float,int,long)):
            zl = lhs
            zr = rhs._value
            z = zl ** zr
            dz_dl = z_to_seq( 0.0 )
            dz_dr = z_to_seq( cmath.log(zl) * z  if zl != 0 else 0 )
            
            return UncertainComplex.bivariate_n_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr,
                rhs.real._context
            )
        else:
            return NotImplemented
    
    #-----------------------------------------------------------------
    def _exp(self):
        """
        Complex exponential function
        
        """
        z = cmath.exp(self.x)
        dz_dx = z_to_seq( z )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _log(self):
        """
        Complex natural log function
        
        There is one branch cut, from 0 along the negative real
        axis to -Inf, continuous from above.
        
        """
        x = complex(self.x)
        z = cmath.log(x)
        dz_dx = z_to_seq( 1./x )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _log10(self):
        """
        Complex base-10 log function
        
        There is one branch cut, from 0 along the negative real
        axis to -Inf, continuous from above.
        
        """
        x = complex(self.x)
        z = cmath.log10(x)
        dz_dx = z_to_seq( LOG10_E/x )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _sqrt(self):
        """
        Complex square root function
        
        There is one branch cut, from 0 along the negative real
        axis to -Inf, continuous from above.
        
        """
        z = cmath.sqrt(self.x)
        dz_dx = z_to_seq( 1.0/(2.0 * z) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _sin(self):
        """
        Complex sine function
        
        """
        z = cmath.sin(self.x)
        dz_dx = z_to_seq( cmath.cos(self.x) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _cos(self):
        """
        Complex cosine function
        
        """
        z = cmath.cos(self.x)
        dz_dx = z_to_seq( -cmath.sin(self.x) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _tan(self):
        """
        Complex tangent function
        
        """
        z = cmath.tan(self.x)
        d = cmath.cos(self.x)
        dz_dx = z_to_seq( 1./d**2 )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _asin(self):
        """
        Inverse complex sine function
        
        There are two branch cuts: one extends right from 1 along the
        real axis to Inf, continuous from below; the other extends
        left from -1 along the real axis to -Inf, continuous from
        above.
        
        """
        x = complex(self.x)
        z = cmath.asin(x)
        dz_dx = z_to_seq( 1./cmath.sqrt(1 - x**2) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _acos(self):
        """
        Inverse complex cosine function
        
        There are two branch cuts: one extends right from 1 along the
        real axis to Inf, continuous from below; the other extends
        left from -1 along the real axis to -Inf, continuous from
        above.
        
        """
        x = complex(self.x)
        z = cmath.acos(x)
        dz_dx = z_to_seq( -1./cmath.sqrt(1 - x**2) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _atan(self):
        """
        Inverse complex tangent function
        
        There are two branch cuts:
        One extends from 1j along the imaginary axis to Inf j,
        continuous from the right. The other extends from -1j
        along the imaginary axis to -Inf j, continuous from the left.
        
        """
        x = complex(self.x)
        z = cmath.atan(x)
        dz_dx = z_to_seq( 1./(1 + x**2) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _sinh(self):
        """
        Complex hyperbolic sine function
        
        """
        z = cmath.sinh(self.x)
        dz_dx = z_to_seq( cmath.cosh(self.x) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _cosh(self):
        """
        Complex hyperbolic cosine function
        
        """
        z = cmath.cosh(self.x)
        dz_dx = z_to_seq( cmath.sinh(self.x) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _tanh(self):
        """
        Complex hyperbolic tangent function
        
        """
        z = cmath.tanh(self.x)
        d = cmath.cosh(self.x)
        dz_dx = z_to_seq( 1./d**2 )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _asinh(self):
        """
        Inverse complex hyperbolic sine function
        
        There are two branch cuts: one extends from 1j along the
        imaginary axis to Inf j, continuous from the right;
        the other extends from -1j along the imaginary axis
        to -Inf j, continuous from the left.
        
        """
        x = complex(self.x)
        z = cmath.asinh(x)
        dz_dx = z_to_seq( 1./cmath.sqrt(1 + x**2) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _acosh(self):
        """
        Inverse complex hyperbolic cosine function
        
        There is one branch cut, extending left from 1 along the
        real axis to -Inf, continuous from above.

        """
        x = complex(self.x)
        z = cmath.acosh(x)
        dz_dx = z_to_seq( 1./cmath.sqrt((x-1)*(x+1)) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _atanh(self):
        """
        Inverse complex hyperbolic tangent function
        
        There are two branch cuts: one extends from 1 along the
        real axis to Inf, continuous from below;
        the other extends from -1 along the real axis to -Inf,
        continuous from above.

        """
        x = complex(self.x)
        z = cmath.atanh(x)
        dz_dx = z_to_seq( 1./((1-x)*(1+x)) )
        return UncertainComplex.univariate_uc(
            self,
            z,
            dz_dx
        )

    #-----------------------------------------------------------------
    def _magnitude(self):
        """
        Return the magnitude.

        Taking the magnitude of an uncertain complex number
        generates an uncertain real number.
        
        Returns
        -------
        UncertainReal
        
        """
        re = self.real
        im = self.imag
        
        x = complex(self.x)
        mag_x = abs(x)
        try:
            dz_dre = x.real/mag_x
            dz_dim = x.imag/mag_x
        except ZeroDivisionError:
            raise ZeroDivisionError(
                "uncertainty(z) is undefined when |z| = 0"
            )
        
        return UncertainReal(
                re._context
            ,   mag_x
            ,   merge_weighted_vectors(re._u_components,dz_dre,im._u_components,dz_dim)
            ,   merge_weighted_vectors(re._d_components,dz_dre,im._d_components,dz_dim)
            ,   merge_weighted_vectors(re._i_components,dz_dre,im._i_components,dz_dim)
            )

    #-----------------------------------------------------------------
    def _mag_squared(self):
        """
        Return the magnitude squared.

        Taking the norm of an uncertain complex number generates
        an uncertain real number.
        
        Returns
        -------
        UncertainReal
        
        """
        re = self.real
        im = self.imag
        
        x = complex(self.x)
        dz_dre = 2.0*x.real
        dz_dim = 2.0*x.imag
        
        return UncertainReal(
                re._context
            ,   abs(x)**2
            ,   merge_weighted_vectors(re._u_components,dz_dre,im._u_components,dz_dim)
            ,   merge_weighted_vectors(re._d_components,dz_dre,im._d_components,dz_dim)
            ,   merge_weighted_vectors(re._i_components,dz_dre,im._i_components,dz_dim)
            )
    
    #-----------------------------------------------------------------
    def _phase(self):
        """
        Return the phase.

        Taking the phase of an uncertain complex number
        generates an uncertain real number.
        
        :rtype: :class:`UncertainReal`
        
        """
        re = self.real
        im = self.imag
        
        return im._atan2(re)

#---------------------------------------------------------------------------
def std_variance_covariance_complex(x):
    """Return the variance-covariance matrix

    The variance-covariance matrix characterises the  uncertainty
    of an uncertain complex number.
    
    Parameter
    ---------
    x : UncertainComplex

    Returns
    -------
    a 4-element sequence of float
    
    """
    re, im = x.real, x.imag
    
    v_r = re.v
    v_i = im.v
    cv = std_covariance_real(re,im)
    
    return VarianceCovariance(v_r,cv,cv,v_i)

#---------------------------------------------------------------------------
def _covariance_submatrix(context,correlations,u_re,u_im):
    """Return v_rr, v_ir, v_ii, the 3 covariance matrix terms

    `u_re` and `u_im` are `GTC.Vector`s containing uid's
    and component of uncertainty values.
    
    """
    # Utility function for `willink_hall(x)`
    # Each of the terms returned is like an 
    # LPU calculation for the variance-covariance
    # of a sub-matrix of the covariance matrix.
    
    v_rr = v_ri = v_ii = 0.0

    # In this algorithm, we need uncertainty components with
    # the same set of influences.
    keys = u_re.keys()
    assert keys == u_im.keys()
    
    for i,x_i in enumerate(keys):

        # In the absence of correlation, just these terms
        x_u_re = u_re[x_i]
        v_rr += x_u_re**2
        
        x_u_im = u_im[x_i]
        v_ii += x_u_im**2

        v_ri += x_u_re * x_u_im

        # Additional terms required when there are correlations
        if x_i in correlations:
            row_x = correlations[x_i]
            
            v_rr += sum(
                2.0 * x_u_re * u_re[y_i] * row_x.get(y_i,0.0)
                    for y_i in keys[i+1:]
            )

            v_ii += sum(
                2.0 * x_u_im * u_im[y_i] * row_x.get(y_i,0.0)
                    for y_i in keys[i+1:]
            )

            # Cross product of `u_re` and `u_im` so we need
            # to iterate over all keys (there is no symmetry
            # allowing us to step over just half). We just
            # skip the term `u_im[x_uid]`, which is already
            # in the sum.
            v_ri += sum(
                x_u_re * u_im[y_i] * row_x.get(y_i,0.0)
                    for y_i in keys if y_i != x_i
            )

    return v_rr, v_ri, v_ii

#---------------------------------------------------------------------------
class _EnsembleComponents(object):
    
    """
    Worker class for the willink_hall function 
    """
    
    __slots__ = ('u_re','u_im','nu')
    
    # Class attributes to accumulate results 
    # In use, `clear` should be called initially;
    # initialising to `None` will cause an error 
    # immediately if this is not done.
    sum_sq_u11 = None
    sum_sq_diag = None
    sum_sq_u22 = None
    
    def __init__(self,nu):
    
        # Instance attributes to hold data
        self.u_re = Vector()
        self.u_im = Vector()
        self.nu = nu
        
    def accumulate(self,cxt,correlations):
        """
        Update the running sums from this object.
        
        """
        # Calculate `v` = u * r * u'
        v_11,v_12,v_22 = _covariance_submatrix(
            cxt,
            correlations,
            self.u_re,
            self.u_im
        )
        
        nu = self.nu 
        
        _EnsembleComponents.sum_sq_u11 += v_11 * v_11 / nu
        _EnsembleComponents.sum_sq_u22 += v_22 * v_22 / nu
        _EnsembleComponents.sum_sq_diag += (v_11*v_22 + v_12**2) / nu
  
    @classmethod 
    def clear(cls):
        """
        Set running totals to zero 
        
        """
        _EnsembleComponents.sum_sq_u11 = 0
        _EnsembleComponents.sum_sq_u22 = 0
        _EnsembleComponents.sum_sq_diag = 0
        
#---------------------------------------------------------------------------
def willink_hall(x):
    """Return the covariance matrix and degrees of freedom

    A 2-element sequence is returned. The first element contains  
    a sequence of variance-covariance elements, the second element
    contains the degrees-of-freedom associated with `x`. 
    
    This calculation is described in Willink and Hall,
    Metrologia 2002, 39, pp 361-369.

    Parameters
    ----------
    x : UncertainComplex

    Returns
    -------
    2-element sequence containing a 4-element sequence of float
    and a float.

    If the calculation of degrees of freedom is illegal, `nan`
    is returned as the second element.
    
    """
    #
    # The main purpose of the code below is to detect illegal cases
    # and accumulate uncertainty components associated with influences
    # that have finite DoF. The variance calculation
    # is delegated to `std_variance_covariance_complex()`,
    # which calls routines in `library_real` to evaluate variance and 
    # covariance regardless of degrees of freedom.
    #
    if not isinstance(x,UncertainComplex):
        raise RuntimeError(
            "expected 'UncertainComplex' got: '{!r}'".format(x)
        )
    
    real = x.real
    imag = x.imag

    context = real._context

    if real.is_elementary:
        return VarianceAndDof(
            std_variance_covariance_complex(x),
            real.df
        )
    else:
        # willink_hall separates the work to be done on truly 
        # independent UNs from that on correlated ones.
        
        # Need all keys for the independent components
        re_u = extend_vector(x.real._u_components,x.imag._u_components)    
        im_u = extend_vector(x.imag._u_components,x.real._u_components)
        
        # Need all keys for the dependent components
        re_d = extend_vector(x.real._d_components,x.imag._d_components)    
        im_d = extend_vector(x.imag._d_components,x.real._d_components)

        if len(re_u) + len(re_d) == 0:
            # A constant
            return VarianceAndDof((0.,0.,0.,0.),inf)
            
        ids_u = re_u.keys()
        ids_d = re_d.keys()
                
        degrees_of_freedom_u = [ k_i.df for k_i in ids_u ]
        degrees_of_freedom_d = [ k_i.df for k_i in ids_d ]
        
        # Everything has infinite DoF?
        if ( 
            degrees_of_freedom_u.count(inf) == len(degrees_of_freedom_u) and 
            degrees_of_freedom_d.count(inf) == len(degrees_of_freedom_d)
        ):
            return VarianceAndDof( std_variance_covariance_complex(x), inf )
        
        # Initially clear the accumulators
        _EnsembleComponents.clear()
        
        # -------------------------------------------------------------------
        # Process independent components.
        #   They cannot belong to an ensemble and
        #   they cannot be correlated.
        #
        for i_re,id_re in enumerate( ids_u ):
            
            nu_i = degrees_of_freedom_u[i_re]

            if not is_infinity( nu_i ):
                # update the sums immediately (does NOT belong to an ensemble)
                v_11 = re_u[id_re]**2
                v_22 = im_u[id_re]**2
                v_12 = re_u[id_re]*im_u[id_re]
                                
                _EnsembleComponents.sum_sq_u11 += v_11*v_11/nu_i
                _EnsembleComponents.sum_sq_u22 += v_22*v_22/nu_i
                _EnsembleComponents.sum_sq_diag += (v_11*v_22 + v_12**2)/nu_i
                
        # -------------------------------------------------------------------
        # Process the dependent components
        #
        len_ids = len(ids_d) 
        
        if len_ids != 0:
        
            correlations = context._correlations._mat
            
            skip_imaginary = False      # Initial value 
            
            ensemble_reg = dict()       # keys: frozenset , values: _EnsembleComponent 
            
            # _EnsembleComponents.clear() # Clear previous accumulation

            # There is one element in `ids` for each real-valued 
            # component (ie for 3 complex influences len(ids) == 6)
            for i_re,id_re in enumerate( ids_d ):
            
                # If an influence is complex, the real and imaginary
                # components are handled in the first pass, so 
                # we need to skip to the next id in the list. 
                
                # TODO: this can probably be relaxed when ensembles 
                # are managed. Just treat complex as an ensemble. 
                if skip_imaginary:
                    skip_imaginary = False
                    continue
                
                # mapping between LeafNodes and correlation coefficients
                row_re = correlations.get(id_re,{})
                # If {}, can we just continue?
                
                nu_i = degrees_of_freedom_d[i_re]
                i_re_infinite = is_infinity( nu_i )         

                ensemble_i = frozenset(
                    context._ensemble.get( id_re, frozenset() )
                )
                if len(ensemble_i) and ensemble_i not in ensemble_reg:
                    # Non-trivial ensemble that has not yet been identified
                    ensemble_reg[ensemble_i] = _EnsembleComponents(nu_i)
                   
                # `components_i` holds the various components 
                # associated with this influence. When it is 
                # part of an ensemble, we reuse the same object.
                components_i = ensemble_reg.get(
                    ensemble_i,
                    _EnsembleComponents(nu_i)
                )
                
                if id_re in context._complex_ids:
                    # This is a complex influence
                    skip_imaginary = True
                    
                    id_im = context._complex_ids[id_re][1]

                    # mapping between LeafNodes and correlation coefficients
                    row_im = correlations.get( id_im, {} )

                    # This steps over the imaginary component, 
                    # which is assumed to follow 
                    next_i = i_re + 2

                    # Check for correlations with any other (real) influence 
                    # and perhaps abort DoF calculation
                    if next_i < len_ids:
                        # `j` is any of the other (real) influences of `i`  
                        for j, j_id in enumerate( ids_d[next_i:] ):
                        
                            # Look for the illegal case of correlation between 
                            # influences when at least one has finite dof and 
                            # they are not in an ensemble together.
                            
                            if i_re_infinite and is_infinity( 
                                    degrees_of_freedom_d[next_i+j]
                                ):  
                                    continue
                                
                            elif (
                                j_id not in ensemble_i  
                                    and ( 
                                        j_id in row_re or j_id in row_im 
                                    )
                            ):
                                # Illegal case: `j` is not in an ensemble with `i`
                                # but `j` is correlated with one of the components
                                # of `i`
                                return VarianceAndDof(
                                    std_variance_covariance_complex(x),
                                    nan
                                )
                        
                    # If we get here, this complex influence
                    # can be used for the DoF calculation. 
                    # Update the buffer.
                    if not i_re_infinite:
                        components_i.u_re.append( id_re,re_d[id_re] )
                        components_i.u_re.append( id_im,re_d[id_im] )
                        components_i.u_im.append( id_re,im_d[id_re] )
                        components_i.u_im.append( id_im,im_d[id_im] )
                    
                else:
                    # This is a real influence. 
                    next_i = i_re+1

                    # Check for correlations and perhaps abort DoF calculation
                    # TODO: this can probably be removed now
                    if not i_re_infinite and next_i < len_ids:
                        
                        for j, j_id in enumerate( ids_d[next_i:] ):                        
                            # Look for the illegal cases
                            if (
                                not is_infinity( 
                                    degrees_of_freedom_d[next_i+j]  
                                ) 
                                and id_re not in ensemble_i
                                and ( j_id in row_re )
                            ):
                                assert False, "should not now occur"
                                # # Illegal case: `j` is correlated with `i`
                                # # but they are not in an ensemble
                                # return VarianceAndDof(
                                    # std_variance_covariance_complex(x),
                                    # nan
                                # )

                    # If we can get here, this real influence can be
                    # used for the DoF calculation. 
                    # Update the buffer. 
                    if not i_re_infinite:
                        components_i.u_re.append( id_re,re[id_re] )
                        components_i.u_im.append( id_re,im[id_re] )

                # If the current influence does NOT belong to an ensemble
                # update the sums immediately, otherwise wait until the end
                if len( ensemble_i ) == 0:
                    components_i.accumulate(context,correlations)
                                
            for ec_i in ensemble_reg.itervalues():
                ec_i.accumulate(context,correlations)

        #------------------------------------------------------                
        # End of for loop
        #
        var = std_variance_covariance_complex(x)
        sum_u11, sum_u12, dum, sum_u22 = var
        
        if sum_u11 == 0.0 and sum_u12 == 0.0 and sum_u22 == 0.0:
            # This is a pathological case that occurs when
            # all components have zero uncertainty. We can't
            # work out the DoF in this case.
            return VarianceAndDof(var,nan)
            
        # Normalisation constant for better numerical stability
        u2_bar = (sum_u11 + sum_u22)**2/4.0  

        sum_sq_u11 = _EnsembleComponents.sum_sq_u11
        sum_sq_u22 = _EnsembleComponents.sum_sq_u22
        sum_sq_diag = _EnsembleComponents.sum_sq_diag
            
        A = 2.0*sum_u11*sum_u11/u2_bar
        D = (sum_u11*sum_u22 + sum_u12*sum_u12)/u2_bar
        F = 2.0* sum_u22*sum_u22/u2_bar

        a = 2.0*sum_sq_u11/u2_bar  
        d = sum_sq_diag/u2_bar
        f = 2.0*sum_sq_u22/u2_bar

        num = (A + D + F)
        den = (a + d + f)

        try:
            dof = num / den
        except ZeroDivisionError:
            dof = inf
            
        return VarianceAndDof(var,dof)
           

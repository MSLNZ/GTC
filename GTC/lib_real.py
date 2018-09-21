"""
"""
from __future__ import division

import math
import cmath
import numbers

from string import Template
from itertools import izip

from GTC2.GTC import nodes 
from GTC2.GTC.named_tuples import VarianceAndDof, GroomedUncertainReal, ComponentOfUncertainty
from GTC2.GTC.vector import *

inf = float('inf')
nan = float('nan') 

is_infinity = math.isinf
is_undefined = math.isnan

LOG10_E = math.log10(math.e)

#----------------------------------------------------------------------------
class UncertainReal(object):
    
    """
    An `UncertainReal` holds information about the estimate 
    of a real-valued quantity
    
    """
    
    __slots__ = [
        '_context'              # The Context object associated with this UN
    ,   '_x'                    # The value estimate
    ,   '_u'                    # The standard uncertainty in the estimate
    ,   '_u_components'         # weighted Jacobian matrix for independent 
    ,   '_d_components'         # weighted Jacobian matrix for dependent cpts
    ,   '_i_components'         # Intermediate components of uncertainty
    ,   'is_elementary'         # True for elementary UNs 
    ,   'is_intermediate'       # True for intermediate UNs 
    ,   '_node'                 # May refer to a Node
    ]            

    #-------------------------------------------------------------------------
    def __init__(self,context,x,u_comp,d_comp,i_comp,node=None):

        self._context = context
        self._x = float(x)
        self._u_components = u_comp
        self._d_components = d_comp
        self._i_components = i_comp
        self._node = node
 
        if node is None:
            self.is_elementary = False 
            self.is_intermediate = False
            
        else: 
        # Constants are Leaf nodes, but the UID is None,
        # they should not be classed as `elementary`
            self.is_elementary = (
                isinstance(self._node,nodes.Leaf)
                    and not self._node.uid is None
            )
            # An intermediate uncertain number has a `Node` object  
            self.is_intermediate = type(self._node) is nodes.Node
            
            assert not(self.is_elementary and self.is_intermediate)

    #-------------------------------------------------------------------------
    def _round(self,digits,df_decimals):
        """
        Return a `RoundedUncertainReal` 
        
        `digits` specifies the number of significant digits of 
        in uncertainty that will be retained. The value will use 
        the same precision. The degrees-of-freedom will be  
        represented using `df_decimals` decimal places.

        `df_decimals` specifies the number of decimal places 
        reported for the degrees-of-freedom.
        
        Degrees-of-freedom are greater than 1E6 are set to `inf`.
        
        """
        if self.u != 0:
            log10_u = math.log10( self.u )
            if log10_u.is_integer(): log10_u += 1 
            
            # The least power of 10 above the value of `u`
            exponent = math.ceil( log10_u ) 
            
            # In fixed-point, precision is the number of decimal places. 
            precision = 0 if exponent-digits >= 0 else int(digits-exponent)
        
            factor = 10**(exponent-digits)
            
            x = factor*round(self.x/factor)
            u = factor*round(self.u/factor)
            
            # Just the digits representing uncertainty 
            # e.g.: 3.141(20)
            u_digits = "{:.0f}".format(self.u/factor)[:digits]

            df_factor = 10**(-df_decimals)       
            df = df_factor*math.floor(self.df/df_factor)
            if df > 1E6: df = float('inf')
            
            if self.label is None:
                return GroomedUncertainReal(
                    x = x,
                    u = u,
                    df = df,
                    label = None,
                    precision = precision,
                    df_decimals = df_decimals,
                    u_digits = u_digits
                )
            else:
                return GroomedUncertainReal(
                    x = x,
                    u = u,
                    df = df,
                    label = self.label,
                    precision = precision,
                    df_decimals = df_decimals,
                    u_digits = u_digits
                )
        else:
            # A constant 
            # Use default fixed-point precision
            if self.label is None:
                return GroomedUncertainReal(
                    x = self.x,
                    u = 0,
                    df = inf,
                    label = None,
                    precision = 6,
                    df_decimals = 0,
                    u_digits = "00"
                )  
            else:
                return GroomedUncertainReal(
                    x = self.x,
                    u = 0,
                    df = inf,
                    label = self.label,
                    precision = 6,
                    df_decimals = 0,
                    u_digits = "00"
                )  
        
    #------------------------------------------------------------------------
    def __repr__(self):
        # repr() is for displaying information, so 
        # use full precision (16 digits) in x and u.
        gself = self._round(16,3)        
        
        if gself.label is None:
            s = "ureal({1:.{0}g},{2:.{0}g},{4:.{3}g})".format( 
                gself.precision,
                gself.x,
                gself.u,
                gself.df_decimals,
                gself.df
            )
        else:
            s = "ureal({1:.{0}g},{2:.{0}g},{4:.{3}g},label='{5!s}')".format( 
                gself.precision,
                gself.x,
                gself.u,
                gself.df_decimals,
                gself.df,
                self._node.tag
            )
                      
        return s

    #------------------------------------------------------------------------
    def __str__(self):
        # str() is for presentation, so assume that the usual 
        # 2-digit fixed point format in the uncertainty will suffice.          
        gself = self._round(2,0)
        return "{1.x:.{0}f}({1.u_digits})".format( gself.precision, gself )
     
    #------------------------------------------------------------------------
    def __nonzero__(self):
        # Used to convert to a Boolean 
        return self._x != 0
        
    #------------------------------------------------------------------------
    # 
    def __abs__(self):
        return abs(self._x)

    #------------------------------------------------------------------------
    @property
    def real(self):
        """Return the real component of this uncertain number
        """
        return self  
    
    #------------------------------------------------------------------------
    @property
    def imag(self):
        """Returns the imaginary component of this uncertain number
        
        Always returns an uncertain real number with
        a value of zero and no uncertainty
        
        """
        # Returning an UN constant ensures that an algorithm
        # expecting an uncertain number will not break
        return self._context.constant_real(0.0,label=None)
 
    #------------------------------------------------------------------------
    def conjugate(self):
        """Return the complex conjugate

        Returns
        -------
        UncertainReal
        
        """
        return self.real
 
    #------------------------------------------------------------------------
    # Comparisons 
    def __eq__(self,other):
        return self._x == other
    def __ne__(self,other):
        return self._x != other
    def __gt__(self,other):
        return self._x > other
    def __ge__(self,other):
        return self._x >= other
    def __lt__(self,other):
        return self._x < other
    def __le__(self,other):
        return self._x <= other

    #------------------------------------------------------------------------
    @property
    def x(self):
        """Return the value

        :returns: float
        
        **Example**::
            >>> ur = ureal(2.5,0.5)
            >>> ur.x
            2.5

        .. note:: ``un.x`` is equivalent to ``float(un)`` and to ``value(un)``
        
        """
        return self._x

    #------------------------------------------------------------------------
    @property
    def u(self):
        """Return the standard uncertainty 
        
        :returns: float
        
        **Example**::
            >>> ur = ureal(2.5,0.5)
            >>> ur.u
            0.5

        .. note:: ``un.u`` is equivalent to ``uncertainty(un)``
        
        """
        try:
            return self._u
            
        except AttributeError:
            # cache the return value
            if self.is_elementary or self.is_intermediate:
                self._u = self._node.u
            else:
                self._u = math.sqrt( std_variance_real(self) )

            return self._u 
            
   #------------------------------------------------------------------------
    @property
    def v(self):
        """Return the standard variance
        
        :returns: float
        
        **Example**::
            >>> ur = ureal(2.5,0.5)
            >>> ur.v
            0.25

        .. note:: ``un.v`` is equivalent to ``variance(un)``
        
        """
        u = self.u 
        return u*u

    #------------------------------------------------------------------------
    @property
    def df(self):
        """Return the degrees of freedom
        
        :returns: float
        
        **Example**::
            >>> ur = ureal(2.5,0.5,3)
            >>> ur.df
            3

        .. note:: ``un.df`` is equivalent to ``dof(un)``
        
        """
        if self.is_elementary:
            return self._node.df
        else:
            return welch_satterthwaite(self)[1]

    #-----------------------------------------------------------------
    @property
    def label(self):
        """The label attribute

        .. note:: ``un.label`` is equivalent to ``label(un)``
        
        **Example**::
            >>> x = ureal(2.5,0.5,label='x')
            >>> x.label
            'x'
        
            >>> label(x)
            'x'
            
        """
        try:
            # Elementary, constant and intermediate UNs
            return self._node.tag
        except AttributeError:
            return None          

    #------------------------------------------------------------------------
    # Arithmetic operations
    def __neg__(self):
        """
        Unary negative
        
        """
        return UncertainReal(
                self._context
            ,   -self.x
            ,   scale_vector(self._u_components,-1.0)
            ,   scale_vector(self._d_components,-1.0)
            ,   scale_vector(self._i_components,-1.0)
            )
            
    #------------------------------------------------------------------------
    def __pos__(self):
        """
        Unary positive

        Copies the uncertain real number

        """     
        # NB the copy is neither elementary nor intermediate
        return UncertainReal(
                self._context
            ,   self.x
            ,   Vector(copy=self._u_components)
            ,   Vector(copy=self._d_components)
            ,   Vector(copy=self._i_components)
            )
    
    #------------------------------------------------------------------------
    def __add__(self,rhs):
        if isinstance(rhs,UncertainReal):
            return _add(self,rhs)
        elif isinstance(rhs,numbers.Real):
            return _add(self,float(rhs))      
        elif isinstance(rhs,complex):
            return _add_re_z(self,complex(rhs))        
        else:
            return NotImplemented
        
    def __radd__(self,lhs):
        if isinstance(lhs,numbers.Real):
            return _radd(float(lhs),self)
        elif isinstance(lhs,complex):
            return _add_z_re(complex(lhs),self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __sub__(self,rhs):
        if isinstance(rhs,UncertainReal):
            return _sub(self,rhs)
        elif isinstance(rhs,numbers.Real):
            return _sub(self,float(rhs))
        elif isinstance(rhs,complex):
            return _sub_re_z(self,complex(rhs))
        else:
            return NotImplemented
        
    def __rsub__(self,lhs):
        if isinstance(lhs,numbers.Real):
            return _rsub(float(lhs),self)
        elif isinstance(lhs,complex):
            return _sub_z_re(complex(lhs),self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __mul__(self,rhs):
        if isinstance(rhs,UncertainReal):
            return _mul(self,rhs)
        elif isinstance(rhs,numbers.Real):
            return _mul(self,float(rhs))
        elif isinstance(rhs,complex):
            return _mul_re_z(self,complex(rhs))
        else:
            return NotImplemented
        
    def __rmul__(self,lhs):
        if isinstance(lhs,numbers.Real):
            return _rmul(float(lhs),self)
        elif isinstance(lhs,complex):
            return _mul_z_re(complex(lhs),self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __truediv__(self,rhs):
        return self.__div__(rhs)
        
    def __div__(self,rhs):
        if isinstance(rhs,UncertainReal):
            return _div(self,rhs)
        elif isinstance(rhs,numbers.Real):
            return _div(self,float(rhs))
        elif isinstance(rhs,complex):
            return _div_re_z(self,complex(rhs))
        else:
            return NotImplemented

    def __rtruediv__(self,lhs):
        return self.__rdiv__(lhs)
        
    def __rdiv__(self,lhs):
        if isinstance(lhs,numbers.Real):
            return _rdiv(float(lhs),self)
        elif isinstance(lhs,complex):
            return _div_z_re(complex(lhs),self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __pow__(self,rhs):
        if isinstance(rhs,UncertainReal):
            return _pow(self,rhs)
        elif isinstance(rhs,numbers.Real):
            return _pow(self,float(rhs))
        elif isinstance(rhs,complex):
            return _pow_re_z(self,complex(rhs))
        else:
            return NotImplemented

    def __rpow__(self,lhs):
        if isinstance(lhs,numbers.Real):
            return _rpow(float(lhs),self)
        elif isinstance(lhs,complex):
            return _pow_z_re(complex(lhs),self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def _exp(self):
        """
        Real exponential function
        
        """
        x = float(self.x)
        y = math.exp( x )
        dy_dx = y
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
        )
        
    #------------------------------------------------------------------------
    def _log(self):
        """
        Real natural log function
        
        """
        x = float(self.x)
        y = math.log( x )
        dy_dx = 1.0/x
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
        )
    
    #------------------------------------------------------------------------
    def _log10(self):
        """
        Real base-10 log function
        
        """
        x = float(self.x)
        y = math.log10( x )
        dy_dx = LOG10_E/x
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
        )
    
    #------------------------------------------------------------------------
    def _sqrt(self):
        """
        Real square root function
        
        """
        y = math.sqrt( self.x )
        dy_dx = 0.5/y
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
        )
        
    #------------------------------------------------------------------------
    def _sin(self):
        """
        Real sine function
        
        """
        y = math.sin( self.x )
        dy_dx = math.cos( self.x )
        
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )
    #------------------------------------------------------------------------
    def _cos(self):
        """
        Real cosine function
        
        """
        y = math.cos( self.x )
        dy_dx = -math.sin( self.x )
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #------------------------------------------------------------------------
    def _tan(self):
        """
        Real tangent function
        
        """
        y = math.tan( self.x )
        d = math.cos( self.x )
        dy_dx = 1.0/(d*d)
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )
            
    #-----------------------------------------------------------------
    def _asin(self):
        """
        Inverse real sine function
        
        """
        x = float(self.x)
        y = math.asin( x )
        dy_dx = 1.0/math.sqrt((1 - x)*(1+x))
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _acos(self):
        """
        Inverse real cosine function
        
        """
        x = float(self.x)
        y = math.acos( x )
        dy_dx = -1.0/math.sqrt((1 - x)*(1+x))
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _atan(self):
        """
        Inverse real tangent function
        
        """
        x = float(self.x)
        y = math.atan( x )
        dy_dx = 1.0/(1 + x*x)
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _atan2(self,rhs):
        """
        Two-argument inverse real tangent function
        
        """
        if isinstance(rhs,UncertainReal):
            return _atan2_re_re(self,rhs)
        elif isinstance(rhs,(float,int,long)):
            return _atan2_re_x(self,float(rhs))
        elif isinstance(rhs,complex):
            raise TypeError,'atan2 is undefined with a complex argument'
        else:
            return NotImplemented

    def _ratan2(self,lhs):
        if isinstance(lhs,(float,int,long)):
            return _atan2_x_re(float(lhs),self)
        elif isinstance(lhs,complex):
            raise TypeError,'atan2 is undefined with a complex argument'
        else:
            return NotImplemented
        
    #-----------------------------------------------------------------
    def _sinh(self):
        """
        Real hyperbolic sine function
        
        """
        y = math.sinh( self.x )
        dy_dx = math.cosh( self.x )
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _cosh(self):
        """
        Real hyperbolic cosine function
        
        """
        y = math.cosh( self.x )
        dy_dx = math.sinh( self.x )
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _tanh(self):
        """
        Real hyperbolic tangent function
        
        """
        y = math.tanh( self.x )
        d = math.cosh( self.x )
        dy_dx = 1.0/(d*d)
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )
            
    #-----------------------------------------------------------------
    def _asinh(self):
        """
        Inverse real hyperbolic sine function
        
        """
        x = float(self.x)
        y = math.asinh( x )
        dy_dx = 1.0/math.sqrt(x**2 + 1)
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _acosh(self):
        """
        Inverse real hyperbolic cosine function
        
        """
        x = float(self.x)
        y = math.acosh( x )
        dy_dx = 1.0/math.sqrt((x - 1)*(x + 1))
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _atanh(self):
        """
        Inverse real hyperbolic tangent function
        
        """
        x = float(self.x)
        y = math.atanh( x )
        dy_dx = 1.0/((1 - x) * (1 + x))
        return UncertainReal(
                self._context
            ,   y
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _magnitude(self):
        """
        Return the magnitude

        The magnitude of an uncertain real number is defined
        everywhere except at the origin. A ``ZeroDivisionError`` is
        raised in that case.

        Returns
        -------
        UncertainReal
        
        """
        x = float(self.x)
        
        if x==0:
            raise ZeroDivisionError(
                  "Cannot take |x| of an uncertain real number when x==0"
            )
        
        dy_dx = math.copysign(1.0,x)

        return UncertainReal(
                self._context
            ,   abs(x)
            ,   scale_vector(self._u_components,dy_dx)
            ,   scale_vector(self._d_components,dy_dx)
            ,   scale_vector(self._i_components,dy_dx)
            )
          
    #-----------------------------------------------------------------
    def _mag_squared(self):
        """
        Returns x**2 
        
        Returns
        -------
        UncertainReal
        
        """
        return self*self
        
    #-----------------------------------------------------------------
    def _phase(self):
        """
        Return the phase

        The phase of an uncertain real number is always 0
        with no uncertainty.
        
        """
        return self._context.constant_real(0,label=None)

#----------------------------------------------------------------
def _atan2_re_re(lhs,rhs): 
    """
    Return the bivariate inverse tan of a pair
    of uncertain real numbers. 
    
    """
    context = lhs._context 
    assert context == rhs._context
    
    x = rhs.x
    y = lhs.x
    
    den = (x**2 + y**2)
    if den == 0.0:
        dz_dx = dz_dy = 0.0
    else:
        dz_dx = -y/den
        dz_dy = x/den
        
    return UncertainReal(
            lhs._context 
        ,   math.atan2(y,x)
        ,   merge_weighted_vectors(lhs._u_components,dz_dy,rhs._u_components,dz_dx)
        ,   merge_weighted_vectors(lhs._d_components,dz_dy,rhs._d_components,dz_dx)
        ,   merge_weighted_vectors(lhs._i_components,dz_dy,rhs._i_components,dz_dx)
        )

def _atan2_x_re(y,rhs):
    x = rhs.x
    den = (x**2 + y**2)
    if den == 0.0:
        dz_dx = 0.0
    else:
        dz_dx = -y/den
    return UncertainReal(
            rhs._context
        ,   math.atan2(y,x)
        ,   scale_vector(rhs._u_components,dz_dx)
        ,   scale_vector(rhs._d_components,dz_dx)
        ,   scale_vector(rhs._i_components,dz_dx)
        )

def _atan2_re_x(lhs,x):
        y = lhs.x
        den = (x**2 + y**2)
        if den == 0.0:
            dz_dy = 0.0
        else:
            dz_dy = x/den
        return UncertainReal(
                lhs._context
            ,   math.atan2(y,x)
            ,   scale_vector(lhs._u_components,dz_dy)
            ,   scale_vector(lhs._d_components,dz_dy)
            ,   scale_vector(lhs._i_components,dz_dy)
            )

#----------------------------------------------------------------------------
def _pow(lhs,rhs):
    """
    Raise the uncertain real number `lhs` to the power of `rhs`
    """
    if isinstance(rhs,UncertainReal):
        
        r = rhs.x
        l = lhs.x

        y = l**r
        dy_dl = r * l**(r-1)
        dy_dr = math.log(abs(l))*y if l != 0 else 0
        
        return UncertainReal(
                lhs._context 
            ,   y
            ,   merge_weighted_vectors(lhs._u_components,dy_dl,rhs._u_components,dy_dr)
            ,   merge_weighted_vectors(lhs._d_components,dy_dl,rhs._d_components,dy_dr)
            ,   merge_weighted_vectors(lhs._i_components,dy_dl,rhs._i_components,dy_dr)
            )
 
    elif isinstance(rhs,numbers.Real): 
        if rhs == 0:
            return 1.0
        elif rhs == 1:
            return lhs
        else:
            # Raise an uncertain real number to the power
            # of a number

            l = lhs.x
            r = rhs 
            
            y = l**r
            dy_dl = r * l**(r-1)
            return UncertainReal(
                    lhs._context
                ,   y
                ,   scale_vector(lhs._u_components,dy_dl)
                ,   scale_vector(lhs._d_components,dy_dl)
                ,   scale_vector(lhs._i_components,dy_dl)
                )
 
#----------------------------------------------------------------------------
def _rpow(lhs,rhs):
    """
    Raise `lhs` to the power of uncertain real number `rhs`
    
    """
    if isinstance(lhs,numbers.Real):    
        l = lhs.x
        r = rhs.x

        y = l**r
        dy_dr = math.log(abs(l))*y if l != 0 else 0
        
        return UncertainReal(
                rhs._context
            ,   y
            ,   scale_vector(rhs._u_components,dy_dr)
            ,   scale_vector(rhs._d_components,dy_dr)
            ,   scale_vector(rhs._i_components,dy_dr)
            )
    elif isinstance(lhs,numbers.Complex):
        return _pow_z_re(lhs,rhs)  
    else:
        raise NotImplemented

#----------------------------------------------------------------------------
def _pow_z_re(lhs,rhs):
    """
    Raise a complex number (lhs) to the power of an uncertain real number (rhs)
    
    """
    c = rhs._context 
    
    l = lhs
    r = rhs.x
    y = l**r
    
    # Sensitivity coefficient
    dy_dr = y * cmath.log(l) if l != 0 else 0
            
    r = UncertainReal(
            c,
            y.real,
            scale_vector(rhs._u_components,dy_dr.real),
            scale_vector(rhs._d_components,dy_dr.real),
            scale_vector(rhs._i_components,dy_dr.real)
        )        
    i = UncertainReal(
            c,
            y.imag,
            scale_vector(rhs._u_components,dy_dr.imag),
            scale_vector(rhs._d_components,dy_dr.imag),
            scale_vector(rhs._i_components,dy_dr.imag)
        )        
    
    return c.uncertain_complex(r,i,None)

#----------------------------------------------------------------------------
def _pow_re_z(lhs,rhs):
    """
    Raise an uncertain real number (lhs) to the power of a complex number (rhs) 
    
    """
    c = lhs._context 
    
    l = lhs.x
    r = rhs
    y = l**r
    
    # Sensitivity coefficient
    dy_dl = r * l**(r-1)

    r = UncertainReal(
            c,
            y.real,
            scale_vector(lhs._u_components,dy_dl.real),
            scale_vector(lhs._d_components,dy_dl.real),
            scale_vector(lhs._i_components,dy_dl.real)
        )        
    i = UncertainReal(
            c,
            y.imag,
            scale_vector(lhs._u_components,dy_dl.imag),
            scale_vector(lhs._d_components,dy_dl.imag),
            scale_vector(lhs._i_components,dy_dl.imag)
        )        
    
    return c.uncertain_complex(r,i,None)
 
#----------------------------------------------------------------------------
def _div(lhs,rhs):
    """
    Divide the uncertain real number `lhs` by `rhs`
    
    """
    if isinstance(rhs,UncertainReal):
    
        r = rhs.x
        l = lhs.x

        y = l/r
        dy_dl = 1.0/r
        dy_dr = -y/r
        
        return UncertainReal(
                lhs._context 
            ,   y
            ,   merge_weighted_vectors(lhs._u_components,dy_dl,rhs._u_components,dy_dr)
            ,   merge_weighted_vectors(lhs._d_components,dy_dl,rhs._d_components,dy_dr)
            ,   merge_weighted_vectors(lhs._i_components,dy_dl,rhs._i_components,dy_dr)
            )
            
    elif isinstance(rhs,numbers.Real):
        if rhs == 1.0:
            return lhs 
        else:
            l = lhs.x

            y = l/rhs
            dy_dl = 1.0/rhs
            
            return UncertainReal(
                    lhs._context
                ,   y
                ,   scale_vector(lhs._u_components,dy_dl)
                ,   scale_vector(lhs._d_components,dy_dl)
                ,   scale_vector(lhs._i_components,dy_dl)
                )
    elif isinstance(rhs,numbers.Complex):
        if rhs == 1.0:
            return lhs
        else:
            return _div_re_z(lhs,rhs)
   
    else:
        raise NotImplemented

#----------------------------------------------------------------------------
def _rdiv(lhs,rhs):
    """
    Divide `lhs` by the uncertain real number `rhs`
    
    """
    if isinstance(lhs,numbers.Real):    
        r = rhs.x

        y = lhs/r
        dy_dr = -y/r
        return UncertainReal(
                rhs._context
            ,   y
            ,   scale_vector(rhs._u_components,dy_dr)
            ,   scale_vector(rhs._d_components,dy_dr)
            ,   scale_vector(rhs._i_components,dy_dr)
            )
    elif isinstance(lhs,numbers.Complex):
        return _div_z_re(lhs,rhs)  
    else:
        raise NotImplemented

#----------------------------------------------------------------------------
def _div_z_re(lhs,rhs):
    """
    Divide a complex number (lhs) by an uncertain real number (rhs)
    
    """
    c = rhs._context
    r = lhs.real / rhs 
    i = lhs.imag / rhs 
    
    return c.uncertain_complex(r,i,None)

def _div_re_z(lhs,rhs):
    """
    Divide an uncertain real number (lhs) and a complex number (rhs) 
    
    """
    c = lhs._context
    
    norm = abs(rhs)**2
    r = lhs * rhs.real/norm
    i = lhs * -rhs.imag/norm
    
    return c.uncertain_complex(r,i,None)

#----------------------------------------------------------------------------
def _mul(lhs,rhs):
    """
    Multiply `rhs` with the uncertain real number `lhs` 
    
    """
    if isinstance(rhs,UncertainReal):
    
        l = lhs.x
        r = rhs.x
        return UncertainReal(
                lhs._context 
            ,   l*r
            ,   merge_weighted_vectors(lhs._u_components,r,rhs._u_components,l)
            ,   merge_weighted_vectors(lhs._d_components,r,rhs._d_components,l)
            ,   merge_weighted_vectors(lhs._i_components,r,rhs._i_components,l)
            )

    elif isinstance(rhs,numbers.Real):
        if rhs == 1.0:
            return lhs
        else:
            return UncertainReal(
                    lhs._context
                ,   float.__mul__(lhs.x,float(rhs))
                ,   scale_vector(lhs._u_components,rhs)
                ,   scale_vector(lhs._d_components,rhs)
                ,   scale_vector(lhs._i_components,rhs)
                )
                
    elif isinstance(rhs,numbers.Complex):
        if rhs == 1.0:
            return lhs
        else:
            return _mul_re_z(lhs,rhs)
   
    else:
        raise NotImplemented

#----------------------------------------------------------------------------
def _rmul(lhs,rhs):
    """
    Multiply `lhs` with the uncertain real number `rhs` 
    
    """
    if isinstance(lhs,numbers.Real):    
        if lhs == 1.0:
            return rhs
        else:
            return UncertainReal(
                    rhs._context
                ,   float.__mul__(float(lhs),rhs.x)
                ,   scale_vector(rhs._u_components,lhs)
                ,   scale_vector(rhs._d_components,lhs)
                ,   scale_vector(rhs._i_components,lhs)
                )
    elif isinstance(lhs,numbers.Complex):
        if lhs == 1.0:
            return rhs
        else:
            return _mul_z_re(lhs,rhs)  
    else:
        raise NotImplemented

#----------------------------------------------------------------------------
def _mul_z_re(lhs,rhs):
    """
    Multiply a complex number (lhs) and an uncertain real number (rhs)
    
    """
    c = rhs._context
    r = lhs.real * rhs 
    i = lhs.imag * rhs 
    
    return c.uncertain_complex(r,i,None)
    
#----------------------------------------------------------------------------
def _mul_re_z(lhs,rhs):
    """
    Multiply a complex number (rhs) and an uncertain real number (lhs)

    """
    c = lhs._context
    r = lhs * rhs.real
    i = lhs * rhs.imag

    return c.uncertain_complex(r,i,None)

#----------------------------------------------------------------------------
def _sub(lhs,rhs):
    """
    Subtract `rhs` from the uncertain real number `lhs` 
    
    """
    if isinstance(rhs,UncertainReal):
        return UncertainReal(
                lhs._context
            ,   lhs.x - rhs.x
            ,   merge_weighted_vectors(lhs._u_components,1.0,rhs._u_components,-1.0)
            ,   merge_weighted_vectors(lhs._d_components,1.0,rhs._d_components,-1.0)
            ,   merge_weighted_vectors(lhs._i_components,1.0,rhs._i_components,-1.0)
            )
    elif isinstance(rhs,numbers.Real):
        if rhs == 0.0:
            return lhs
        else:
            return UncertainReal(
                    lhs._context
                ,   float.__sub__(lhs.x,float(rhs))
                ,   scale_vector(lhs._u_components,1.0)
                ,   scale_vector(lhs._d_components,1.0)
                ,   scale_vector(lhs._i_components,1.0)
                )
    elif isinstance(rhs,numbers.Complex):
        if rhs == 0.0:
            return lhs
        else:
            return _sub_re_z(lhs,rhs)
   
    else:
        raise NotImplementedError()
  
#----------------------------------------------------------------------------
def _rsub(lhs,rhs):  
    """
    Subtract the uncertain real number `rhs` from `lhs`
    
    """
    if isinstance(lhs,numbers.Real):    
        if lhs == 0.0:
            return -rhs
        else:
            return UncertainReal(
                rhs._context
            ,   float.__sub__(float(lhs),rhs.x)
            ,   scale_vector(rhs._u_components,-1.0)
            ,   scale_vector(rhs._d_components,-1.0)
            ,   scale_vector(rhs._i_components,-1.0)
            )
                
    elif isinstance(lhs,numbers.Complex):
        if lhs == 0.0:
            return -rhs
        else:
            return _sub_z_re(lhs,rhs)  
    else:
        raise NotImplementedError()

#----------------------------------------------------------------------------
def _sub_z_re(lhs,rhs):
    """
    Subtract an uncertain real number `rhs` from a complex `lhs`
    """
    c = rhs._context
    r = lhs.real - rhs 
    i = c.constant_real( lhs.imag, None )
    
    return c.uncertain_complex(r,i,None)
 
#----------------------------------------------------------------------------
def _sub_re_z(lhs,rhs):
    """
    Subtract a complex `rhs` from an uncertain real number `lhs` 
    """
    c = lhs._context
    r = lhs - rhs.real
    i = c.constant_real( -rhs.imag, None )
    
    return c.uncertain_complex(r,i,None)
 
#----------------------------------------------------------------------------
def _add(lhs,rhs):
    """
    Add the uncertain real number `lhs` to `rhs`
    
    """
    if isinstance(rhs,UncertainReal):
            
        return UncertainReal(
                lhs._context
            ,   lhs.x + rhs.x
            ,   merge_vectors(lhs._u_components,rhs._u_components)
            ,   merge_vectors(lhs._d_components,rhs._d_components)
            ,   merge_vectors(lhs._i_components,rhs._i_components)
            )
            
    elif isinstance(rhs,numbers.Real):
        if rhs == 0.0:
            return lhs
        else:
            return UncertainReal(
                    lhs._context
                ,   lhs.x + rhs
                ,   scale_vector(lhs._u_components,1.0)
                ,   scale_vector(lhs._d_components,1.0)
                ,   scale_vector(lhs._i_components,1.0)
                )
    elif isinstance(rhs,numbers.Complex):
        if rhs == 0.0:
            return lhs
        else:
            return _add_re_z(lhs,rhs)
   
    else:
        raise NotImplemented
#----------------------------------------------------------------------------
def _radd(lhs,rhs):
    """
    Add `lhs` to the uncertain real number `rhs` 
    
    """
    if isinstance(lhs,numbers.Real):    
        if lhs == 0.0:
            return rhs
        else:
            return UncertainReal(
                    rhs._context
                ,   float.__add__(float(lhs),rhs.x)
                ,   scale_vector(rhs._u_components,1.0)
                ,   scale_vector(rhs._d_components,1.0)
                ,   scale_vector(rhs._i_components,1.0)
                )
                
    elif isinstance(lhs,numbers.Complex):
        if lhs == 0.0:
            return rhs
        else:
            return _add_z_re(lhs,rhs)  
    else:
        raise NotImplemented

#----------------------------------------------------------------------------
def _add_re_z(lhs,rhs):
    """
    Add a complex number `rhs` to an uncertain real number `lhs`
    
    """
    c = lhs._context
    r = lhs + rhs.real 
    i = c.constant_real( rhs.imag, None )
    
    return c.uncertain_complex(r,i,None)

#----------------------------------------------------------------------------
def _add_z_re(lhs,rhs):
    """
    Add a complex number `lhs` to an uncertain real number `rhs`
    
    """
    c = rhs._context
    r = lhs.real + rhs
    i = c.constant_real( lhs.imag, None )
    
    return c.uncertain_complex(r,i,None)

#----------------------------------------------------------------------------
def set_correlation_real(x1,x2,r):
    """
    Assign a correlation coefficient between ``x1`` and ``x2``

    Parameters
    ----------
    x1, x2 : UncertainReal
    r: float
    
    """
    x1._context.set_correlation(x1,x2,r)
        
#----------------------------------------------------------------------
def get_correlation_real(x1,x2):
    """Return the correlation between ``x1`` and `x2``
    
    A correlation coefficient may be calculated between a pair 
    of uncertain real numbers ``x1`` and `x2``, which need not
    be elementary.
    
    Parameters
    ----------
    x1, x2 : UncertainReal
    
    Returns
    -------
    float
    
    """
    if x1.is_elementary and x2.is_elementary:
        context = x1._context
        assert context is x2._context,"Different contexts!"
        
        return context.get_correlation(x1,x2)
    else:
        v1 = std_variance_real(x1)
        v2 = std_variance_real(x2)
        
        num = std_covariance_real(x1,x2)
        den =  math.sqrt( v1*v2 )
        
        return num / den if num != 0.0 else 0.0
    
#----------------------------------------------------------------------------
def std_variance_real(x):
    """
    Return the standard variance

    Parameter
    ---------
    x : UncertainReal

    Returns
    -------
    float
    
    """
    c = x._context    
    
    # The independent components of uncertainty
    # ( Faster this way than with reduce()! )
    var = 0.0
    for u_i in x._u_components.itervalues():
        var += u_i * u_i
    
    # Only evaluate the following terms when correlations 
    # have been declared, which is rare.
    if len(x._d_components) != 0:
        # All correlations 
        c_mat = c._correlations._mat  
        
        # Components declared as perhaps being correlated 
        cpts = x._d_components
        keys = cpts.keys()
        values = cpts.values()
        
        for i,k_i in enumerate(keys):
            u_i = values[i]
            var += u_i * u_i 
            try:
                row_i = c_mat[k_i]   
            except KeyError:
                # No entry, next key
                continue
                
            for j,k_j in enumerate( keys[i+1:] ):
                var += 2.0*u_i*row_i.get(k_j,0.0)*values[j+i+1]
                            
    return var

#----------------------------------------------------------------------
def std_covariance_real(x1,x2):
    """
    Return the covariance between 'x1' and 'x2'

    Parameter
    ---------
    x1, x2 : UncertainReal

    Returns
    -------
    float
        
    """
    # Assume that `x1` and `x2` are not elementary UNs 
    
    context = x1._context
    
    cv = 0.0

    # Keys are references to Leaf objects
    
    for k1_i,u1_i in x1._u_components.iteritems():
        # `k1` is not correlated with anything, so only if 
        # `k1` happens to influence x2 do we get any contribution.
        cv += u1_i*x2._u_components.get(k1_i,0.0)
        
    for k1_i,u1_i in x1._d_components.iteritems():
        # `k1` could be correlated with `k2`
        try:
            row_i = context._correlations._mat[k1_i]
            # Some correlations declared, so check. 
            for k2_i,u2_i in x2._d_components.iteritems():
                cv += u1_i*row_i.get(k2_i,0.0)*u2_i
        except KeyError:
            # `k1_i` has no declared correlations
            cv += u1_i*x2._d_components.get(k1_i,0.0)

    return cv

#----------------------------------------------------------------------
def get_covariance_real(x1,x2):
    """Return the covariance between ``x1`` and `x2``
    
    Covariance may be calculated between a pair 
    of uncertain real numbers ``x1`` and `x2``, 
    which need not be elementary.
    
    Returns
    -------
    float
    
    """
    if x1.is_elementary and x2.is_elementary:
        context = x1._context
        assert context is x2._context,"Different contexts!"
        
        r = context.get_correlation(x1,x2)
        return x1.u * r * x2.u
    else:        
        return std_covariance_real(x1,x2) 
        
#----------------------------------------------------------------------------
def welch_satterthwaite(x):
    """Return the variance and degrees-of-freedom.

    Uses the Welch Satterthwaite calculation of dof
    for an uncertain real number ``x``
    
    Parameters
    ----------
    x : UncertainReal

    Returns
    -------
    The variance and degrees-of-freedom
    
    """    
    if not isinstance(x,UncertainReal):
        raise RuntimeError(
            "UncertainReal required, got: '{!r}'".format(x)
        )
    
    c = x._context

    if x.is_elementary:
        return VarianceAndDof(x.v,x.df)
     
    elif len(x._u_components) == 0 and len(x._d_components) == 0:
        # This is a constant_real
        return VarianceAndDof(0.0,inf)
    else:      
        u_cpts = x._u_components
        u_keys = u_cpts.keys()      # Leaf objects
        u_values = u_cpts.values()  # uncertainties
        u_dof = [ k_i.df for k_i in u_keys ]

        d_cpts = x._d_components
        d_keys = d_cpts.keys()      # Leaf objects
        d_values = d_cpts.values()  # uncertainties
        d_dof = [ k_i.df for k_i in d_keys ]

        # If everything has infinite DoF we don't need to use WS
        degrees_of_freedom = u_dof + d_dof
        if degrees_of_freedom.count(inf) == len(degrees_of_freedom):
            return VarianceAndDof(x.v,inf)
         
        #----------------------------------------------------------
        var = 0.0                       # the standard variance
        cpts_lst = []                   # the variance of components  
        dof_lst = []
 
        # Independent components are un-correlated.
        # So, no worries about ensembles   
        for i,u_i in enumerate(u_values):
                                    
            v_i = u_i * u_i
            var += v_i
            df_i = u_dof[i] 

            cpts_lst.append(v_i)
            dof_lst.append(df_i)

        #-----------------------------------------        
        # The restriction on correlations between influences
        # is relaxed for complex quantities. 
        # We follow the method of 'Result 2' in
        # Willink, Metrologia 44 (2007) 340-349. Note,
        # however, that this would not be valid if the
        # correlation was the result of a functional
        # combination of other influences that
        # have finite degrees of freedom (see
        # Willink, Metrologia 45 (2008) 63-67.
        #
        # The restriction on correlations between influences
        # is relaxed for ensembles of real quantities that are 
        # considered to sampled from a multivariate distribution. 
        # They are treated as a single component when evaluating WS, 
        # in keeping with the Willink reference above.
        #
        # NB, we may not need to identify complex numbers per se, 
        # because  they could be handled in the same way as uncertain  
        # numbers in an ensemble. However, this might be slightly slower.
        
        # Control value that may get changed to NaN below        
        df = 0.0
                
        # This is used to accumulate the sums of variances
        # of all components in the ensembles.
        cpts_map = {}   

        # flag for handling the ensemble of a complex number
        finish_complex = False  
        
        if len(d_keys):
            # The last value of the sequence is treated at the bottom
            for i,k_i in enumerate(d_keys[:-1]):
                                        
                u_i = d_values[i]
                v_i = u_i * u_i
                var += v_i
                df_i = d_dof[i]
                                    
                # Control of complex. The `_complex_ids` register has a pair of
                # ids for the complex quantity and is indexed by either of them. 
                if k_i in c._complex_ids:
                    complex_id = c._complex_ids[k_i]
                else:
                    complex_id = (None,None)

                # Context._ensemble is a WeakKeyDictionary of WeakSets
                # Need to freeze this set here, to use it as a dict key
                # that identifies the ensemble.
                ensemble_i = frozenset( c._ensemble.get(k_i,frozenset()) )
                
                if len(ensemble_i) !=0 and ensemble_i not in cpts_map:
                    # Create a new component entry for this new ensemble
                    cpts_map[ensemble_i] = [0,df_i]

                if ensemble_i in cpts_map:
                    # Update the total variance of this ensemble
                    
                    # assert cpts_map[ensemble_i][1] == df_i
                    cpts_map[ensemble_i][0] += v_i
                    
                # It may be possible to treat 
                # complex as an ensemble. In this routine the
                # processing would be the same. However, check
                # the WH routine, because it may have different
                # requirements
                elif finish_complex:
                    # In this case, we are continuing to evaluate 
                    # a single component for the WS calculation,
                    # because this is the imaginary component of
                    # a complex number.
                    v_old =  cpts_lst[-1]
                    cpts_lst[-1] = v_old + v_i
                
                else:   
                    # Update the list of variance components for a single influence
                    cpts_lst.append(v_i)
                    dof_lst.append(df_i)

                # Set flag only when both complex components are identified,
                # because it is possible that just one or other component may
                # be given as the argument `x`.
                if (k_i,d_keys[i+1]) != complex_id:
                    finish_complex = False
                else:
                    # i.e., we need to process the next component
                    finish_complex =  True
                    
                # ---------------------------------------------------------
                nid_i = k_i
                if k_i in c._correlations._mat:
                    # Correlations associated with the influence `k_i`
                    row_i = c._correlations._mat[k_i]
                    
                    nu_i_infinite = is_infinity( df_i ) 
                    
                    # Look at the remaining influences 
                    for j,k_j in enumerate(d_keys[i+1:]):
                    
                        if k_j in row_i:  
                            # Special cases are handled here.
                            
                            u_j = d_values[i+1+j]
                            r = row_i[k_j]
                            covar_ij = 2.0*u_i*r*u_j
                            var += covar_ij    

                            if nu_i_infinite and is_infinity( k_j.df ):
                                # The correlated influences both have  
                                # infinite dof so it is still valid to use WS.
                                # For a single entry, we add 
                                # the covariance term here.
                                # The loop over `i` takes care of
                                # the variance.
                                # Since infinite DoF are not summed
                                # in WS, we do not need to modify `cpts_lst`
                                #
                                continue
                                            
                            # An ensemble of real quantities may be correlated.
                            # This is also true for the real and imaginary  
                            # components of a complex quantity.
                            elif k_j in ensemble_i:
                                cpts_map[ensemble_i][0] += covar_ij
                                continue
                                    
                            # The real and imaginary components of a
                            # complex quantity may be correlated.
                            # This section of code is executed when the
                            # imaginary component associated with id_i
                            # (a real component) is encountered. It puts
                            # the covariance term into cpts_lst. The
                            # variance associated with `id_j` will be
                            # put in when the outer loop next increments.
                            # I.e., here we are only worried about off-diagonal
                            # components of the covariance matrix.
                            #
                            elif (k_i,k_j) == complex_id:
                                cpts_lst[-1] += covar_ij
                                continue

                            else:
                                # Correlation with no excuse, illegal!
                                df = nan
                                # Don't expect to ever get here now, because of 
                                # controls on using set_correlation
                                assert False 
                                continue

            # Last value cannot be correlated with anything not already processed,
            # but it might be the final component of an ensemble 
            k_i = d_keys[-1]
            u_i = d_values[-1]
            v_i = u_i * u_i
            df_i = d_dof[-1]
            var += v_i
                
            ensemble_i = frozenset( c._ensemble.get(k_i,frozenset()) )

            # It may be part of an ensemble 
            if ensemble_i in cpts_map:
                cpts_map[ensemble_i][0] += v_i
            elif finish_complex:
                v_old =  cpts_lst[-1]
                cpts_lst[-1] = v_old + v_i
            else:
                cpts_lst.append(v_i)
                dof_lst.append(df_i)
        
        # Finish building cpts_lst and dof_lst
        # by using the values accumulated in `cpts_map`
        for v_i,df_i in cpts_map.itervalues():
            cpts_lst.append(v_i)
            dof_lst.append(df_i)   
            
        # There is a pathological case in which var == 0.
        # It can occur in a product of zero-valued uncertain factors.
        if var == 0: df = nan
                
        #--------------------------------------------------------------------        
        if is_undefined(df):
            return VarianceAndDof(var,nan)
        else:
            # Final calculation of WS 
            den = 0.0
            for v_i,dof_i in izip(cpts_lst,dof_lst):
                if not is_infinity(dof_i):
                    u2 = v_i / var
                    den += u2 * u2 / dof_i
                   
            try:
                return VarianceAndDof(var,1.0/den)
            except ZeroDivisionError:
                return VarianceAndDof(var,inf)
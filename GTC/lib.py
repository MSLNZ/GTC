"""
Defines :class:`UncertainReal` and implements the mathematical 
operations on this class of objects.

"""
from __future__ import division

import math
import cmath
import numbers
try:
    from itertools import izip  # Python 2
    PY2 = True
except ImportError:
    izip = zip
    long = int
    PY2 = False

from GTC import nodes
from GTC import vector 
from GTC import context

from GTC.named_tuples import (
    VarianceCovariance, 
    VarianceAndDof, 
    StandardUncertainty,
    GroomedUncertainReal,
    GroomedUncertainComplex,
)

from GTC import (
    inf, 
    nan, 
    inf_dof, 
)

LOG10_E = math.log10(math.e)

#----------------------------------------------------------------------------
def _is_uncertain_real_constant(x):
    if isinstance(x,UncertainReal):
        return bool( 
            len(x._u_components) == 0 and 
            len(x._d_components) == 0
        )
    else:
        raise TypeError(
            "UncertainReal required: {!r}".format(x)
        )

#----------------------------------------------------------------------------
def _is_uncertain_complex_constant(z):
    if isinstance(z,UncertainComplex):
        return bool( 
            _is_uncertain_real_constant(z.real) and 
            _is_uncertain_real_constant(z.imag)
        )
    else:
        raise TypeError(
            "UncertainComplex required: {!r}".format(z)
        )
          
#----------------------------------------------------------------------------
class UncertainReal(object):
    
    """
    An :class:`UncertainReal` holds information about the measured 
    value of a real-valued quantity
    
    """
    
    __slots__ = [
        '_x'                    # The estimate
    ,   '_u'                    # The standard uncertainty in the estimate
    ,   '_u_components'         # weighted Jacobian matrix  
    ,   '_d_components'         # weighted Jacobian matrix 
    ,   '_i_components'         # Intermediate components of uncertainty
    ,   'is_elementary'         
    ,   'is_intermediate'       
    ,   '_node'                 # May refer to a node
    ]            

    #-------------------------------------------------------------------------
    def __init__(self,x,u_comp,d_comp,i_comp,node=None):

        self._x = float(x)
        self._u_components = u_comp
        self._d_components = d_comp
        self._i_components = i_comp
        self._node = node
 
        if node is None:
            self.is_elementary = False 
            self.is_intermediate = False
            
        else: 
            # Constants have Leaf nodes, but the UID is None;
            # they will not be classed as `elementary`
            self.is_elementary = (
                isinstance(self._node,nodes.Leaf)
                    and not self._node.uid is None
            )
            # An intermediate uncertain number has a ``Node`` object  
            self.is_intermediate = type(self._node) is nodes.Node
            
            assert not(self.is_elementary and self.is_intermediate)

    #----------------------------------------------------------------------------
    @classmethod
    def _constant(cls,x,label=None):
        """
        Return a constant uncertain real number with value ``x`` 
        
        A constant uncertain real number has no uncertainty
        and infinite degrees of freedom.        
        
        Parameters
        ----------
        x : float
        label : str or None
        rtype : UncertainReal
            
        """
        # A constant will not be archived,
        # so it has no need for a UID. 
        return UncertainReal(
                x
            ,   vector.Vector( )
            ,   vector.Vector( )
            ,   vector.Vector( )
            ,   nodes.Leaf(uid=None,label=label,u=0.0,df=inf)
        )
        
    #------------------------------------------------------------------------
    @classmethod
    def _elementary(cls,x,u,df,label,independent):
        """
        Return an elementary uncertain real number.

        The uncertain number will have a value ``x``, standard
        uncertainty ``u`` and degrees of freedom ``df``.

        A ``ValueError`` is raised if the value of 
        `u` is less than zero or the value of `df` is less than 1.

        The ``independent`` argument controls whether this
        uncertain number may be correlated with others.
        
        Parameters
        ----------
        x : float
        u : float
        df : float
        label : str, or None
        independent : bool

        Returns
        -------
        :class:`UncertainReal`
        
        """
        if df < 1:
            raise ValueError(
                "invalid degrees of freedom: {!r}".format(df) 
            )
        if u < 0:
            # u == 0 can occur in complex UNs.
            raise ValueError(
                "invalid uncertainty: {!r}".format(u)
            )
                    
        uid = context._context._next_elementary_id()
        ln = context._context.new_leaf(uid,label,u,df,independent=independent)
        
        if independent:
            return UncertainReal(
                    x
                ,   vector.Vector( index=[ln],value=[u] )
                ,   vector.Vector( )
                ,   vector.Vector( )
                ,   ln
                )
        else:
            return UncertainReal(
                    x
                ,   vector.Vector( )
                ,   vector.Vector( index=[ln],value=[u] )
                ,   vector.Vector( )
                ,   ln
                )
    #------------------------------------------------------------------------
    @classmethod
    def _intermediate(cls,un,label):
        """
        Create an intermediate uncertain number
        
        To investigate the sensitivity of subsequent results,
        an intermediate UN must be declared.
        
        Parameters
        ----------
        :arg un: :class:`UncertainReal`
        :arg label: str
        
        """
        if not un.is_elementary:
            if not un.is_intermediate:                     
                # A new registration 
                uid = context._context._next_intermediate_id()
                
                u = un.u
                un._node = context._context.new_node(uid,label,u)

                # Seed the Vector of intermediate components 
                # with this new Node object, so that uncertainty 
                # will be propagated.
                un._i_components = vector.merge_vectors(
                    un._i_components,
                    vector.Vector( index=[un._node], value=[u] )
                )
                un.is_intermediate = True
                            
            # else:
                # Assume that it has been registered, perhaps the 
                # user has repeated the registration process.

        # else:
            # There should be no harm in ignoring elementary UNs.
            # They will be archived properly and they are not dependent
            # on anything. It is convenient for the user not to worry
            # whether or not something is elementary or not. 
            
    #------------------------------------------------------------------------
    @classmethod
    def _archived_elementary(cls,uid,x):
        """
        Restore an uncertain number that has been archived. 

        Parameters
        ----------
        uid : unique identifier
        x : float

        Returns
        -------
        UncertainReal
        
        """
        # Use the context cache `_registered_leaf_nodes` 
        # to avoid creating multiple Leaf objects.
        l = context._context._registered_leaf_nodes[uid]

        # The Leaf object is used to seed one 
        # Vector component, so that uncertainty 
        # will be propagated
        if l.independent:
            un = UncertainReal(
                    x
                ,   vector.Vector( index=[l],value=[l.u] )
                ,   vector.Vector( )
                ,   vector.Vector( )
                ,   l
                )
        else:
            un = UncertainReal(
                    x
                ,   vector.Vector( )
                ,   vector.Vector( index=[l],value=[l.u] )
                ,   vector.Vector( )
                ,   l
                )
        
        return un  
        
    #-------------------------------------------------------------------------
    def _round(self,digits,df_decimals):
        """
        Return a ``GroomedUncertainReal`` 
        
        `digits` specifies the number of significant digits 
        in the uncertainty that will be retained. The value will use 
        the same precision. The degrees-of-freedom will be  
        represented using `df_decimals` decimal places.

        `df_decimals` specifies the number of decimal places 
        reported for the degrees-of-freedom.
        
        Degrees-of-freedom are greater than 1E6 are set to `inf`.
        
        """
        if (
            math.isnan(self.x) or math.isinf(self.x) or 
            math.isnan(self.u) or math.isinf(self.u)
        ):
            return GroomedUncertainReal(
                x = self.x,
                u = self.u,
                df = self.df,
                label = self.label,  
                precision = digits,
                df_decimals = df_decimals,
                u_digits = '({})'.format(self.u)
            )
            
        elif self.u != 0:
            log10_u = math.log10( self.u )
            if log10_u.is_integer(): log10_u += 1 
            
            # The least power of 10 above the value of `u`
            exponent = math.ceil( log10_u ) 
            
            # In fixed-point, precision is the number of decimal places. 
            decimal_places = 0 if exponent-digits >= 0 else int(digits-exponent)
        
            factor = 10**(exponent-digits)
            
            x = factor*round(self.x/factor)
            u = factor*round(self.u/factor)
            
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
                u_digits = "{1:.{0}f}".format(decimal_places,u)
            else:
                u_digits = "{:.0f}".format(self.u/factor)

            df = self.df
            if not math.isnan(df):
                if df > inf_dof:
                    df = inf
                else:
                    df_factor = 10 ** (-df_decimals)
                    df = df_factor * math.floor(df / df_factor)

            return GroomedUncertainReal(
                x = x,
                u = u,
                df = df,
                label = self.label,
                precision = decimal_places,
                df_decimals = df_decimals,
                u_digits = "({})".format(u_digits)
            )
            
        elif _is_uncertain_real_constant(self):
            # Use default fixed-point precision
            return GroomedUncertainReal(
                x = self.x,
                u = 0,
                df = inf,
                label = self.label,
                precision = 6,
                df_decimals = 0,
                u_digits = ""
            )  
        else:
            assert False, "unexpected"
        
    #------------------------------------------------------------------------
    def __repr__(self):
        x = self.x
        u = self.u
        df = self.df
        
        if not math.isnan(df) and df > inf_dof:
            df = inf 
        
        if self.label is None:
            s = "ureal({!r},{!r},{!r})".format( 
                x,u,df
            )            
        else:
            s = "ureal({!r},{!r},{!r}, label={!r})".format( 
                x,u,df,self.label
            )                  
        
        return s

    #------------------------------------------------------------------------
    def __str__(self):
        # Use 2-digit fixed-point format for the uncertainty          
        gself = self._round(2,0)
        return "{1.x:.{0}f}{1.u_digits}".format( gself.precision, gself )
             
    #------------------------------------------------------------------------
    # 
    def __abs__(self):
        return abs(self._x)

    #------------------------------------------------------------------------
    @property
    def real(self):
        """Return the real component 
        """
        return self  
    
    #------------------------------------------------------------------------
    @property
    def imag(self):
        """Returns the imaginary component 
        
        """
        # Returning an UN constant ensures that an algorithm
        # expecting an uncertain number will not break
        return UncertainReal._constant(0.0)
 
    #------------------------------------------------------------------------
    def conjugate(self):
        """Return the complex conjugate

        :rtype: :class:`~lib.UncertainReal`
        
        """
        return self.real
 
    #------------------------------------------------------------------------
    # Comparisons are made with the value
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
        
    def __bool__(self):
        # Used to coerce to Boolean 
        return UncertainReal.__ne__(self, 0.0)
    __nonzero__ = __bool__

    #------------------------------------------------------------------------
    @property
    def x(self):
        """Return the value

        :rtype: float
        
        Note that ``un.x`` is equivalent to :func:`value(un)<core.value>`
        
        **Example**::
            >>> ur = ureal(2.5,0.5)
            >>> ur.x
            2.5
            
        """
        return self._x

    #------------------------------------------------------------------------
    @property
    def u(self):
        """Return the standard uncertainty 
        
        :rtype: float
        
        Note that ``un.u`` is equivalent to :func:`uncertainty(un)<core.uncertainty>`
        
        **Example**::

            >>> ur = ureal(2.5,0.5)
            >>> ur.u
            0.5
            
        """
        if hasattr(self,'_u'):
            return self._u
        else:
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
        
        :rtype: float
        
        Note that ``un.v`` is equivalent to :func:`variance(un)<core.variance>`
        
        **Example**::
            >>> ur = ureal(2.5,0.5)
            >>> ur.v
            0.25
            
        """
        if hasattr(self,'_u'):
            u = self.u 
            return u*u
        else:
            if self.is_elementary or self.is_intermediate:
                u = self._u = self._node.u
                return u*u
            else:
                v = std_variance_real(self)
                self._u = math.sqrt( v )
                return v

    #------------------------------------------------------------------------
    @property
    def df(self):
        """Return the degrees of freedom
        
        :rtype: float

        Note ``un.df`` is equivalent to :func:`dof(un)<core.dof>`
        
        **Example**::
            >>> ur = ureal(2.5,0.5,3)
            >>> ur.df
            3.0        
            
        """
        if self.is_elementary:
            return self._node.df
        else:
            v_df = welch_satterthwaite(self)
            if not hasattr(self,"_u"):
                self._u = math.sqrt( v_df.cv )
                
        return v_df.df 

    #-----------------------------------------------------------------
    @property
    def label(self):
        """The uncertain-number label
        
        :rtype: str

        Note ``un.label`` is equivalent to :func:`label(un)<core.label>`
        
        **Example**::
            >>> x = ureal(2.5,0.5,label='x')
            >>> x.label
            'x'
        
            >>> label(x)
            'x'
            
        """
        try:
            # Elementary, constant and intermediate UNs
            return self._node.label
        except AttributeError:
            return None          

    #------------------------------------------------------------------------
    # Arithmetic operations
    def __neg__(self):
        """
        Unary negative operator
        
        """
        return UncertainReal(
                -self.x
            ,   vector.scale_vector(self._u_components,-1.0)
            ,   vector.scale_vector(self._d_components,-1.0)
            ,   vector.scale_vector(self._i_components,-1.0)
            )
            
    #------------------------------------------------------------------------
    def __pos__(self):
        """
        Unary positive operator

        """     
        # This is a copy but not a clone,
        # because if ``self`` had a node this 
        # object does not have it too.
        return UncertainReal(
                self.x
            ,   vector.Vector(copy=self._u_components)
            ,   vector.Vector(copy=self._d_components)
            ,   vector.Vector(copy=self._i_components)
            )
    
    #------------------------------------------------------------------------
    def __add__(self,rhs):
        if isinstance( rhs,(UncertainReal,numbers.Complex) ):
            return _add(self,rhs)
        else:
            return NotImplemented
        
    def __radd__(self,lhs):
        if isinstance(lhs,numbers.Complex):
            return _radd(lhs,self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __sub__(self,rhs):
        if isinstance(rhs,(UncertainReal,numbers.Complex)):
            return _sub(self,rhs)
        else:
            return NotImplemented
        
    def __rsub__(self,lhs):
        if isinstance(lhs,numbers.Complex):
            return _rsub(lhs,self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __mul__(self,rhs):
        if isinstance(rhs,(UncertainReal,numbers.Complex)):
            return _mul(self,rhs)
        else:
            return NotImplemented
        
    def __rmul__(self,lhs):
        if isinstance(lhs,numbers.Complex):
            return _rmul(lhs,self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __truediv__(self,rhs):
        return self.__div__(rhs)
        
    def __div__(self,rhs):
        if isinstance(rhs,(UncertainReal,numbers.Complex)):
            return _div(self,rhs)
        else:
            return NotImplemented

    def __rtruediv__(self,lhs):
        return self.__rdiv__(lhs)
        
    def __rdiv__(self,lhs):
        if isinstance(lhs,numbers.Complex):
            return _rdiv(lhs,self)
        else:
            return NotImplemented

    #------------------------------------------------------------------------
    def __pow__(self,rhs):
        if isinstance(rhs,(UncertainReal,numbers.Complex)):
            return _pow(self,rhs)
        else:
            return NotImplemented

    def __rpow__(self,lhs):
        if isinstance(lhs,numbers.Complex):
            return _rpow(lhs,self)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
        )
    
    #------------------------------------------------------------------------
    def _sqrt(self):
        """
        Real square root function
        
        """
        y = math.sqrt( self.x )
        dy_dx = 0.5/y
        return UncertainReal(
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
        )
        
    #------------------------------------------------------------------------
    def _sin(self):
        """
        Real sine function
        
        """
        y = math.sin( self.x )
        dy_dx = math.cos( self.x )
        
        return UncertainReal(
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
            )
    #------------------------------------------------------------------------
    def _cos(self):
        """
        Real cosine function
        
        """
        y = math.cos( self.x )
        dy_dx = -math.sin( self.x )
        return UncertainReal(
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _atan2(self,rhs):
        """
        Two-argument inverse real tangent function
        
        """
        if isinstance(rhs,UncertainReal):
            return _atan2_re_re(self,rhs)
        elif isinstance(rhs,numbers.Real):
            return _atan2_re_x(self,float(rhs))
        elif isinstance(rhs,numbers.Complex):
            raise TypeError('atan2 is undefined with a complex argument')
        else:
            return NotImplemented

    def _ratan2(self,lhs):
        if isinstance(lhs,numbers.Real):
            return _atan2_x_re(float(lhs),self)
        elif isinstance(lhs,numbers.Complex):
            raise TypeError('atan2 is undefined with a complex argument')
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _cosh(self):
        """
        Real hyperbolic cosine function
        
        """
        y = math.cosh( self.x )
        dy_dx = math.sinh( self.x )
        return UncertainReal(
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
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
                y
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
            )

    #-----------------------------------------------------------------
    def _magnitude(self):
        """
        Magnitude function

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
                abs(x)
            ,   vector.scale_vector(self._u_components,dy_dx)
            ,   vector.scale_vector(self._d_components,dy_dx)
            ,   vector.scale_vector(self._i_components,dy_dx)
            )
          
    #-----------------------------------------------------------------
    def _mag_squared(self):
        """
        Return x**2 
        
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
        return UncertainReal._constant(0.0)

#----------------------------------------------------------------------------
def _atan2_re_re(lhs,rhs): 
    """
    Return the bivariate arctan of a pair
    of uncertain real numbers. 
    
    """
    x = rhs.x
    y = lhs.x
    
    den = (x**2 + y**2)
    if den == 0.0:
        dz_dx = dz_dy = 0.0
    else:
        dz_dx = -y/den
        dz_dy = x/den
        
    return UncertainReal(
            math.atan2(y,x)
        ,   vector.merge_weighted_vectors(lhs._u_components,dz_dy,rhs._u_components,dz_dx)
        ,   vector.merge_weighted_vectors(lhs._d_components,dz_dy,rhs._d_components,dz_dx)
        ,   vector.merge_weighted_vectors(lhs._i_components,dz_dy,rhs._i_components,dz_dx)
        )

#----------------------------------------------------------------------------
def _atan2_x_re(y,rhs):
    x = rhs.x
    den = (x**2 + y**2)
    if den == 0.0:
        dz_dx = 0.0
    else:
        dz_dx = -y/den
    return UncertainReal(
            math.atan2(y,x)
        ,   vector.scale_vector(rhs._u_components,dz_dx)
        ,   vector.scale_vector(rhs._d_components,dz_dx)
        ,   vector.scale_vector(rhs._i_components,dz_dx)
        )

#----------------------------------------------------------------------------
def _atan2_re_x(lhs,x):
        y = lhs.x
        den = (x**2 + y**2)
        if den == 0.0:
            dz_dy = 0.0
        else:
            dz_dy = x/den
        return UncertainReal(
                math.atan2(y,x)
            ,   vector.scale_vector(lhs._u_components,dz_dy)
            ,   vector.scale_vector(lhs._d_components,dz_dy)
            ,   vector.scale_vector(lhs._i_components,dz_dy)
            )

#----------------------------------------------------------------------------
def _pow(lhs,rhs):
    """
    Raise uncertain real `lhs` to the power of `rhs`
    `lhs` is guaranteed UncertainReal, `rhs` is a number 
    or an UncertainReal
    
    """
    if isinstance(rhs,UncertainReal):
        
        r = rhs.x
        l = lhs.x

        try:
            y = l**r
        except ValueError:
            # py 2.7 does not handle fractional powers of negative numbers 
            # but py3 does by returning a complex. 
            # We patch the py2 case by casting `lhs` to a ucomplex
            return (lhs + 0j)**rhs 

        if isinstance(y,numbers.Real):
            
            dy_dl = r * l**(r-1)
            dy_dr = math.log(abs(l))*y if l != 0 else 0
            
            return UncertainReal(
                    y
                ,   vector.merge_weighted_vectors(lhs._u_components,dy_dl,rhs._u_components,dy_dr)
                ,   vector.merge_weighted_vectors(lhs._d_components,dy_dl,rhs._d_components,dy_dr)
                ,   vector.merge_weighted_vectors(lhs._i_components,dy_dl,rhs._i_components,dy_dr)
                )
        elif isinstance(y,numbers.Complex):
            # If `y` is complex, do this as a ucomplex problem 
            # This is only possible in py3
            return (lhs + 0j)**rhs
        else:
            assert False,'unexpected'
 
    elif isinstance(rhs,numbers.Real): 
        if rhs == 0:
            return 1.0
        elif rhs == 1:
            return lhs
        else:
            l = lhs.x
            r = rhs 
            
            try:
                y = l**r
            except ValueError:
                # py 2.7 does not handle fractional powers of negative numbers 
                # but py3 does by returning a complex. 
                # We patch the py2 case by casting `lhs` to a ucomplex
                return (lhs + 0j)**rhs 
                
            if isinstance(y,numbers.Real):
                dy_dl = r * l**(r-1)
                return UncertainReal(
                        y
                    ,   vector.scale_vector(lhs._u_components,dy_dl)
                    ,   vector.scale_vector(lhs._d_components,dy_dl)
                    ,   vector.scale_vector(lhs._i_components,dy_dl)
                    )
            elif isinstance(y,numbers.Complex):
                # If `y` is complex, do this as a ucomplex problem 
                # This is only possible in py3
                return (lhs + 0j)**rhs
            else:
                assert False,'unexpected'
 
    elif isinstance(rhs,numbers.Complex): 
        return (lhs + 0j)**rhs

    else:
        assert False, 'unexpected'


#----------------------------------------------------------------------------
def _rpow(lhs,rhs):
    """
    Raise `lhs` to the power of uncertain real `rhs`
    
    """
    # Called from __rpow__, so we know that `rhs` is UncertainReal 
    # `lhs` will be either float or complex
    if isinstance(lhs,numbers.Real):    
        l = lhs
        r = rhs.x

        try:
            y = l**r
        except ValueError:
            # py 2.7 does not handle fractional powers of negative numbers 
            # but py3 does by returning a complex. 
            # We patch the py2 case by casting `lhs` to a ucomplex
            return lhs**(rhs + 0j) 
                
        if isinstance(y,numbers.Real):
            dy_dr = math.log(abs(l))*y if l != 0 else 0
            
            return UncertainReal(
                    y
                ,   vector.scale_vector(rhs._u_components,dy_dr)
                ,   vector.scale_vector(rhs._d_components,dy_dr)
                ,   vector.scale_vector(rhs._i_components,dy_dr)
                )
        elif isinstance(y,numbers.Complex):
            # If `y` is complex, do this as a ucomplex problem 
            return lhs**(rhs + 0j)
        else:
            assert False,'unexpected'     
            
    elif isinstance(lhs,numbers.Complex):
        return lhs**(rhs + 0j)
        
    else:
        assert False, 'unexpected'
 
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
                y
            ,   vector.merge_weighted_vectors(
                    lhs._u_components,dy_dl,rhs._u_components,dy_dr
                )
            ,   vector.merge_weighted_vectors(
                    lhs._d_components,dy_dl,rhs._d_components,dy_dr
                )
            ,   vector.merge_weighted_vectors(
                    lhs._i_components,dy_dl,rhs._i_components,dy_dr
                )
            )
            
    elif isinstance(rhs,numbers.Real):
        if rhs == 1.0:
            return lhs 
        else:
            l = lhs.x

            y = l/rhs
            dy_dl = 1.0/rhs
            
            return UncertainReal(
                    y
                ,   vector.scale_vector(lhs._u_components,dy_dl)
                ,   vector.scale_vector(lhs._d_components,dy_dl)
                ,   vector.scale_vector(lhs._i_components,dy_dl)
                )
    elif isinstance(rhs,numbers.Complex):
        if rhs == 1.0:
            r = +lhs
            i = UncertainReal._constant(0.0)
        else:
            norm = abs(rhs)**2
            r = lhs * rhs.real/norm
            i = lhs * -rhs.imag/norm
            
        return UncertainComplex(r,i)   
    else:
        assert False, 'unexpected'

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
                y
            ,   vector.scale_vector(rhs._u_components,dy_dr)
            ,   vector.scale_vector(rhs._d_components,dy_dr)
            ,   vector.scale_vector(rhs._i_components,dy_dr)
            )
    elif isinstance(lhs,numbers.Complex):
        r = lhs.real / rhs 
        i = lhs.imag / rhs 
        
        return UncertainComplex(r,i)
    else:
        assert False, 'unexpected'

#----------------------------------------------------------------------------
def _mul(lhs,rhs):
    """
    Multiply `rhs` with the uncertain real number `lhs` 
    
    """
    if isinstance(rhs,UncertainReal):
    
        l = lhs.x
        r = rhs.x
        return UncertainReal(
                l*r
            ,   vector.merge_weighted_vectors(
                    lhs._u_components,r,rhs._u_components,l
                )
            ,   vector.merge_weighted_vectors(
                    lhs._d_components,r,rhs._d_components,l
                )
            ,   vector.merge_weighted_vectors(
                    lhs._i_components,r,rhs._i_components,l
                )
            )

    elif isinstance(rhs,numbers.Real):
        if rhs == 1.0:
            return lhs
        else:
            return UncertainReal(
                    float.__mul__(lhs.x,float(rhs))
                ,   vector.scale_vector(lhs._u_components,rhs)
                ,   vector.scale_vector(lhs._d_components,rhs)
                ,   vector.scale_vector(lhs._i_components,rhs)
                )
                
    elif isinstance(rhs,numbers.Complex):
        if rhs == 1.0:
            r = +lhs 
            i = UncertainReal._constant(0.0)
        else:
            r = lhs * rhs.real
            i = lhs * rhs.imag

        return UncertainComplex(r,i)
   
    else:
        assert False, 'unexpected'

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
                    float.__mul__(float(lhs),rhs.x)
                ,   vector.scale_vector(rhs._u_components,lhs)
                ,   vector.scale_vector(rhs._d_components,lhs)
                ,   vector.scale_vector(rhs._i_components,lhs)
                )
    elif isinstance(lhs,numbers.Complex):
        if lhs == 1.0:
            r = +rhs 
            i = UncertainReal._constant(0.0)
        else:
            r = lhs.real * rhs 
            i = lhs.imag * rhs 
    
        return UncertainComplex(r,i)
    else:
        assert False, 'unexpected'

#----------------------------------------------------------------------------
def _sub(lhs,rhs):
    """
    Subtract `rhs` from the uncertain real number `lhs` 
    
    """
    if isinstance(rhs,UncertainReal):
        return UncertainReal(
                lhs.x - rhs.x
            ,   vector.merge_weighted_vectors(
                    lhs._u_components,1.0,rhs._u_components,-1.0
                )
            ,   vector.merge_weighted_vectors(
                    lhs._d_components,1.0,rhs._d_components,-1.0
                )
            ,   vector.merge_weighted_vectors(
                    lhs._i_components,1.0,rhs._i_components,-1.0
                )
            )
            
    elif isinstance(rhs,numbers.Real):
        if rhs == 0.0:
            return lhs
        else:
            return UncertainReal(
                    float.__sub__(lhs.x,float(rhs))
                ,   vector.scale_vector(lhs._u_components,1.0)
                ,   vector.scale_vector(lhs._d_components,1.0)
                ,   vector.scale_vector(lhs._i_components,1.0)
                )
                
    elif isinstance(rhs,numbers.Complex):
        if rhs == 0.0:
            r = +lhs 
            i = UncertainReal._constant( -rhs.imag )
        else:
            r = lhs - rhs.real
            i = UncertainReal._constant( -rhs.imag )
            
        return UncertainComplex(r,i)
   
    else:
        assert False, 'unexpected'
  
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
                float.__sub__(float(lhs),rhs.x)
            ,   vector.scale_vector(rhs._u_components,-1.0)
            ,   vector.scale_vector(rhs._d_components,-1.0)
            ,   vector.scale_vector(rhs._i_components,-1.0)
            )
                
    elif isinstance(lhs,numbers.Complex):
        if lhs == 0.0:
            r = -rhs 
            i = UncertainReal._constant(0.0)
        else:
            r = lhs.real - rhs 
            i = UncertainReal._constant(lhs.imag)
        
        return UncertainComplex(r,i)

    else:
        raise NotImplementedError()

#----------------------------------------------------------------------------
def _add(lhs,rhs):
    """
    Add the uncertain real number `lhs` to `rhs`
    
    `lhs` is guaranteed UncertainReal
    `rhs` can be UncertainReal or a number
    
    """
    if isinstance(rhs,UncertainReal):
            
        return UncertainReal(
                lhs.x + rhs.x
            ,   vector.merge_vectors(lhs._u_components,rhs._u_components)
            ,   vector.merge_vectors(lhs._d_components,rhs._d_components)
            ,   vector.merge_vectors(lhs._i_components,rhs._i_components)
            )
            
    elif isinstance(rhs,numbers.Real):
        if rhs == 0.0:
            return lhs
        else:
            return UncertainReal(
                    lhs.x + rhs
                ,   vector.scale_vector(lhs._u_components,1.0)
                ,   vector.scale_vector(lhs._d_components,1.0)
                ,   vector.scale_vector(lhs._i_components,1.0)
                )
    elif isinstance(rhs,numbers.Complex):
        if rhs == 0.0:
            r = +lhs 
            i = UncertainReal._constant(0.0)
        else:
            r = lhs + rhs.real 
            i = UncertainReal._constant(rhs.imag)
            
        return UncertainComplex(r,i)
   
    else:
        assert False, 'unexpected'
        
#----------------------------------------------------------------------------
def _radd(lhs,rhs):
    """
    Add `lhs` to the uncertain real number `rhs`
    `rhs` is guaranteed UncertainReal
    
    """
    if isinstance(lhs,numbers.Real):    
        if lhs == 0.0:
            return rhs
        else:
            return UncertainReal(
                    float.__add__(float(lhs),rhs.x)
                ,   vector.scale_vector(rhs._u_components,1.0)
                ,   vector.scale_vector(rhs._d_components,1.0)
                ,   vector.scale_vector(rhs._i_components,1.0)
                )
                
    elif isinstance(lhs,numbers.Complex):
        if lhs == 0.0:
            r = +rhs
            i = UncertainReal._constant(0.0)
        else:
            r = lhs.real + rhs 
            i = UncertainReal._constant(lhs.imag)
            
        # Addition of a complex changes the type
        return UncertainComplex(r,i)
        
    else:
        assert False, 'unexpected'

#----------------------------------------------------------------------------
def set_correlation_real(x1,x2,r):
    """
    Assign a correlation coefficient between ``x1`` and ``x2``

    Parameters
    ----------
    x1, x2 : UncertainReal
    r: float
    
    """
    if (
        x1.is_elementary and 
        x2.is_elementary
    ):        
    
        ln1 = x1._node
        ln2 = x2._node
        
        if (
            not ln1.independent and
            not ln2.independent
        ):
            if ln1 is ln2 and r != 1.0:
                raise ValueError(
                    "value should be 1.0, got: '{}'".format(r)
                )
            else:
                ln1.correlation[ln2.uid] = r 
                ln2.correlation[ln1.uid] = r 
        else:
            raise RuntimeError( 
                "`set_correlation` called on independent node"
            )
    else:
        raise TypeError(
            "Arguments must be elementary uncertain numbers, \
            got: {!r} and {!r}".format(x1,x2)
        )
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
        ln1 = x1._node
        ln2 = x2._node
        
        if ln1 is ln2:
            return 1.0
        else:
            return ln1.correlation.get( ln2.uid, 0.0 )
        
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
    # The independent components of uncertainty
    # ( Faster this way than with reduce()! )
    var = 0.0
    if len(x._u_components) != 0:    
        var += math.fsum(
            u_i * u_i 
                for u_i in x._u_components.itervalues()
        )
    
    # For these terms correlations are possible
    if len(x._d_components) != 0:
        cpts = x._d_components
        keys = cpts.keys()
        values = cpts.values()
        
        for i,k_i in enumerate(keys):
            u_i = values[i]
            var += u_i * u_i 
            var += math.fsum(
                2.0 *
                u_i *
                k_i.correlation.get(k_j.uid,0.0) *
                values[j+i+1]
                    for j,k_j in enumerate( keys[i+1:] )
            )
                            
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
    
    cv = 0.0
    
    # `k1_i` is not correlated with anything, but if 
    # `k1_i` happens to influence x2 we get a contribution.    
    cv += math.fsum(
        u1_i * x2._u_components.get(k1_i,0.0)
            for k1_i,u1_i in x1._u_components.iteritems()
    )
                    
    # `k1_i` may be correlated with `k2_i`
    for k1_i,u1_i in x1._d_components.iteritems():
        cv += math.fsum(
            u1_i *
            k1_i.correlation.get(k2_i.uid,0.0) *
            u2_i
                for k2_i,u2_i in x2._d_components.iteritems()
        )

    return cv

#----------------------------------------------------------------------
def get_covariance_real(x1,x2):
    """Return the covariance between ``x1`` and `x2``
    
    Covariance may be calculated between a pair 
    of uncertain real numbers ``x1`` and `x2``, 
    which are not elementary.
    
    Returns
    -------
    float
    
    """
    if x1.is_elementary and x2.is_elementary:
        n1 = x1._node
        n2 = x2._node
        return n1.u*n1.correlation.get(n2.uid,0.0)*n2.u
    else:        
        return std_covariance_real(x1,x2) 
        
#----------------------------------------------------------------------------
def welch_satterthwaite(x):
    """Return the variance and degrees-of-freedom.

    Uses the Welch Satterthwaite calculation of dof
    for an uncertain real number ``x``.
    
    Uses the extensions described by Willink
    in Metrologia 44 (2007).
    
    Parameters
    ----------
    x : UncertainReal

    Returns
    -------
    The variance and degrees-of-freedom
    
    """    
    if not isinstance(x,UncertainReal):
        raise TypeError(
            "UncertainReal required, got: '{!r}'".format(x)
        )
    
    if x.is_elementary:
        # This isn't used in the current implementation
        return VarianceAndDof(x.v,x.df)
     
    elif _is_uncertain_real_constant(x):
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
        var = 0.0                       # combined standard variance
        cpts_lst = []                   # component variances  
        dof_lst = []
 
        # Independent components. So, no worries about ensembles   
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
        # is relaxed for ensembles of real quantities. 
        # They are treated as a single component when evaluating WS, 
        # in keeping with the Willink reference above.
        #
        
        # Control value that may get changed to NaN below        
        df = 0.0
                
        # Used to accumulate the sums of variances
        # of all components in ensembles.
        cpts_map = {}   

        # flag for handling the case of a complex number
        finish_complex = False  
        
        if len(d_keys):
            # The final item in the sequence is treated separately below
            for i,k_i in enumerate(d_keys[:-1]):
                                        
                u_i = d_values[i]
                v_i = u_i * u_i
                var += v_i
                df_i = d_dof[i]
                                    
                # Control of complex. 
                complex_id = getattr( k_i, 'complex', (None,None) )

                # Need to freeze this set to use it as a dict key
                # that identifies the ensemble.
                ensemble_i = frozenset( k_i.ensemble )
                
                if len(ensemble_i) !=0 and ensemble_i not in cpts_map:
                    # Create a new component entry for this ensemble
                    cpts_map[ensemble_i] = [0,df_i]

                if ensemble_i in cpts_map:
                    # Update the total variance of this ensemble
                    cpts_map[ensemble_i][0] += v_i
                    
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
                # because it is possible that just one component may
                # be given as the argument `x`.
                # NB, consecutive keys are assumed here
                if (k_i.uid,d_keys[i+1].uid) != complex_id:
                    finish_complex = False
                else:
                    # i.e., we need to process the next component
                    finish_complex =  True
                    
                # ---------------------------------------------------------
                nu_i_infinite = math.isinf( df_i ) 
                
                # Look at the remaining influences 
                for j,k_j in enumerate(d_keys[i+1:]):
                
                    if k_j.uid in k_i.correlation:                        
                        u_j = d_values[i+1+j]
                        r = k_i.correlation[k_j.uid]
                        covar_ij = 2.0*u_i*r*u_j
                        var += covar_ij    

                        if nu_i_infinite and math.isinf( k_j.df ):
                            # The correlated influences both have  
                            # infinite dof, so it is OK to use WS.
                            # Since infinite DoF are not summed
                            # in WS, we do not need to modify `cpts_lst`
                            #
                            continue
                                        
                        # Members of an ensemble may be correlated.
                        elif k_j.uid in ensemble_i:
                            cpts_map[ensemble_i][0] += covar_ij
                            continue
                                
                        # The real and imaginary components of a
                        # complex quantity may be correlated.
                        # This code is executed when the
                        # imaginary component associated with k_i
                        # (a real component) is encountered. It puts
                        # the covariance term into cpts_lst. The
                        # variance associated with `k_j` will be
                        # included in when the outer loop next increments.
                        # I.e., here we are only worried about off-diagonal
                        # components of the covariance matrix.
                        #
                        elif (k_i.uid,k_j.uid) == complex_id:
                            cpts_lst[-1] += covar_ij
                            continue

                        else:
                            # Correlation with no excuse, illegal!
                            df = nan
                            # Don't expect to get here, because of 
                            # controls on using ``set_correlation``
                            assert False 

            # Last value cannot be correlated with anything 
            # that has not already been processed,
            # but it might be the final component of an ensemble 
            k_i = d_keys[-1]
            u_i = d_values[-1]
            v_i = u_i * u_i
            df_i = d_dof[-1]
            var += v_i
                
            ensemble_i = frozenset( k_i.ensemble )
            if ensemble_i in cpts_map:
                cpts_map[ensemble_i][0] += v_i
                
            elif finish_complex:
                v_old =  cpts_lst[-1]
                cpts_lst[-1] = v_old + v_i
                
            else:
                cpts_lst.append(v_i)
                dof_lst.append(df_i)
        
        # Finish building cpts_lst and dof_lst
        # using the values accumulated in `cpts_map`
        values = cpts_map.itervalues() if PY2 else cpts_map.values()
        for v_i,df_i in values:
            cpts_lst.append(v_i)
            dof_lst.append(df_i)   
            
        # There is a pathological case where var == 0.
        # It can occur in a product of zero-valued uncertain factors.
        if var == 0: df = nan
                
        #--------------------------------------------------------------------        
        if math.isnan(df):
            return VarianceAndDof(var,nan)
        else:
            # Final calculation of WS 
            den = 0.0
            for v_i,dof_i in izip(cpts_lst,dof_lst):
                if not math.isinf(dof_i):
                    u2 = v_i / var
                    den += u2 * u2 / dof_i
            try:
                return VarianceAndDof(var,1.0/den)
            except ZeroDivisionError:
                return VarianceAndDof(var,inf)
                
#----------------------------------------------------------------------------
def real_ensemble(seq,df):
    """
    Declare the uncertain numbers in ``seq`` to be an ensemble.

    The elements of ``seq`` must be elementary
    and have the same number of degrees of freedom. 
    
    It is permissible for members of an ensemble to be correlated 
    and have finite degrees of freedom without causing problems 
    when evaluating the effective degrees of freedom. See: 
    
    R Willink, Metrologia 44 (2007) 340-349, section 4.1.1

    Effectively, members of an ensemble are treated 
    as simultaneous independent measurements of 
    a multivariate distribution. 
    
    """
    # TODO: assertions not required in release version
    # have been declared independent=False 
    assert all( s_i._node.independent == False for s_i in seq )

    # ensemble members must have the same degrees of freedom
    assert all( s_i.df == df for s_i in seq )

    # ensemble members must be elementary
    assert all( s_i.is_elementary for s_i in seq )
            
    ensemble = set( x._node.uid for x in seq )
    for s_i in seq:
        s_i._node.ensemble = ensemble     
                
#----------------------------------------------------------------------------
def append_real_ensemble(member,x):
    """
    Append an element to the an existing ensemble

    The uncertain number ``x`` must be elementary and have the 
    same number of degrees of freedom as other members 
    of the ensemble (not checked). 
    
    """
    # TODO: remove assertions in release version, because 
    # this function is only called from within GTC modules. 
    assert x.df == member._node.df
    assert x.is_elementary
    assert x._node.independent == False
    
    # All Leaf nodes refer to the same ensemble object 
    # So by adding a member here, all the other Leaf nodes 
    # see the change.
    member._node.ensemble.add(x._node)
    x._node.ensemble = member._node.ensemble

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
    An :class:`UncertainComplex` holds information about the measured
    value of a complex-valued quantity

    """
    
    __slots__ = (
        'real'
    ,   'imag'
    ,   '_value'
    ,   '_u'                    
    ,   '_r'                    
    ,   '_v'                    
    ,   '_label'
    ,   'is_elementary'         
    ,   'is_intermediate'       
    )

    #------------------------------------------------------------------------
    def __init__(self,r,i):
        """
        UncertainComplex(r,i)
        
        An :class:`UncertainComplex` object encapsulates a pair 
        of :class:`UncertainReal` objects

        Parameters
        ----------
        r, i : UncertainReal
        
        """
        # TODO: real and imaginary components do not always 
        # have the same `is_elementary` status!
        #
        # For example, doing arithmetic between an elementary 
        # UncertainReal and an UncertainComplex produces a 
        # constant UncertainReal of zero for the imaginary 
        # component. There may be other cases.
        # We could make this go away by forcing 
        # trivial addition and subtraction of zero to 
        # produce a new uncertain number.
        # 
        assert (i.is_elementary == r.is_elementary) or\
            (i.is_elementary and _is_uncertain_real_constant(r)) or\
            (r.is_elementary and _is_uncertain_real_constant(i))
            
        assert i.is_intermediate == r.is_intermediate
        
        self.real = r
        self.imag = i
        self._value = complex(r.x,i.x)
        
        self.is_elementary = r.is_elementary or i.is_elementary
        self.is_intermediate = r.is_intermediate

    #----------------------------------------------------------------------------
    @classmethod
    def _constant(cls,z,label=None):
        """
        Return a constant uncertain complex number.
        
        A constant uncertain complex number has no uncertainty
        and infinite degrees of freedom.        

        The real and imaginary components are given labels 
        with the suffixes '_re' and '_im' to added ``label``.
        
        Parameters
        ----------
        z : complex
        label : string, or None

        Returns
        -------
        UncertainComplex
        
        """
        if label is None:
            label_r,label_i = None,None
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
        real = UncertainReal._constant(z.real,label_r)
        imag = UncertainReal._constant(z.imag,label_i)

        ucomplex = UncertainComplex(real,imag)    
        ucomplex._label = label
            
        return ucomplex        

    #----------------------------------------------------------------------------
    @classmethod
    def _elementary(cls,z,u_r,u_i,r,df,label,independent):
        """
        Return an elementary uncertain complex number.

        Parameters
        ----------
        x : complex
        u_r, u_i : standard uncertainties 
        r : correlation coefficient
        df : float
        label : string, or None

        Returns
        -------
        UncertainComplex
        
        The real and imaginary components are given labels 
        with the suffixes '_re' and '_im' to added ``label``.

        The ``independent`` argument controls whether this
        uncertain number may be correlated with others.
        
        """
        if label is None:
            label_r,label_i = None,None
            
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
        # `independent` will be False if `r != 0`
        real = UncertainReal._elementary(z.real,u_r,df,label_r,independent)
        imag = UncertainReal._elementary(z.imag,u_i,df,label_i,independent)

        # We need to be able to look up complex pairs
        # The integer part of the IDs are consecutive.
        complex_id = (real._node.uid,imag._node.uid)
        real._node.complex = complex_id 
        imag._node.complex = complex_id
        
        if r is not None:
            real._node.correlation[imag._node.uid] = r 
            imag._node.correlation[real._node.uid] = r 
            
        ucomplex = UncertainComplex(real,imag)
        ucomplex.is_elementary = True
        
        ucomplex._label = label
            
        return ucomplex   
        
    #----------------------------------------------------------------------------
    @classmethod
    def _intermediate(cls,z,label):
        """
        Return an intermediate uncertain complex number

        :arg z: the uncertain complex number
        :type z: :class:`UncertainComplex`

        :arg label: str

        If ``label`` is not ``None`` the label will be applied
        to the uncertain complex number and labels with
        a suitable suffix will be applied to the
        real and imaginary components.
        
        """
        if label is None:
            UncertainReal._intermediate(z.real,None)
            UncertainReal._intermediate(z.imag,None) 
        else:
            label_r = "{}_re".format(label)
            label_i = "{}_im".format(label)
            
            UncertainReal._intermediate(z.real,label_r)
            UncertainReal._intermediate(z.imag,label_i) 
            
        z._label = label
        
    #------------------------------------------------------------------------
    def _round(self,digits,df_decimals):
        """
        `digits` specifies the number of significant digits of 
        in the least component uncertainty that will be retained. 
        
        The components of the value will use the same precision. 
        
        The degrees-of-freedom will be represented using 
        `df_decimals` decimal places.

        `df_decimals` specifies the number of decimal places 
        reported for the degrees-of-freedom.
        
        Degrees-of-freedom are greater than `inf_dof` are set to `inf`.
        
        """
        v11, v12, v21, v22 = self.v 
        re_u = math.sqrt( v11 )
        im_u = math.sqrt( v22 )
        
        if (
            cmath.isnan(self.x) or cmath.isinf(self.x) or 
            math.isnan(re_u) or math.isinf(re_u) or 
            math.isnan(im_u) or math.isinf(im_u) 
        ):
            return GroomedUncertainComplex(
                x = self.x,
                u = [re_u,im_u],
                r = None,
                df = self.df,
                label = self.label,
                precision = digits,
                df_decimals = df_decimals,
                re_u_digits = '{}'.format(re_u),
                im_u_digits = '{}'.format(im_u)
            )
            
        elif v11 != 0 or v22 != 0:
            re = self.real 
            im = self.imag 
            
            # Real and imaginary component uncertainties are different
            # find the lesser uncertainty and round to two digits,
            # then express the results in this precision.
            u = min(re.u, im.u)
            # However, if one component has no uncertainty use the other 
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

            # Get the decimal places representing uncertainty 
            # When the uncertainty is to the left of the 
            # decimal point there will be `digits` 
            # but to the right of the decimal point there must
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

            den = (re_u*im_u)
            r = v12/den if v12 != 0.0 else 0.0
            r_factor = 10**(-3)
            r = r_factor*round(r/r_factor)

            df = self.df
            if not math.isnan(df):
                if df > inf_dof:
                    df = inf
                else:
                    df_factor = 10**(-df_decimals)
                    df = df_factor*math.floor(self.df/df_factor)
            
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
        elif _is_uncertain_complex_constant(self):
            # Just use Python's default fixed-point precision
            return GroomedUncertainComplex(
                x = self.x,
                u = [0.0, 0.0],
                r = None,
                df = inf,
                label = self.label,
                precision = 6,
                df_decimals = 0,
                re_u_digits = 0,
                im_u_digits = 0
            )
            
        else:
            assert False, 'unexpected'
            
    #------------------------------------------------------------------------
    def __repr__(self):
        
        x = self.x
        u = self.u
        r = self.r  
        df = self.df
        
        if not math.isnan(df) and df > inf_dof:
            df = inf 

        if self.label is None:
            s = ("ucomplex(({0.real:.16g}{0.imag:+.16g}j), "
                "u=[{1[0]!r},{1[1]!r}], "
                "r={2!r}, df={3!r}"
                ")").format( 
                x,u,r,df
            )        
        else:
            s = ("ucomplex(({0.real:.16g}{0.imag:+.16g}j), "
                "u=[{1[0]!r},{1[1]!r}], "
                "r={2!r}, df={3!r}, "
                "label={4}"
                ")").format( 
                x,u,r,df,self.label
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
    # For coercion to Boolean 
    def __bool__(self):
        return UncertainComplex.__ne__(self,0.0)
    __nonzero__ = __bool__

    #------------------------------------------------------------------------
    def __abs__(self):
        return abs( self._value )
    
    #------------------------------------------------------------------------
    def conjugate(self):
        """Return the complex conjugate

        An UncertainComplex object is created by negating the imaginary
        component.

        :rtype: :class:`~lib.UncertainComplex`
        
        """
        # NB unary '+' makes a new object with the same uncertainty and value
        return UncertainComplex(+self.real,-self.imag)  
        
    #------------------------------------------------------------------------
    @property
    def x(self):
        """Return the value 

        :rtype: complex
        
        Note that ``uc.x`` is equivalent to :func:`value(uc)<core.value>`
        
        **Example**::
            >>> uc = ucomplex(1+2j,(.3,.2))
            >>> uc.x
            (1+2j)

        """
        return self._value

    #------------------------------------------------------------------------
    @property
    def u(self):
        """Return standard uncertainties for the real and imaginary components

        :rtype: 2-element sequence of float
        
        Note that ``uc.u`` is equivalent to :func:`uncertainty(uc)<core.uncertainty>`
        
        **Example**::

            >>> uc = ucomplex(1+2j,(.5,.5))
            >>> uc.u
            StandardUncertainty(real=0.5, imag=0.5)
        
        """        
        if not hasattr(self,"_u"):
            self.real.u
            self.imag.u
            self._u = StandardUncertainty(self.real.u,self.imag.u)
            
        return self._u 

    #------------------------------------------------------------------------
    @property
    def v(self):
        """Return the variance-covariance matrix

        The uncertainty of an uncertain complex number can be associated with
        a 4-element variance-covariance matrix.

        :rtype: 4-element sequence of float
        
        Note that ``uc.v`` is equivalent to :func:`variance(uc)<core.variance>`
        
        **Example**::

            >>> uc = ucomplex(1+2j,(.5,.5))
            >>> uc.v
            VarianceCovariance(rr=0.25, ri=0.0, ir=0.0, ii=0.25)

        """
        if not hasattr(self,"_v"):
            self._v = std_variance_covariance_complex(self)
            
        return self._v

    #------------------------------------------------------------------------
    @property
    def r(self):
        """Return the correlation coefficient between real 
        and imaginary components

        :rtype: float
        
        """
        if not hasattr(self,"_r"):
            cv = self.v
            self._r = cv[1]/(cv[0]*cv[3]) if cv[1] != 0.0 else 0.0
        
        return self._r
            
    #------------------------------------------------------------------------
    @property
    def df(self):
        """Return the degrees-of-freedom 

        When the object is not an elementary uncertain number, the 
        effective degrees-of-freedom is calculated using the method
        described by Willink and Hall in Metrologia 2002, 39, pp 361-369.

        :rtype: float
        
        Note that ``uc.df`` is equivalent to :func:`dof(uc)<core.dof>`
        
        **Example**::
            >>> uc = ucomplex(1+2j,(.3,.2),3)
            >>> uc.df
            3.0
        
        """
        cv_df = willink_hall(self)
        if not hasattr(self,"_v"):
            self._v = cv_df.cv 
            
        return cv_df.df 

    #--------------------------------------------
    @property
    def label(self):
        """The `label` attribute

        :rtype: str
        
        Note that``un.label`` is equivalent to :func:`label(un)<core.label>`
        
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
        # NB in case like x+0 -> x, we do absolutely nothing.
        # This, means an elementary UN remains elementary even
        # when you might expect it to loose that following the addition.
        # We would have a slightly more consistent system if we didn't
        # take such short cuts, at the expense of unnecessary extra steps.
        # TODO: should decide whether to stick with the shortcuts. 
        
        lhs = self
        if isinstance(rhs,UncertainComplex):
            r = self.real + rhs.real
            i = self.imag + rhs.imag
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,UncertainReal):
            r = self.real + rhs
            # Force `i` to be an intermediate uncertain number,
            # which `self.imag + 0` will not do.
            i = +self.imag 
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,numbers.Real):
            if rhs == 0.0:
                return self
            else:            
                r = self.real + rhs
                # Force `i` to be an intermediate uncertain number,
                # which `self.imag + 0` will not do.
                i = +self.imag
                return UncertainComplex(r,i)
            
        elif isinstance(rhs,numbers.Complex):
            if rhs == 0.0:
                return self
            else:
                # # Force addition between uncertain numbers
                r = self.real + UncertainReal._constant( rhs.real )
                i = self.imag + UncertainReal._constant( rhs.imag )
                return UncertainComplex(r,i)
                
        else:
            return NotImplemented
        
    def __radd__(self,lhs):
        if isinstance(lhs,UncertainReal):
            r = lhs + self.real
            # Force `i` to be an intermediate uncertain number,
            # which `self.imag + 0` will not do.
            i = +self.imag
            return UncertainComplex(r,i)
            
        elif isinstance(lhs,numbers.Real):
            if lhs == 0.0:
                return self
            else:            
                r = lhs + self.real
                # Force `i` to be an intermediate uncertain number,
                # which `self.imag + 0` will not do.
                i = +self.imag
                return UncertainComplex(r,i)
                
        elif isinstance(lhs,numbers.Complex):
            if lhs == 0.0:
                return self
            else:
                # Force addition between uncertain numbers
                r = UncertainReal._constant( lhs.real ) + self.real
                i = UncertainReal._constant( lhs.imag ) + self.imag
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
            i = +self.imag
            return UncertainComplex(r,i)
            
        elif isinstance(rhs,numbers.Real):
            if rhs == 0.0:
                return self
            else:
                r = self.real - rhs
                i = +self.imag
                return UncertainComplex(r,i)
                
        elif isinstance(rhs,numbers.Complex):
            if rhs == 0.0:
                return self
            else:
                r = self.real - UncertainReal._constant( rhs.real )
                i = self.imag - UncertainReal._constant( rhs.imag )
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
                
        elif isinstance(lhs,numbers.Complex):
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
        
            return _bivariate_uc_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
            
        elif isinstance(rhs,UncertainReal):
            l = lhs._value
            r = rhs.x
            z = l * r
            
            dz_dl = z_to_seq( r )                
            dz_dr = z_to_seq( l )            

            return _bivariate_uc_ur(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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
            
            return _bivariate_uc_n(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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

            return _bivariate_ur_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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
            
            return _bivariate_n_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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

            dz_dl = z_to_seq( 1.0 / r ) 
            dz_dr = z_to_seq( -z / r )            
        
            return _bivariate_uc_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
            
        elif isinstance(rhs,UncertainReal):
            l = lhs._value
            r = rhs.x
            
            z = l / r
            
            dz_dl = z_to_seq( 1.0 / r ) 
            dz_dr = z_to_seq( -z / r )            

            return _bivariate_uc_ur(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
            
        elif isinstance(rhs,numbers.Complex):
            if rhs == 1.0:
                return self
            else:            
                l = lhs._value
                r = 1.0 * rhs  # avoid integer division problems
            
            z = l / r

            dz_dl = z_to_seq( 1.0 / r ) 
            dz_dr = z_to_seq( 0.0 )            
            
            return _bivariate_uc_n(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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
            dz_dl = z_to_seq( 1.0 / r )     

            return _bivariate_ur_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
            
        elif isinstance(lhs,numbers.Complex):
            r = rhs._value
            l = 1.0 * lhs # avoid integer division problems
            
            z = l / r

            dz_dr = z_to_seq( -z / r )                
            dz_dl = z_to_seq( 0.0 )            
            
            return _bivariate_n_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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
        
            return _bivariate_uc_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
        elif isinstance(rhs,UncertainReal):
            zl = lhs._value
            zr = rhs.x
            z = zl ** zr
            dz_dl = z_to_seq( zr * z / zl )
            dz_dr = z_to_seq( cmath.log(zl) * z if zl != 0 else 0  )

            return _bivariate_uc_ur(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
        elif isinstance(rhs,numbers.Complex):
            if rhs == 1.0:
                return self
            else:
                zl = lhs._value
                zr = rhs
                z = zl ** zr
                dz_dl = z_to_seq( zr * z / zl )
                dz_dr = z_to_seq( 0.0 )
                   
                return _bivariate_uc_n(
                    lhs,rhs,
                    z,
                    dz_dl,
                    dz_dr
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

            return _bivariate_ur_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
            )
        elif isinstance(lhs,numbers.Complex):
            zl = lhs
            zr = rhs._value
            z = zl ** zr
            dz_dl = z_to_seq( 0.0 )
            dz_dr = z_to_seq( cmath.log(zl) * z  if zl != 0 else 0 )
            
            return _bivariate_n_uc(
                lhs,rhs,
                z,
                dz_dl,
                dz_dr
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
        return _univariate_uc(
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
                mag_x
            ,   vector.merge_weighted_vectors(
                    re._u_components,dz_dre,im._u_components,dz_dim
                )
            ,   vector.merge_weighted_vectors(
                    re._d_components,dz_dre,im._d_components,dz_dim
                )
            ,   vector.merge_weighted_vectors(
                    re._i_components,dz_dre,im._i_components,dz_dim
                )
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
                abs(x)**2
            ,   vector.merge_weighted_vectors(
                    re._u_components,dz_dre,im._u_components,dz_dim
                )
            ,   vector.merge_weighted_vectors(
                    re._d_components,dz_dre,im._d_components,dz_dim
                )
            ,   vector.merge_weighted_vectors(
                    re._i_components,dz_dre,im._i_components,dz_dim
                )
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

#----------------------------------------------------------------------------
def _univariate_uc(arg,z,dz_dx):
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
    arg : :class:`UncertainComplex`
    z : complex
    dz_dx : 4-element sequence of float
    
    Returns
    -------
    :class:`UncertainComplex`
    
    """
    return UncertainComplex(
        UncertainReal(
            z.real,
            vector.merge_weighted_vectors(
                arg.real._u_components,dz_dx[0],
                arg.imag._u_components,dz_dx[1],
            ),
            vector.merge_weighted_vectors(
                arg.real._d_components,dz_dx[0],
                arg.imag._d_components,dz_dx[1],
            ),
            vector.merge_weighted_vectors(
                arg.real._i_components,dz_dx[0],
                arg.imag._i_components,dz_dx[1],
            ),
        ),
        UncertainReal(
            z.imag,
            vector.merge_weighted_vectors(
                arg.real._u_components,dz_dx[2],
                arg.imag._u_components,dz_dx[3],
            ),
            vector.merge_weighted_vectors(
                arg.real._d_components,dz_dx[2],
                arg.imag._d_components,dz_dx[3],
            ),
            vector.merge_weighted_vectors(
                arg.real._i_components,dz_dx[2],
                arg.imag._i_components,dz_dx[3],
            )
        )
    )
#----------------------------------------------------------------------------
def _bivariate_uc_uc(
    lhs,rhs,
    z,
    dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
    dz_dr  # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
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
    lhs, rhs : :class:`UncertainComplex`
    z : complex
    dz_dl, dz_dr : 4-element sequence of float
    
    Returns
    -------
    :class:`UncertainComplex`
    
    """
    lhs_r = lhs.real
    lhs_i = lhs.imag
    rhs_r = rhs.real
    rhs_i = rhs.imag

    u_lhs_real, u_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._u_components,(dz_dl[0],dz_dl[2]),
        lhs_i._u_components,(dz_dl[1],dz_dl[3])
    )
    u_rhs_real, u_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._u_components,(dz_dr[0],dz_dr[2]),
        rhs_i._u_components,(dz_dr[1],dz_dr[3])
    )
    d_lhs_real, d_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._d_components,(dz_dl[0],dz_dl[2]),
        lhs_i._d_components,(dz_dl[1],dz_dl[3])
    )
    d_rhs_real, d_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._d_components,(dz_dr[0],dz_dr[2]),
        rhs_i._d_components,(dz_dr[1],dz_dr[3])
    )
    i_lhs_real, i_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._i_components,(dz_dl[0],dz_dl[2]),
        lhs_i._i_components,(dz_dl[1],dz_dl[3])
    )
    i_rhs_real, i_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._i_components,(dz_dr[0],dz_dr[2]),
        rhs_i._i_components,(dz_dr[1],dz_dr[3])
    )
    return UncertainComplex(
        UncertainReal(
            z.real,
            vector.merge_vectors(
                u_lhs_real, u_rhs_real
            ),
            vector.merge_vectors(
                d_lhs_real, d_rhs_real
            ),
            vector.merge_vectors(
                i_lhs_real, i_rhs_real
            )
        ),
        UncertainReal(
            z.imag,
            vector.merge_vectors(
                u_lhs_imag,u_rhs_imag
            ),
            vector.merge_vectors(
                d_lhs_imag,d_rhs_imag
            ),
            vector.merge_vectors(
                i_lhs_imag, i_rhs_imag
            )
        )
    )
#----------------------------------------------------------------------------
def _bivariate_uc_ur(
    lhs,rhs,
    z,
    dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
    dz_dr  # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
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
    lhs : :class:`UncertainComplex`
    rhs : :class:`UncertainReal`
    z : complex
    dz_dl, dz_dr : 4-element sequence of float
    
    Returns
    -------
    Uncert:class:`UncertainComplex`ainComplex
            
    """
    lhs_r = lhs.real
    lhs_i = lhs.imag

    u_lhs_real, u_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._u_components,(dz_dl[0],dz_dl[2]),
        lhs_i._u_components,(dz_dl[1],dz_dl[3])
    )

    u_rhs_real, u_rhs_imag = vector.scale_vector_twice(
        rhs._u_components,(dz_dr[0],dz_dr[2])
    )
    
    d_lhs_real, d_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._d_components,(dz_dl[0],dz_dl[2]),
        lhs_i._d_components,(dz_dl[1],dz_dl[3])
    )

    d_rhs_real, d_rhs_imag = vector.scale_vector_twice(
        rhs._d_components,(dz_dr[0],dz_dr[2])
    )

    i_lhs_real, i_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._i_components,(dz_dl[0],dz_dl[2]),
        lhs_i._i_components,(dz_dl[1],dz_dl[3])
    )

    i_rhs_real, i_rhs_imag = vector.scale_vector_twice(
        rhs._i_components,(dz_dr[0],dz_dr[2])
    )
    
    return UncertainComplex(
        UncertainReal(
            z.real,
            vector.merge_vectors(
                u_lhs_real,u_rhs_real
            ),
            vector.merge_vectors(
                d_lhs_real,d_rhs_real
            ),
            vector.merge_vectors(
                i_lhs_real, i_rhs_real
            )
        ),
        UncertainReal(
            z.imag,
            vector.merge_vectors(
                u_lhs_imag,u_rhs_imag
            ),
            vector.merge_vectors(
                d_lhs_imag,d_rhs_imag
            ),
            vector.merge_vectors(
                i_lhs_imag, i_rhs_imag
            )
        )
    )
#----------------------------------------------------------------------------
def _bivariate_uc_n(
    lhs,rhs,
    z,
    dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
    dz_dr  # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
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
    lhs : :class:`UncertainComplex`
    rhs : float
    z : complex
    dz_dl, dz_dr : 4-element sequence of float
    
    Returns
    -------
    :class:`UncertainComplex`
            
    """
    lhs_r = lhs.real
    lhs_i = lhs.imag

    u_lhs_real, u_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._u_components,(dz_dl[0],dz_dl[2]),
        lhs_i._u_components,(dz_dl[1],dz_dl[3])
    )

    d_lhs_real, d_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._d_components,(dz_dl[0],dz_dl[2]),
        lhs_i._d_components,(dz_dl[1],dz_dl[3])
    )

    i_lhs_real, i_lhs_imag = vector.merge_weighted_vectors_twice(
        lhs_r._i_components,(dz_dl[0],dz_dl[2]),
        lhs_i._i_components,(dz_dl[1],dz_dl[3])
    )

    return UncertainComplex(
        UncertainReal(
            z.real,
            u_lhs_real,
            d_lhs_real,
            i_lhs_real
        ),
        UncertainReal(
            z.imag,
            u_lhs_imag,
            d_lhs_imag,
            i_lhs_imag
        )
    )
#----------------------------------------------------------------------------
def _bivariate_ur_uc(
    lhs,rhs,
    z,
    dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
    dz_dr  # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
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
    lhs : :class:`UncertainReal`
    rhs : :class:`UncertainComplex` 
    z : complex
    dz_dl, dz_dr : 4-element sequence of float
    
    Returns
    -------
    :class:`UncertainComplex`
            
    """
    rhs_r = rhs.real
    rhs_i = rhs.imag

    u_lhs_real, u_lhs_imag = vector.scale_vector_twice(
        lhs._u_components,(dz_dl[0],dz_dl[2])
    )
    
    u_rhs_real, u_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._u_components,(dz_dr[0],dz_dr[2]),
        rhs_i._u_components,(dz_dr[1],dz_dr[3])
    )

    d_lhs_real, d_lhs_imag = vector.scale_vector_twice(
        lhs._d_components,(dz_dl[0],dz_dl[2])
    )
    
    d_rhs_real, d_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._d_components,(dz_dr[0],dz_dr[2]),
        rhs_i._d_components,(dz_dr[1],dz_dr[3])
    )

    i_lhs_real, i_lhs_imag = vector.scale_vector_twice(
        lhs._i_components,(dz_dl[0],dz_dl[2])
    )
    
    i_rhs_real, i_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._i_components,(dz_dr[0],dz_dr[2]),
        rhs_i._i_components,(dz_dr[1],dz_dr[3])
    )

    return UncertainComplex(
        UncertainReal(
            z.real,
            vector.merge_vectors(
                u_lhs_real,u_rhs_real
            ),
            vector.merge_vectors(
                d_lhs_real,d_rhs_real
            ),
            vector.merge_vectors(
                i_lhs_real, i_rhs_real
            )
        ),
        UncertainReal(
            z.imag,
            vector.merge_vectors(
                u_lhs_imag,u_rhs_imag
            ),
            vector.merge_vectors(
                d_lhs_imag,d_rhs_imag
            ),
            vector.merge_vectors(
                i_lhs_imag, i_rhs_imag
            )
        )
    )
#----------------------------------------------------------------------------
def _bivariate_n_uc(
    lhs,rhs,
    z,
    dz_dl, # (dz_re_dl_re, dz_re_dl_im, dz_im_dl_re, dz_im_dl_im)
    dz_dr  # (dz_re_dr_re, dz_re_dr_im, dz_im_dr_re, dz_im_dr_im)
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
    lhs : float 
    rhs : :class:`UncertainComplex`
    z : complex
    dz_dl, dz_dr : 4-element sequence of float
    
    Returns
    -------
    :class:`UncertainComplex`
            
    """
    rhs_r = rhs.real
    rhs_i = rhs.imag

    u_rhs_real, u_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._u_components,(dz_dr[0],dz_dr[2]),
        rhs_i._u_components,(dz_dr[1],dz_dr[3])
    )

    d_rhs_real, d_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._d_components,(dz_dr[0],dz_dr[2]),
        rhs_i._d_components,(dz_dr[1],dz_dr[3])
    )

    i_rhs_real, i_rhs_imag = vector.merge_weighted_vectors_twice(
        rhs_r._i_components,(dz_dr[0],dz_dr[2]),
        rhs_i._i_components,(dz_dr[1],dz_dr[3])
    )
    
    return UncertainComplex(
        UncertainReal(
            z.real,
            u_rhs_real,
            d_rhs_real,
            i_rhs_real
        ),
        UncertainReal(
            z.imag,
            u_rhs_imag,
            d_rhs_imag,
            i_rhs_imag
        )
    )
#---------------------------------------------------------------------------
def std_variance_covariance_complex(x):
    """Return the variance-covariance matrix

    The variance-covariance matrix characterises the  uncertainty
    of an uncertain complex number.
    
    :arg :class:`~lib.UncertainComplex`

    :returns: a 4-element sequence of float
    :rtype: :class:`~named_tuples.VarianceCovariance`
    
    """
    re, im = x.real, x.imag
    
    v_r = re.v
    v_i = im.v
    cv = std_covariance_real(re,im)
    
    return VarianceCovariance(v_r,cv,cv,v_i)

#---------------------------------------------------------------------------
def _covariance_submatrix(u_re,u_im):
    """Return v_rr, v_ir, v_ii, the 3 covariance matrix terms

    `u_re` and `u_im` are `GTC.Vector`s Leaf nodes and 
    component of uncertainty values.
    The nodes are all independent==False`
    
    """
    # Utility function for `willink_hall(x)`
    # Each of the terms returned is like an 
    # LPU calculation for the variance-covariance
    # of a sub-matrix of the covariance matrix.
    
    v_rr = v_ri = v_ii = 0.0

    # We need uncertainty components with
    # the same set of influences.
    assert u_re.keys() == u_im.keys()
    
    keys = u_re.keys()
    for i,x_i in enumerate(keys):

        # In the absence of correlation, just these terms
        x_u_re = u_re[x_i]
        v_rr += x_u_re**2
        
        x_u_im = u_im[x_i]
        v_ii += x_u_im**2

        v_ri += x_u_re * x_u_im

        # Additional terms required when there are correlations
        row_x = x_i.correlation

        v_rr += math.fsum(
            2.0 * x_u_re * u_re[y_i] * row_x.get(y_i.uid,0.0)
                for y_i in keys[i+1:]
        )

        v_ii += math.fsum(
            2.0 * x_u_im * u_im[y_i] * row_x.get(y_i.uid,0.0)
                for y_i in keys[i+1:]
        )

        # Cross product of `u_re` and `u_im` so we need
        # to iterate over all keys (there is no symmetry
        # allowing us to step over just half). We just
        # skip the term `y_i == x_i`, which is already
        # in the sum.
        v_ri +=math.fsum(
            x_u_re * u_im[y_i] * row_x.get(y_i.uid,0.0)
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
    
        # Instance attributes hold data
        self.u_re = vector.Vector()
        self.u_im = vector.Vector()
        self.nu = nu
        
    def accumulate(self):
        """
        Update the running sums from this object.
        
        """
        # Calculate `v` = u * r * u'
        v_11,v_12,v_22 = _covariance_submatrix(
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
    # The main purpose of the code is to detect illegal cases
    # and accumulate uncertainty components associated with influences
    # that have finite DoF. The variance calculation
    # is delegated to `std_variance_covariance_complex()`,
    # which calls routines in `library_real` to evaluate variance and 
    # covariance regardless of degrees of freedom.
    #
    if not isinstance(x,UncertainComplex):
        raise TypeError(
            "expected 'UncertainComplex' got: '{!r}'".format(x)
        )
    
    if _is_uncertain_complex_constant(x):
        return VarianceAndDof((0.,0.,0.,0.),inf)
        
    real = x.real
    imag = x.imag

    if real.is_elementary:
        assert imag.is_elementary
        return VarianceAndDof(
            std_variance_covariance_complex(x),
            real.df
        )
    else:
        # Separate the work to be done on 
        # independent UNs from the work 
        # on possibly correlated UNs.
        
        # All keys for the independent components
        re_u = vector.extend_vector(
            x.real._u_components,x.imag._u_components
        )    
        im_u = vector.extend_vector(
            x.imag._u_components,x.real._u_components
        )
        
        # All keys for the dependent components
        re_d = vector.extend_vector(
            x.real._d_components,x.imag._d_components
        )    
        im_d = vector.extend_vector(
            x.imag._d_components,x.real._d_components
        )
            
        ids_u = re_u.keys()
        ids_d = re_d.keys()
                
        degrees_of_freedom_u = [ k_i.df for k_i in ids_u ]
        degrees_of_freedom_d = [ k_i.df for k_i in ids_d ]
        
        # Perhaps everything has infinite DoF?
        if ( 
            degrees_of_freedom_u.count(inf) == len(degrees_of_freedom_u) and 
            degrees_of_freedom_d.count(inf) == len(degrees_of_freedom_d)
        ):
            return VarianceAndDof( std_variance_covariance_complex(x), inf )
        
        # -------------------------------------------------------------------
        # Initially clear the accumulators
        _EnsembleComponents.clear()
        
        # -------------------------------------------------------------------
        # Process independent components.
        #   They cannot belong to an ensemble and
        #   they cannot be correlated.
        #
        for i_re,id_re in enumerate( ids_u ):
            
            nu_i = degrees_of_freedom_u[i_re]

            if not math.isinf( nu_i ):
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
            skip_imaginary = False      # Initial value 
            
            ensemble_reg = dict()       # keys: frozenset , values: _EnsembleComponent 
            
            # There is one element in `ids_d` for each real-valued 
            # component (ie for 3 complex influences len(ids) == 6)
            for i_re,id_re in enumerate( ids_d ):
            
                # If an influence is complex, the real and imaginary
                # components are handled in the first pass, so 
                # we need to skip to the next id in the list.   
                if skip_imaginary:
                    skip_imaginary = False
                    continue
                
                # mapping between uid's and correlation coefficients
                row_re = id_re.correlation
                
                nu_i = degrees_of_freedom_d[i_re]
                i_re_infinite = math.isinf( nu_i )         

                ensemble_i = frozenset(id_re.ensemble)
                if len(ensemble_i) and ensemble_i not in ensemble_reg:
                    # A non-trivial ensemble not yet identified
                    ensemble_reg[ensemble_i] = _EnsembleComponents(nu_i)
                   
                # `components_i` holds the components 
                # associated with this influence. When it is 
                # part of an ensemble, reuse the same object.
                components_i = ensemble_reg.get(
                    ensemble_i,
                    _EnsembleComponents(nu_i)
                )
                
                if hasattr(id_re,'complex'):
                    # This is a complex influence
                    skip_imaginary = True
                    
                    # Assumes consecutive nodes!!
                    id_im = ids_d[i_re + 1]

                    # mapping between uid's and correlation coefficients
                    row_im = id_im.correlation

                    # Step over the imaginary component, 
                    # which is assumed to follow 
                    next_i = i_re + 2

                    # Check for correlations with any other (real) influence 
                    if next_i < len_ids:
                        # `j` is any of the other (real) influences of `i`  
                        for j, j_id in enumerate( ids_d[next_i:] ):
                        
                            # Look for the illegal case of correlation between 
                            # influences when at least one has finite dof and 
                            # they are not in an ensemble together.                            
                            if i_re_infinite and math.isinf( 
                                    degrees_of_freedom_d[next_i+j]
                                ):  
                                    continue
                                
                            elif (
                                j_id.uid not in ensemble_i  
                                    and ( 
                                        j_id.uid in row_re or 
                                        j_id.uid in row_im 
                                    )
                            ):
                                # Illegal: `j` is not in an ensemble with `i` 
                                # but `j` is correlated with a component of `i`
                                # Do not expect this case to be allowed
                                assert False, 'unexpected'
                                # return VarianceAndDof(
                                    # std_variance_covariance_complex(x),
                                    # nan
                                # )
                        
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

                    assert i_re_infinite or next_i >= len_ids, "unexpected"
                    # TODO: this can probably be removed 
                    # Check for correlations, perhaps abort DoF calculation
                    # if not i_re_infinite and next_i < len_ids:
                        
                        # for j, j_id in enumerate( ids_d[next_i:] ):                        
                            # # Look for the illegal cases
                            # if (
                                # not math.isinf( 
                                    # degrees_of_freedom_d[next_i+j]  
                                # ) 
                                # and id_re.uid not in ensemble_i
                                # and ( j_id.uid in row_re )
                            # ):
                                # assert False, "should not now occur"

                    # If we can get here, this real influence can be
                    # used for the DoF calculation. 
                    # Update the buffer. 
                    if not i_re_infinite:
                        components_i.u_re.append( id_re,re_d[id_re] )
                        components_i.u_im.append( id_re,im_d[id_re] )

                # If the current influence does NOT belong to an ensemble
                # update the sums immediately, otherwise wait until the end
                if len( ensemble_i ) == 0:
                    components_i.accumulate()

            values = ensemble_reg.itervalues() if PY2 else ensemble_reg.values()
            for ec_i in values:
                ec_i.accumulate()

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
           
#----------------------------------------------------------------------------
def complex_ensemble(seq,df):
    """
    Declare the uncertain numbers in ``seq`` to be an ensemble.

    The uncertain numbers in ``seq`` must be elementary
    and have the same numbers of degrees of freedom. 
    
    It is permissible for members of an ensemble to be correlated 
    and have finite degrees of freedom without causing problems 
    when evaluating the effective degrees of freedom. See: 
    
    R Willink, Metrologia 44 (2007) 340-349, section 4.1.1

    Effectively, members of an ensemble are treated 
    as simultaneous independent measurements of 
    a multivariate distribution. 
    
    """
    # NB, we simply assign ``dof`` without checking for previous values. 
    # This avoids overhead and should not be a risk, because 
    # users call this method via functions in the ``core`` module.
    
    # TODO: assertions not required in release version
    # ensemble members must have the same degrees of freedom
    assert all( s_i.df == df for s_i in seq )

    # ensemble members must be elementary
    assert all( s_i.is_elementary for s_i in seq )
    
    # All UNs will have been declared with ``independent=False`` 
    if not all( 
        x._node.independent == False 
            for pair in seq 
                for x in (pair.real,pair.imag) 
    ):
        raise RuntimeError(
            "members of an ensemble must be elementary and dependent"
        )
        
    ensemble = set( 
        x._node.uid 
            for pair in seq 
                for x in (pair.real,pair.imag) 
    )
    # This object is referenced from the Leaf node of each member
    for pair in seq:
        for x in (pair.real,pair.imag):
            x._node.ensemble = ensemble
        